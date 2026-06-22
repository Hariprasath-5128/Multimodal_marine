"""
evaluate_heads_5fold.py — 5-Fold Stratified Cross-Validation for Open & Closed Projection Heads
============================================================================================
Features:
  1. Loads raw test samples (images, texts, audios) and extracts raw features EXACTLY ONCE.
  2. Loads target model checkpoints (Closed and/or Open).
  3. Projects raw features using projection heads.
  4. Splits image queries into 5 folds using StratifiedKFold (stratified by species label).
  5. Evaluates retrieval (Image->Text, Image->Audio) on each fold.
  6. Reports fold-by-fold results, mean, and standard deviation for all recall metrics.
  7. Saves a JSON report containing fold details and summary statistics.
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

# Suppress sklearn UserWarning regarding n_splits larger than class member count
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Set up module paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CHECKPOINT_PATH_CLOSED, CHECKPOINT_PATH_OPEN, DEVICE,
    IMAGE_MODEL_DIR, TEXT_MODEL_DIR, AUDIO_MODEL_PATH, AST_PRETRAINED_ID,
    AUDIO_SAMPLE_RATE, AUDIO_TARGET_SECONDS,
    RECALL_WEIGHT_R1, RECALL_WEIGHT_R5, RECALL_WEIGHT_R10,
)

from dataset import (
    get_test_image_split,
    get_test_text_split,
    get_test_audio_split,
)


# ─────────────────────────────────────────────────────────────────────────────
# Encoders & Raw Feature Loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_image_encoder(device: str):
    import timm
    from torchvision import transforms
    from pathlib import Path

    ckpts = sorted(Path(IMAGE_MODEL_DIR).glob("*_seed42.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No *_seed42.pth found in {IMAGE_MODEL_DIR}")
    ckpt = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    bb_name = ckpt.get("backbone", "convnextv2_base")
    img_size = ckpt.get("img_size", 288)

    model = timm.create_model(bb_name, pretrained=False, num_classes=0, global_pool="avg")
    bb_sd = {k[len("backbone."):]: v for k, v in ckpt["model_state_dict"].items() if k.startswith("backbone.")}
    model.load_state_dict(bb_sd, strict=True)
    model.eval().to(device)

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return model, tfm


@torch.no_grad()
def _encode_raw_image(bb, tfm, img_path: str, device: str) -> torch.Tensor:
    from PIL import Image
    import pillow_avif
    img = Image.open(img_path).convert("RGB")
    t = tfm(img).unsqueeze(0).to(device)
    return bb(t).squeeze(0).cpu()


def _load_text_encoder(device: str):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(TEXT_MODEL_DIR, device=device)
    m.eval()
    return m


def _encode_raw_texts(text_model, file_paths: list[str]) -> torch.Tensor | None:
    """Read and mean-pool all text documents for one species."""
    texts = []
    for fp in file_paths:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                c = f.read().strip()
            if len(c.split()) > 5:
                texts.append(c)
        except Exception:
            pass
    if not texts:
        return None
    embs = text_model.encode(
        texts, convert_to_tensor=True,
        show_progress_bar=False, normalize_embeddings=True
    )
    mean_emb = embs.mean(dim=0)
    return F.normalize(mean_emb, p=2, dim=0).cpu()


def _load_audio_encoder(device: str):
    from transformers import ASTForAudioClassification, ASTFeatureExtractor
    fe = ASTFeatureExtractor.from_pretrained(AST_PRETRAINED_ID)
    sd = torch.load(AUDIO_MODEL_PATH, map_location="cpu", weights_only=False)
    clf_key = next(
        (k for k in sd if "classifier" in k and "weight" in k and sd[k].ndim == 2),
        None
    )
    num_labels = sd[clf_key].shape[0] if clf_key else 32
    model = ASTForAudioClassification.from_pretrained(
        AST_PRETRAINED_ID, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    hook = {"hidden": None}
    def _fwd_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        hook["hidden"] = h.mean(dim=1)
    model.audio_spectrogram_transformer.encoder.layer[-1].register_forward_hook(_fwd_hook)

    return model, fe, hook


@torch.no_grad()
def _encode_raw_audio_file(model, fe, hook, wav_path: str, device: str) -> torch.Tensor | None:
    import torchaudio
    try:
        wav, sr = torchaudio.load(wav_path)
        wav = wav.mean(dim=0, keepdim=True)
        if sr != AUDIO_SAMPLE_RATE:
            wav = torchaudio.transforms.Resample(sr, AUDIO_SAMPLE_RATE)(wav)
        target = AUDIO_TARGET_SECONDS * AUDIO_SAMPLE_RATE
        if wav.shape[1] > target:
            wav = wav[:, :target]
        else:
            wav = F.pad(wav, (0, target - wav.shape[1]))
        inp = fe(wav.squeeze().numpy(), sampling_rate=AUDIO_SAMPLE_RATE, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        hook["hidden"] = None
        model(**inp)
        feat = hook["hidden"]
        return feat.squeeze(0).cpu() if feat is not None else None
    except Exception:
        return None


def _encode_raw_audio_species(model, fe, hook, wav_paths: list[str], device: str) -> torch.Tensor | None:
    """Mean-pool all wav clips for one species."""
    embs = [_encode_raw_audio_file(model, fe, hook, w, device) for w in wav_paths]
    embs = [e for e in embs if e is not None]
    if not embs:
        return None
    mean_emb = torch.stack(embs).mean(dim=0)
    return F.normalize(mean_emb, p=2, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Recall Computation Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_recall(
    query_embs:    torch.Tensor,   # [Q, D]
    query_labels:  torch.Tensor,   # [Q]
    target_embs:   torch.Tensor,   # [T, D]
    target_labels: torch.Tensor,   # [T]
    ks: tuple = (1, 5, 10),
) -> dict[int, float]:
    if query_embs.size(0) == 0 or target_embs.size(0) == 0:
        return {k: 0.0 for k in ks}
    sim = torch.matmul(query_embs, target_embs.T)           # [Q, T]
    results = {}
    for k in ks:
        eff_k       = min(k, sim.size(1))
        topk_idx    = sim.topk(eff_k, dim=1).indices         # [Q, k]
        topk_labels = target_labels[topk_idx]                 # [Q, k]
        correct     = (topk_labels == query_labels.unsqueeze(1)).any(dim=1)
        results[k]  = correct.float().mean().item()
    return results


def _composite(r: dict) -> float:
    return (RECALL_WEIGHT_R1 * r.get(1, 0.0)
            + RECALL_WEIGHT_R5  * r.get(5, 0.0)
            + RECALL_WEIGHT_R10 * r.get(10, 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Model Pipeline Loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_pipeline(model_type: str, device: str):
    """Dynamically import and instantiate the correct model architecture."""
    if model_type == "closed":
        import models_closed
        pipeline = models_closed.MarineImageBindPipeline().to(device)
        ckpt_path = CHECKPOINT_PATH_CLOSED
    elif model_type == "open":
        import models_open
        pipeline = models_open.MarineImageBindPipeline().to(device)
        ckpt_path = CHECKPOINT_PATH_OPEN
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pipeline.load_state_dict(ckpt["model_state"], strict=False)
    pipeline.eval()
    return pipeline, ckpt["epoch"], ckpt["composite_score"]


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation Function
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(model_type: str, raw_data: dict, device: str):
    KS = (1, 5, 10)
    
    print("\n" + "=" * 64)
    print(f"  Running 5-Fold Validation: {model_type.upper()} Model")
    print("=" * 64)

    # 1. Load pipeline
    try:
        pipeline, epoch, val_comp = _load_pipeline(model_type, device)
        print(f"[Loaded Checkpoint] Epoch {epoch}, Val composite = {val_comp:.4f}")
    except Exception as e:
        print(f"[ERROR] Failed to load {model_type} pipeline: {e}")
        return None

    # 2. Extract mappings
    sp2id = raw_data["sp2id"]
    raw_images = raw_data["images"]
    raw_texts = raw_data["texts"]
    raw_audios = raw_data["audios"]

    # 3. Project Raw Features using the Projection Heads
    print("Projecting raw features through heads...")
    
    # Image
    img_emb_list = []
    img_lbl_list = []
    for sp, raw_feat in raw_images:
        with torch.no_grad():
            if model_type == "closed":
                proj = pipeline.project_image(raw_feat.unsqueeze(0).to(device)).squeeze(0).cpu()
            else:
                proj = pipeline.image_head(raw_feat.unsqueeze(0).to(device)).squeeze(0).cpu()
        img_emb_list.append(proj)
        img_lbl_list.append(sp2id[sp])
    
    img_embs = torch.stack(img_emb_list)
    img_labels = torch.tensor(img_lbl_list)

    # Text
    txt_emb_list = []
    txt_lbl_list = []
    for sp, raw_feat in raw_texts.items():
        with torch.no_grad():
            proj = pipeline.text_head(raw_feat.unsqueeze(0).to(device)).squeeze(0).cpu()
        txt_emb_list.append(proj)
        txt_lbl_list.append(sp2id[sp])

    txt_embs = torch.stack(txt_emb_list) if txt_emb_list else torch.zeros(0, 768)
    txt_labels = torch.tensor(txt_lbl_list) if txt_lbl_list else torch.zeros(0, dtype=torch.long)

    # Audio
    aud_emb_list = []
    aud_lbl_list = []
    for sp, raw_feat in raw_audios.items():
        with torch.no_grad():
            proj = pipeline.audio_head(raw_feat.unsqueeze(0).to(device)).squeeze(0).cpu()
        aud_emb_list.append(proj)
        aud_lbl_list.append(sp2id[sp])

    aud_embs = torch.stack(aud_emb_list) if aud_emb_list else torch.zeros(0, 768)
    aud_labels = torch.tensor(aud_lbl_list) if aud_lbl_list else torch.zeros(0, dtype=torch.long)

    # 4. Perform 5-Fold Stratified K-Fold Split on Image Queries
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_metrics = []

    # Identify audio species subset for correct audio query filtering
    aud_species_ids = set(aud_labels.tolist()) if aud_labels.numel() > 0 else set()

    for fold_idx, (_, val_idx) in enumerate(skf.split(img_embs, img_labels)):
        fold_img_embs = img_embs[val_idx]
        fold_img_labels = img_labels[val_idx]

        # Calculate Image -> Text Recall for the fold
        t_recall = _compute_recall(fold_img_embs, fold_img_labels, txt_embs, txt_labels, KS)
        t_comp = _composite(t_recall)

        # Filter queries for Image -> Audio (only queries whose labels exist in audio gallery)
        if aud_species_ids:
            aud_query_mask = torch.tensor([lbl.item() in aud_species_ids for lbl in fold_img_labels])
            fold_img_embs_aud = fold_img_embs[aud_query_mask]
            fold_img_labels_aud = fold_img_labels[aud_query_mask]
        else:
            fold_img_embs_aud = torch.zeros(0, 768)
            fold_img_labels_aud = torch.zeros(0, dtype=torch.long)

        a_recall = _compute_recall(fold_img_embs_aud, fold_img_labels_aud, aud_embs, aud_labels, KS)
        a_comp = _composite(a_recall)

        overall = (t_comp + a_comp) / 2

        metrics = {
            "fold": fold_idx + 1,
            "img2txt": {f"R@{k}": t_recall[k] for k in KS},
            "img2txt_composite": t_comp,
            "img2aud": {f"R@{k}": a_recall[k] for k in KS},
            "img2aud_composite": a_comp,
            "overall_composite": overall,
            "n_queries_txt": int(fold_img_embs.size(0)),
            "n_queries_aud": int(fold_img_embs_aud.size(0)),
        }
        fold_metrics.append(metrics)

        print(f"  Fold {fold_idx + 1}: Img2Txt Comp = {t_comp:.4f} | Img2Aud Comp = {a_comp:.4f} | Overall Comp = {overall:.4f}")

    # 5. Aggregate metrics across folds
    summary = {}
    
    def get_stats(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    # Image -> Text
    summary["img2txt"] = {
        f"R@{k}": get_stats([m["img2txt"][f"R@{k}"] for m in fold_metrics]) for k in KS
    }
    summary["img2txt_composite"] = get_stats([m["img2txt_composite"] for m in fold_metrics])

    # Image -> Audio
    summary["img2aud"] = {
        f"R@{k}": get_stats([m["img2aud"][f"R@{k}"] for m in fold_metrics]) for k in KS
    }
    summary["img2aud_composite"] = get_stats([m["img2aud_composite"] for m in fold_metrics])

    # Overall
    summary["overall_composite"] = get_stats([m["overall_composite"] for m in fold_metrics])

    # 6. Print Summary
    print("\n" + "=" * 64)
    print(f"  5-FOLD CROSS-VALIDATION SUMMARY: {model_type.upper()}")
    print("=" * 64)
    
    print("\n  Image -> Text Retrieval:")
    for k in KS:
        mu = summary["img2txt"][f"R@{k}"]["mean"]
        std = summary["img2txt"][f"R@{k}"]["std"]
        print(f"    R@{k:<3}: {mu:.4f} \u00b1 {std:.4f}  ({mu*100:.1f}%)")
    print(f"    Composite : {summary['img2txt_composite']['mean']:.4f} \u00b1 {summary['img2txt_composite']['std']:.4f}")

    print("\n  Image -> Audio Retrieval:")
    for k in KS:
        mu = summary["img2aud"][f"R@{k}"]["mean"]
        std = summary["img2aud"][f"R@{k}"]["std"]
        print(f"    R@{k:<3}: {mu:.4f} \u00b1 {std:.4f}  ({mu*100:.1f}%)")
    print(f"    Composite : {summary['img2aud_composite']['mean']:.4f} \u00b1 {summary['img2aud_composite']['std']:.4f}")

    print(f"\n  Overall Composite Score : {summary['overall_composite']['mean']:.4f} \u00b1 {summary['overall_composite']['std']:.4f}")
    print("=" * 64)

    # 7. Save JSON report
    report = {
        "model_type": model_type,
        "checkpoint_epoch": epoch,
        "checkpoint_val_composite": val_comp,
        "folds": fold_metrics,
        "summary": summary
    }
    
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"test_evaluation_report_5fold_{model_type}.json"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {report_path}\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="Run 5-fold cross-validation on projection heads.")
    parser.add_argument("--model", type=str, choices=["open", "closed", "both"], default="both",
                        help="Which model(s) to evaluate: open, closed, or both")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device to use for model forward passes")
    args = parser.parse_args()

    print("=" * 64)
    print("  Marine Multimodal Alignment -- 5-Fold Cross-Validation")
    print("=" * 64)

    # 1. Resolve test splits using dataset.py
    print("\nResolving test splits from dataset.py ...")
    img_samples, img_meta = get_test_image_split()
    txt_files, txt_meta = get_test_text_split()
    aud_files, aud_meta = get_test_audio_split()

    print(f"  Image  : {img_meta['n_images']:>5} images, {img_meta['n_species']:>3} species")
    print(f"  Text   : {txt_meta['n_docs']:>5} docs,   {txt_meta['n_species']:>3} species")
    print(f"  Audio  : {len(aud_files):>5} species with audio")

    # Build unique species label maps
    all_species = sorted(
        set(sp for sp, _ in img_samples)
        | set(txt_files.keys())
        | set(aud_files.keys())
    )
    sp2id = {sp: i for i, sp in enumerate(all_species)}

    # 2. Load frozen encoders
    print("\nLoading frozen encoders ...")
    img_enc, img_tfm = _load_image_encoder(args.device)
    print("  [Image] Loaded ConvNeXtV2 model and transforms")
    
    txt_enc = _load_text_encoder(args.device)
    print("  [Text]  Loaded SentenceTransformer")
    
    aud_enc, aud_fe, aud_hook = _load_audio_encoder(args.device)
    print("  [Audio] Loaded AST model")

    # 3. Extract Raw Features (Bottle-neck operation, done ONCE)
    print("\n" + "-" * 64)
    print("  STEP 1: Extracting raw features (heavy computation)...")
    print("-" * 64)
    
    # Image
    print("Encoding images...")
    raw_images = []
    for sp, img_path in tqdm(img_samples, desc="Images", unit="img"):
        try:
            raw_feat = _encode_raw_image(img_enc, img_tfm, img_path, args.device)
            raw_images.append((sp, raw_feat))
        except Exception as e:
            print(f"  [WARN] Skipped {img_path}: {e}")

    # Text
    print("Encoding text docs...")
    raw_texts = {}
    for sp, fpaths in tqdm(txt_files.items(), desc="Text", unit="species"):
        if sp in sp2id:
            raw_feat = _encode_raw_texts(txt_enc, fpaths)
            if raw_feat is not None:
                raw_texts[sp] = raw_feat

    # Audio
    print("Encoding audio files...")
    raw_audios = {}
    for sp, wav_paths in tqdm(aud_files.items(), desc="Audio", unit="species"):
        if sp in sp2id:
            raw_feat = _encode_raw_audio_species(aud_enc, aud_fe, aud_hook, wav_paths, args.device)
            if raw_feat is not None:
                raw_audios[sp] = raw_feat

    # Package raw data
    raw_data = {
        "sp2id": sp2id,
        "images": raw_images,
        "texts": raw_texts,
        "audios": raw_audios
    }

    # Free memory occupied by frozen encoders to make room
    del img_enc, txt_enc, aud_enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Run cross-validation evaluations
    models_to_run = []
    if args.model in ["closed", "both"]:
        models_to_run.append("closed")
    if args.model in ["open", "both"]:
        models_to_run.append("open")

    for model_type in models_to_run:
        run_evaluation(model_type, raw_data, args.device)

    print("\nAll evaluations complete!")


if __name__ == "__main__":
    main()
