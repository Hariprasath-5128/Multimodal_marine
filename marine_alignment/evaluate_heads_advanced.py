"""
evaluate_heads_advanced.py — Advanced Evaluation of Projection Heads
===================================================================
Features:
  1. Loads raw test samples (images, texts, audios).
  2. Extracts raw image features using Test-Time Augmentation (TTA)
     (averaging features of original and horizontally flipped images).
  3. Projects raw features using the target projection head (Open or Closed).
  4. Runs 5-fold stratified cross-validation on two methods:
     - Baseline (standard projected features).
     - Query Expansion (QE) (refines query embedding using top gallery retrieved item).
  5. Outputs side-by-side comparison tables to clearly show the accuracy boost.
  6. Saves a JSON report containing advanced metrics.
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

# Suppress sklearn UserWarning
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
def _encode_raw_image(bb, tfm, img_path: str, device: str, use_tta: bool = True) -> torch.Tensor:
    from PIL import Image
    import pillow_avif
    img = Image.open(img_path).convert("RGB")
    
    # Original image
    t_orig = tfm(img).unsqueeze(0).to(device)
    feat_orig = bb(t_orig).squeeze(0)
    
    if use_tta:
        # Flipped image (horizontal flip)
        img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
        t_flip = tfm(img_flip).unsqueeze(0).to(device)
        feat_flip = bb(t_flip).squeeze(0)
        
        # Average the features
        mean_feat = 0.5 * feat_orig + 0.5 * feat_flip
        return mean_feat.cpu()
    else:
        return feat_orig.cpu()


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
# Recall Computation & Post-Processing Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_recall_from_similarity(
    sim: torch.Tensor,             # [Q, T] similarity matrix
    query_labels: torch.Tensor,    # [Q]
    target_labels: torch.Tensor,   # [T]
    ks: tuple = (1, 5, 10),
) -> dict[int, float]:
    if sim.size(0) == 0 or sim.size(1) == 0:
        return {k: 0.0 for k in ks}
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


def apply_query_expansion(
    query_embs: torch.Tensor,      # [Q, D]
    gallery_embs: torch.Tensor,    # [T, D]
    qe_weight: float = 0.20
) -> torch.Tensor:                 # [Q, D]
    """
    Update each query embedding by mixing in the top-1 retrieved gallery item's embedding.
    """
    if query_embs.size(0) == 0 or gallery_embs.size(0) == 0:
        return query_embs
    sim = torch.matmul(query_embs, gallery_embs.T)           # [Q, T]
    top1_idx = sim.argmax(dim=1)                             # [Q]
    top1_gallery_embs = gallery_embs[top1_idx]               # [Q, D]
    
    expanded = query_embs + qe_weight * top1_gallery_embs
    return F.normalize(expanded, p=2, dim=-1)


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

def run_evaluation(model_type: str, raw_data: dict, device: str, use_tta: bool, qe_weight: float):
    KS = (1, 5, 10)
    
    print("\n" + "=" * 64)
    print(f"  Evaluating Model: {model_type.upper()} (TTA={use_tta}, QE_Weight={qe_weight})")
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
    
    baseline_folds = []
    advanced_folds = []
    aud_species_ids = set(aud_labels.tolist()) if aud_labels.numel() > 0 else set()

    for fold_idx, (_, val_idx) in enumerate(skf.split(img_embs, img_labels)):
        fold_img_embs = img_embs[val_idx]
        fold_img_labels = img_labels[val_idx]

        # ── Method 1: Baseline ──
        # Image -> Text
        sim_t_base = torch.matmul(fold_img_embs, txt_embs.T)
        t_rec_base = _compute_recall_from_similarity(sim_t_base, fold_img_labels, txt_labels, KS)
        t_comp_base = _composite(t_rec_base)

        # Image -> Audio
        if aud_species_ids:
            aud_mask = torch.tensor([lbl.item() in aud_species_ids for lbl in fold_img_labels])
            fold_img_embs_aud = fold_img_embs[aud_mask]
            fold_img_labels_aud = fold_img_labels[aud_mask]
        else:
            fold_img_embs_aud = torch.zeros(0, 768)
            fold_img_labels_aud = torch.zeros(0, dtype=torch.long)

        sim_a_base = torch.matmul(fold_img_embs_aud, aud_embs.T)
        a_rec_base = _compute_recall_from_similarity(sim_a_base, fold_img_labels_aud, aud_labels, KS)
        a_comp_base = _composite(a_rec_base)
        
        overall_base = (t_comp_base + a_comp_base) / 2
        baseline_folds.append({
            "t_rec": t_rec_base, "t_comp": t_comp_base,
            "a_rec": a_rec_base, "a_comp": a_comp_base,
            "overall": overall_base
        })

        # ── Method 2: Advanced (with Query Expansion) ──
        # Apply Query Expansion in the shared latent space
        fold_img_embs_qe_txt = apply_query_expansion(fold_img_embs, txt_embs, qe_weight)
        fold_img_embs_qe_aud = apply_query_expansion(fold_img_embs_aud, aud_embs, qe_weight)

        # Image -> Text (QE)
        sim_t_qe = torch.matmul(fold_img_embs_qe_txt, txt_embs.T)
        t_rec_qe = _compute_recall_from_similarity(sim_t_qe, fold_img_labels, txt_labels, KS)
        t_comp_qe = _composite(t_rec_qe)

        # Image -> Audio (QE)
        sim_a_qe = torch.matmul(fold_img_embs_qe_aud, aud_embs.T)
        a_rec_qe = _compute_recall_from_similarity(sim_a_qe, fold_img_labels_aud, aud_labels, KS)
        a_comp_qe = _composite(a_rec_qe)

        overall_qe = (t_comp_qe + a_comp_qe) / 2
        advanced_folds.append({
            "t_rec": t_rec_qe, "t_comp": t_comp_qe,
            "a_rec": a_rec_qe, "a_comp": a_comp_qe,
            "overall": overall_qe
        })

    # 5. Summarize statistics
    def summarize(folds_list):
        summary = {}
        for d in ["img2txt", "img2aud"]:
            rec_key = "t_rec" if d == "img2txt" else "a_rec"
            comp_key = "t_comp" if d == "img2txt" else "a_comp"
            
            summary[d] = {}
            for k in KS:
                vals = [f[rec_key][k] for f in folds_list]
                summary[d][f"R@{k}"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            
            vals_comp = [f[comp_key] for f in folds_list]
            summary[f"{d}_composite"] = {"mean": float(np.mean(vals_comp)), "std": float(np.std(vals_comp))}
            
        vals_overall = [f["overall"] for f in folds_list]
        summary["overall_composite"] = {"mean": float(np.mean(vals_overall)), "std": float(np.std(vals_overall))}
        return summary

    base_summary = summarize(baseline_folds)
    adv_summary = summarize(advanced_folds)

    # 6. Print Side-by-Side Comparison
    print("\n" + "=" * 80)
    print(f"  5-FOLD COMPARISON REPORT: {model_type.upper()}")
    print("=" * 80)
    print(f"  {'Metric':<25} | {'Baseline (TTA)':<24} | {'Advanced (TTA + QE)':<24}")
    print("-" * 80)
    
    # Image -> Text
    print("  Image -> Text:")
    for k in KS:
        m = f"img2txt"
        b_mu, b_std = base_summary[m][f"R@{k}"]["mean"], base_summary[m][f"R@{k}"]["std"]
        a_mu, a_std = adv_summary[m][f"R@{k}"]["mean"], adv_summary[m][f"R@{k}"]["std"]
        diff = a_mu - b_mu
        print(f"    R@{k:<3}                 | {b_mu:.4f} \u00b1 {b_std:.4f}           | {a_mu:.4f} \u00b1 {a_std:.4f}  ({diff:+.1%})")
    
    b_comp = base_summary["img2txt_composite"]["mean"]
    b_c_std = base_summary["img2txt_composite"]["std"]
    a_comp = adv_summary["img2txt_composite"]["mean"]
    a_c_std = adv_summary["img2txt_composite"]["std"]
    print(f"    Composite           | {b_comp:.4f} \u00b1 {b_c_std:.4f}           | {a_comp:.4f} \u00b1 {a_c_std:.4f}  ({a_comp - b_comp:+.1%})")
    print("-" * 80)

    # Image -> Audio
    print("  Image -> Audio:")
    for k in KS:
        m = f"img2aud"
        b_mu, b_std = base_summary[m][f"R@{k}"]["mean"], base_summary[m][f"R@{k}"]["std"]
        a_mu, a_std = adv_summary[m][f"R@{k}"]["mean"], adv_summary[m][f"R@{k}"]["std"]
        diff = a_mu - b_mu
        print(f"    R@{k:<3}                 | {b_mu:.4f} \u00b1 {b_std:.4f}           | {a_mu:.4f} \u00b1 {a_std:.4f}  ({diff:+.1%})")
        
    b_comp = base_summary["img2aud_composite"]["mean"]
    b_c_std = base_summary["img2aud_composite"]["std"]
    a_comp = adv_summary["img2aud_composite"]["mean"]
    a_c_std = adv_summary["img2aud_composite"]["std"]
    print(f"    Composite           | {b_comp:.4f} \u00b1 {b_c_std:.4f}           | {a_comp:.4f} \u00b1 {a_c_std:.4f}  ({a_comp - b_comp:+.1%})")
    print("-" * 80)

    # Overall Composite
    b_ov = base_summary["overall_composite"]["mean"]
    b_o_std = base_summary["overall_composite"]["std"]
    a_ov = adv_summary["overall_composite"]["mean"]
    a_o_std = adv_summary["overall_composite"]["std"]
    print(f"  Overall Composite     | {b_ov:.4f} \u00b1 {b_o_std:.4f}           | {a_ov:.4f} \u00b1 {a_o_std:.4f}  ({a_ov - b_ov:+.1%})")
    print("=" * 80)

    # 7. Save JSON report
    report = {
        "model_type": model_type,
        "checkpoint_epoch": epoch,
        "checkpoint_val_composite": val_comp,
        "config": {
            "use_tta": use_tta,
            "qe_weight": qe_weight
        },
        "baseline_summary": base_summary,
        "advanced_summary": adv_summary
    }
    
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"test_evaluation_report_advanced_{model_type}.json"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {report_path}\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="Run advanced evaluation on projection heads.")
    parser.add_argument("--model", type=str, choices=["open", "closed"], required=True,
                        help="Which model to evaluate: open or closed")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device to use for model forward passes")
    parser.add_argument("--no_tta", action="store_true", help="Disable Test-Time Augmentation (TTA) for images")
    parser.add_argument("--qe_weight", type=float, default=0.20, help="Weight parameter for Query Expansion (QE)")
    args = parser.parse_args()

    print("=" * 64)
    print("  Marine Multimodal Alignment -- Advanced Evaluation Script")
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
    print(f"  Test-Time Augmentation (TTA) = {not args.no_tta}")
    print("-" * 64)
    
    # Image
    print("Encoding images...")
    raw_images = []
    for sp, img_path in tqdm(img_samples, desc="Images", unit="img"):
        try:
            raw_feat = _encode_raw_image(img_enc, img_tfm, img_path, args.device, use_tta=(not args.no_tta))
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

    # 4. Run evaluation
    run_evaluation(args.model, raw_data, args.device, use_tta=(not args.no_tta), qe_weight=args.qe_weight)

    print("\nAll evaluations complete!")


if __name__ == "__main__":
    main()
