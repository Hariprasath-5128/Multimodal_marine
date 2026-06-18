"""
evaluate_heads.py — Test-Set Evaluation of the Trained Projection Heads
=========================================================================
All data-split decisions are delegated to dataset.py.
This script only handles:
  1. Loading the trained checkpoint.
  2. Loading the three frozen encoders on-the-fly.
  3. Encoding raw test samples through the frozen encoder + trained head.
  4. Computing Recall@1/5/10 for Image->Text and Image->Audio retrieval.
  5. Printing a results table and saving a JSON report.

Data Sources (resolved by dataset.py, with automatic fallbacks)
---------------------------------------------------------------
  Image  : image_dataset/test/           -> fallback: image_dataset/train/
  Text   : text_dataset/test/expanded_test_dataset/ -> fallback: train/
  Audio  : audio_split/val/              -> per-species fallback: train/

Usage
-----
    python marine_alignment/evaluate_heads.py
    python marine_alignment/evaluate_heads.py --device cpu
    python marine_alignment/evaluate_heads.py --per_species
"""

import os
import sys
import json
import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CHECKPOINT_PATH, DEVICE,
    IMAGE_MODEL_DIR, TEXT_MODEL_DIR, AUDIO_MODEL_PATH, AST_PRETRAINED_ID,
    AUDIO_SAMPLE_RATE, AUDIO_TARGET_SECONDS,
    RECALL_WEIGHT_R1, RECALL_WEIGHT_R5, RECALL_WEIGHT_R10,
)

# All split logic lives in dataset.py
from dataset import (
    get_test_image_split,
    get_test_text_split,
    get_test_audio_split,
)

from models import MarineImageBindPipeline

REPORT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "test_evaluation_report.json"
)


# ─────────────────────────────────────────────────────────────────────────────
# Frozen encoder loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_image_encoder(device: str):
    import timm
    from torchvision import transforms
    from pathlib import Path

    ckpts = sorted(Path(IMAGE_MODEL_DIR).glob("*_seed42.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No *_seed42.pth in {IMAGE_MODEL_DIR}")
    ck = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    bb_name  = ck.get("backbone", "convnextv2_base")
    img_size = ck.get("img_size", 288)

    backbone = timm.create_model(bb_name, pretrained=False,
                                  num_classes=0, global_pool="avg")
    bb_sd = {k[len("backbone."):]: v
             for k, v in ck["model_state_dict"].items()
             if k.startswith("backbone.")}
    backbone.load_state_dict(bb_sd, strict=True)
    backbone.eval().to(device)

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return backbone, tfm


@torch.no_grad()
def _encode_image(backbone, tfm, img_path: str, device: str) -> torch.Tensor:
    from PIL import Image
    import pillow_avif
    img = Image.open(img_path).convert("RGB")
    t   = tfm(img).unsqueeze(0).to(device)
    return backbone(t).squeeze(0).cpu()


def _load_text_encoder(device: str):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(TEXT_MODEL_DIR, device=device)
    m.eval()
    return m


def _encode_texts(text_model, file_paths: list[str]) -> torch.Tensor | None:
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
    clf_key    = next(
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
    model.audio_spectrogram_transformer.encoder.layer[-1] \
        .register_forward_hook(_fwd_hook)

    return model, fe, hook


@torch.no_grad()
def _encode_audio_file(model, fe, hook, wav_path: str,
                        device: str) -> torch.Tensor | None:
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
        inp  = fe(wav.squeeze().numpy(), sampling_rate=AUDIO_SAMPLE_RATE,
                  return_tensors="pt")
        inp  = {k: v.to(device) for k, v in inp.items()}
        hook["hidden"] = None
        model(**inp)
        feat = hook["hidden"]
        return feat.squeeze(0).cpu() if feat is not None else None
    except Exception:
        return None


def _encode_audio_species(model, fe, hook, wav_paths: list[str],
                            device: str) -> torch.Tensor | None:
    """Mean-pool all wav clips for one species."""
    embs = [_encode_audio_file(model, fe, hook, w, device) for w in wav_paths]
    embs = [e for e in embs if e is not None]
    if not embs:
        return None
    mean_emb = torch.stack(embs).mean(dim=0)
    return F.normalize(mean_emb, p=2, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Recall computation
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
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = args.device
    KS     = (1, 5, 10)

    print("=" * 64)
    print("  Marine Multimodal Alignment -- Head Evaluation")
    print("=" * 64)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n[ERROR] No checkpoint at:\n  {CHECKPOINT_PATH}")
        print("Run train.py first.\n")
        sys.exit(1)

    # ── Load trained pipeline ─────────────────────────────────────────────────
    pipeline = MarineImageBindPipeline().to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    pipeline.load_state_dict(ckpt["model_state"])
    pipeline.eval()
    print(f"\n[Checkpoint] Epoch {ckpt['epoch']}, "
          f"Val composite = {ckpt['composite_score']:.4f}")

    # ── Resolve test splits (dataset.py decides source + fallback) ────────────
    print("\nResolving test splits from dataset.py ...")

    img_samples, img_meta = get_test_image_split()
    txt_files,   txt_meta = get_test_text_split()
    aud_files,   aud_meta = get_test_audio_split()

    print(f"  Image  : {img_meta['n_images']:>5} images, "
          f"{img_meta['n_species']:>3} species  "
          f"[{img_meta['source']}]")
    print(f"  Text   : {txt_meta['n_docs']:>5} docs,   "
          f"{txt_meta['n_species']:>3} species  "
          f"[{txt_meta['source']}]")
    print(f"  Audio  : {len(aud_files):>5} species  "
          f"[val={aud_meta['n_val']}, "
          f"train-fallback={aud_meta['n_train_fallback']}, "
          f"missing={aud_meta['n_missing']}]")

    # Build a global species->int label map from everything we have
    all_species = sorted(
        set(sp for sp, _ in img_samples)
        | set(txt_files.keys())
        | set(aud_files.keys())
    )
    sp2id = {sp: i for i, sp in enumerate(all_species)}

    # ── Load frozen encoders ──────────────────────────────────────────────────
    print("\nLoading frozen encoders ...")
    print("  [Image] ConvNeXtV2 backbone ...")
    img_bb, img_tfm = _load_image_encoder(device)

    print("  [Text]  SentenceTransformer ...")
    txt_enc = _load_text_encoder(device)

    print("  [Audio] AST model ...")
    aud_enc, aud_fe, aud_hook = _load_audio_encoder(device)

    # ── 1. Build TEXT gallery ─────────────────────────────────────────────────
    print("\nEncoding test TEXT gallery ...")
    txt_emb_list, txt_lbl_list = [], []

    for sp, fpaths in tqdm(txt_files.items(), desc="text", unit="species"):
        if sp not in sp2id:
            continue
        raw = _encode_texts(txt_enc, fpaths)
        if raw is None:
            continue
        with torch.no_grad():
            proj = pipeline.text_head(raw.unsqueeze(0).to(device)).squeeze(0).cpu()
        txt_emb_list.append(proj)
        txt_lbl_list.append(sp2id[sp])

    txt_embs   = torch.stack(txt_emb_list)    if txt_emb_list else torch.zeros(0, 512)
    txt_labels = torch.tensor(txt_lbl_list)   if txt_lbl_list else torch.zeros(0, dtype=torch.long)
    print(f"  Text gallery: {txt_embs.size(0)} species")

    # ── 2. Build AUDIO gallery ────────────────────────────────────────────────
    print("\nEncoding test AUDIO gallery ...")
    aud_emb_list, aud_lbl_list = [], []

    for sp, wav_paths in tqdm(aud_files.items(), desc="audio", unit="species"):
        if sp not in sp2id:
            continue
        raw = _encode_audio_species(aud_enc, aud_fe, aud_hook, wav_paths, device)
        if raw is None:
            continue
        with torch.no_grad():
            proj = pipeline.audio_head(raw.unsqueeze(0).to(device)).squeeze(0).cpu()
        aud_emb_list.append(proj)
        aud_lbl_list.append(sp2id[sp])

    aud_embs   = torch.stack(aud_emb_list)   if aud_emb_list else torch.zeros(0, 512)
    aud_labels = torch.tensor(aud_lbl_list)  if aud_lbl_list else torch.zeros(0, dtype=torch.long)
    print(f"  Audio gallery: {aud_embs.size(0)} species")

    # ── 3. Build IMAGE queries ────────────────────────────────────────────────
    print("\nEncoding test IMAGE queries ...")
    img_emb_list, img_lbl_list, img_sp_list = [], [], []

    for sp, img_path in tqdm(img_samples, desc="images", unit="img"):
        if sp not in sp2id:
            continue
        try:
            raw  = _encode_image(img_bb, img_tfm, img_path, device)
            with torch.no_grad():
                proj = pipeline.image_head(raw.unsqueeze(0).to(device)).squeeze(0).cpu()
            img_emb_list.append(proj)
            img_lbl_list.append(sp2id[sp])
            img_sp_list.append(sp)
        except Exception as e:
            print(f"  [WARN] Skipped {img_path}: {e}")

    img_embs   = torch.stack(img_emb_list)   if img_emb_list else torch.zeros(0, 512)
    img_labels = torch.tensor(img_lbl_list)  if img_lbl_list else torch.zeros(0, dtype=torch.long)
    print(f"  Image queries: {img_embs.size(0)} images, "
          f"{len(set(img_lbl_list))} species")

    # ── 4. Compute Recall ─────────────────────────────────────────────────────
    print("\nComputing Recall@K ...")
    txt_recall = _compute_recall(img_embs, img_labels, txt_embs, txt_labels, KS)
    
    # For audio recall, only score images whose species is IN the audio gallery.
    aud_species_ids = set(aud_labels.tolist()) if aud_labels.numel() > 0 else set()
    if aud_species_ids:
        aud_query_mask = torch.tensor(
            [lbl.item() in aud_species_ids for lbl in img_labels]
        )
        img_embs_aud   = img_embs[aud_query_mask]
        img_labels_aud = img_labels[aud_query_mask]
    else:
        img_embs_aud, img_labels_aud = img_embs, img_labels

    print(f"  Audio recall computed over {img_embs_aud.size(0)} images "
          f"({len(aud_species_ids)} species with audio / "
          f"{img_embs.size(0)} total images)")

    aud_recall = _compute_recall(img_embs_aud, img_labels_aud, aud_embs, aud_labels, KS)

    txt_comp = _composite(txt_recall)
    aud_comp = _composite(aud_recall)
    overall  = (txt_comp + aud_comp) / 2

    # ── 5. Per-species breakdown ──────────────────────────────────────────────
    sp_breakdown = {}
    for sp in all_species:
        sp_id    = sp2id.get(sp, -1)
        sp_mask  = (img_labels == sp_id)
        if sp_mask.sum() == 0:
            continue
        q_emb = img_embs[sp_mask]
        q_lbl = img_labels[sp_mask]
        r_txt = _compute_recall(q_emb, q_lbl, txt_embs, txt_labels, KS) \
                if txt_embs.size(0) > 0 else {k: 0.0 for k in KS}
        r_aud = _compute_recall(q_emb, q_lbl, aud_embs, aud_labels, KS) \
                if aud_embs.size(0) > 0 else {k: 0.0 for k in KS}
        sp_breakdown[sp] = {
            "n_images":  int(sp_mask.sum()),
            "img2txt":   {f"R@{k}": round(r_txt[k], 4) for k in KS},
            "img2aud":   {f"R@{k}": round(r_aud[k], 4) for k in KS},
            "audio_src": aud_meta["source_per_species"].get(sp, "missing"),
        }

    # ── 6. Print results ──────────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("  EVALUATION RESULTS  (test / held-out data)")
    print("=" * 64)

    print("\n  Image -> Text Retrieval:")
    for k in KS:
        print(f"    R@{k:<3}: {txt_recall[k]:.4f}  ({txt_recall[k]*100:.1f}%)")
    print(f"    Composite : {txt_comp:.4f}")

    print("\n  Image -> Audio Retrieval:")
    for k in KS:
        print(f"    R@{k:<3}: {aud_recall[k]:.4f}  ({aud_recall[k]*100:.1f}%)")
    print(f"    Composite : {aud_comp:.4f}")

    print(f"\n  Overall Composite Score : {overall:.4f}")

    if args.per_species:
        print()
        print("  Per-Species Breakdown:")
        hdr = f"  {'Species':<42} {'N':>4}  {'TxtR@1':>6}  {'AudR@1':>6}  {'AudSrc'}"
        print(hdr)
        print("  " + "-" * 70)
        for sp, info in sorted(sp_breakdown.items()):
            print(
                f"  {sp:<42} {info['n_images']:>4}  "
                f"{info['img2txt']['R@1']:>6.3f}  "
                f"{info['img2aud']['R@1']:>6.3f}  "
                f"{info['audio_src']}"
            )

    # ── 7. Save JSON report ───────────────────────────────────────────────────
    report = {
        "checkpoint_epoch":        ckpt["epoch"],
        "val_composite_score":     round(ckpt["composite_score"], 4),
        "data_sources": {
            "image":  img_meta,
            "text":   txt_meta,
            "audio": {
                "n_val":            aud_meta["n_val"],
                "n_train_fallback": aud_meta["n_train_fallback"],
                "n_missing":        aud_meta["n_missing"],
            },
        },
        "test_img2txt":            {f"R@{k}": round(txt_recall[k], 4) for k in KS},
        "test_img2aud":            {f"R@{k}": round(aud_recall[k], 4) for k in KS},
        "test_img2aud_n_queries":  int(img_embs_aud.size(0)),
        "test_img2aud_n_species":  len(aud_species_ids),
        "test_img2aud_note":       "Recall computed only over images whose species has audio in gallery",
        "test_img2txt_composite":  round(txt_comp, 4),
        "test_img2aud_composite":  round(aud_comp, 4),
        "overall_composite":       round(overall, 4),
        "per_species":             sp_breakdown,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Report saved -> {REPORT_PATH}")
    print("=" * 64)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate trained projection heads on test data"
    )
    p.add_argument("--device",      type=str,  default=DEVICE)
    p.add_argument("--per_species", action="store_true",
                   help="Print a per-species R@1 breakdown table")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
