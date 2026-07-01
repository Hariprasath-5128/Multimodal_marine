"""
verify_features.py — Sanity-Check Script for Generated Embeddings
===================================================================
Run this immediately after feature_extractor.py to confirm:
  1.  Every .pt file loads without error.
  2.  Tensor shapes match config dimensions.
  3.  Image embedding is always present (never None).
  4.  L2 norms are approximately 1.0 (i.e. pre-normalised).
  5.  No NaN / Inf values exist.
  6.  Modality coverage table is printed per species.

Usage
-----
    python verify_features.py              # verifies EMBEDDING_DIR from config
    python verify_features.py --verbose    # also print per-file anomalies
    python verify_features.py --fix_norms  # re-normalise any off-sphere vectors in-place
"""

import os
import sys
import glob
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import EMBEDDING_DIR, IMG_INPUT_DIM, TXT_INPUT_DIM, AUD_INPUT_DIM

# ANSI colour helpers
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):    return f"{GREEN}✓{RESET} {msg}"
def warn(msg):  return f"{YELLOW}⚠ {RESET}{msg}"
def err(msg):   return f"{RED}✗{RESET} {msg}"


def verify(embedding_dir: str, verbose: bool, fix_norms: bool) -> int:
    """
    Returns the number of fatal errors found (0 = all clear).
    """
    pt_files = sorted(glob.glob(os.path.join(embedding_dir, "*.pt")))

    if not pt_files:
        print(err(f"No .pt files found in: {embedding_dir}"))
        print("  Run feature_extractor.py first.")
        return 1

    print(f"\n{BOLD}{'='*64}{RESET}")
    print(f"{BOLD}  Marine Feature Verification{RESET}")
    print(f"{BOLD}{'='*64}{RESET}")
    print(f"  Directory : {embedding_dir}")
    print(f"  Files     : {len(pt_files)}")
    print()

    # ── Per-file scan ──────────────────────────────────────────────────────────
    errors   = 0
    warnings = 0

    species_has_image: dict[str, list[bool]]  = defaultdict(list)
    species_has_text:  dict[str, list[bool]]  = defaultdict(list)
    species_has_audio: dict[str, list[bool]]  = defaultdict(list)

    img_dims_seen:  set[tuple] = set()
    txt_dims_seen:  set[tuple] = set()
    aud_dims_seen:  set[tuple] = set()

    img_norms: list[float] = []
    txt_norms: list[float] = []
    aud_norms: list[float] = []

    for fpath in pt_files:
        fname = os.path.basename(fpath)

        try:
            data = torch.load(fpath, weights_only=True, map_location="cpu")
        except Exception as e:
            print(err(f"{fname}: load failed — {e}"))
            errors += 1
            continue

        sp = data.get("species_name", "UNKNOWN")

        # ── Image ──────────────────────────────────────────────────────────────
        img_emb = data.get("image_emb")
        if img_emb is None:
            print(err(f"{fname}: image_emb is None!"))
            errors += 1
            species_has_image[sp].append(False)
        else:
            img_emb = img_emb.float().squeeze()
            img_dims_seen.add(tuple(img_emb.shape))
            if img_emb.shape != torch.Size([IMG_INPUT_DIM]):
                print(err(f"{fname}: image shape {tuple(img_emb.shape)} ≠ [{IMG_INPUT_DIM}]"))
                errors += 1
            if torch.isnan(img_emb).any() or torch.isinf(img_emb).any():
                print(err(f"{fname}: image_emb contains NaN/Inf"))
                errors += 1
            norm = img_emb.norm().item()
            img_norms.append(norm)
            if abs(norm - 1.0) > 0.05:
                msg = f"{fname}: image L2-norm={norm:.4f} (expected ≈1.0)"
                if fix_norms:
                    data["image_emb"] = F.normalize(img_emb, p=2, dim=0)
                    torch.save(data, fpath)
                    if verbose:
                        print(warn(msg + " → fixed"))
                    warnings += 1
                else:
                    if verbose:
                        print(warn(msg))
                    warnings += 1
            species_has_image[sp].append(True)

        # ── Text ───────────────────────────────────────────────────────────────
        txt_emb = data.get("text_embs")
        if txt_emb is None:
            txt_emb = data.get("text_emb") # fallback for old files
        has_txt = txt_emb is not None
        species_has_text[sp].append(has_txt)
        if has_txt:
            txt_emb = txt_emb.float()
            if txt_emb.dim() == 1:
                txt_emb = txt_emb.unsqueeze(0)
            txt_dims_seen.add(tuple(txt_emb.shape[1:]))
            if txt_emb.shape[1] != TXT_INPUT_DIM:
                print(err(f"{fname}: text shape {tuple(txt_emb.shape)} ≠ [N, {TXT_INPUT_DIM}]"))
                errors += 1
            if torch.isnan(txt_emb).any() or torch.isinf(txt_emb).any():
                print(err(f"{fname}: text_embs contains NaN/Inf"))
                errors += 1
            norms = txt_emb.norm(dim=1).tolist()
            txt_norms.extend(norms)

        # ── Audio ──────────────────────────────────────────────────────────────
        aud_emb = data.get("audio_embs")
        if aud_emb is None:
            aud_emb = data.get("audio_emb") # fallback
        has_aud = aud_emb is not None
        species_has_audio[sp].append(has_aud)
        if has_aud:
            aud_emb = aud_emb.float()
            if aud_emb.dim() == 1:
                aud_emb = aud_emb.unsqueeze(0)
            aud_dims_seen.add(tuple(aud_emb.shape[1:]))
            if aud_dims_seen and verbose and len(aud_dims_seen) == 1:
                pass  # will report in summary
            if torch.isnan(aud_emb).any() or torch.isinf(aud_emb).any():
                print(err(f"{fname}: audio_embs contains NaN/Inf"))
                errors += 1
            norms = aud_emb.norm(dim=1).tolist()
            aud_norms.extend(norms)

    # ── Dimension summary ──────────────────────────────────────────────────────
    print(f"{BOLD}Embedding Dimensions{RESET}")
    print(f"  Image dims seen : {img_dims_seen or 'none'}  (expected ({IMG_INPUT_DIM},))")
    print(f"  Text  dims seen : {txt_dims_seen or 'none'}  (expected ({TXT_INPUT_DIM},))")
    print(f"  Audio dims seen : {aud_dims_seen or 'none'}  (expected ({AUD_INPUT_DIM},))")

    if aud_dims_seen:
        unique_aud = list(aud_dims_seen)
        actual_aud_dim = unique_aud[0][0] if unique_aud else "?"
        marker = ok(str(actual_aud_dim)) if actual_aud_dim == AUD_INPUT_DIM else warn(str(actual_aud_dim))
        print(f"  -> Actual audio dim: {marker}")
    print()

    # ── L2 Norm statistics ────────────────────────────────────────────────────
    def norm_stats(norms, label):
        if not norms:
            return
        t = torch.tensor(norms)
        print(f"  {label:6s} L2-norms : "
              f"min={t.min():.4f}  max={t.max():.4f}  "
              f"mean={t.mean():.4f}  "
              f"(off-sphere >{0.05:.0%}: "
              f"{(t.sub(1).abs() > 0.05).sum().item()}/{len(norms)})")

    print(f"{BOLD}L2 Norm Statistics{RESET}")
    norm_stats(img_norms, "Image")
    norm_stats(txt_norms, "Text")
    norm_stats(aud_norms, "Audio")
    print()

    # ── Modality Coverage Table ───────────────────────────────────────────────
    all_species = sorted(set(
        list(species_has_image) + list(species_has_text) + list(species_has_audio)
    ))
    n_img = n_txt = n_aud = n_all = 0

    print(f"{BOLD}Modality Coverage per Species ({len(all_species)} species){RESET}")
    header = f"  {'Species':<45s} {'Img':>5} {'Txt':>5} {'Aud':>5} {'Files':>6}"
    print(header)
    print("  " + "-" * 62)

    for sp in all_species:
        imgs  = species_has_image.get(sp, [])
        txts  = species_has_text.get(sp, [])
        auds  = species_has_audio.get(sp, [])
        n     = len(imgs)
        has_i = any(imgs)
        has_t = any(txts)
        has_a = any(auds)

        i_sym = f"{GREEN}✓{RESET}" if has_i else f"{RED}✗{RESET}"
        t_sym = f"{GREEN}✓{RESET}" if has_t else f"{YELLOW}–{RESET}"
        a_sym = f"{GREEN}✓{RESET}" if has_a else f"{YELLOW}–{RESET}"

        print(f"  {sp:<45s}  {i_sym}     {t_sym}     {a_sym}   {n:>4d}")

        if has_i: n_img += 1
        if has_t: n_txt += 1
        if has_a: n_aud += 1
        if has_i and has_t and has_a: n_all += 1

    print("  " + "-" * 62)
    print(f"  {'TOTAL':<45s}  {n_img:>4d}  {n_txt:>4d}  {n_aud:>4d}  {n_all:>4d} (all-modal)")
    print()

    # ── Final verdict ──────────────────────────────────────────────────────────
    print(f"{BOLD}Summary{RESET}")
    total_files = len(pt_files)
    print(f"  Total .pt files : {total_files}")
    print(f"  Errors          : {RED}{errors}{RESET}" if errors else f"  Errors          : {GREEN}0{RESET}")
    print(f"  Warnings        : {YELLOW}{warnings}{RESET}" if warnings else f"  Warnings        : {GREEN}0{RESET}")

    if errors == 0:
        print(f"\n{GREEN}{BOLD}✓ All features verified successfully.{RESET}")
        print("  You can now run:  python train_open.py and python train_closed.py")
    else:
        print(f"\n{RED}{BOLD}✗ {errors} error(s) found. Fix before training.{RESET}")

    return errors


def parse_args():
    p = argparse.ArgumentParser(description="Verify marine feature embeddings")
    p.add_argument("--embedding_dir", type=str,  default=EMBEDDING_DIR)
    p.add_argument("--verbose",       action="store_true",
                   help="Print per-file warnings")
    p.add_argument("--fix_norms",     action="store_true",
                   help="Re-normalise off-sphere vectors in-place")
    return p.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    n_errs = verify(args.embedding_dir, args.verbose, args.fix_norms)
    sys.exit(n_errs)
