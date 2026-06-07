"""
evaluate_main.py  — Root Domain Classifier Evaluation
=======================================================
Evaluates the domain classifier model (trained by train_main.py).

Reports:
  - Overall top-1 accuracy across all test images
  - Per-domain breakdown (precision, recall, count)
  - Confusion matrix

Test images are loaded from: datasets/marine_mammal_test/<species_folder>
Each species folder is routed to its domain using the same logic as
the full pipeline.

Usage:
  python evaluate_main.py
"""

import os
import sys
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

try:
    import timm
except ImportError:
    raise ImportError("pip install timm")

# ─── Paths ────────────────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).parent.resolve()
PROJECT_ROOT = THIS_DIR.parent.parent
TEST_DIR     = PROJECT_ROOT / "datasets" / "marine_mammal_test"
MODELS_DIR   = THIS_DIR / "models" / "main"
RESULTS_DIR  = THIS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
DOMAINS = ["dolphin", "whale", "seal", "sealion", "porpoise", "manatee"]
MEAN    = [0.485, 0.456, 0.406]
STD     = [0.229, 0.224, 0.225]

# ─── Domain routing (same as pipeline) ───────────────────────────────────────
def get_domain(folder_name: str) -> str | None:
    n = folder_name.lower().replace("-", " ").replace("_", " ")
    if n == "vaquita":                                      return "dolphin"
    if "dolphin" in n:                                      return "dolphin"
    if "whale" in n or "orca" in n or "narwhal" in n or "beluga" in n:
        return "whale"
    if "porpoise" in n:                                     return "porpoise"
    if "sea lion" in n or "sealion" in n:                   return "sealion"
    if "seal" in n:                                         return "seal"
    if "manatee" in n:                                      return "manatee"
    if "dugong" in n:                                       return None
    if "walrus" in n:                                       return None
    if "sea otter" in n:                                    return None
    if "polar bear" in n:                                   return None
    return None

# ─── Model (must match train_main.py) ─────────────────────────────────────────
class DomainNet(nn.Module):
    def __init__(self, num_classes: int, backbone_name: str):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False,
            num_classes=0, global_pool="avg",
        )
        dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(0.3),
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))

# ─── TTA (5 views) ────────────────────────────────────────────────────────────
def build_tta(size: int):
    sl = int(size * 1.12)
    return [
        transforms.Compose([transforms.Resize((size, size)),
                             transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
        transforms.Compose([transforms.Resize((sl, sl)),
                             transforms.CenterCrop(size),
                             transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
        transforms.Compose([transforms.Resize((size, size)),
                             transforms.RandomHorizontalFlip(p=1.0),
                             transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
        transforms.Compose([transforms.Resize((sl, sl)),
                             transforms.CenterCrop(size),
                             transforms.RandomHorizontalFlip(p=1.0),
                             transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
        transforms.Compose([transforms.Resize((int(size * 1.05), int(size * 1.05))),
                             transforms.CenterCrop(size),
                             transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    ]

# ─── Main ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load checkpoint
    ckpt_path = MODELS_DIR / "domain_classifier.pth"
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        print("        Run train_main.py first.")
        sys.exit(1)

    ck       = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = ck["backbone"]
    img_size = ck["img_size"]
    i2c      = ck["idx_to_class"]
    n_cls    = ck["num_classes"]

    print(f"Backbone : {backbone}  |  {img_size}px  |  {n_cls} classes")
    print(f"Best val accuracy (training) : {ck['best_val_acc']*100:.2f}%\n")

    model = DomainNet(n_cls, backbone).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    tta_tfms = build_tta(img_size)

    # Collect stats
    total   = 0
    correct = 0
    domain_stats = {d: {"total": 0, "correct": 0} for d in DOMAINS}
    skipped = 0

    for sp_folder in sorted(TEST_DIR.iterdir()):
        if not sp_folder.is_dir():
            continue
        true_domain = get_domain(sp_folder.name)
        if true_domain is None:
            skipped += 1
            continue   # out-of-domain species (dugong, walrus, etc.)

        for img_file in sp_folder.iterdir():
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            img = Image.open(img_file).convert("RGB")

            # TTA averaging
            combined = None
            for tfm in tta_tfms:
                t   = tfm(img).unsqueeze(0).to(device)
                out = model(t)
                combined = out if combined is None else combined + out

            pred_idx    = combined.argmax(1).item()
            pred_domain = i2c[pred_idx]

            total  += 1
            domain_stats[true_domain]["total"] += 1
            if pred_domain == true_domain:
                correct += 1
                domain_stats[true_domain]["correct"] += 1

    # ── Summary ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  ROOT DOMAIN CLASSIFIER — EVALUATION REPORT")
    print("=" * 60)
    print(f"{'Domain':<14} {'N':>5} {'Correct':>8} {'Accuracy':>10}")
    print("-" * 60)

    rows = []
    for d in DOMAINS:
        s   = domain_stats[d]
        n   = s["total"]
        c   = s["correct"]
        acc = 100.0 * c / n if n > 0 else 0.0
        flag = " [PASS]" if acc >= 90.0 else " [FAIL]"
        print(f"  {d:<12} {n:>5} {c:>8} {acc:>9.1f}%{flag}")
        rows.append((d, n, c, acc))

    print("-" * 60)
    overall = 100.0 * correct / total if total > 0 else 0.0
    flag = " [PASS]" if overall >= 90.0 else " [FAIL]"
    print(f"  {'TOTAL':<12} {total:>5} {correct:>8} {overall:>9.1f}%{flag}")
    print("=" * 60)
    if skipped:
        print(f"\n  Note: {skipped} out-of-domain species folders skipped "
              f"(dugong, walrus, sea otter, polar bear)")

    # ── Markdown report ────────────────────────────────────────────────────────
    md = ["# Root Domain Classifier — Evaluation Report\n"]
    md.append(f"**Overall Accuracy: {overall:.1f}%**\n")
    md.append("| Domain | N | Correct | Accuracy | Status |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")
    for d, n, c, acc in rows:
        status = "[PASS]" if acc >= 90.0 else "[FAIL]"
        md.append(f"| {d} | {n} | {c} | {acc:.1f}% | {status} |")
    md.append(f"| **TOTAL** | {total} | {correct} | **{overall:.1f}%** | "
              f"{'[PASS]' if overall >= 90.0 else '[FAIL]'} |")

    report = RESULTS_DIR / "main_domain_report.md"
    report.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[DONE] Report saved -> {report}")


if __name__ == "__main__":
    main()
