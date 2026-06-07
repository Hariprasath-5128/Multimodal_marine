"""
evaluate.py  — Full Two-Stage Pipeline Evaluation
===================================================
Runs every test image through the two-stage pipeline:

  Stage 1: Main domain classifier  (models/main/domain_classifier.pth)
           → predicts which domain bucket (dolphin / whale / seal / …)

  Stage 2: Domain-specific species classifier  (models/<domain>_seed42.pth)
           → predicts top-3 species within that domain

Accuracy weighting:  1st = 100%  |  2nd = 35%  |  3rd = 15%
Target: ≥ 75% weighted accuracy per domain.

Usage:
  python evaluate.py               # all domains
  python evaluate.py --domain seal # single domain

Notes
-----
* If the main model checkpoint is missing, the script falls back to
  keyword-based domain routing (old behaviour) with a warning.
* Domain routing errors (Stage-1 predicts wrong domain) count as a miss.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ─── Pipeline ─────────────────────────────────────────────────────────────────
# Import shared objects from pipeline.py (same directory)
THIS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(THIS_DIR))

from pipeline import (
    MarinePipeline,
    canonical,
    heuristic_domain,
    DOMAINS,
)

import torch

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = THIS_DIR.parent.parent
TEST_DIR     = PROJECT_ROOT / "datasets" / "image_dataset" / "test"
RESULTS_DIR  = THIS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Accuracy weights (Top-1 / Top-2 / Top-3) ────────────────────────────────
TOP3_WEIGHTS = [1.0, 0.35, 0.15]

# ─── Out-of-domain species (no domain model trained for these) ────────────────
OUT_OF_DOMAIN = {"dugong", "walrus", "sea otter", "polar bear",
                 "sea_otter", "polar_bear"}


# ─── Evaluate one domain ──────────────────────────────────────────────────────
def evaluate_domain(domain: str, pipeline: MarinePipeline) -> dict:
    stats = {
        "total": 0, "top1": 0, "top2": 0, "top3": 0,
        "miss": 0, "domain_error": 0,
        "weighted_sum": 0.0,
        "y_true": [],
        "y_pred": [],
        "per_species": defaultdict(
            lambda: {"total": 0, "top1": 0, "top2": 0, "top3": 0,
                     "domain_error": 0}
        ),
    }

    for sp_folder in sorted(TEST_DIR.iterdir()):
        if not sp_folder.is_dir():
            continue

        # Determine the ground-truth domain from the folder name
        true_domain = heuristic_domain(sp_folder.name)
        if true_domain != domain:
            continue    # not this domain's turn

        true_key = canonical(sp_folder.name)
        sp_stats = stats["per_species"][sp_folder.name]

        for img_file in sorted(sp_folder.iterdir()):
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            stats["total"]    += 1
            sp_stats["total"] += 1

            # ── Full pipeline predict ──────────────────────────────────────
            result = pipeline.predict(
                str(img_file),
                hint_domain=true_domain,    # fallback hint if no main model
            )

            pred_domain = result["domain"]
            top3        = result["top3"]   # [(species_name, conf), ...]

            # Stage-1 domain error → automatic miss
            if pred_domain != true_domain:
                stats["miss"]              += 1
                stats["domain_error"]      += 1
                sp_stats["domain_error"]   += 1
                stats["y_true"].append(true_key)
                stats["y_pred"].append("unknown_domain")
                continue

            top3_names = [name for name, _ in top3]
            stats["y_true"].append(true_key)
            stats["y_pred"].append(top3_names[0] if top3_names else "unknown")

            if top3_names and true_key == top3_names[0]:
                stats["top1"]           += 1
                sp_stats["top1"]        += 1
                stats["weighted_sum"]   += TOP3_WEIGHTS[0]
            elif len(top3_names) > 1 and true_key == top3_names[1]:
                stats["top2"]           += 1
                sp_stats["top2"]        += 1
                stats["weighted_sum"]   += TOP3_WEIGHTS[1]
            elif len(top3_names) > 2 and true_key == top3_names[2]:
                stats["top3"]           += 1
                sp_stats["top3"]        += 1
                stats["weighted_sum"]   += TOP3_WEIGHTS[2]
            else:
                stats["miss"] += 1

    return stats


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="all",
                    help="Domain to evaluate, or 'all'")
    args = ap.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domains = DOMAINS if args.domain == "all" else [args.domain]

    print(f"\nDevice : {device}")
    print(f"Weights: Top-1={TOP3_WEIGHTS[0]}  "
          f"Top-2={TOP3_WEIGHTS[1]}  Top-3={TOP3_WEIGHTS[2]}")

    # ── Load full pipeline ─────────────────────────────────────────────────────
    print("\nLoading pipeline ...")
    pipe = MarinePipeline()
    pipe.load(device, domains=domains)

    # ── Evaluate each domain ───────────────────────────────────────────────────
    all_stats = {}
    for dom in domains:
        if dom not in pipe.species_models:
            print(f"\n  [{dom}] skipped — no species model found")
            continue
        print(f"\nEvaluating [{dom}] ...")
        all_stats[dom] = evaluate_domain(dom, pipe)

    # ── Per-domain summary ─────────────────────────────────────────────────────
    sep = "=" * 100
    print(f"\n\n{sep}")
    print("TOP-3 WEIGHTED ACCURACY  —  FULL PIPELINE SUMMARY")
    print(f"  (Stage-1 domain classifier -> Stage-2 species classifier)")
    print(sep)
    print(f"{'Domain':<12} {'N':>5} {'Top-1':>7} {'Top-2':>7} {'Top-3':>7} "
          f"{'Miss':>6} {'DomErr':>7} {'Coverage':>9} {'WgtAcc':>8}  Status")
    print("-" * 100)

    grand_total = grand_wsum = 0
    rows = []

    for dom in domains:
        if dom not in all_stats:
            continue
        s    = all_stats[dom]
        n    = s["total"]
        if n == 0:
            continue
        in3  = s["top1"] + s["top2"] + s["top3"]
        cov  = 100.0 * in3 / n
        wacc = 100.0 * s["weighted_sum"] / n
        flag = "[PASS]" if wacc >= 75.0 else "[FAIL]"
        grand_total += n
        grand_wsum  += s["weighted_sum"]
        rows.append((dom, n, s["top1"], s["top2"], s["top3"],
                     s["miss"], s["domain_error"], cov, wacc, flag))
        print(f"{dom:<12} {n:>5} {s['top1']:>7} {s['top2']:>7} {s['top3']:>7} "
              f"{s['miss']:>6} {s['domain_error']:>7} {cov:>8.1f}% "
              f"{wacc:>7.1f}%  {flag}")

    print("-" * 100)
    if grand_total > 0:
        g_wacc = 100.0 * grand_wsum / grand_total
        g_flag = "[PASS]" if g_wacc >= 75.0 else "[FAIL]"
        print(f"{'TOTAL':<12} {grand_total:>5} {' ':>7} {' ':>7} {' ':>7} "
              f"{' ':>6} {' ':>7} {' ':>9} {g_wacc:>7.1f}%  {g_flag}")
    print(sep)

    # ── Per-species breakdown ──────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("PER-SPECIES BREAKDOWN")
    print(f"{'='*70}")
    for dom in domains:
        if dom not in all_stats:
            continue
        s = all_stats[dom]
        print(f"\n[{dom.upper()}]")
        print(f"  {'Species':<42} {'N':>3} {'T1':>4} {'T2':>4} {'T3':>4} "
              f"{'DomErr':>7} {'WgtAcc':>8}")
        print(f"  {'-'*75}")
        for sp, ss in sorted(s["per_species"].items()):
            n = ss["total"]
            if n == 0:
                continue
            w    = ss["top1"]*TOP3_WEIGHTS[0] + ss["top2"]*TOP3_WEIGHTS[1] + \
                   ss["top3"]*TOP3_WEIGHTS[2]
            wacc = 100.0 * w / n
            de   = ss.get("domain_error", 0)
            print(f"  {sp:<42} {n:>3} {ss['top1']:>4} {ss['top2']:>4} "
                  f"{ss['top3']:>4} {de:>7} {wacc:>7.1f}%")

    # ── Markdown report ────────────────────────────────────────────────────────
    md = [
        "# Two-Stage Marine Pipeline — Evaluation Report\n",
        f"**Weights:** Top-1={TOP3_WEIGHTS[0]} | "
        f"Top-2={TOP3_WEIGHTS[1]} | Top-3={TOP3_WEIGHTS[2]}\n",
        "## Domain Summary\n",
        "| Domain | N | Top-1 | Top-2 | Top-3 | Miss | DomErr "
        "| Coverage | WgtAcc | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: "
        "| :---: | :---: | :---: |",
    ]
    for dom, n, t1, t2, t3, miss, de, cov, wacc, flag in rows:
        md.append(f"| {dom} | {n} | {t1} | {t2} | {t3} | {miss} | {de} "
                  f"| {cov:.1f}% | **{wacc:.1f}%** | {flag} |")
    if grand_total > 0:
        md.append(f"| **TOTAL** | {grand_total} | | | | | | "
                  f"| **{g_wacc:.1f}%** | {g_flag} |")

    md.append("\n## Per-Species Detail\n")
    for dom in domains:
        if dom not in all_stats:
            continue
        s = all_stats[dom]
        md.append(f"\n### {dom.capitalize()}\n")
        md.append("| Species | N | Top-1 | Top-2 | Top-3 | DomErr | WgtAcc |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for sp, ss in sorted(s["per_species"].items()):
            n = ss["total"]
            if n == 0:
                continue
            w    = ss["top1"]*TOP3_WEIGHTS[0] + ss["top2"]*TOP3_WEIGHTS[1] + \
                   ss["top3"]*TOP3_WEIGHTS[2]
            wacc = 100.0 * w / n
            de   = ss.get("domain_error", 0)
            md.append(f"| {sp} | {n} | {ss['top1']} | {ss['top2']} | "
                      f"{ss['top3']} | {de} | {wacc:.1f}% |")

    report = RESULTS_DIR / "top3_report.md"
    report.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[DONE] Report saved -> {report}")

    # Generate Confusion Matrix Heatmap and Classification Report
    y_true_all = []
    y_pred_all = []
    for dom in domains:
        if dom in all_stats:
            y_true_all.extend(all_stats[dom]["y_true"])
            y_pred_all.extend(all_stats[dom]["y_pred"])
            
    if y_true_all:
        labels = sorted(list(set(y_true_all) | set(y_pred_all)))
        print("\n\nClassification Report (Top-1):")
        print(classification_report(y_true_all, y_pred_all, zero_division=0))
        
        cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
        plt.figure(figsize=(24, 20))
        sns.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap="Blues", cbar=False)
        plt.title("Image Classification Confusion Matrix (Top-1)")
        plt.xlabel("Predicted Species")
        plt.ylabel("Actual Species")
        plt.tight_layout()
        plt.savefig(THIS_DIR / "confusion_matrix.png")
        plt.close()
        print(f"Confusion matrix saved to {THIS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
