"""
sweep_text_thresholds.py — Text Modality Zero-Shot Threshold Sweeper
==================================================================
Steps:
  1. Loads the closed or open model pipeline and SentenceTransformer text encoder.
  2. Fits a One-Class SVM on the training set text projections.
  3. Pre-computes the target Image Gallery using species-level test image centroids.
  4. Encodes and projects all OOD texts in datasets/ood_text_dataset/*.txt.
  5. Encodes and projects a subset of known test texts in datasets/text_dataset/test/*.txt.
  6. Sweeps Cosine Similarity and Confidence Margin thresholds to find optimal OOD values.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from sklearn.svm import OneClassSVM

# Add marine_alignment folder to path
ALIGNMENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "marine_alignment")
sys.path.insert(0, ALIGNMENT_DIR)

from config import (
    CHECKPOINT_PATH_CLOSED, CHECKPOINT_PATH_OPEN, DEVICE,
    TEXT_MODEL_DIR,
)
from dataset import get_test_text_split, make_splits, EMBEDDING_DIR, canonical
import models_closed
import models_open

def _load_text_encoder(device: str):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(TEXT_MODEL_DIR, device=device)
    m.eval()
    return m

def _encode_texts(text_model, file_paths: list[str]) -> torch.Tensor | None:
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

def collect_text_files(root_dir: str) -> list[str]:
    text_paths = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".txt") and not f.startswith("_"):
                text_paths.append(os.path.join(root, f))
    return sorted(text_paths)

def main():
    parser = argparse.ArgumentParser(description="Sweep Text Thresholds for OOD text detection.")
    parser.add_argument("--model", type=str, choices=["closed", "open"], default="closed",
                        help="Model type: closed or open")
    parser.add_argument("--max_test_texts", type=int, default=100,
                        help="Maximum number of known test texts to process")
    parser.add_argument("--device", type=str, default=DEVICE, help="Execution device")
    args = parser.parse_args()

    print("=" * 72)
    print(f"  OOD Text Zero-Shot Discovery Sweep Evaluator ({args.model.upper()} Model)")
    print("=" * 72)

    # 1. Collect OOD Text Files
    ood_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "ood_text_dataset")
    ood_paths = collect_text_files(ood_dir)
    print(f"Found {len(ood_paths)} OOD test text profiles in {ood_dir}")
    if len(ood_paths) == 0:
        print("[ERROR]: No text files found inside datasets/ood_text_dataset/.")
        return

    # 2. Collect Known Test Text Files
    test_text_files, _ = get_test_text_split()
    # Flatten test_text_files dict of species -> file paths
    known_paths = []
    for sp, fpaths in test_text_files.items():
        known_paths.extend(fpaths)
    print(f"Found {len(known_paths)} total known test text files.")
    
    if len(known_paths) > args.max_test_texts:
        np.random.seed(42)
        sampled_indices = np.random.choice(len(known_paths), args.max_test_texts, replace=False)
        known_paths = [known_paths[i] for i in sampled_indices]
        print(f"Sampled {len(known_paths)} known test texts for evaluation.")

    # 3. Load Model Pipelines and Text Encoder
    print("\nLoading model pipelines...")
    txt_enc = _load_text_encoder(args.device)

    if args.model == "closed":
        pipeline = models_closed.MarineImageBindPipeline().to(args.device)
        ckpt = torch.load(CHECKPOINT_PATH_CLOSED, map_location=args.device, weights_only=False)
    else:
        pipeline = models_open.MarineImageBindPipeline().to(args.device)
        ckpt = torch.load(CHECKPOINT_PATH_OPEN, map_location=args.device, weights_only=False)
    pipeline.load_state_dict(ckpt["model_state"], strict=False)
    pipeline.eval()

    # 4. Train One-Class SVM on known training text projections
    print("\nTraining One-Class SVM on known training text database...")
    train_files, val_files = make_splits()
    raw_train_feats = []
    for fname in train_files:
        path = os.path.join(EMBEDDING_DIR, fname)
        data = torch.load(path, map_location="cpu", weights_only=True)
        if "text_embs" in data and data["text_embs"] is not None:
            raw_train_feats.append(data["text_embs"].float())
    
    raw_train_feats = torch.cat(raw_train_feats, dim=0).to(args.device)
    with torch.no_grad():
        proj_train_feats = pipeline.text_head(raw_train_feats).cpu().numpy()
        
    oc_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    oc_svm.fit(proj_train_feats)
    print("One-Class SVM fitted.")

    # 5. Pre-compute Target Image Gallery (Species Centroids)
    print("\nConstructing target species image gallery...")
    species_imgs = defaultdict(list)
    for fname in val_files:
        path = os.path.join(EMBEDDING_DIR, fname)
        data = torch.load(path, map_location="cpu", weights_only=True)
        if "image_emb" in data and data["image_emb"] is not None:
            species_imgs[data["species_name"]].append(data["image_emb"].float().squeeze())
            
    gallery_species = []
    gallery_embs = []
    for sp, img_list in species_imgs.items():
        img_stack = torch.stack(img_list).to(args.device)
        with torch.no_grad():
            if args.model == "closed":
                proj = pipeline.project_image(img_stack).mean(dim=0).cpu()
            else:
                proj = pipeline.image_head(img_stack).mean(dim=0).cpu()
        gallery_species.append(sp)
        gallery_embs.append(F.normalize(proj, p=2, dim=0))
    gallery_tensor = torch.stack(gallery_embs)
    print(f"Image gallery built with {gallery_tensor.size(0)} species centroids.")

    # 6. Extract or Load cached projected text embeddings
    cache_path = f"sweep_text_features_cache_{args.model}.pt"
    use_cache = False
    if os.path.exists(cache_path):
        try:
            cache_data = torch.load(cache_path, map_location="cpu", weights_only=True)
            if cache_data.get("ood_paths") == ood_paths and cache_data.get("known_paths") == known_paths:
                print("\nLoading pre-extracted text projections from cache...")
                ood_tensor = cache_data["ood_tensor"]
                known_tensor = cache_data["known_tensor"]
                use_cache = True
        except Exception:
            pass

    if not use_cache:
        print("\nExtracting projections for OOD text files...")
        ood_embeddings = []
        for p in ood_paths:
            raw = _encode_texts(txt_enc, [p])
            if raw is None:
                # Use zero vector fallback if text was un-encodable
                raw = torch.zeros(768)
            with torch.no_grad():
                proj = pipeline.text_head(raw.unsqueeze(0).to(args.device)).squeeze(0).cpu()
                ood_embeddings.append(proj)
        ood_tensor = torch.stack(ood_embeddings)

        print("Extracting projections for Known test texts...")
        known_embeddings = []
        for p in known_paths:
            raw = _encode_texts(txt_enc, [p])
            if raw is None:
                raw = torch.zeros(768)
            with torch.no_grad():
                proj = pipeline.text_head(raw.unsqueeze(0).to(args.device)).squeeze(0).cpu()
                known_embeddings.append(proj)
        known_tensor = torch.stack(known_embeddings)
        
        # Save to cache
        torch.save({
            "ood_paths": ood_paths,
            "known_paths": known_paths,
            "ood_tensor": ood_tensor,
            "known_tensor": known_tensor
        }, cache_path)

    # 7. Pre-compute SVM predictions, maximum similarities, and margins
    print("\nComputing retrieval similarity values against image gallery...")
    
    # OOD
    ood_svm_preds = oc_svm.predict(ood_tensor.numpy())
    ood_sims = torch.matmul(ood_tensor, gallery_tensor.T)
    ood_topk = ood_sims.topk(5, dim=-1)
    ood_max_sims = ood_topk.values[:, 0].tolist()
    ood_margins = (ood_topk.values[:, 0] - ood_topk.values[:, 1:].mean(dim=-1)).tolist()

    # Known
    known_svm_preds = oc_svm.predict(known_tensor.numpy())
    known_sims = torch.matmul(known_tensor, gallery_tensor.T)
    known_topk = known_sims.topk(5, dim=-1)
    known_max_sims = known_topk.values[:, 0].tolist()
    known_margins = (known_topk.values[:, 0] - known_topk.values[:, 1:].mean(dim=-1)).tolist()

    # 8. Sweep Threshold Values to optimize F1-Score
    print("\nRunning threshold sweep optimization...")
    best_f1 = -1.0
    best_sim_thresh = 0.0
    best_margin_thresh = 0.0
    best_stats = {}

    sim_values = np.arange(0.35, 0.61, 0.01)
    margin_values = np.arange(0.02, 0.16, 0.01)

    for s_t in sim_values:
        for m_t in margin_values:
            # Evaluate OOD queries (True Positives = correctly rejected)
            tp = 0
            fn = 0
            for i in range(len(ood_paths)):
                is_outlier_svm = (ood_svm_preds[i] == -1)
                is_outlier_sim = (ood_max_sims[i] < s_t)
                is_outlier_marg = (ood_margins[i] < m_t)
                
                is_confident = (not is_outlier_sim) and (not is_outlier_marg)
                rejected = is_outlier_svm or is_outlier_sim or is_outlier_marg if not is_confident else False
                
                if rejected:
                    tp += 1
                else:
                    fn += 1
            
            # Evaluate Known queries (False Positives = incorrectly rejected)
            fp = 0
            tn = 0
            for i in range(len(known_paths)):
                is_outlier_svm = (known_svm_preds[i] == -1)
                is_outlier_sim = (known_max_sims[i] < s_t)
                is_outlier_marg = (known_margins[i] < m_t)
                
                is_confident = (not is_outlier_sim) and (not is_outlier_marg)
                rejected = is_outlier_svm or is_outlier_sim or is_outlier_marg if not is_confident else False
                
                if rejected:
                    fp += 1
                else:
                    tn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_sim_thresh = s_t
                best_margin_thresh = m_t
                best_stats = {
                    "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                    "precision": precision, "recall": recall
                }

    print("\n" + "=" * 72)
    print("  OPTIMIZATION SWEEP RESULTS (TEXT MODALITY)")
    print("=" * 72)
    print(f"  Best F1-Score for OOD text detection: {best_f1:.4f}")
    print(f"  Optimal Cosine Similarity Threshold : {best_sim_thresh:.2f}")
    print(f"  Optimal Confidence Margin Threshold  : {best_margin_thresh:.2f}")
    print("-" * 72)
    print(f"  Detailed Statistics for Optimal Thresholds:")
    print(f"    - True Positives (OOD correctly rejected)    : {best_stats['tp']}/{len(ood_paths)} ({best_stats['recall']*100:.1f}%)")
    print(f"    - False Positives (Known incorrectly rejected): {best_stats['fp']}/{len(known_paths)}")
    print(f"    - True Negatives (Known correctly accepted)  : {best_stats['tn']}/{len(known_paths)}")
    print(f"    - False Negatives (OOD incorrectly accepted) : {best_stats['fn']}/{len(ood_paths)}")
    print(f"    - OOD Detection Precision                     : {best_stats['precision']*100:.1f}%")
    print(f"    - OOD Detection Recall (Sensitivity)         : {best_stats['recall']*100:.1f}%")
    print("=" * 72)

if __name__ == "__main__":
    main()
