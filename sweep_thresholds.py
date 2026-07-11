"""
sweep_thresholds.py — Optimal Threshold and Margin Sweeper for Zero-Shot Discovery
================================================================================
Steps:
  1. Fits the One-Class SVM on the known training set.
  2. Encodes all OOD images manually placed inside `datasets/ood_dataset/*`.
  3. Encodes a representative subset of the known test images in `datasets/image_dataset/test/*`.
  4. Computes similarity scores and confidence margins for all OOD and known test images.
  5. Sweeps absolute similarity thresholds (0.35 to 0.60) and margin thresholds (0.02 to 0.15).
  6. Computes Precision, Recall, and F1-score to find the mathematically optimal values.
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.svm import OneClassSVM

# Add marine_alignment folder to path
ALIGNMENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "marine_alignment")
sys.path.insert(0, ALIGNMENT_DIR)

from config import (
    CHECKPOINT_PATH_CLOSED, CHECKPOINT_PATH_OPEN, DEVICE,
    IMAGE_MODEL_DIR, TEXT_MODEL_DIR,
)
from dataset import get_test_text_split, make_splits, EMBEDDING_DIR, canonical
import models_closed
import models_open

def _get_image_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

def _load_image_encoder(device: str):
    import timm
    from pathlib import Path
    ckpt_paths = sorted(Path(IMAGE_MODEL_DIR).glob("*_seed42.pth"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No *_seed42.pth found in {IMAGE_MODEL_DIR}")
    ckpt = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)
    model = timm.create_model("convnextv2_base", pretrained=False, num_classes=0, global_pool="avg")
    bb_sd = {k[len("backbone."):]: v for k, v in ckpt["model_state_dict"].items() if k.startswith("backbone.")}
    model.load_state_dict(bb_sd, strict=True)
    model.eval()
    model.to(device)
    return model

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

def collect_image_files(root_dir: str) -> list[str]:
    """Walks directory and finds all raw image paths (nested or direct)."""
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                image_paths.append(os.path.join(root, f))
    return sorted(image_paths)

def main():
    parser = argparse.ArgumentParser(description="Sweep Thresholds for OOD species detection.")
    parser.add_argument("--model", type=str, choices=["closed", "open"], default="closed",
                        help="Model type: closed or open")
    parser.add_argument("--max_test_images", type=int, default=100,
                        help="Maximum number of known test images to process on CPU to keep it fast")
    parser.add_argument("--device", type=str, default=DEVICE, help="Execution device")
    args = parser.parse_args()

    print("=" * 72)
    print(f"  OOD Zero-Shot Discovery Sweep Evaluator ({args.model.upper()} Model)")
    print("=" * 72)

    # 1. Collect OOD Images
    ood_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "ood_dataset")
    ood_paths = collect_image_files(ood_dir)
    print(f"Found {len(ood_paths)} OOD test images manually placed in {ood_dir}")
    if len(ood_paths) == 0:
        print("\n[WARNING]: No images found inside your datasets/ood_dataset subfolders yet!")
        print("Please copy some test images into your manually created subfolders, then run this sweep.")
        return

    # 2. Collect Known Test Images
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "image_dataset", "test")
    test_paths = collect_image_files(test_dir)
    print(f"Found {len(test_paths)} total known test images in {test_dir}")
    
    # Sample subset to prevent long CPU wait times
    if len(test_paths) > args.max_test_images:
        np.random.seed(42)
        sampled_indices = np.random.choice(len(test_paths), args.max_test_images, replace=False)
        test_paths = [test_paths[i] for i in sampled_indices]
        print(f"Sampled {len(test_paths)} known test images for efficient evaluation on CPU.")

    # 3. Load Model pipelines and Text Encoder
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

    # 4. Train One-Class SVM on known training embeddings
    print("\nTraining One-Class SVM on known training dataset...")
    train_files, _ = make_splits()
    raw_train_feats = []
    for fname in train_files:
        path = os.path.join(EMBEDDING_DIR, fname)
        data = torch.load(path, map_location="cpu", weights_only=True)
        if "image_emb" in data and data["image_emb"] is not None:
            raw_train_feats.append(data["image_emb"].float().squeeze())
    
    raw_train_feats = torch.stack(raw_train_feats).to(args.device)
    with torch.no_grad():
        if args.model == "closed":
            proj_train_feats = pipeline.project_image(raw_train_feats).cpu().numpy()
        else:
            proj_train_feats = pipeline.image_head(raw_train_feats).cpu().numpy()
            
    oc_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    oc_svm.fit(proj_train_feats)
    print("One-Class SVM fitted.")

    # 5. Encode text gallery
    print("\nEncoding text database gallery...")
    txt_files, _ = get_test_text_split()
    txt_sp_list = []
    raw_txt_list = []
    for sp, fpaths in txt_files.items():
        raw = _encode_texts(txt_enc, fpaths)
        if raw is not None:
            txt_sp_list.append(sp)
            raw_txt_list.append(raw)
    raw_txt_tensor = torch.stack(raw_txt_list).to(args.device)
    with torch.no_grad():
        txt_embs = pipeline.text_head(raw_txt_tensor).cpu()

    # 6. Extract or Load cached projected embeddings
    cache_path = f"sweep_features_cache_{args.model}.pt"
    use_cache = False
    if os.path.exists(cache_path):
        try:
            cache_data = torch.load(cache_path, map_location="cpu", weights_only=True)
            if cache_data.get("ood_paths") == ood_paths and cache_data.get("test_paths") == test_paths:
                print("\nLoading pre-extracted query projections from cache...")
                ood_tensor = cache_data["ood_tensor"]
                known_tensor = cache_data["known_tensor"]
                use_cache = True
        except Exception:
            pass

    if not use_cache:
        print("\nLoading ConvNeXt image encoder...")
        img_enc = _load_image_encoder(args.device)
        tfm = _get_image_transform()

        print("\nExtracting projections for OOD query images...")
        ood_embeddings = []
        for p in ood_paths:
            img = Image.open(p).convert("RGB")
            t = tfm(img).unsqueeze(0).to(args.device)
            with torch.no_grad():
                raw_feat = img_enc(t)
                if args.model == "closed":
                    proj = pipeline.project_image(raw_feat).squeeze(0).cpu()
                else:
                    proj = pipeline.image_head(raw_feat).squeeze(0).cpu()
                ood_embeddings.append(proj)
        ood_tensor = torch.stack(ood_embeddings)

        print("Extracting projections for Known test images...")
        known_embeddings = []
        for p in test_paths:
            img = Image.open(p).convert("RGB")
            t = tfm(img).unsqueeze(0).to(args.device)
            with torch.no_grad():
                raw_feat = img_enc(t)
                if args.model == "closed":
                    proj = pipeline.project_image(raw_feat).squeeze(0).cpu()
                else:
                    proj = pipeline.image_head(raw_feat).squeeze(0).cpu()
                known_embeddings.append(proj)
        known_tensor = torch.stack(known_embeddings)
        
        # Save to cache
        torch.save({
            "ood_paths": ood_paths,
            "test_paths": test_paths,
            "ood_tensor": ood_tensor,
            "known_tensor": known_tensor
        }, cache_path)

    # 7. Pre-compute SVM predictions, maximum similarities, and margins
    print("\nComputing retrieval and SVM diagnostic values...")
    
    # OOD
    ood_svm_preds = oc_svm.predict(ood_tensor.numpy()) # [N_ood]
    ood_sims = torch.matmul(ood_tensor, txt_embs.T) # [N_ood, N_classes]
    ood_topk = ood_sims.topk(5, dim=-1)
    ood_max_sims = ood_topk.values[:, 0].tolist()
    ood_margins = (ood_topk.values[:, 0] - ood_topk.values[:, 1:].mean(dim=-1)).tolist()

    # Known
    known_svm_preds = oc_svm.predict(known_tensor.numpy()) # [N_known]
    known_sims = torch.matmul(known_tensor, txt_embs.T) # [N_known, N_classes]
    known_topk = known_sims.topk(5, dim=-1)
    known_max_sims = known_topk.values[:, 0].tolist()
    known_margins = (known_topk.values[:, 0] - known_topk.values[:, 1:].mean(dim=-1)).tolist()

    # 8. Sweep Threshold Values to optimize OOD Detection F1-Score
    print("\nRunning threshold sweep optimization...")
    best_f1 = -1.0
    best_sim_thresh = 0.0
    best_margin_thresh = 0.0
    best_stats = {}

    sim_values = np.arange(0.35, 0.61, 0.01)
    margin_values = np.arange(0.02, 0.16, 0.01)

    for s_t in sim_values:
        for m_t in margin_values:
            # Evaluate OOD queries (True Positives = correctly rejected as unknown)
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
            
            # Evaluate Known queries (False Positives = incorrectly rejected as unknown)
            fp = 0
            tn = 0
            for i in range(len(test_paths)):
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
    print("  OPTIMIZATION SWEEP RESULTS")
    print("=" * 72)
    print(f"  Best F1-Score for OOD detection: {best_f1:.4f}")
    print(f"  Optimal Cosine Similarity Threshold : {best_sim_thresh:.2f}")
    print(f"  Optimal Confidence Margin Threshold  : {best_margin_thresh:.2f}")
    print("-" * 72)
    print(f"  Detailed Statistics for Optimal Thresholds:")
    print(f"    - True Positives (OOD correctly rejected)    : {best_stats['tp']}/{len(ood_paths)} ({best_stats['recall']*100:.1f}%)")
    print(f"    - False Positives (Known incorrectly rejected): {best_stats['fp']}/{len(test_paths)}")
    print(f"    - True Negatives (Known correctly accepted)  : {best_stats['tn']}/{len(test_paths)}")
    print(f"    - False Negatives (OOD incorrectly accepted) : {best_stats['fn']}/{len(ood_paths)}")
    print(f"    - OOD Detection Precision                     : {best_stats['precision']*100:.1f}%")
    print(f"    - OOD Detection Recall (Sensitivity)         : {best_stats['recall']*100:.1f}%")
    print("=" * 72)

if __name__ == "__main__":
    main()