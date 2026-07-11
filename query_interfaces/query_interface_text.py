"""
query_interface_text.py — Zero-Shot Discovery & Retrieval Text Query Interface
=============================================================================
Features:
  1. Fits a One-Class SVM (OC-SVM) on the training set text projections.
  2. Pre-computes the target Image Gallery using species-level test image centroids.
  3. Encodes any input text query (string or file path).
  4. Projects the query embedding to the 768-D shared latent space.
  5. Runs open-set recognition checks:
     - Check A: Out-of-distribution (OOD) outlier detection using OC-SVM.
     - Check B: Absolute similarity threshold against the species image gallery.
     - Check C: Confidence margin threshold between top-1 and other top ranks.
  6. Outputs "New/Unknown semantic topic found!" if any confidence criteria fail.
"""

import os
import sys
import argparse
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

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

def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Discovery & Retrieval Text Query Interface.")
    parser.add_argument("--text", type=str, required=True,
                        help="Raw query text string OR path to a text file")
    parser.add_argument("--model", type=str, choices=["open", "closed"], default="closed",
                        help="Model architecture: open or closed")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Cosine similarity threshold (defaults: closed=0.46, open=0.36)")
    parser.add_argument("--margin_threshold", type=float, default=None,
                        help="Confidence margin threshold (defaults: closed=0.02, open=0.02)")
    parser.add_argument("--device", type=str, default=DEVICE, help="Execution device")
    args = parser.parse_args()

    # Set default similarity and margin thresholds based on optimal sweeps
    if args.threshold is None:
        args.threshold = 0.38 if args.model == "closed" else 0.30
    if args.margin_threshold is None:
        args.margin_threshold = 0.02 if args.model == "closed" else 0.02

    print("=" * 64)
    print("  Marine Multimodal Text Zero-Shot Query & Discovery Interface")
    print("=" * 64)

    # 1. Resolve text input
    query_string = ""
    if os.path.exists(args.text):
        try:
            with open(args.text, "r", encoding="utf-8") as f:
                query_string = f.read().strip()
            print(f"Loaded query text from file: {args.text}")
        except Exception as e:
            print(f"[ERROR] Failed to read text file at {args.text}: {e}")
            return
    else:
        query_string = args.text.strip()
        print(f"Loaded raw query string: \"{query_string[:60]}...\"")

    if not query_string or len(query_string.split()) < 3:
        print("[ERROR] Query text is too short. Please provide a descriptive paragraph.")
        return

    # 2. Load model pipelines and Text Encoder
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
    print(f"Loaded {args.model.upper()} checkpoint (Epoch {ckpt['epoch']})")

    # 3. Train One-Class SVM on known training text projections
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

    # 4. Construct Species Image Gallery (Centroids)
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

    # 5. Process query text
    print("\nProcessing query text features...")
    # Encode using SentenceTransformer
    embs = txt_enc.encode(
        [query_string], convert_to_tensor=True,
        show_progress_bar=False, normalize_embeddings=True
    )
    raw_txt = F.normalize(embs.mean(dim=0), p=2, dim=0)

    with torch.no_grad():
        proj_txt = pipeline.text_head(raw_txt.unsqueeze(0).to(args.device)).squeeze(0).cpu()
            
    proj_txt_np = proj_txt.numpy().reshape(1, -1)

    # 6. Open-Set Classification Decision
    
    # Check A: One-Class SVM outlier check
    svm_prediction = oc_svm.predict(proj_txt_np)[0]
    
    # Check B & C: Similarity and Margin Check against visual gallery
    sims = torch.matmul(proj_txt, gallery_tensor.T)
    topk = sims.topk(min(5, sims.size(0)))
    scores = topk.values.tolist()
    indices = topk.indices.tolist()
    
    max_similarity = scores[0]
    margin = max_similarity - np.mean(scores[1:])

    is_outlier_svm = (svm_prediction == -1)
    is_outlier_similarity = (max_similarity < args.threshold)
    is_outlier_margin = (margin < args.margin_threshold)

    print("\n" + "-" * 50)
    print("  DIAGNOSTIC CRITERIA")
    print("-" * 50)
    print(f"  One-Class SVM Outlier Check : {'OUTLIER (Reject)' if is_outlier_svm else 'INLIER (Accept)'}")
    print(f"  Maximum Cosine Similarity  : {max_similarity:.4f}  (Threshold: {args.threshold})")
    print(f"  Confidence Margin (Top1-Rest): {margin:.4f}  (Threshold: {args.margin_threshold})")
    print("-" * 50)

    # Decision Rule: Override global SVM outlier if similarity and margin both exceed thresholds
    is_confident_match = (not is_outlier_similarity) and (not is_outlier_margin)
    
    if is_confident_match:
        reject_query = False
    else:
        reject_query = is_outlier_svm or is_outlier_similarity or is_outlier_margin

    if reject_query:
        print("\n[DECISION]: New/Unknown semantic topic found!")
        print("   This text description does not confidently match any of the 75 species.")
    else:
        print("\n[DECISION]: Confident species match found.")
        print("\nTop Matched Species (Visual Gallery):")
        for rank, (idx, score) in enumerate(zip(indices, scores), 1):
            print(f"  {rank}. {gallery_species[idx]:<30} (Similarity: {score:.4f})")
    print("=" * 64)


if __name__ == "__main__":
    main()
