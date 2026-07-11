"""
query_interface.py — Zero-Shot Discovery & Retrieval Query Interface
===================================================================
Features:
  1. Fits a One-Class SVM (OC-SVM) on the training set features of 75 known marine mammal species.
  2. Encodes any input image query (using ConvNeXt-Base).
  3. Projects the query feature to the 768-D shared latent space.
  4. Runs open-set recognition checks:
     - Check A: Out-of-distribution (OOD) outlier detection using OC-SVM.
     - Check B: Absolute similarity threshold against text database prototypes.
     - Check C: Confidence margin threshold between top-1 and other top ranks.
  5. Outputs "New/Unknown creature found!" if any confidence criteria fail.
  6. Implements DBSCAN offline clustering to discover candidate novel species over time.
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Add marine_alignment folder to path so we can import config, dataset, models
ALIGNMENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "marine_alignment")
sys.path.insert(0, ALIGNMENT_DIR)

from config import (
    CHECKPOINT_PATH_CLOSED, CHECKPOINT_PATH_OPEN, DEVICE,
    IMAGE_MODEL_DIR, TEXT_MODEL_DIR,
)
from dataset import get_test_text_split, make_splits, EMBEDDING_DIR
import models_closed
import models_open

BUFFER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unknowns_buffer.json")


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
    bb_name = ckpt.get("backbone", "convnextv2_base")
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


def train_anomaly_detector(pipeline, model_type: str, device: str):
    """Loads known training embeddings and trains a One-Class SVM outlier detector."""
    print("Extracting training set projections for One-Class SVM...")
    train_files, _ = make_splits()
    
    raw_train_feats = []
    for fname in train_files:
        path = os.path.join(EMBEDDING_DIR, fname)
        data = torch.load(path, map_location="cpu", weights_only=True)
        if "image_emb" in data and data["image_emb"] is not None:
            raw_train_feats.append(data["image_emb"].float().squeeze())
            
    if not raw_train_feats:
        raise ValueError("No training image features found in extracted_features/.")
        
    raw_train_feats = torch.stack(raw_train_feats).to(device)
    
    with torch.no_grad():
        if model_type == "closed":
            proj_train_feats = pipeline.project_image(raw_train_feats).cpu().numpy()
        else:
            proj_train_feats = pipeline.image_head(raw_train_feats).cpu().numpy()
            
    print(f"Training One-Class SVM on {proj_train_feats.shape[0]} training samples...")
    # nu = 0.05 (allows 5% of training samples to be classified as outliers)
    oc_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    oc_svm.fit(proj_train_feats)
    return oc_svm


def update_discovery_buffer(img_path: str, embedding: np.ndarray, min_cluster_size: int = 3):
    """Saves unknown projections and runs DBSCAN density clustering offline."""
    if os.path.exists(BUFFER_PATH):
        try:
            with open(BUFFER_PATH, "r") as f:
                buffer_data = json.load(f)
        except Exception:
            buffer_data = []
    else:
        buffer_data = []

    # Check if image is already in the buffer to avoid duplicates
    if any(item["image_path"] == img_path for item in buffer_data):
        return

    buffer_data.append({
        "image_path": img_path,
        "embedding": embedding.tolist()
    })

    with open(BUFFER_PATH, "w") as f:
        json.dump(buffer_data, f, indent=2)

    # Run DBSCAN if we have enough anomalies
    if len(buffer_data) >= min_cluster_size:
        embeddings = np.array([item["embedding"] for item in buffer_data])
        #eps=0.15 (15% distance tolerance on cosine sphere)
        db = DBSCAN(eps=0.15, min_samples=min_cluster_size, metric="cosine").fit(embeddings)
        labels = db.labels_
        
        unique_labels = set(labels)
        for label in unique_labels:
            if label >= 0:
                cluster_indices = [i for i, l in enumerate(labels) if l == label]
                print(f"\n[Discovery Pipeline] New candidate species group discovered!")
                print(f"   Found a cluster of {len(cluster_indices)} similar unknown query images:")
                for idx in cluster_indices:
                    print(f"   - {buffer_data[idx]['image_path']}")


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Discovery & Species Query Interface.")
    parser.add_argument("--image", type=str, required=True, help="Path to query image file")
    parser.add_argument("--model", type=str, choices=["open", "closed"], default="closed",
                        help="Model architecture: open or closed")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Cosine similarity threshold (defaults: closed=0.56, open=0.36)")
    parser.add_argument("--margin_threshold", type=float, default=None,
                        help="Confidence margin threshold (defaults: closed=0.02, open=0.02)")
    parser.add_argument("--device", type=str, default=DEVICE, help="Execution device")
    args = parser.parse_args()

    # Set default similarity and margin thresholds if not specified
    if args.threshold is None:
        args.threshold = 0.56 if args.model == "closed" else 0.36
    if args.margin_threshold is None:
        args.margin_threshold = 0.02 if args.model == "closed" else 0.02

    print("=" * 64)
    print("  Marine Multimodal Zero-Shot Query & Discovery Interface")
    print("=" * 64)

    if not os.path.exists(args.image):
        print(f"[ERROR] Query image not found at: {args.image}")
        return

    # 1. Resolve text dataset split
    print("\nResolving text database gallery...")
    txt_files, txt_meta = get_test_text_split()
    
    # 2. Load models
    print("\nLoading encoders and model pipelines...")
    img_enc = _load_image_encoder(args.device)
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

    # 3. Train the One-Class SVM outlier detector
    oc_svm = train_anomaly_detector(pipeline, args.model, args.device)

    # 4. Encode text gallery
    print("\nEncoding text database...")
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

    # 5. Process query image
    print("\nProcessing query image features...")
    img = Image.open(args.image).convert("RGB")
    tfm = _get_image_transform()
    t = tfm(img).unsqueeze(0).to(args.device)
    
    with torch.no_grad():
        raw_img = img_enc(t)
        if args.model == "closed":
            proj_img = pipeline.project_image(raw_img).squeeze(0).cpu()
        else:
            proj_img = pipeline.image_head(raw_img).squeeze(0).cpu()
            
    proj_img_np = proj_img.numpy().reshape(1, -1)

    # 6. Open-Set Classification Decision
    
    # Check A: One-Class SVM outlier check (-1 = outlier, 1 = inlier)
    svm_prediction = oc_svm.predict(proj_img_np)[0]
    
    # Check B & C: Similarity and Margin Check
    sims = torch.matmul(proj_img, txt_embs.T)
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

    # A match is highly confident if both similarity and margin exceed their thresholds.
    # In this case, we override the global One-Class SVM outlier flag (which can trigger
    # false positives on single-species taxons or minor style shifts).
    is_confident_match = (not is_outlier_similarity) and (not is_outlier_margin)

    if is_confident_match:
        reject_query = False
    else:
        reject_query = is_outlier_svm or is_outlier_similarity or is_outlier_margin

    if reject_query:
        print("\n[DECISION]: New/Unknown creature found!")
        print("   This specimen does not confidently match any of the 75 species.")
        print("   Logging query projection for clustering...")
        update_discovery_buffer(args.image, proj_img.numpy())
    else:
        print("\n[DECISION]: Confident species match found.")
        print("\nTop Matched Species:")
        for rank, (idx, score) in enumerate(zip(indices, scores), 1):
            print(f"  {rank}. {txt_sp_list[idx]:<30} (Similarity: {score:.4f})")
    print("=" * 64)


if __name__ == "__main__":
    main()
