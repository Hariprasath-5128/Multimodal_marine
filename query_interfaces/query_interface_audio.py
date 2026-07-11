"""
query_interface_audio.py — Zero-Shot Discovery & Retrieval Audio Query Interface
=============================================================================
Features:
  1. Fits a One-Class SVM (OC-SVM) on the training set audio projections.
  2. Pre-computes the target Text prototype Gallery.
  3. Encodes any input audio query (.wav file).
  4. Projects the query embedding to the 768-D shared latent space.
  5. Runs open-set recognition checks:
     - Check A: Out-of-distribution (OOD) outlier detection using OC-SVM.
     - Check B: Absolute similarity threshold against the species text gallery.
     - Check C: Confidence margin threshold between top-1 and other top ranks.
  6. Outputs "New/Unknown acoustic vocalization found!" if any confidence criteria fail.
"""

import os
import sys
import argparse
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Add marine_alignment folder to path
ALIGNMENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "marine_alignment")
sys.path.insert(0, ALIGNMENT_DIR)

from config import (
    CHECKPOINT_PATH_CLOSED, CHECKPOINT_PATH_OPEN, DEVICE,
    TEXT_MODEL_DIR,
)
from dataset import get_test_text_split, make_splits, EMBEDDING_DIR
import models_closed
import models_open
from feature_extractor import (
    _load_audio_encoder,
    extract_audio_embedding_from_file
)

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
    parser = argparse.ArgumentParser(description="Zero-Shot Discovery & Retrieval Audio Query Interface.")
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to query audio file (.wav)")
    parser.add_argument("--model", type=str, choices=["open", "closed"], default="closed",
                        help="Model architecture: open or closed")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Cosine similarity threshold (defaults: closed=0.45, open=0.30)")
    parser.add_argument("--margin_threshold", type=float, default=None,
                        help="Confidence margin threshold (defaults: closed=0.02, open=0.02)")
    parser.add_argument("--device", type=str, default=DEVICE, help="Execution device")
    args = parser.parse_args()

    # Set default similarity and margin thresholds based on typical optimal sweeps
    if args.threshold is None:
        args.threshold = 0.55 if args.model == "closed" else 0.55
    if args.margin_threshold is None:
        args.margin_threshold = 0.015 if args.model == "closed" else 0.015

    print("=" * 64)
    print("  Marine Multimodal Audio Zero-Shot Query & Discovery Interface")
    print("=" * 64)

    if not os.path.exists(args.audio):
        print(f"[ERROR] Query audio file not found at: {args.audio}")
        return

    # 1. Load model pipelines and Text Encoder
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

    # 2. Train One-Class SVM on known training audio projections
    print("\nTraining One-Class SVM on known training audio database...")
    train_files, _ = make_splits()
    raw_train_feats = []
    for fname in train_files:
        path = os.path.join(EMBEDDING_DIR, fname)
        data = torch.load(path, map_location="cpu", weights_only=True)
        if "audio_embs" in data and data["audio_embs"] is not None:
            feats = data["audio_embs"].float()
            if feats.ndim == 1:
                raw_train_feats.append(feats)
            else:
                for idx in range(feats.size(0)):
                    raw_train_feats.append(feats[idx])
    
    raw_train_feats = torch.stack(raw_train_feats).to(args.device)
    with torch.no_grad():
        if args.model == "closed":
            proj_train_feats = pipeline.project_audio(raw_train_feats).cpu().numpy()
        else:
            proj_train_feats = pipeline.audio_head(raw_train_feats).cpu().numpy()
            
    oc_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    oc_svm.fit(proj_train_feats)
    print("One-Class SVM fitted.")

    # 3. Construct Species Text Gallery (Prototypes)
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
    print(f"Text gallery built with {txt_embs.size(0)} species prototypes.")

    # 4. Process query audio file
    print("\nLoading AST audio encoder model...")
    ast_model, ast_extractor, ast_hook = _load_audio_encoder(args.device)

    print("\nProcessing query audio features...")
    raw_audio = extract_audio_embedding_from_file(ast_model, ast_extractor, ast_hook, args.audio, args.device)
    if raw_audio is None:
        print("[ERROR] Failed to extract audio feature from file.")
        return

    with torch.no_grad():
        if args.model == "closed":
            proj_audio = pipeline.project_audio(raw_audio.unsqueeze(0).to(args.device)).squeeze(0).cpu()
        else:
            proj_audio = pipeline.audio_head(raw_audio.unsqueeze(0).to(args.device)).squeeze(0).cpu()
            
    proj_audio_np = proj_audio.numpy().reshape(1, -1)

    # 5. Open-Set Classification Decision
    
    # Check A: One-Class SVM outlier check
    svm_prediction = oc_svm.predict(proj_audio_np)[0]
    
    # Check B & C: Similarity and Margin Check against text gallery
    sims = torch.matmul(proj_audio, txt_embs.T)
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
        print("\n[DECISION]: New/Unknown acoustic vocalization found!")
        print("   This sound recording does not confidently match any of the 75 species.")
    else:
        print("\n[DECISION]: Confident species match found.")
        print("\nTop Matched Species (Text Gallery):")
        for rank, (idx, score) in enumerate(zip(indices, scores), 1):
            print(f"  {rank}. {txt_sp_list[idx]:<30} (Similarity: {score:.4f})")
    print("=" * 64)


if __name__ == "__main__":
    main()
