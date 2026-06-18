"""
config.py — Centralized Hyperparameters & Path Registry
=========================================================
All tunable constants and directory paths for the marine multimodal
alignment pipeline live here.  Import this module everywhere else;
never hard-code magic numbers in other files.
"""

import os
import torch

# ── Directory Layout ───────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Where feature_extractor.py writes its .pt files
EMBEDDING_DIR = os.path.join(BASE_DIR, "extracted_features")

# Where train.py saves checkpoints
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_multimodal_pipeline.pth")

# ── Upstream Dataset Paths (used by feature_extractor.py) ─────────────────────
IMAGE_DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets", "image_dataset", "train")
TEXT_DATASET_ROOT  = os.path.join(
    PROJECT_ROOT, "datasets", "text_dataset", "train", "expanded_train_dataset"
)
AUDIO_DATASET_ROOT = os.path.join(
    PROJECT_ROOT, "datasets", "audio_dataset", "audio_split", "train"
)

# ── Upstream Model Paths (used by feature_extractor.py) ───────────────────────
IMAGE_MODEL_DIR   = os.path.join(PROJECT_ROOT, "training", "image_classification", "models")
TEXT_MODEL_DIR    = os.path.join(
    PROJECT_ROOT, "training", "text_classification", "marine_text_reasoning_model_v4"
)
AUDIO_MODEL_PATH  = os.path.join(
    PROJECT_ROOT, "training", "audio_classification",
    "marine_audio_classification_model", "best_marine_ast_optimized.pth"
)
AST_PRETRAINED_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"

# ── Frozen Encoder Output Dimensions ──────────────────────────────────────────
# ConvNeXtV2-Base backbone  → global_pool="avg"  → 1024-D
IMG_INPUT_DIM = 1024
# all-mpnet-base-v2 fine-tuned → SentenceTransformer → 768-D
TXT_INPUT_DIM = 768
# AST hidden states (mean-pooled before classifier head) → 768-D
# NOTE: verify_features.py will print the actual shape at extraction time.
AUD_INPUT_DIM = 768

# ── Target Latent Space ────────────────────────────────────────────────────────
# Priority 5: Increased from 512 to 768 for better retrieval capacity.
# Experiment A: 768 (same as TXT/AUD encoder output dim, no info bottleneck).
SHARED_DIM = 768

# ── Training Hyperparameters ───────────────────────────────────────────────────
TEMPERATURE  = 0.07
BATCH_SIZE   = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 5e-5
EPOCHS        = 50

# Train / validation split ratio
VAL_SPLIT = 0.20
RANDOM_SEED = 42

# Minimum samples in a masked sub-batch to compute a valid contrastive loss
# (need at least 2 anchors for a meaningful similarity matrix)
MIN_VALID_SAMPLES = 2

# ── Recall Weights for Composite Checkpoint Criterion ─────────────────────────
RECALL_WEIGHT_R1  = 0.6
RECALL_WEIGHT_R5  = 0.3
RECALL_WEIGHT_R10 = 0.1

# ── Dynamic Subset Averaging (Priority 2) ──────────────────────────────────────
# Number of text/audio embeddings randomly sampled per species per __getitem__ call.
# Fallback: min(K, available_samples) if fewer embeddings are stored.
K_TEXT_SUBSET  = 3
K_AUDIO_SUBSET = 3

# ── Hard Negative Mining (Priority 4) ─────────────────────────────────────────
# Top-N most similar negatives retained in the contrastive denominator.
# All non-top-N negatives are masked out with -inf before logsumexp.
HARD_NEG_COUNT = 20

# ── Audio Pre-processing ───────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE    = 16_000   # AST requirement
AUDIO_TARGET_SECONDS = 10

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Ensure output directories exist at import time ────────────────────────────
os.makedirs(EMBEDDING_DIR,  exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
