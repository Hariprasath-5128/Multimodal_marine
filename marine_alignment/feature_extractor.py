"""
feature_extractor.py — Offline Multi-Modal Embedding Generator
================================================================
Runs the three frozen, pretrained encoders once over the entire dataset
and writes one .pt file per image sample to extracted_features/.

Each .pt file is a dict:
    {
        "image_emb"    : FloatTensor [1024]         — always present
        "text_embs"    : FloatTensor [N_txt, 768]   — all text embeddings for species
                         None when no text found
        "audio_embs"   : FloatTensor [N_aud, 768]   — all audio embeddings for species
                         None when no audio found
        "species_name" : str                         — canonical underscore key
    }

Priority 2 change: instead of storing a single mean-pooled text/audio
embedding per species, we now store the FULL STACK of per-document
(text) or per-clip (audio) embeddings.  dataset.py __getitem__() then
randomly samples K of them and averages on-the-fly each epoch, creating
diverse semantic views without additional disk I/O.

Naming convention:  <species_name>_img_<NNNN>.pt
  e.g.  humpback_whale_img_0003.pt

Feature Extraction Strategy
-----------------------------
IMAGE  — ConvNeXtV2-Base from timm, global_pool="avg", no classification head.
         Backbone weights are loaded from the best domain-species checkpoint
         (arbitrary seed42 file from the image_classification/models/ dir).
         Image size: 288 x 288 (matches train.py species model img_size).

TEXT   — marine_text_reasoning_model_v4 SentenceTransformer.
         All .txt files for a species are encoded individually -> [N_txt, 768].
         The full stack is stored in each .pt file under "text_embs".
         dataset.py samples K_TEXT_SUBSET of them per __getitem__ call.

AUDIO  — ASTForAudioClassification reloaded from best_marine_ast_optimized.pth.
         Hook captures the mean-pooled hidden state from the last encoder layer
         (before the classifier head) -> 768-D per clip.
         All audio clips for a species are stacked -> [N_aud, 768] under "audio_embs".
         dataset.py samples K_AUDIO_SUBSET of them per __getitem__ call.

Run
---
    python feature_extractor.py               # all modalities
    python feature_extractor.py --no_audio   # skip audio (fast dry-run)
    python feature_extractor.py --device cpu
    python feature_extractor.py --overwrite  # re-extract existing files
"""

import os
import re
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# ── Allow running from any CWD ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    EMBEDDING_DIR,
    IMAGE_DATASET_ROOT, TEXT_DATASET_ROOT, AUDIO_DATASET_ROOT,
    IMAGE_MODEL_DIR, TEXT_MODEL_DIR, AUDIO_MODEL_PATH, AST_PRETRAINED_ID,
    AUDIO_SAMPLE_RATE, AUDIO_TARGET_SECONDS,
    DEVICE,
)

# ──────────────────────────────────────────────────────────────────────────────
# 0.  Species name normalisation
# ──────────────────────────────────────────────────────────────────────────────

def canonical_species(name: str) -> str:
    """
    Convert any folder/file species name to canonical underscore key.
    e.g. "Atlantic Spotted Dolphin" -> "atlantic_spotted_dolphin"
         "rissos_dolphin"           -> "rissos_dolphin"
    """
    s = name.lower().strip()
    s = re.sub(r"['\-\s]+", "_", s)          # hyphens / spaces -> underscore
    s = re.sub(r"[^a-z0-9_]", "", s)         # strip anything else
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# ──────────────────────────────────────────────────────────────────────────────
# 1.  IMAGE encoder
# ──────────────────────────────────────────────────────────────────────────────

def _load_image_encoder(device: str):
    """
    Load ConvNeXtV2-Base backbone with global_pool="avg" (no head).
    Borrows backbone name and weights from the first available seed42
    species checkpoint to guarantee weight consistency.

    Returns (model, transform)
    """
    try:
        import timm
        from torchvision import transforms
    except ImportError:
        raise ImportError("pip install timm torchvision")

    # Find first available species checkpoint for backbone name
    ckpt_paths = sorted(Path(IMAGE_MODEL_DIR).glob("*_seed42.pth"))
    if not ckpt_paths:
        raise FileNotFoundError(
            f"No *_seed42.pth found in {IMAGE_MODEL_DIR}"
        )

    ckpt = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)
    backbone_name = ckpt.get("backbone", "convnextv2_base")
    img_size      = ckpt.get("img_size", 288)
    print(f"  [Image] backbone={backbone_name}  img_size={img_size}")

    # Build backbone only (no head)
    backbone = timm.create_model(
        backbone_name,
        pretrained=False,
        num_classes=0,
        global_pool="avg",
    )

    # Strip classification head weights from the checkpoint, keep backbone
    raw_sd    = ckpt["model_state_dict"]
    bb_sd     = {k[len("backbone."):]: v
                 for k, v in raw_sd.items()
                 if k.startswith("backbone.")}
    backbone.load_state_dict(bb_sd, strict=True)
    backbone.eval().to(device)

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    return backbone, transform, img_size


@torch.no_grad()
def extract_image_embedding(backbone, transform, img_path: str, device: str) -> torch.Tensor:
    """
    Return 1-D feature tensor [D] for a single image file.
    """
    from PIL import Image
    import pillow_avif
    img = Image.open(img_path).convert("RGB")
    t   = transform(img).unsqueeze(0).to(device)      # [1, C, H, W]
    feat = backbone(t)                                  # [1, D]
    return feat.squeeze(0).cpu()                        # [D]


# ──────────────────────────────────────────────────────────────────────────────
# 2.  TEXT encoder
# ──────────────────────────────────────────────────────────────────────────────

def _load_text_encoder(device: str):
    """Return the marine SentenceTransformer model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("pip install sentence-transformers")

    print(f"  [Text] Loading SentenceTransformer from {TEXT_MODEL_DIR}")
    model = SentenceTransformer(TEXT_MODEL_DIR, device=device)
    model.eval()
    return model


def extract_species_text_embeddings(
    text_model,
    species_key: str,
    text_root:   str = TEXT_DATASET_ROOT,
    device:      str = "cpu",
) -> torch.Tensor | None:
    """
    Priority 2: Encode each .txt document separately and return the
    full stack as a 2-D tensor [N_docs, 768].

    Returns None if no valid text files are found.
    """
    pattern   = os.path.join(text_root, f"{species_key}_*.txt")
    txt_files = sorted(glob.glob(pattern))

    if not txt_files:
        return None

    texts = []
    for fp in txt_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if len(content.split()) > 10:
                texts.append(content)
        except Exception:
            pass

    if not texts:
        return None

    embeddings = text_model.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )                                               # [N, 768]
    # Return the full stack, L2 norm already applied by normalize_embeddings=True
    return embeddings.cpu()                         # [N_docs, 768]


# ──────────────────────────────────────────────────────────────────────────────
# 3.  AUDIO encoder
# ──────────────────────────────────────────────────────────────────────────────

def _load_audio_encoder(device: str):
    """
    Reload ASTForAudioClassification from the fine-tuned checkpoint.
    Registers a forward hook on the last encoder layer to capture
    mean-pooled hidden states (768-D) before the classifier head.

    Returns (model, feature_extractor, hook_container)
    """
    try:
        import torchaudio
        from transformers import ASTForAudioClassification, ASTFeatureExtractor
    except ImportError:
        raise ImportError("pip install transformers torchaudio")

    print(f"  [Audio] Loading AST checkpoint from {AUDIO_MODEL_PATH}")
    print(f"  [Audio] Pretrained config id: {AST_PRETRAINED_ID}")

    feature_extractor = ASTFeatureExtractor.from_pretrained(AST_PRETRAINED_ID)

    # Determine num_labels from the saved state dict
    sd = torch.load(AUDIO_MODEL_PATH, map_location="cpu", weights_only=False)
    classifier_key = None
    for k in sd.keys():
        if "classifier" in k and "weight" in k and sd[k].ndim == 2:
            classifier_key = k
            break
    num_labels = sd[classifier_key].shape[0] if classifier_key else 32

    model = ASTForAudioClassification.from_pretrained(
        AST_PRETRAINED_ID,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    # Hook: capture output of the last transformer encoder layer
    hook_container = {"hidden": None}

    def _hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hook_container["hidden"] = hidden.mean(dim=1)   # [B, 768]

    last_layer = model.audio_spectrogram_transformer.encoder.layer[-1]
    last_layer.register_forward_hook(_hook)

    return model, feature_extractor, hook_container


@torch.no_grad()
def extract_audio_embedding_from_file(
    model,
    feature_extractor,
    hook_container: dict,
    wav_path:       str,
    device:         str,
) -> torch.Tensor | None:
    """
    Extract mean-pooled AST hidden state for one .wav file.
    Returns [768] tensor or None on error.
    """
    try:
        import torchaudio
    except ImportError:
        return None

    try:
        waveform, sr = torchaudio.load(wav_path)
        waveform = waveform.mean(dim=0, keepdim=True)   # mono

        if sr != AUDIO_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, AUDIO_SAMPLE_RATE)
            waveform  = resampler(waveform)

        # Pad / trim to TARGET_SECONDS
        target_len = AUDIO_TARGET_SECONDS * AUDIO_SAMPLE_RATE
        if waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]
        else:
            pad = target_len - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad))

        inputs = feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=AUDIO_SAMPLE_RATE,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        hook_container["hidden"] = None
        _ = model(**inputs)                             # triggers hook
        feat = hook_container["hidden"]                 # [1, 768]

        if feat is None:
            return None

        return feat.squeeze(0).cpu()                    # [768]

    except Exception as e:
        print(f"      [WARN] Audio load failed for {wav_path}: {e}")
        return None


def extract_species_audio_embeddings(
    model,
    feature_extractor,
    hook_container: dict,
    species_key:    str,
    audio_root:     str = AUDIO_DATASET_ROOT,
    device:         str = "cpu",
) -> torch.Tensor | None:
    """
    Priority 2: Extract embedding for each .wav clip individually and
    return the full stack as a 2-D tensor [N_clips, 768].

    Returns None if no audio directory or no valid clips are found.
    """
    species_dir = os.path.join(audio_root, species_key)
    if not os.path.isdir(species_dir):
        # Try common alternate naming conventions
        for d in os.listdir(audio_root):
            if canonical_species(d) == species_key:
                species_dir = os.path.join(audio_root, d)
                break
        else:
            return None

    wav_files = sorted(glob.glob(os.path.join(species_dir, "*.wav")))
    if not wav_files:
        return None

    embeddings = []
    for wf in wav_files:
        emb = extract_audio_embedding_from_file(
            model, feature_extractor, hook_container, wf, device
        )
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        return None

    # Stack all per-clip embeddings: [N_clips, 768]
    stacked = torch.stack(embeddings, dim=0)        # [N_clips, 768]
    # L2-normalise each clip embedding independently
    stacked = F.normalize(stacked, p=2, dim=1)
    return stacked                                  # [N_clips, 768]


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Discover image samples
# ──────────────────────────────────────────────────────────────────────────────

def collect_image_samples(image_root: str) -> list[tuple[str, str, str]]:
    """
    Walk image_dataset/train/<domain>/<species>/<image_files>
    and return list of (species_key, domain, abs_image_path).
    """
    samples = []
    for domain in sorted(os.listdir(image_root)):
        domain_dir = os.path.join(image_root, domain)
        if not os.path.isdir(domain_dir):
            continue
        for species_folder in sorted(os.listdir(domain_dir)):
            species_dir = os.path.join(domain_dir, species_folder)
            if not os.path.isdir(species_dir):
                continue
            species_key = canonical_species(species_folder)
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                for img_path in sorted(glob.glob(os.path.join(species_dir, ext))):
                    samples.append((species_key, domain, img_path))
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Main extraction loop
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    device    = args.device
    skip_text  = args.no_text
    skip_audio = args.no_audio
    overwrite  = args.overwrite

    print("=" * 64)
    print("  Marine Feature Extractor  (Priority 2: full embedding stacks)")
    print("=" * 64)
    print(f"  Device        : {device}")
    print(f"  Output dir    : {EMBEDDING_DIR}")
    print(f"  Skip text     : {skip_text}")
    print(f"  Skip audio    : {skip_audio}")
    print()

    os.makedirs(EMBEDDING_DIR, exist_ok=True)

    # ── Load encoders ──────────────────────────────────────────────────────────
    print("Loading encoders ...")
    image_backbone, img_transform, _ = _load_image_encoder(device)

    text_model = None
    if not skip_text:
        text_model = _load_text_encoder(device)

    audio_model = audio_fe = audio_hook = None
    if not skip_audio:
        audio_model, audio_fe, audio_hook = _load_audio_encoder(device)

    # ── Collect image samples ──────────────────────────────────────────────────
    print("\nCollecting image samples ...")
    samples = collect_image_samples(IMAGE_DATASET_ROOT)
    print(f"  Found {len(samples)} image files across "
          f"{len(set(s[0] for s in samples))} species")

    # ── Pre-compute per-species text & audio embedding stacks ──────────────────
    # Priority 2: store full [N, 768] tensors, not mean-pooled averages.
    print("\nPre-computing species-level text embedding stacks ...")
    species_keys  = sorted(set(s[0] for s in samples))
    text_cache:  dict[str, torch.Tensor | None] = {}
    audio_cache: dict[str, torch.Tensor | None] = {}

    if not skip_text:
        for sp in tqdm(species_keys, desc="text", unit="species"):
            text_cache[sp] = extract_species_text_embeddings(
                text_model, sp, TEXT_DATASET_ROOT, device
            )
        n_txt = sum(v is not None for v in text_cache.values())
        total_docs = sum(v.shape[0] for v in text_cache.values() if v is not None)
        print(f"  Text coverage : {n_txt}/{len(species_keys)} species, {total_docs} total docs")

    if not skip_audio:
        print("\nPre-computing species-level audio embedding stacks ...")
        shape_printed = False
        for sp in tqdm(species_keys, desc="audio", unit="species"):
            emb = extract_species_audio_embeddings(
                audio_model, audio_fe, audio_hook, sp, AUDIO_DATASET_ROOT, device
            )
            audio_cache[sp] = emb
            if emb is not None and not shape_printed:
                print(f"\n  [OK] Audio stack shape: {tuple(emb.shape)}  (N_clips x 768)")
                shape_printed = True
        n_aud = sum(v is not None for v in audio_cache.values())
        total_clips = sum(v.shape[0] for v in audio_cache.values() if v is not None)
        print(f"  Audio coverage: {n_aud}/{len(species_keys)} species, {total_clips} total clips")

    # ── Per-image .pt files ────────────────────────────────────────────────────
    print("\nExtracting image embeddings and writing .pt files ...")

    species_counter: dict[str, int] = {}
    skipped = written = 0

    for species_key, domain, img_path in tqdm(samples, desc="images", unit="img"):
        idx = species_counter.get(species_key, 0)
        species_counter[species_key] = idx + 1

        out_fname = f"{species_key}_img_{idx:04d}.pt"
        out_path  = os.path.join(EMBEDDING_DIR, out_fname)

        if os.path.exists(out_path) and not overwrite:
            skipped += 1
            continue

        # Image embedding [1024]
        img_emb = extract_image_embedding(
            image_backbone, img_transform, img_path, device
        )

        # Text & audio — full stacks from species cache
        # Keys changed from "text_emb"/"audio_emb" -> "text_embs"/"audio_embs"
        # to signal the new format (plural).
        txt_embs = text_cache.get(species_key)  if not skip_text  else None
        aud_embs = audio_cache.get(species_key) if not skip_audio else None

        record = {
            "image_emb":    img_emb,          # [1024]        FloatTensor
            "text_embs":    txt_embs,          # [N_txt, 768]  FloatTensor | None
            "audio_embs":   aud_embs,          # [N_aud, 768]  FloatTensor | None
            "species_name": species_key,
        }
        torch.save(record, out_path)
        written += 1

    print(f"\n{'='*64}")
    print(f"  Done.  Written={written}  Skipped(existing)={skipped}")
    print(f"  Output: {EMBEDDING_DIR}")
    print(f"{'='*64}")
    print("\nRun verify_features.py next to confirm all embeddings are valid.")


def parse_args():
    p = argparse.ArgumentParser(description="Marine Offline Feature Extractor")
    p.add_argument("--device",    type=str, default=DEVICE,
                   help="torch device (cuda/cpu)")
    p.add_argument("--no_text",   action="store_true",
                   help="Skip text embedding extraction")
    p.add_argument("--no_audio",  action="store_true",
                   help="Skip audio embedding extraction")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-extract even if .pt file already exists")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
