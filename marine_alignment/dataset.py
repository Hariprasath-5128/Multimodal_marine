"""
dataset.py — Dynamic Species Registry, Splits & Raw Test Discovery
====================================================================
This is the SINGLE SOURCE OF TRUTH for all data-split logic in the
marine alignment pipeline.  Everything — training splits, validation
splits, and raw test splits — is resolved here.

Sections
--------
  1. Name Canonicalisation
  2. Species Registry  (built from extracted_features/*.pt)
  3. Train / Val Split (of the .pt embedding files)
  4. MarineFeatureDataset (PyTorch Dataset over .pt files)
  5. Raw Test Split Discovery  ← NEW
       image  : datasets/image_dataset/test/  (flat or domain layout)
       text   : datasets/text_dataset/test/expanded_test_dataset/
       audio  : datasets/audio_dataset/audio_split/val/
                  → fallback: datasets/audio_dataset/audio_split/train/

All callers (train.py, evaluate_heads.py, …) import from here.
No split logic lives anywhere else.
"""

import os
import re
import glob
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset, Sampler

from config import (
    EMBEDDING_DIR,
    IMG_INPUT_DIM, TXT_INPUT_DIM, AUD_INPUT_DIM,
    VAL_SPLIT, RANDOM_SEED,
    PROJECT_ROOT,
    K_TEXT_SUBSET, K_AUDIO_SUBSET,
)

# ─────────────────────────────────────────────────────────────────────────────
# Raw dataset root paths (used by test-split discovery)
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_TEST_ROOT  = os.path.join(PROJECT_ROOT, "datasets", "image_dataset",  "test")
TEXT_TEST_ROOT   = os.path.join(PROJECT_ROOT, "datasets", "text_dataset",   "test", "expanded_test_dataset")
AUDIO_VAL_ROOT   = os.path.join(PROJECT_ROOT, "datasets", "audio_dataset",  "audio_split", "val")
AUDIO_TRAIN_ROOT = os.path.join(PROJECT_ROOT, "datasets", "audio_dataset",  "audio_split", "train")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Name Canonicalisation
# ─────────────────────────────────────────────────────────────────────────────

def canonical(name: str) -> str:
    """
    Convert any folder/filename species key to a uniform underscore form.
    Examples
    --------
    "Humpback Whale"        -> "humpback_whale"
    "risso's_dolphin"       -> "rissos_dolphin"
    "indo-pacific dolphin"  -> "indo_pacific_dolphin"
    """
    s = name.lower().strip()
    s = re.sub(r"['\-\s]+", "_", s)        # apostrophes / hyphens / spaces
    s = re.sub(r"[^a-z0-9_]", "", s)       # remove everything else
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# ─────────────────────────────────────────────────────────────────────────────
# 2. Species Registry  (from extracted_features/*.pt)
# ─────────────────────────────────────────────────────────────────────────────

def _build_species_registry(embedding_dir: str):
    """
    Scan embedding_dir for .pt files, collect unique species_name values,
    and return sorted bidirectional maps plus the full file list.

    Returns
    -------
    species_to_idx : dict[str, int]
    idx_to_species : dict[int, str]
    file_list      : list[str]  — bare filenames (not full paths)
    """
    pt_files = sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(embedding_dir, "*.pt"))
    )

    if not pt_files:
        raise FileNotFoundError(
            f"No .pt embedding files found in '{embedding_dir}'.\n"
            "Run feature_extractor.py first to generate them."
        )

    species_set = set()
    for fname in pt_files:
        data = torch.load(
            os.path.join(embedding_dir, fname),
            weights_only=True,
            map_location="cpu",
        )
        species_set.add(data["species_name"])

    sorted_species = sorted(species_set)
    species_to_idx = {name: idx for idx, name in enumerate(sorted_species)}
    idx_to_species = {idx: name for name, idx in species_to_idx.items()}

    return species_to_idx, idx_to_species, pt_files


# Module-level lazy cache
_REGISTRY_CACHE: dict | None = None


def get_registry(embedding_dir: str = EMBEDDING_DIR):
    """Return (species_to_idx, idx_to_species, file_list), cached."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        s2i, i2s, files = _build_species_registry(embedding_dir)
        _REGISTRY_CACHE = {
            "species_to_idx": s2i,
            "idx_to_species": i2s,
            "file_list":      files,
        }
    return (
        _REGISTRY_CACHE["species_to_idx"],
        _REGISTRY_CACHE["idx_to_species"],
        _REGISTRY_CACHE["file_list"],
    )


def get_idx_to_species(embedding_dir: str = EMBEDDING_DIR) -> dict:
    _, i2s, _ = get_registry(embedding_dir)
    return i2s


# ─────────────────────────────────────────────────────────────────────────────
# 3. Train / Val Split  (of .pt embedding files)
# ─────────────────────────────────────────────────────────────────────────────

def make_splits(
    embedding_dir: str = EMBEDDING_DIR,
    val_frac: float = VAL_SPLIT,
    seed: int = RANDOM_SEED,
) -> tuple[list[str], list[str]]:
    """
    Stratified split of all .pt files into train and val lists.
    Stratified by species_name so every species appears in val.

    Returns
    -------
    train_files : list[str]
    val_files   : list[str]
    """
    _, _, all_files = get_registry(embedding_dir)

    species_files: dict[str, list[str]] = defaultdict(list)
    for fname in all_files:
        data = torch.load(
            os.path.join(embedding_dir, fname),
            weights_only=True,
            map_location="cpu",
        )
        species_files[data["species_name"]].append(fname)

    rng = random.Random(seed)
    train_files, val_files = [], []
    for sp, files in species_files.items():
        shuffled = files[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_frac))
        val_files.extend(shuffled[:n_val])
        train_files.extend(shuffled[n_val:])

    return sorted(train_files), sorted(val_files)


# ─────────────────────────────────────────────────────────────────────────────
# 4. MarineFeatureDataset  (PyTorch Dataset over .pt files)
# ─────────────────────────────────────────────────────────────────────────────

class MarineFeatureDataset(Dataset):
    """
    Loads pre-computed multi-modal embedding .pt files.

    Priority 2 — Dynamic Subset Averaging
    ----------------------------------------
    Instead of loading a single pre-averaged text/audio embedding, we
    load the full embedding stack stored by feature_extractor.py and
    randomly sample K_TEXT_SUBSET (or K_AUDIO_SUBSET) embeddings, then
    average them on-the-fly.  Because this sampling is random, each
    epoch presents different semantic views of the same species, which
    creates richer training signal and improves audio alignment.

    Backward compatibility: also handles old-format .pt files that store
    a single 1-D tensor under "text_emb"/"audio_emb" (pre-Priority-2).

    Each sample dict:
        image_emb  : FloatTensor [IMG_INPUT_DIM]   — always present
        text_emb   : FloatTensor [TXT_INPUT_DIM]   — dynamically averaged subset
                                                      zeros if missing
        audio_emb  : FloatTensor [AUD_INPUT_DIM]   — dynamically averaged subset
                                                      zeros if missing
        has_text   : BoolTensor  scalar
        has_audio  : BoolTensor  scalar
        species_id : LongTensor  scalar
        file_name  : str
    """

    def __init__(self, file_list: list[str], embedding_dir: str = EMBEDDING_DIR):
        self.file_list     = file_list
        self.embedding_dir = embedding_dir
        self.species_to_idx, _, _ = get_registry(embedding_dir)

    def _build_image_mapping(self):
        import glob
        from collections import defaultdict
        self.image_paths_per_species = defaultdict(list)
        # Using the exact same logic as feature_extractor.py to guarantee matching indices
        train_root = os.path.join(PROJECT_ROOT, "datasets", "image_dataset", "train")
        for domain in sorted(os.listdir(train_root)):
            domain_dir = os.path.join(train_root, domain)
            if not os.path.isdir(domain_dir): continue
            for species_folder in sorted(os.listdir(domain_dir)):
                species_dir = os.path.join(domain_dir, species_folder)
                if not os.path.isdir(species_dir): continue
                species_key = canonical(species_folder)
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.avif"):
                    for p in sorted(glob.glob(os.path.join(species_dir, ext))):
                        self.image_paths_per_species[species_key].append(p)

    def __len__(self) -> int:
        return len(self.file_list)

    @staticmethod
    def _dynamic_average(
        stack:   torch.Tensor,     # [N, D]  — full embedding stack
        k:       int,              # target sample count
    ) -> torch.Tensor:             # [D]     — averaged subset
        """
        Randomly sample min(k, N) rows from stack and return their mean.
        The result is L2-normalised so it stays on the unit sphere.
        """
        n = stack.size(0)
        k_actual = min(k, n)
        if k_actual == n:
            # Use all (e.g. species with few docs/clips)
            subset = stack
        else:
            idx    = torch.randperm(n)[:k_actual]
            subset = stack[idx]         # [k_actual, D]
        mean_emb = subset.mean(dim=0)   # [D]
        return torch.nn.functional.normalize(mean_emb, p=2, dim=0)

    def __getitem__(self, idx: int) -> dict:
        fname     = self.file_list[idx]
        file_path = os.path.join(self.embedding_dir, fname)
        data      = torch.load(file_path, weights_only=True, map_location="cpu")

        species_name = data["species_name"]
        species_id   = self.species_to_idx[species_name]

        # ── Image — Pre-computed Embeddings ──────────────────────────────────────────
        # Load the frozen 1024-D image embedding from the .pt file directly.
        # This reduces I/O bottleneck and removes the need for ConvNeXt forward passes.
        if "image_emb" in data and data["image_emb"] is not None:
            image_tensor = data["image_emb"].float().squeeze()
        else:
            image_tensor = torch.zeros(IMG_INPUT_DIM, dtype=torch.float32)

        # ── Text — dynamic subset average (Priority 2) ─────────────────────────
        # New format: "text_embs" -> [N_docs, 768]
        # Old format: "text_emb"  -> [768]  (backward compat)
        if "text_embs" in data and data["text_embs"] is not None:
            stack    = data["text_embs"].float()        # [N_docs, 768]
            if stack.dim() == 1:
                # Fallback: single vector stored without batch dimension
                text_emb = stack
            else:
                text_emb = self._dynamic_average(stack, K_TEXT_SUBSET)
            has_text = True
        elif "text_emb" in data and data["text_emb"] is not None:
            # Legacy single-vector format
            text_emb = data["text_emb"].float().squeeze()
            has_text = True
        else:
            text_emb = torch.zeros(TXT_INPUT_DIM, dtype=torch.float32)
            has_text = False

        # ── Audio — dynamic subset average (Priority 2) ────────────────────────
        # New format: "audio_embs" -> [N_clips, 768]
        # Old format: "audio_emb"  -> [768]  (backward compat)
        if "audio_embs" in data and data["audio_embs"] is not None:
            stack = data["audio_embs"].float()          # [N_clips, 768]
            if stack.dim() == 1:
                audio_emb = stack
            else:
                audio_emb = self._dynamic_average(stack, K_AUDIO_SUBSET)
            has_audio = True
        elif "audio_emb" in data and data["audio_emb"] is not None:
            # Legacy single-vector format
            audio_emb = data["audio_emb"].float().squeeze()
            has_audio = True
        else:
            audio_emb = torch.zeros(AUD_INPUT_DIM, dtype=torch.float32)
            has_audio = False

        return {
            "image_tensor": image_tensor,
            "text_emb":     text_emb,
            "audio_emb":    audio_emb,
            "has_text":     torch.tensor(has_text,   dtype=torch.bool),
            "has_audio":    torch.tensor(has_audio,  dtype=torch.bool),
            "species_id":   torch.tensor(species_id, dtype=torch.long),
            "file_name":    fname,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. SpeciesBalancedSampler (Priority 7 — Fix for SupCon)
# ─────────────────────────────────────────────────────────────────────────────

class SpeciesBalancedSampler(Sampler):
    """
    Yields batches where each species has exactly `k_samples` instances.
    This guarantees that the `positive_mask` in the Supervised Contrastive Loss
    is never completely empty for any sample, preventing 0-loss collapse.

    If a species has fewer than `k_samples` files, we sample with replacement.
    """
    def __init__(self, dataset: MarineFeatureDataset, batch_size: int, k_samples: int = 2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k_samples = k_samples
        self.num_species_per_batch = batch_size // k_samples
        
        self.species_to_indices = defaultdict(list)
        
        # Build mapping of species -> indices in the given dataset split
        for idx, fname in enumerate(dataset.file_list):
            file_path = os.path.join(dataset.embedding_dir, fname)
            # Load only the tiny metadata parts if possible, but standard torch.load is fast enough
            data = torch.load(file_path, weights_only=True, map_location="cpu")
            sp_name = data["species_name"]
            sp_idx = dataset.species_to_idx[sp_name]
            self.species_to_indices[sp_idx].append(idx)
            
        self.species_ids = list(self.species_to_indices.keys())
        
        # Determine number of batches to cover approximately the dataset size
        self.num_batches = len(dataset) // batch_size
        if len(dataset) % batch_size != 0:
            self.num_batches += 1

    def __iter__(self):
        for _ in range(self.num_batches):
            if len(self.species_ids) >= self.num_species_per_batch:
                selected_species = random.sample(self.species_ids, self.num_species_per_batch)
            else:
                selected_species = random.choices(self.species_ids, k=self.num_species_per_batch)
                
            batch_indices = []
            for sp_id in selected_species:
                sp_indices = self.species_to_indices[sp_id]
                if len(sp_indices) >= self.k_samples:
                    batch_indices.extend(random.sample(sp_indices, self.k_samples))
                else:
                    batch_indices.extend(random.choices(sp_indices, k=self.k_samples))
                    
            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return self.num_batches


# ─────────────────────────────────────────────────────────────────────────────
# 6. Raw Test Split Discovery
# ─────────────────────────────────────────────────────────────────────────────
#
# Each function below follows the same contract:
#   - Try the dedicated test/val split first.
#   - If it does not exist (or is empty), fall back to the train split.
#   - Return a list of (species_key: str, path: str) tuples so callers
#     can iterate and extract features on-the-fly.
#   - Also return a metadata dict that describes which source was used,
#     so evaluate_heads.py can report it transparently.
#
# ─────────────────────────────────────────────────────────────────────────────

IMG_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.avif")


def get_test_image_split() -> tuple[list[tuple[str, str]], dict]:
    """
    Discover test image files.

    Layout auto-detection
    ---------------------
    The image test folder supports two layouts:
      Flat   : test/<species>/<images>          (no domain subfolder)
      Domain : test/<domain>/<species>/<images> (extra level)
    Detected automatically: if a child dir of test/ contains images
    directly -> flat; otherwise it drills one more level.

    Fallback
    --------
    If IMAGE_TEST_ROOT does not exist or is empty, falls back to
    datasets/image_dataset/train/.

    Returns
    -------
    samples  : list of (species_key, abs_image_path)
    meta     : {"source": "test"|"train", "layout": "flat"|"domain",
                "n_species": int, "n_images": int}
    """
    def _collect_from_root(root: str) -> list[tuple[str, str]]:
        samples = []
        for entry in sorted(os.listdir(root)):
            entry_path = os.path.join(root, entry)
            if not os.path.isdir(entry_path):
                continue
            # Check if this folder directly contains images (flat layout)
            has_images = any(
                glob.glob(os.path.join(entry_path, ext))
                for ext in IMG_EXTENSIONS
            )
            if has_images:
                sp = canonical(entry)
                for ext in IMG_EXTENSIONS:
                    for p in sorted(glob.glob(os.path.join(entry_path, ext))):
                        samples.append((sp, p))
            else:
                # Domain layout — drill one level deeper
                for sp_folder in sorted(os.listdir(entry_path)):
                    sp_path = os.path.join(entry_path, sp_folder)
                    if not os.path.isdir(sp_path):
                        continue
                    sp = canonical(sp_folder)
                    for ext in IMG_EXTENSIONS:
                        for p in sorted(glob.glob(os.path.join(sp_path, ext))):
                            samples.append((sp, p))
        return samples

    # Try test split first
    if os.path.isdir(IMAGE_TEST_ROOT):
        samples = _collect_from_root(IMAGE_TEST_ROOT)
        if samples:
            species = set(s for s, _ in samples)
            return samples, {
                "source":   "test",
                "n_species": len(species),
                "n_images":  len(samples),
            }

    # Fallback: train split
    train_root = os.path.join(PROJECT_ROOT, "datasets", "image_dataset", "train")
    samples = _collect_from_root(train_root)
    species = set(s for s, _ in samples)
    return samples, {
        "source":    "train (fallback — no test split found)",
        "n_species": len(species),
        "n_images":  len(samples),
    }


def get_test_text_split() -> tuple[dict[str, list[str]], dict]:
    """
    Discover test text files per species.

    Split priority
    --------------
    1. datasets/text_dataset/test/expanded_test_dataset/
       Pattern: <species>_test_<N>_<Type>.txt
    2. Fallback: datasets/text_dataset/train/expanded_train_dataset/
       Pattern: <species>_<N>_<Type>.txt

    Returns
    -------
    texts  : dict[species_key -> list[abs_file_path]]
    meta   : {"source": str, "n_species": int, "n_docs": int}
    """
    def _collect_test(root: str) -> dict[str, list[str]]:
        result = defaultdict(list)
        if not os.path.isdir(root):
            return result
        for fname in sorted(os.listdir(root)):
            if not fname.endswith(".txt") or fname.startswith("_"):
                continue
            # Pattern: <species>_test_<N>_<Type>.txt
            m = re.match(r"^(.+?)_test_\d+_", fname)
            if m:
                sp = canonical(m.group(1))
                result[sp].append(os.path.join(root, fname))
        return result

    def _collect_train(root: str) -> dict[str, list[str]]:
        result = defaultdict(list)
        if not os.path.isdir(root):
            return result
        for fname in sorted(os.listdir(root)):
            if not fname.endswith(".txt") or fname.startswith("_"):
                continue
            # Pattern: <species>_<N>_<Type>.txt
            m = re.match(r"^(.+?)_\d+_", fname)
            if m:
                sp = canonical(m.group(1))
                result[sp].append(os.path.join(root, fname))
        return result

    # Try test split
    texts = _collect_test(TEXT_TEST_ROOT)
    if texts:
        n_docs = sum(len(v) for v in texts.values())
        return dict(texts), {
            "source":    "test",
            "n_species": len(texts),
            "n_docs":    n_docs,
        }

    # Fallback: train split
    train_root = os.path.join(
        PROJECT_ROOT, "datasets", "text_dataset", "train", "expanded_train_dataset"
    )
    texts = _collect_train(train_root)
    n_docs = sum(len(v) for v in texts.values())
    return dict(texts), {
        "source":    "train (fallback — no test split found)",
        "n_species": len(texts),
        "n_docs":    n_docs,
    }


def get_test_audio_split() -> tuple[dict[str, list[str]], dict]:
    """
    Discover test audio (.wav) files per species.

    Split priority (per species independently)
    ------------------------------------------
    1. audio_split/val/<species>/*.wav   — dedicated val set
    2. audio_split/train/<species>/*.wav — fallback when val is empty
    3. Species skipped if neither split has audio.

    Returns
    -------
    audio  : dict[species_key -> list[abs_wav_path]]
              Only species with at least one .wav file are included.
    meta   : {"source_per_species": dict[sp -> "val"|"train"|"missing"],
              "n_val": int, "n_train_fallback": int, "n_missing": int}
    """
    # Build lookup: canonical_name -> original_folder_name, for both roots
    def _folder_map(root: str) -> dict[str, str]:
        m = {}
        if not os.path.isdir(root):
            return m
        for d in os.listdir(root):
            if os.path.isdir(os.path.join(root, d)):
                m[canonical(d)] = d
        return m

    val_map   = _folder_map(AUDIO_VAL_ROOT)
    train_map = _folder_map(AUDIO_TRAIN_ROOT)

    # Candidate species = union of everything found in both splits
    all_sp = sorted(set(val_map) | set(train_map))

    audio: dict[str, list[str]] = {}
    source_per_species: dict[str, str] = {}

    for sp in all_sp:
        wavs = []

        # Try val first
        if sp in val_map:
            d = os.path.join(AUDIO_VAL_ROOT, val_map[sp])
            wavs = sorted(glob.glob(os.path.join(d, "*.wav")))
            if wavs:
                source_per_species[sp] = "val"

        # Fallback to train if val was empty
        if not wavs and sp in train_map:
            d = os.path.join(AUDIO_TRAIN_ROOT, train_map[sp])
            wavs = sorted(glob.glob(os.path.join(d, "*.wav")))
            if wavs:
                source_per_species[sp] = "train (fallback)"

        if wavs:
            audio[sp] = wavs
        else:
            source_per_species[sp] = "missing"

    n_val      = sum(1 for s in source_per_species.values() if s == "val")
    n_fallback = sum(1 for s in source_per_species.values() if "fallback" in s)
    n_missing  = sum(1 for s in source_per_species.values() if s == "missing")

    return audio, {
        "source_per_species":  source_per_species,
        "n_val":               n_val,
        "n_train_fallback":    n_fallback,
        "n_missing":           n_missing,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── .pt registry ──────────────────────────────────────────────────────────
    print("=== .pt Registry ===")
    try:
        s2i, i2s, files = get_registry()
        print(f"  {len(s2i)} species, {len(files)} .pt files")
        train_f, val_f = make_splits()
        print(f"  Train={len(train_f)}  Val={len(val_f)}")
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")

    # ── Raw test splits ────────────────────────────────────────────────────────
    print("\n=== Raw Test Splits ===")

    img_samples, img_meta = get_test_image_split()
    print(f"  Image  : {img_meta['n_images']} images, "
          f"{img_meta['n_species']} species  [source={img_meta['source']}]")

    txt_files, txt_meta = get_test_text_split()
    print(f"  Text   : {txt_meta['n_docs']} docs, "
          f"{txt_meta['n_species']} species  [source={txt_meta['source']}]")

    aud_files, aud_meta = get_test_audio_split()
    print(f"  Audio  : {len(aud_files)} species with audio  "
          f"[val={aud_meta['n_val']}, "
          f"train-fallback={aud_meta['n_train_fallback']}, "
          f"missing={aud_meta['n_missing']}]")
