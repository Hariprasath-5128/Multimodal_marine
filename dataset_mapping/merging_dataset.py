"""
============================================================================
 MARINE MULTIMODAL DATASET MERGING PIPELINE  (v2 — from scratch)
 ============================================================================
 Pipeline order:
   1. Import image dataset  →  save locally to datasets/images/
   2. Import audio dataset  →  save locally to datasets/audio/
   3. Match/unmatch report   (with source tracking)
   4. Download missing images & audio from trustable sites
   5. Map everything with shared common labels
   6. Generate text descriptions for ALL species
   7. Final combined CSV with shared labels for all 3 modalities
============================================================================
"""

import os
import re
import json
import time
import shutil
import requests
import numpy as np
import pandas as pd
from io import BytesIO
from pathlib import Path
from difflib import SequenceMatcher
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from PIL import Image
import soundfile as sf

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

BASE_DIR    = Path(r"c:\Projects\marine")
DATASET_DIR = BASE_DIR / "datasets"
IMAGE_DIR   = DATASET_DIR / "images"
AUDIO_DIR   = DATASET_DIR / "audio"
TEXT_DIR    = DATASET_DIR / "text"
OUTPUT_CSV  = DATASET_DIR / "marine_multimodal_complete.csv"
REPORT_CSV  = DATASET_DIR / "species_match_report.csv"
LABEL_JSON  = DATASET_DIR / "shared_label_ids.json"

for d in [DATASET_DIR, IMAGE_DIR, AUDIO_DIR, TEXT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TIMEOUT   = 10
API_DELAY = 0.3

# ─────────────────────────────────────────────────────
# LABEL NORMALIZATION  &  MANUAL MAPPING TABLE
# ─────────────────────────────────────────────────────

def normalize(name: str) -> str:
    """Lowercase, remove underscores/commas, collapse spaces."""
    if pd.isna(name):
        return ""
    s = str(name).strip().lower()
    s = s.replace("_", " ").replace(",", "")
    return re.sub(r"\s+", " ", s).strip()


# Audio "species" column  →  canonical form
AUDIO_MAP = {
    "atlantic spotted dolphin":         "atlantic spotted dolphin",
    "bearded seal":                     "bearded seal",
    "beluga white whale":               "beluga whale",
    "bottlenose dolphin":               "bottlenose dolphin",
    "bowhead whale":                    "bowhead whale",
    "clymene dolphin":                  "clymene dolphin",
    "common dolphin":                   "common dolphin",
    "false killer whale":               "false killer whale",
    "fin finback whale":                "fin whale",
    "frasers dolphin":                  "fraser's dolphin",
    "grampus rissos dolphin":           "risso's dolphin",
    "harp seal":                        "harp seal",
    "humpback whale":                   "humpback whale",
    "killer whale":                     "killer whale",
    "leopard seal":                     "leopard seal",
    "long-finned pilot whale":          "long-finned pilot whale",
    "melon headed whale":               "melon-headed whale",
    "minke whale":                      "minke whale",
    "narwhal":                          "narwhal",
    "northern right whale":             "north atlantic right whale",
    "pantropical spotted dolphin":      "pantropical spotted dolphin",
    "ross seal":                        "ross seal",
    "rough-toothed dolphin":            "rough-toothed dolphin",
    "short-finned pacific pilot whale": "short-finned pilot whale",
    "southern right whale":             "southern right whale",
    "sperm whale":                      "sperm whale",
    "spinner dolphin":                  "spinner dolphin",
    "striped dolphin":                  "striped dolphin",
    "walrus":                           "walrus",
    "weddell seal":                     "weddell seal",
    "white-beaked dolphin":             "white-beaked dolphin",
    "white-sided dolphin":              "white-sided dolphin",
}

# Image "animal_name" overrides  →  canonical form
IMAGE_MAP = {
    "beluga":                           "beluga whale",
    "risso's dolphin":                  "risso's dolphin",
    "fraser's dolphin":                 "fraser's dolphin",
    "north atlantic right whale":       "north atlantic right whale",
}

TAXONOMY_KW = {
    "cetaceans": ["whale", "dolphin", "porpoise", "narwhal", "beluga", "orca", "killer whale", "vaquita"],
    "pinnipeds": ["seal", "sea lion", "walrus", "fur seal"],
    "sirenians": ["manatee", "dugong", "sea cow"],
    "mustelids": ["sea otter", "otter"],
    "ursids":    ["polar bear"],
}

def taxonomy(label: str) -> str:
    l = label.lower()
    for grp, kws in TAXONOMY_KW.items():
        for kw in kws:
            if kw in l:
                return grp
    return "other"

def fuzzy(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def safe_name(label: str) -> str:
    return label.replace(" ", "_").replace("'", "")


# ═════════════════════════════════════════════
#  STEP 1 ▸ IMPORT IMAGE DATASET
# ═════════════════════════════════════════════

print("=" * 70)
print(" STEP 1 / 7 — Importing Image Dataset")
print("=" * 70)

# Retry wrapper for slow/flaky connections
def load_with_retry(dataset_name, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Loading {dataset_name} (attempt {attempt}/{max_retries}) ...")
            ds = load_dataset(dataset_name)
            return ds
        except Exception as e:
            print(f"  ⚠ Attempt {attempt} failed: {type(e).__name__}")
            if attempt < max_retries:
                wait = 10 * attempt
                print(f"    Retrying in {wait}s ...")
                time.sleep(wait)
            else:
                print(f"  ✗ All {max_retries} attempts failed. Check your internet.")
                raise

image_ds  = load_with_retry("yeyimilk/LLM-Vision-Marine-Animals")
image_all = concatenate_datasets([image_ds[s] for s in image_ds])
image_df  = image_all.to_pandas()

print(f"  Loaded {len(image_df)} images, columns: {list(image_df.columns)}")

# Save every image to datasets/images/<species>/<species>_NNN.jpg
print("  Saving images to disk ...")
img_saved = 0
img_paths = []   # parallel list — one path per row

for idx, row in tqdm(image_df.iterrows(), total=len(image_df), desc="  Images"):
    raw_label = row["animal_name"]
    norm      = normalize(raw_label)
    species_dir = IMAGE_DIR / safe_name(norm)
    species_dir.mkdir(exist_ok=True)
    fpath = species_dir / f"{safe_name(norm)}_{idx}.jpg"

    try:
        obj = row["image"]
        if isinstance(obj, dict) and "bytes" in obj and obj["bytes"]:
            img = Image.open(BytesIO(obj["bytes"]))
        elif isinstance(obj, Image.Image):
            img = obj
        elif isinstance(obj, dict) and "path" in obj and obj["path"]:
            img = Image.open(obj["path"])
        else:
            img = Image.open(BytesIO(obj))
        img.convert("RGB").save(str(fpath), "JPEG", quality=90)
        img_saved += 1
        img_paths.append(str(fpath))
    except Exception as e:
        img_paths.append(None)
        if img_saved < 3:
            print(f"    ⚠ image {idx}: {e}")

image_df["local_path"] = img_paths
print(f"  ✓ Images saved: {img_saved}/{len(image_df)}")


# ═════════════════════════════════════════════
#  STEP 2 ▸ IMPORT AUDIO DATASET
# ═════════════════════════════════════════════

print("\n" + "=" * 70)
print(" STEP 2 / 7 — Importing Audio Dataset")
print("=" * 70)

audio_ds  = load_with_retry("ardavey/marine_ocean_mammal_sound")
audio_all = concatenate_datasets([audio_ds[s] for s in audio_ds])
audio_df  = audio_all.to_pandas()

print(f"  Loaded {len(audio_df)} audio clips, columns: {list(audio_df.columns)}")

# Save every audio to datasets/audio/<species>/<species>_NNN.wav
# to_pandas() gives audio as {'bytes':..., 'path':...}  — decode with soundfile
print("  Saving audio to disk ...")
aud_saved = 0
aud_fail  = 0
aud_paths = []

for idx, row in tqdm(audio_df.iterrows(), total=len(audio_df), desc="  Audio"):
    raw_label = row["species"]
    norm      = normalize(raw_label)
    species_dir = AUDIO_DIR / safe_name(norm)
    species_dir.mkdir(exist_ok=True)
    fpath = species_dir / f"{safe_name(norm)}_{idx}.wav"

    try:
        obj = row["audio"]
        if isinstance(obj, dict) and "bytes" in obj and obj["bytes"]:
            arr, sr = sf.read(BytesIO(obj["bytes"]))
            sf.write(str(fpath), np.asarray(arr, dtype=np.float32), sr)
            aud_saved += 1
            aud_paths.append(str(fpath))
        elif isinstance(obj, dict) and "path" in obj and obj["path"] and os.path.exists(obj["path"]):
            shutil.copy2(obj["path"], str(fpath))
            aud_saved += 1
            aud_paths.append(str(fpath))
        elif isinstance(obj, dict) and "array" in obj:
            arr = np.asarray(obj["array"], dtype=np.float32)
            sr  = obj.get("sampling_rate", 16000)
            sf.write(str(fpath), arr, sr)
            aud_saved += 1
            aud_paths.append(str(fpath))
        else:
            aud_paths.append(None)
            aud_fail += 1
    except Exception as e:
        aud_paths.append(None)
        aud_fail += 1
        if aud_fail <= 3:
            print(f"    ⚠ audio {idx}: {e}")

audio_df["local_path"] = aud_paths
print(f"  ✓ Audio saved: {aud_saved}/{len(audio_df)}")
if aud_fail:
    print(f"  ⚠ Audio failed: {aud_fail}")


# ═════════════════════════════════════════════
#  STEP 3 ▸ MATCH / UNMATCH REPORT
# ═════════════════════════════════════════════

print("\n" + "=" * 70)
print(" STEP 3 / 7 — Species Match / Unmatch Report")
print("=" * 70)

# Build canonical labels
image_df["norm"] = image_df["animal_name"].apply(normalize)
audio_df["norm"] = audio_df["species"].apply(normalize)

# Map to canonical
def canon_audio(n):
    return AUDIO_MAP.get(n, n)

def canon_image(n):
    if n in IMAGE_MAP:
        return IMAGE_MAP[n]
    # fuzzy match against audio canonical values
    best, best_s = n, 0.0
    for cv in set(AUDIO_MAP.values()):
        s = fuzzy(n, cv)
        if s > best_s:
            best_s, best = s, cv
    return best if best_s >= 0.80 else n

image_df["shared_label"] = image_df["norm"].apply(canon_image)
audio_df["shared_label"] = audio_df["norm"].apply(canon_audio)

img_species = set(image_df["shared_label"].unique())
aud_species = set(audio_df["shared_label"].unique())

matched      = sorted(img_species & aud_species)
only_image   = sorted(img_species - aud_species)
only_audio   = sorted(aud_species - img_species)
all_species  = sorted(img_species | aud_species)

# Build report dataframe
report_rows = []
for sp in all_species:
    has_img = sp in img_species
    has_aud = sp in aud_species
    n_img   = len(image_df[image_df["shared_label"] == sp])
    n_aud   = len(audio_df[audio_df["shared_label"] == sp])
    report_rows.append({
        "shared_label":  sp,
        "taxonomy":      taxonomy(sp),
        "has_image":     has_img,
        "image_count":   n_img,
        "image_source":  "HuggingFace dataset" if has_img else "MISSING → will download",
        "has_audio":     has_aud,
        "audio_count":   n_aud,
        "audio_source":  "HuggingFace dataset" if has_aud else "MISSING → will download",
        "status":        "MATCHED" if (has_img and has_aud) else "UNMATCHED",
    })

report_df = pd.DataFrame(report_rows)
report_df.to_csv(REPORT_CSV, index=False)

# Print summary
print(f"\n  Total unique species : {len(all_species)}")
print(f"  ✓ Matched (both)    : {len(matched)}")
print(f"  ✗ Image-only         : {len(only_image)}")
print(f"  ✗ Audio-only         : {len(only_audio)}")

print(f"\n  {'Species':<40s} {'Image':>7s} {'Audio':>7s}  Status")
print("  " + "─" * 70)
for _, r in report_df.iterrows():
    m = "✓" if r["status"] == "MATCHED" else "✗"
    print(f"  {m} {r['shared_label']:<38s} {r['image_count']:>7d} {r['audio_count']:>7d}  {r['status']}")
print("  " + "─" * 70)

print(f"\n  🖼  UNMATCHED — Missing images (have audio only): {len(only_audio)}")
for i, sp in enumerate(only_audio, 1):
    print(f"      {i:>2d}. {sp}")

print(f"\n  🔊 UNMATCHED — Missing audio (have images only): {len(only_image)}")
for i, sp in enumerate(only_image, 1):
    print(f"      {i:>2d}. {sp}")

print(f"\n  📄 Report saved → {REPORT_CSV}")


# ═════════════════════════════════════════════
#  STEP 4 ▸ DOWNLOAD MISSING IMAGES & AUDIO
# ═════════════════════════════════════════════

print("\n" + "=" * 70)
print(" STEP 4 / 7 — Downloading Missing Data")
print("=" * 70)

# ── 4A  Missing IMAGES ──
# Using Wikimedia Commons — trusted free media repository.
NUM_IMAGES = 9

def download_wikimedia_images(species, save_dir, num_images=9):
    """Download images from Wikimedia Commons search."""
    from urllib.parse import quote

    headers = {
        "User-Agent": "MarineResearchBot/1.0 (hariprasath1528@gmail.com)"
    }

    paths = []

    try:
        search_url = (
            "https://commons.wikimedia.org/w/api.php"
            "?action=query"
            "&format=json"
            "&generator=search"
            "&gsrnamespace=6"
            f"&gsrsearch={quote(species)}"
            f"&gsrlimit={num_images}"
            "&prop=imageinfo"
            "&iiprop=url"
        )

        r = requests.get(search_url, headers=headers, timeout=15)

        if r.status_code != 200:
            print(f"    ⚠ HTTP {r.status_code}")
            return []

        data = r.json()

        if "query" not in data:
            return []

        pages = data["query"].get("pages", {})

        for page in pages.values():
            if len(paths) >= num_images:
                break

            imageinfo = page.get("imageinfo")
            if not imageinfo:
                continue

            img_url = imageinfo[0]["url"]

            # Skip non-image files (SVGs, PDFs, etc.)
            if any(img_url.lower().endswith(ext) for ext in (".svg", ".pdf", ".ogv", ".webm")):
                continue

            try:
                img_resp = requests.get(img_url, headers=headers, timeout=15)
                img = Image.open(BytesIO(img_resp.content)).convert("RGB")
                save_path = save_dir / f"{safe_name(species)}_wiki_{len(paths)}.jpg"
                img.save(str(save_path), "JPEG", quality=90)
                paths.append(str(save_path))
            except Exception:
                continue  # skip unreadable files

    except Exception as e:
        print(f"    ⚠ Wikimedia error: {e}")

    return paths

downloaded_images = {}  # species -> list of paths

print("\n  🖼  Downloading missing images (Wikimedia Commons) ...")
for sp in tqdm(only_audio, desc="  Missing images"):
    d = IMAGE_DIR / safe_name(sp)
    d.mkdir(exist_ok=True)

    # Skip if images already exist
    existing = list(d.glob("*.jpg")) + list(d.glob("*.png"))
    if existing:
        print(f"    ⏭  {sp} — already has {len(existing)} images")
        downloaded_images[sp] = [str(p) for p in existing]
        continue

    paths = download_wikimedia_images(sp, d, NUM_IMAGES)
    if paths:
        downloaded_images[sp] = paths
        print(f"    ✓ {sp}  ({len(paths)} images)")
    else:
        print(f"    ✗ {sp}")
    time.sleep(0.5)

print(f"  ✓ Downloaded images: {len(downloaded_images)}/{len(only_audio)} species")




# ═══════════════════════════════════════════════════════════
#  STEP 5 ▸ MAP EVERYTHING WITH SHARED COMMON LABELS
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" STEP 5 / 7 — Mapping with Shared Common Labels")
print("=" * 70)

records = []

# ── images from HuggingFace dataset ──
for idx, row in image_df.iterrows():
    if row["local_path"] is None:
        continue
    records.append({
        "shared_label":    row["shared_label"],
        "shared_label_id": None,          # filled later
        "modality":        "image",
        "taxonomy_group":  taxonomy(row["shared_label"]),
        "data_path":       row["local_path"],
        "original_label":  row["animal_name"],
        "source":          "HuggingFace:yeyimilk/LLM-Vision-Marine-Animals",
    })

# ── images downloaded for audio-only species ──
for sp, paths in downloaded_images.items():
    for path in paths:
        records.append({
            "shared_label":    sp,
            "shared_label_id": None,
            "modality":        "image",
            "taxonomy_group":  taxonomy(sp),
            "data_path":       path,
            "original_label":  sp,
            "source":          "Downloaded:WikimediaCommons",
        })

# ── audio from HuggingFace dataset ──
for idx, row in audio_df.iterrows():
    if row["local_path"] is None:
        continue
    records.append({
        "shared_label":    row["shared_label"],
        "shared_label_id": None,
        "modality":        "audio",
        "taxonomy_group":  taxonomy(row["shared_label"]),
        "data_path":       row["local_path"],
        "original_label":  row["species"],
        "source":          "HuggingFace:ardavey/marine_ocean_mammal_sound",
    })



# ── assign shared_label_id ──
label_list = sorted({r["shared_label"] for r in records})
label2id   = {lab: i for i, lab in enumerate(label_list)}

for r in records:
    r["shared_label_id"] = label2id[r["shared_label"]]

print(f"  Records so far (image + audio): {len(records)}")
print(f"  Unique species with data      : {len(label_list)}")


# ═══════════════════════════════════════════════════════════
#  STEP 6 ▸ TEXT DESCRIPTIONS FOR ALL SPECIES
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" STEP 6 / 7 — Generating Text Descriptions (Wikipedia)")
print("=" * 70)

def get_wiki_text(species: str):
    """Fetch Wikipedia intro text. Three attempts with different query forms."""
    from urllib.parse import quote
    attempts = [
        species.replace(" ", "_").title(),
        species.replace(" ", "_"),
        species.replace(" ", "_") + "_(animal)",
    ]
    for q in attempts:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(q)}"
        try:
            r = requests.get(url, timeout=TIMEOUT,
                             headers={"User-Agent": "MarineBot/1.0"})
            if r.status_code == 200:
                txt = r.json().get("extract", "")
                if txt and len(txt) > 30:
                    return txt
        except Exception:
            pass
    return None

text_ok   = 0
text_fall = []

print("  Fetching descriptions ...")
for sp in tqdm(label_list, desc="  Wikipedia text"):
    txt = get_wiki_text(sp)

    if txt is None:
        grp = taxonomy(sp)
        txt = (f"The {sp} is a marine mammal belonging to the {grp} group. "
               f"It inhabits ocean and coastal habitats worldwide and plays "
               f"an important role in marine ecosystems.")
        text_fall.append(sp)

    # Save text file
    txt_path = TEXT_DIR / f"{safe_name(sp)}.txt"
    txt_path.write_text(txt, encoding="utf-8")

    # Ensure it's in label2id (it should be)
    sid = label2id.get(sp)
    if sid is None:
        sid = len(label2id)
        label2id[sp] = sid

    records.append({
        "shared_label":    sp,
        "shared_label_id": sid,
        "modality":        "text",
        "taxonomy_group":  taxonomy(sp),
        "data_path":       str(txt_path),
        "original_label":  sp,
        "source":          "Wikipedia" if sp not in text_fall else "Fallback placeholder",
    })
    text_ok += 1
    time.sleep(API_DELAY)

print(f"  ✓ Text entries: {text_ok}")
if text_fall:
    print(f"  ⚠ Fallback text for {len(text_fall)} species: {text_fall}")


# ═══════════════════════════════════════════════════════════
#  STEP 7 ▸ FINAL COMBINED DATASET
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(" STEP 7 / 7 — Building Final Combined Dataset")
print("=" * 70)

final_df = pd.DataFrame(records)
final_df = final_df.sort_values(["shared_label", "modality"]).reset_index(drop=True)
final_df = final_df[[
    "shared_label", "shared_label_id", "modality",
    "taxonomy_group", "data_path", "original_label", "source"
]]

final_df.to_csv(OUTPUT_CSV, index=False)

with open(LABEL_JSON, "w") as f:
    json.dump(label2id, f, indent=2)

# ── Coverage report ──
mod_counts = final_df["modality"].value_counts()
print(f"\n  📊 Total rows: {len(final_df)}")
for m, c in mod_counts.items():
    print(f"      {m:8s} : {c:,}")

print(f"\n  {'Species':<40s} {'Img':>5s} {'Aud':>5s} {'Txt':>5s} {'Source(img)':>25s} {'Source(aud)':>25s}")
print("  " + "─" * 110)

full_cov = 0
for sp in label_list:
    sub  = final_df[final_df["shared_label"] == sp]
    ni   = len(sub[sub["modality"] == "image"])
    na   = len(sub[sub["modality"] == "audio"])
    nt   = len(sub[sub["modality"] == "text"])

    # Determine sources
    img_src = sub[sub["modality"] == "image"]["source"].iloc[0] if ni else "—"
    aud_src = sub[sub["modality"] == "audio"]["source"].iloc[0] if na else "—"

    # Shorten for display
    img_src_short = "HF dataset" if "HuggingFace" in img_src else ("Downloaded" if "Download" in img_src else img_src)
    aud_src_short = "HF dataset" if "HuggingFace" in aud_src else ("Downloaded" if "Download" in aud_src else aud_src[:25])

    ok = ni > 0 and na > 0 and nt > 0
    if ok:
        full_cov += 1
    mark = "✓" if ok else "✗"
    print(f"  {mark} {sp:<38s} {ni:>5d} {na:>5d} {nt:>5d} {img_src_short:>25s} {aud_src_short:>25s}")

print("  " + "─" * 110)
print(f"\n  Full 3-modal coverage : {full_cov}/{len(label_list)} species")
print(f"  Partial coverage     : {len(label_list) - full_cov}/{len(label_list)} species")

# Taxonomy breakdown
print("\n  Taxonomy groups:")
for grp, cnt in final_df.groupby("taxonomy_group")["shared_label"].nunique().items():
    print(f"      {grp:12s} : {cnt} species")

print("\n" + "=" * 70)
print(" ✅  PIPELINE COMPLETE")
print("=" * 70)
print(f"   Output CSV    : {OUTPUT_CSV}")
print(f"   Match report  : {REPORT_CSV}")
print(f"   Label IDs     : {LABEL_JSON}")
print(f"   Images        : {IMAGE_DIR}/")
print(f"   Audio         : {AUDIO_DIR}/")
print(f"   Text          : {TEXT_DIR}/")
print("=" * 70)