"""
download_ood_audio.py — Dynamic ESC-50 OOD Audio Downloader
===========================================================
Downloads the ESC-50 metadata CSV from GitHub, parses it, and dynamically
downloads 20 WAV files belonging to non-marine-mammal sound categories.
"""

import os
import csv
import urllib.request

def main():
    csv_url = "https://raw.githubusercontent.com/karoldvl/ESC-50/master/meta/esc50.csv"
    audio_base_url = "https://raw.githubusercontent.com/karoldvl/ESC-50/master/audio/"
    
    dest_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "ood_audio_dataset")
    os.makedirs(dest_dir, exist_ok=True)

    print("=" * 64)
    print("  Dynamic ESC-50 Audio Downloader")
    print("=" * 64)
    print(f"Target directory: {dest_dir}")

    headers = {
        "User-Agent": "MarineMultimodalAudioBot/1.0 (contact: user@example.com)"
    }

    # 1. Download esc50.csv
    csv_path = os.path.join(dest_dir, "esc50.csv")
    print("\nDownloading ESC-50 metadata CSV ...")
    req = urllib.request.Request(csv_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(response.read().decode("utf-8"))
        print("  [OK] Metadata downloaded successfully.")
    except Exception as e:
        print(f"  [ERROR] Failed to download ESC-50 metadata: {e}")
        return

    # 2. Parse metadata CSV
    # We want to select 1-2 files for each of these non-marine categories
    target_categories = {
        "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "sheep", "crow",
        "rain", "thunderstorm", "wind", "crying_baby", "sneezing", "clapping"
    }

    files_to_download = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row["category"]
            fname = row["filename"]
            if cat in target_categories:
                # Store up to 12 files per target category
                if cat not in files_to_download:
                    files_to_download[cat] = []
                if len(files_to_download[cat]) < 12:
                    files_to_download[cat].append(fname)

    # 3. Download the audio files
    print(f"\nDiscovered {sum(len(v) for v in files_to_download.values())} files across target categories.")
    
    successful = 0
    total_files = 0
    
    for cat, fnames in files_to_download.items():
        for i, source_fname in enumerate(fnames, 1):
            target_name = f"{cat}_{i}.wav"
            filepath = os.path.join(dest_dir, target_name)
            url = audio_base_url + source_fname
            total_files += 1
            
            # Skip if already exists
            if os.path.exists(filepath):
                print(f"File already exists: {target_name} (skipping)")
                successful += 1
                continue
                
            print(f"Downloading {target_name} ({source_fname}) ...")
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=15) as response:
                    with open(filepath, "wb") as f_out:
                        f_out.write(response.read())
                print("  [OK] Saved.")
                successful += 1
            except Exception as e:
                print(f"  [WARN] Failed to download {target_name}: {e}")
            print()

    # Clean up CSV
    try:
        os.remove(csv_path)
    except Exception:
        pass

    print("=" * 64)
    print(f"  Done. Successfully downloaded {successful}/{total_files} audio profiles.")
    print("=" * 64)

if __name__ == "__main__":
    main()
