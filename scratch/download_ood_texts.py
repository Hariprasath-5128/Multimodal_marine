"""
download_ood_texts.py — Automated Wikipedia OOD Text Downloader
==============================================================
Queries the official Wikipedia Summary API for 24 non-marine-mammal topics
and saves their descriptions as text files inside datasets/ood_text_dataset/.
"""

import os
import json
import urllib.request
import urllib.parse

def fetch_wikipedia_summary(topic: str) -> str | None:
    encoded_topic = urllib.parse.quote(topic)
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_topic}"
    
    headers = {
        "User-Agent": "MarineMultimodalOODBot/1.0 (contact: user@example.com)"
    }
    
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("extract")
    except Exception as e:
        print(f"  [WARN] Failed to fetch Wikipedia summary for {topic}: {e}")
        return None

def main():
    topics = [
        # Fish
        "Shark", "Ray", "Tuna", "Salmon", "Clownfish", "Seahorse", "Octopus", "Squid",
        "Cuttlefish", "Crab", "Lobster", "Shrimp", "Jellyfish", "Sea_turtle", "Starfish",
        "Sea_urchin", "Sea_cucumber", "Coral", "Sea_anemone", "Oyster", "Mussel", "Clam",
        "Sea_snake", "Penguin", "Trout", "Cod", "Catfish", "Pike", "Perch", "Halibut",
        "Haddock", "Flounder", "Mackerel", "Herring", "Anchovy", "Sardine", "Goldfish",
        "Carp", "Guppy", "Angelfish", "Betta", "Discus_fish",
        # Mollusks & Crustaceans
        "Snail", "Slug", "Nudibranch", "Nautilus", "Chiton", "Limpet", "Abalone", "Conch",
        "Hermit_crab", "Crayfish", "Krill", "Barnacle", "Woodlouse",
        # Cnidarians & Worms
        "Portuguese_man_o'_war", "Sea_pen", "Sea_fan", "Feather_duster_worm", "Bobbit_worm",
        # Marine & Terrestrial Birds
        "Albatross", "Seagull", "Pelican", "Cormorant", "Puffin", "Tern", "Eagle", "Hawk",
        "Falcon", "Owl", "Duck", "Swan", "Goose", "Pigeon", "Parrot", "Sparrow", "Robin", "Crow",
        # Reptiles & Amphibians
        "Alligator", "Crocodile", "Lizard", "Chameleon", "Gecko", "Iguana", "Snake", "Python",
        "Cobra", "Viper", "Frog", "Toad", "Salamander", "Newt",
        # Terrestrial Mammals
        "Lion", "Elephant", "Kangaroo", "Grizzly_bear", "Chimpanzee", "Giraffe", "Panda", "Koala",
        "Wolf", "Tiger", "Leopard", "Cheetah", "Jaguar", "Raccoon", "Zebra", "Camel", "Deer", "Fox"
    ]

    dest_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "ood_text_dataset")
    os.makedirs(dest_dir, exist_ok=True)

    print("=" * 64)
    print("  Downloading Out-of-Distribution Wikipedia Text Descriptions")
    print("=" * 64)
    print(f"Target directory: {dest_dir}\n")

    successful = 0
    for topic in topics:
        print(f"Fetching summary for: {topic} ...")
        summary = fetch_wikipedia_summary(topic)
        if summary:
            filename = f"{topic.lower()}.txt"
            filepath = os.path.join(dest_dir, filename)
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(summary)
                print(f"  [OK] Saved to {filename} ({len(summary.split())} words)")
                successful += 1
            except Exception as e:
                print(f"  [ERROR] Failed to save {filename}: {e}")
        print()

    print("=" * 64)
    print(f"  Done. Successfully downloaded {successful}/{len(topics)} text profiles.")
    print("=" * 64)

if __name__ == "__main__":
    main()
