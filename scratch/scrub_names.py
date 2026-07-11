"""
scrub_names.py — Anonymous Text Scrubber for OOD Dataset (Strict Generic Edition)
=============================================================================
Reads all text files in datasets/ood_text_dataset/ and:
  1. Removes explicit species names (singular & plural).
  2. Replaces any taxonomy group words ("fish", "fishes", "bird", "reptile", etc.)
     with generic terms like "creature", "creatures", "animal", or "animals".
"""

import os
import re

def anonymous_replace(text: str, name: str) -> str:
    term = name.replace("_", " ").lower().strip()
    
    # Plural forms of the species name
    plurals = [term + "s", term + "es"]
    if term.endswith("y"):
        plurals.append(term[:-1] + "ies")
    elif term == "octopus":
        plurals.append("octopuses")
        plurals.append("octopi")
    elif term == "jellyfish":
        plurals.append("jellyfishes")
    
    # Sort terms by length descending to match longest first
    search_terms = sorted([term] + plurals, key=len, reverse=True)
    
    # Define completely generic replacements
    generic_singular = "this creature"
    generic_plural = "these creatures"
        
    scrubbed = text
    # Phase 1: Replace explicit species names with generic terms
    for t in search_terms:
        pattern = re.compile(r'\b' + re.escape(t) + r'\b', re.IGNORECASE)
        rep = generic_plural if t.endswith("s") or t.endswith("i") or t.endswith("es") else generic_singular
        
        def sub_fn(match):
            m = match.group(0)
            if m[0].isupper():
                return rep.capitalize()
            return rep
            
        scrubbed = pattern.sub(sub_fn, scrubbed)
        
    # Phase 2: Scrub general taxonomic words to prevent leakage (e.g. fish, fishes, bird, reptile, etc.)
    scrub_rules = {
        r'\bfish\b': 'creature',
        r'\bfishes\b': 'creatures',
        r'\breptile\b': 'creature',
        r'\breptiles\b': 'creatures',
        r'\bbird\b': 'creature',
        r'\bbirds\b': 'creatures',
        r'\bmollusk\b': 'creature',
        r'\bmollusks\b': 'creatures',
        r'\bmollusc\b': 'creature',
        r'\bmolluscs\b': 'creatures',
        r'\bcrustacean\b': 'creature',
        r'\bcrustaceans\b': 'creatures',
        r'\bworm\b': 'creature',
        r'\bworms\b': 'creatures',
        r'\banimal\b': 'creature',
        r'\banimals\b': 'creatures',
    }
    
    for pat, rep in scrub_rules.items():
        pattern = re.compile(pat, re.IGNORECASE)
        
        def sub_fn_tax(match):
            m = match.group(0)
            if m[0].isupper():
                return rep.capitalize()
            return rep
            
        scrubbed = pattern.sub(sub_fn_tax, scrubbed)
        
    return scrubbed

def main():
    root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "ood_text_dataset")
    if not os.path.exists(root_dir):
        print(f"Directory not found: {root_dir}")
        return

    print("=" * 64)
    print("  Scrubbing Species Names & Taxonomic Clues (Strict)")
    print("=" * 64)

    files = [f for f in os.listdir(root_dir) if f.endswith(".txt")]
    scrubbed_count = 0

    for filename in files:
        name = filename[:-4] # strip .txt
        filepath = os.path.join(root_dir, filename)
        
        try:
            # Re-read raw summaries by downloading fresh versions if they were already scrubbed,
            # or just process the current text files.
            # To ensure clean state, let's download the fresh Wikipedia pages first in download_ood_texts.py
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
            scrubbed_content = anonymous_replace(content, name)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(scrubbed_content)
                
            print(f"  [OK] Strictly scrubbed: {filename}")
            scrubbed_count += 1
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {filename}: {e}")

    print("=" * 64)
    print(f"  Done. Strict anonymous scrub completed for {scrubbed_count}/{len(files)} files.")
    print("=" * 64)

if __name__ == "__main__":
    main()
