import os
import re
import random
from collections import defaultdict
from itertools import combinations
import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses, models
from torch.utils.data import DataLoader

# ==========================================================
# REPRODUCIBILITY
# ==========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ==========================================================
# CONFIG
# ==========================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "text_dataset", "train", "expanded_train_dataset")
OUTPUT_DIR = os.path.join(THIS_DIR, "marine_text_reasoning_model_v4")

BASE_MODEL = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 8
EPOCHS = 10 
LR = 1e-5

MAX_POS_PAIRS_PER_SPECIES = 60 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# HARDCODED LOOKUPS (Safe Entity Masking)
# ==========================================================
COMMON_NAME_ALIASES = {
    "amazon_river_dolphin": ["boto", "pink river dolphin", "bufeo"],
    "killer_whale": ["orca", "blackfish"],
    "dugong": ["sea cow", "seacow"],
}

# The definitive solution to prevent destroying geographic names like "Atlantic ocean"
LATIN_GENUS_WHITELIST = {
    "Inia", "Sousa", "Orcinus", "Dugong", "Delphinus", "Monodon", "Phocoena",
    "Tursiops", "Balaenoptera", "Physeter", "Eschrichtius", "Megaptera",
    "Odobenus", "Phoca", "Zalophus", "Ursus", "Enhydra", "Trichechus",
    "Neophocaena", "Platanista", "Lipotes", "Pontoporia", "Cephalorhynchus",
    "Lagenorhynchus", "Grampus", "Peponocephala", "Feresa", "Pseudorca",
    "Globicephala", "Steno", "Sotalia", "Stenella", "Kogia", "Eubalaena",
    "Caperea", "Halichoerus", "Cystophora", "Erignathus", "Hydrurga",
    "Leptonychotes", "Ommatophoca", "Mirounga", "Monachus", "Callorhinus",
    "Arctocephalus", "Eumetopias", "Otaria", "Phocarctos", "Neophoca"
}

# ==========================================================
# TEXT VIEW UTILITIES
# ==========================================================

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def apply_dynamic_entity_dropout(text, label):
    """
    Dynamically strips the species name, known aliases, AND scientific names.
    Uses safe pronouns and a strict genus whitelist to protect geographic data.
    """
    terms_to_remove = [label.replace('_', ' ')]
    
    if label in COMMON_NAME_ALIASES:
        terms_to_remove.extend(COMMON_NAME_ALIASES[label])
        
    masked_text = text
    
    # 1. Mask common names and aliases
    for term in terms_to_remove:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        # Using only universally safe pronouns
        replacement = random.choice(["this species", "this animal"])
        masked_text = pattern.sub(replacement, masked_text)
        
    # 2. Safely Mask Binomial Nomenclature using the Whitelist
    scientific_pattern = re.compile(r"\b[A-Z][a-z]{3,}\s[a-z]{3,}\b")
    
    def replace_scientific(match):
        genus = match.group(0).split()[0]
        if genus in LATIN_GENUS_WHITELIST:
            return random.choice(["this species", "this animal"])
        return match.group(0) # If not a known genus (e.g., "Atlantic ocean"), leave it alone

    masked_text = scientific_pattern.sub(replace_scientific, masked_text)
        
    return masked_text

# ==========================================================
# LOAD DATA & MERGE CLASSES
# ==========================================================
print("Loading expanded dataset...")

species_views = defaultdict(list)

for filename in os.listdir(DATA_PATH):
    if not filename.endswith(".txt") or filename.startswith("_"):
        continue 

    match = re.match(r"(.+?)_\d+_", filename)
    if match:
        label = match.group(1)
    else:
        label = filename.replace('.txt', '') 

    if label == "orca":
        label = "killer_whale"

    path = os.path.join(DATA_PATH, filename)
    with open(path, 'r', encoding='utf-8') as f:
        text = clean_text(f.read())
        if len(text.split()) > 20: 
            species_views[label].append(text)

print(f"Species loaded (after merges): {len(species_views)}")

# --- Dataset QA: Sanity Checks & Masking Verification ---
print("\n--- Masking Diagnostic Check ---")
check_count = 0
for lbl, views in species_views.items():
    if len(views) < 5:
        print(f"  [WARNING] '{lbl}' only has {len(views)} views. Check generation logs.")
        
    # Print the dropout logic for the first few species to verify behavior
    if check_count < 3 and len(views) > 0:
        sample = views[0]
        masked = apply_dynamic_entity_dropout(sample, lbl)
        print(f"\nLABEL: {lbl}")
        print(f"ORIGINAL: {sample[:200]}...")
        print(f"MASKED:   {masked[:200]}...")
        check_count += 1
print("--------------------------------\n")

# ==========================================================
# CREATE TRAINING PAIRS 
# ==========================================================
print("Creating positive training pairs with Entity Dropout...")
train_examples = []

for label, views in species_views.items():
    
    masked_views = [apply_dynamic_entity_dropout(v, label) for v in views]
    
    # --- 1. STANDARD PAIRS (Name to Name) -> Capped at 15 ---
    orig_pairs = list(combinations(views, 2))
    if len(orig_pairs) > 15:
        orig_pairs = random.sample(orig_pairs, 15)
        
    # --- 2. ANCHOR PAIRS (Name to Nameless) -> Capped at 45 ---
    anchor_pairs = []
    
    for i in range(len(views)):
        for j in range(len(masked_views)):
            if i != j:  
                anchor_pairs.append((views[i], masked_views[j]))
                
    if len(anchor_pairs) > 45:
        anchor_pairs = random.sample(anchor_pairs, 45)
        
    for a, p in orig_pairs + anchor_pairs:
        train_examples.append(InputExample(texts=[a, p]))

# --- Dataset QA: Pair Statistics ---
view_counts = [len(v) for v in species_views.values()]
print(f"Total positive pairs generated: {len(train_examples)}")
print(f"Average views/species: {np.mean(view_counts):.1f}")
print(f"Min views/species: {min(view_counts)}")
print(f"Max views/species: {max(view_counts)}\n")

train_loader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=BATCH_SIZE,
    drop_last=False,
    pin_memory=(device == 'cuda')
)

# ==========================================================
# MODEL SETUP
# ==========================================================
print("Preparing model...")
word_model = models.Transformer(BASE_MODEL, max_seq_length=384)
pool = models.Pooling(word_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
normalize = models.Normalize()
model = SentenceTransformer(modules=[word_model, pool, normalize], device=device)

train_loss = losses.MultipleNegativesRankingLoss(model=model)

# ==========================================================
# TRAIN
# ==========================================================
print("Starting training...")
warmup_steps = max(100, int(len(train_loader) * EPOCHS * 0.06))
use_amp = False

model.fit(
    train_objectives=[(train_loader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    optimizer_params={"lr": LR},
    output_path=OUTPUT_DIR,
    show_progress_bar=True,
    use_amp=use_amp,
)

print(f"Training finished. Model saved to: {OUTPUT_DIR}")