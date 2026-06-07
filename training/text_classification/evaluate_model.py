import os
import torch
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Configuration
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

TEST_DIR_1 = os.path.join(PROJECT_ROOT, "datasets", "text_dataset", "test", "test_data")
TEST_DIR_2 = os.path.join(PROJECT_ROOT, "datasets", "text_dataset", "test", "expanded_test_dataset")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "datasets", "text_dataset", "train", "train_dataset")
MODEL_PATH = os.path.join(THIS_DIR, "marine_text_reasoning_model_v4")

# 1. Load corpus
corpus_texts = []
species_labels = []

for file in os.listdir(TRAIN_DIR):
    if file.endswith(".txt") and not file.startswith("_"):
        name = file.replace(".txt", "")
        if name == "orca":
            name = "killer_whale"
        with open(os.path.join(TRAIN_DIR, file), "r", encoding="utf-8") as f:
            corpus_texts.append(f.read().strip())
        species_labels.append(name)

# 2. Load test samples
test_samples = []

def parse_filename(filename):
    match = re.match(r"(.+?)_test_\d+_(.+)\.txt", filename)
    if match:
        species = match.group(1)
        test_type = match.group(2)
        return species, test_type
    else:
        return filename.replace(".txt", ""), "Standard"

for test_dir in [TEST_DIR_1, TEST_DIR_2]:
    for file in os.listdir(test_dir):
        if not file.endswith(".txt"):
            continue
        if file.startswith("_"):
            continue 
            
        species, test_type = parse_filename(file)
        if species == "orca":
            species = "killer_whale"
            
        with open(os.path.join(test_dir, file), "r", encoding="utf-8") as f:
            text = f.read().strip()
            
        if test_type == "FeatureOnly":
            text = text.replace("[MASK]", "").replace("  ", " ")
            
        # Replace the true species name with 'the species' dynamically in memory
        term_to_remove = species.replace('_', ' ')
        pattern = re.compile(re.escape(term_to_remove), re.IGNORECASE)
        text = pattern.sub("the species", text)
            
        test_samples.append({
            "text": text,
            "true_species": species,
            "test_type": test_type
        })

print(f"Loaded {len(corpus_texts)} reference species.")
print(f"Loaded {len(test_samples)} test samples.")

# 3. Load Model and Encode
print("Loading model and encoding...")
model = SentenceTransformer(MODEL_PATH)
corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=False)

query_texts = [sample["text"] for sample in test_samples]
query_embeddings = model.encode(query_texts, convert_to_tensor=True, show_progress_bar=True)

# 4. Compute metrics
correct_total = 0
species_correct = defaultdict(int)
species_total = defaultdict(int)
test_type_correct = defaultdict(int)
test_type_total = defaultdict(int)

scores = util.cos_sim(query_embeddings, corpus_embeddings)
predictions = torch.argmax(scores, dim=1)

for idx, sample in enumerate(test_samples):
    true_species = sample["true_species"]
    test_type = sample["test_type"]
    
    pred_idx = predictions[idx].item()
    pred_species = species_labels[pred_idx]
    
    is_correct = (pred_species == true_species)
    
    if is_correct:
        correct_total += 1
        species_correct[true_species] += 1
        test_type_correct[test_type] += 1
        
    species_total[true_species] += 1
    test_type_total[test_type] += 1

# Print Results
print("\n" + "="*50)
print("EVALUATION METRICS")
print("="*50)

overall_acc = (correct_total / len(test_samples)) * 100
print(f"\n1. Overall Accuracy")
print(f"Accuracy: {overall_acc:.2f}% ({correct_total}/{len(test_samples)})\n")

y_true = [sample["true_species"] for sample in test_samples]
y_pred = [species_labels[predictions[idx].item()] for idx in range(len(test_samples))]

print("2. Classification Report (Precision, Recall, F1)")
print(classification_report(y_true, y_pred, zero_division=0))

# Confusion Matrix Heatmap
cm = confusion_matrix(y_true, y_pred, labels=sorted(list(set(y_true))))
plt.figure(figsize=(20, 18))
sns.heatmap(cm, xticklabels=sorted(list(set(y_true))), yticklabels=sorted(list(set(y_true))), cmap="Blues", cbar=False)
plt.title("Text Classification Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "confusion_matrix.png"))
plt.close()
print("Confusion matrix saved to confusion_matrix.png")

print("\n2. Per-Species Accuracy")
for species in sorted(species_total.keys()):
    acc = (species_correct[species] / species_total[species]) * 100
    print(f"  - {species}: {acc:.2f}% ({species_correct[species]}/{species_total[species]})")

print("\n3. Accuracy by Test Type")
for t_type in sorted(test_type_total.keys()):
    acc = (test_type_correct[t_type] / test_type_total[t_type]) * 100
    print(f"  - {t_type}: {acc:.2f}% ({test_type_correct[t_type]}/{test_type_total[t_type]})")
