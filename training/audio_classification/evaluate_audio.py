import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "audio_dataset", "audio_split")
VAL_DIR   = os.path.join(DATA_ROOT, "val")
MODEL_PATH = os.path.join(THIS_DIR, "marine_audio_classification_model", "best_marine_ast_optimized.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
TARGET_DURATION = 10

# =========================================================
# DATASET
# =========================================================

class MarineAudioDataset(Dataset):
    def __init__(self, root_dir, feature_extractor):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        
        # Determine classes from directory structure
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.filepaths = []
        self.labels = []
        
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            for file in os.listdir(cls_path):
                if file.endswith(".wav"):
                    self.filepaths.append(os.path.join(cls_path, file))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.filepaths)

    def pad_or_trim(self, waveform, sr):
        target_length = TARGET_DURATION * sr
        if waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        else:
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.filepaths[idx])

        # Convert to mono
        waveform = waveform.mean(dim=0, keepdim=True)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000

        waveform = self.pad_or_trim(waveform, sr)

        inputs = self.feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=sr,
            return_tensors="pt"
        )

        return inputs["input_values"].squeeze(0), torch.tensor(self.labels[idx])

# =========================================================
# LOAD DATA & MODEL
# =========================================================

print("Loading feature extractor...")
feature_extractor = ASTFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)

print(f"Loading validation dataset from {VAL_DIR}...")
val_dataset = MarineAudioDataset(VAL_DIR, feature_extractor)
val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE)
NUM_CLASSES = len(val_dataset.classes)

print(f"Instantiating model with {NUM_CLASSES} classes...")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)

print(f"Loading state dict from {MODEL_PATH}...")
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# =========================================================
# EVALUATION LOOP
# =========================================================

preds_all = []
labels_all = []

print("Running inference on validation set...")
with torch.no_grad():
    for inputs, labels in tqdm(val_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs).logits
            preds = torch.argmax(outputs, dim=1)

        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

# Map back to string labels
y_true = [val_dataset.classes[idx] for idx in labels_all]
y_pred = [val_dataset.classes[idx] for idx in preds_all]
class_names = val_dataset.classes

print("\n==================================================")
print("AUDIO CLASSIFICATION METRICS")
print("==================================================\n")

print("Classification Report:")
print(classification_report(y_true, y_pred, labels=class_names, zero_division=0))

# Confusion Matrix Heatmap
cm = confusion_matrix(y_true, y_pred, labels=class_names)
plt.figure(figsize=(20, 18))
sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names, cmap="Blues", cbar=False)
plt.title("Audio Classification Confusion Matrix")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("Confusion matrix saved to confusion_matrix.png")
