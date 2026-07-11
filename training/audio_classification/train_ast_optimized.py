# =========================================================
# HIGH PERFORMANCE MARINE AUDIO CLASSIFIER
# RTX 2050 (4GB VRAM) OPTIMIZED
# =========================================================

import os
import torch
import torchaudio
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    ASTFeatureExtractor,
    ASTForAudioClassification,
    get_cosine_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from collections import Counter

# =========================================================
# CONFIG
# =========================================================

DATA_ROOT = "../datasets/audio/audio_split"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR   = os.path.join(DATA_ROOT, "val")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4          # reduced for 4GB GPU
EPOCHS = 40
LR = 1e-5               # lower for fine-tuning
TARGET_DURATION = 10
ACCUMULATION_STEPS = 2  # simulate batch size 8

# =========================================================
# FOCAL LOSS (BETTER FOR IMBALANCE)
# =========================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# =========================================================
# DATASET
# =========================================================

class MarineAudioDataset(Dataset):
    def __init__(self, root_dir, feature_extractor):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        
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

        # 🔥 RESAMPLE TO 16kHz (AST requirement)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000

        # Normalize duration AFTER resampling
        waveform = self.pad_or_trim(waveform, sr)

        inputs = self.feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=sr,
            return_tensors="pt"
        )

        return inputs["input_values"].squeeze(0), torch.tensor(self.labels[idx])

# =========================================================
# LOAD DATA
# =========================================================

feature_extractor = ASTFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)

train_dataset = MarineAudioDataset(TRAIN_DIR, feature_extractor)
val_dataset   = MarineAudioDataset(VAL_DIR, feature_extractor)

NUM_CLASSES = len(train_dataset.classes)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# =========================================================
# MODEL
# =========================================================

model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)

model.to(DEVICE)

# Freeze all layers first
for param in model.base_model.parameters():
    param.requires_grad = False

# Unfreeze last transformer block (memory safe)
for param in model.base_model.encoder.layer[-1].parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

total_steps = len(train_loader) * EPOCHS
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

criterion = FocalLoss()
scaler = torch.cuda.amp.GradScaler()

# =========================================================
# TRAINING LOOP
# =========================================================

best_f1 = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, (inputs, labels) in enumerate(tqdm(train_loader)):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        with torch.cuda.amp.autocast():
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            loss = loss / ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item()

    # ================= VALIDATION =================
    model.eval()
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs).logits
            preds = torch.argmax(outputs, dim=1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="macro")

    print(f"\nEpoch {epoch+1}")
    print("Loss:", total_loss)
    print("Val Accuracy:", acc)
    print("Macro F1:", f1)

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "best_marine_ast_optimized.pth")
        print("Best model saved.")

print("\nTraining Complete.")