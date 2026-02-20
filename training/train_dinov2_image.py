import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoModel

# =====================================================
# CONFIG
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== RELATIVE PATH SETUP =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "images")

# ⭐ MODIFIED: save model inside training/models
SAVE_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 40
LR_BACKBONE = 1e-5
LR_HEAD = 3e-4
VAL_SPLIT = 0.2

print("Device:", DEVICE)

# =====================================================
# STRONG AUGMENTATION
# =====================================================

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomResizedCrop(224, scale=(0.6,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3,0.3,0.3,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
num_classes = len(dataset.classes)

print("Images:", len(dataset))
print("Classes:", num_classes)

val_size = int(len(dataset)*VAL_SPLIT)
train_size = len(dataset)-val_size
train_ds, val_ds = random_split(dataset,[train_size,val_size])

train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
val_loader   = DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)

# =====================================================
# LOAD DINOv2
# =====================================================

print("Loading DINOv2...")
backbone = AutoModel.from_pretrained("facebook/dinov2-base")
backbone.to(DEVICE)

# Freeze everything first
for p in backbone.parameters():
    p.requires_grad=False

# Unfreeze last transformer block
for p in backbone.encoder.layer[-1].parameters():
    p.requires_grad=True

feature_dim = backbone.config.hidden_size

# =====================================================
# ARC FACE HEAD
# =====================================================

class ArcFace(nn.Module):
    def __init__(self,in_features,out_features,s=30.0,m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features,in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s=s
        self.m=m

    def forward(self,x,labels):
        cosine = nn.functional.linear(
            nn.functional.normalize(x),
            nn.functional.normalize(self.weight)
        )

        theta = torch.acos(torch.clamp(cosine,-1+1e-7,1-1e-7))
        target = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1,labels.view(-1,1),1)

        output = cosine*(1-one_hot)+target*one_hot
        output *= self.s
        return output

head = ArcFace(feature_dim,num_classes).to(DEVICE)

# =====================================================
# OPTIMIZER
# =====================================================

optimizer = optim.AdamW([
    {"params": backbone.parameters(), "lr": LR_BACKBONE},
    {"params": head.parameters(), "lr": LR_HEAD}
], weight_decay=1e-4)

criterion = nn.CrossEntropyLoss()

# =====================================================
# TRAIN LOOP
# =====================================================

best_acc=0

for epoch in range(1,EPOCHS+1):
    backbone.train()
    head.train()

    total_loss=0
    correct=0
    total=0

    for imgs,labels in tqdm(train_loader):
        imgs,labels = imgs.to(DEVICE),labels.to(DEVICE)

        feats = backbone(imgs).last_hidden_state[:,0]
        logits = head(feats,labels)

        loss = criterion(logits,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
        pred = logits.argmax(1)
        correct+=(pred==labels).sum().item()
        total+=labels.size(0)

    train_acc = correct/total

    # VALIDATION
    backbone.eval()
    head.eval()
    correct=0
    total=0

    with torch.no_grad():
        for imgs,labels in val_loader:
            imgs,labels = imgs.to(DEVICE),labels.to(DEVICE)
            feats = backbone(imgs).last_hidden_state[:,0]

            cosine = nn.functional.linear(
                nn.functional.normalize(feats),
                nn.functional.normalize(head.weight)
            )
            pred = cosine.argmax(1)

            correct+=(pred==labels).sum().item()
            total+=labels.size(0)

    val_acc = correct/total

    if val_acc>best_acc:
        best_acc=val_acc
        torch.save({
            "backbone":backbone.state_dict(),
            "head":head.state_dict(),
            "classes":dataset.classes
        },os.path.join(SAVE_DIR,"best_research_model.pth"))

    print(f"\nEpoch {epoch}/{EPOCHS}")
    print(f"Train Acc: {train_acc:.4f}")
    print(f"Val Acc  : {val_acc:.4f}")
    print(f"Best Acc : {best_acc:.4f}")

print("\n🏆 FINAL BEST ACC:",best_acc)