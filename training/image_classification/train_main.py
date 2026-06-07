"""
train_main.py  — Root Domain Classifier
=========================================
Trains a single model to classify an image into one of 6 marine domains:
  dolphin | whale | seal | sealion | porpoise | manatee

All species sub-folders inside each domain folder are aggregated under
that domain label. The 80/20 train/val split is done in code — the
dataset directory is NEVER physically modified.

Target: >90% top-1 accuracy on the validation split.
Backbone: convnext_small.in12k_ft_in1k  (50M params, fast, accurate)

Usage:
  python train_main.py
"""

import os
import sys
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

try:
    import timm
except ImportError:
    raise ImportError("pip install timm")

# ─── Paths ────────────────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).parent.resolve()
PROJECT_ROOT = THIS_DIR.parent.parent
DATA_ROOT    = PROJECT_ROOT / "datasets" / "image_dataset"
MODELS_DIR   = THIS_DIR / "models" / "main"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
DOMAINS    = ["dolphin", "whale", "seal", "sealion", "porpoise", "manatee"]
SEED       = 42
BACKBONE   = "convnext_small.in12k_ft_in1k"  # strong 22k pre-training, ~50M
IMG_SIZE   = 224
BATCH_SIZE = 32
VAL_SPLIT  = 0.20        # 80/20 split — done in code only

PHASE1_EPOCHS = 5        # head only warmup
PHASE2_EPOCHS = 40       # full fine-tune
PATIENCE      = 12       # early stopping

HEAD_LR      = 3e-4
BACKBONE_LR  = 1e-5
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0 if sys.platform.startswith("win") else 4
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ─── Reproducibility ─────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─── Transforms ───────────────────────────────────────────────────────────────
def train_tfm():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.10)),
    ])

def val_tfm():
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

# ─── Dataset ──────────────────────────────────────────────────────────────────
class DomainDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), int(self.labels[i])


def load_domain_data(seed):
    """
    Collect all images from image_splitted/<domain>/<species>/*.jpg
    Assign label = domain index.
    80/20 split per-domain, in code only.
    """
    c2i = {d: i for i, d in enumerate(DOMAINS)}
    i2c = {i: d for d, i in c2i.items()}

    all_paths, all_labels = [], []
    for dom in DOMAINS:
        dom_path = DATA_ROOT / dom
        if not dom_path.exists():
            print(f"  [WARN] Domain folder not found: {dom_path}")
            continue
        for sp_dir in dom_path.iterdir():
            if not sp_dir.is_dir():
                continue
            for f in sp_dir.iterdir():
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    all_paths.append(str(f))
                    all_labels.append(c2i[dom])

    all_paths  = np.array(all_paths)
    all_labels = np.array(all_labels)
    rng        = np.random.RandomState(seed)

    tr_idx, va_idx = [], []
    for cls in np.unique(all_labels):
        idx = np.where(all_labels == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * VAL_SPLIT))
        va_idx.extend(idx[:n_val].tolist())
        tr_idx.extend(idx[n_val:].tolist())

    print(f"\n  Total images : {len(all_paths)}")
    print(f"  Train        : {len(tr_idx)}")
    print(f"  Val          : {len(va_idx)}")
    for d in DOMAINS:
        n = np.sum(all_labels == c2i[d])
        print(f"    {d:<12} {n} images")

    return (all_paths[tr_idx], all_labels[tr_idx]), \
           (all_paths[va_idx], all_labels[va_idx]), c2i, i2c


def weighted_sampler(labels):
    counts  = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(
        torch.from_numpy(weights).float(),
        num_samples=len(labels), replacement=True
    )

# ─── Model ────────────────────────────────────────────────────────────────────
class DomainNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE, pretrained=True,
            num_classes=0, global_pool="avg",
        )
        dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(0.3),
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
        nn.init.trunc_normal_(self.head[2].weight, std=0.02)
        nn.init.zeros_(self.head[2].bias)
        nn.init.trunc_normal_(self.head[5].weight, std=0.02)
        nn.init.zeros_(self.head[5].bias)

    def forward(self, x):
        return self.head(self.backbone(x))

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

# ─── Training ─────────────────────────────────────────────────────────────────
def run_epoch(model, loader, opt, crit, device, scaler):
    model.train()
    tot_loss = 0.0
    opt.zero_grad(set_to_none=True)
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            out  = model(imgs)
            loss = crit(out, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        opt.zero_grad(set_to_none=True)
        tot_loss += loss.item() * len(imgs)
    return tot_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct = total = 0
    tot_loss = 0.0
    crit = nn.CrossEntropyLoss()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        tot_loss += crit(out, labels).item() * len(imgs)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += len(imgs)
    return correct / total, tot_loss / max(total, 1)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice   : {device}")
    if device.type == "cuda":
        print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"Backbone : {BACKBONE}  |  {IMG_SIZE}px  |  batch={BATCH_SIZE}")

    set_seed(SEED)

    (tr_p, tr_l), (va_p, va_l), c2i, i2c = load_domain_data(SEED)
    n_cls = len(c2i)

    tr_ds = DomainDataset(tr_p, tr_l, train_tfm())
    va_ds = DomainDataset(va_p, va_l, val_tfm())

    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, sampler=weighted_sampler(tr_l),
                       num_workers=NUM_WORKERS, pin_memory=False)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=False)

    model  = DomainNet(n_cls).to(device)
    crit   = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ── Phase 1: head warmup ─────────────────────────────────────────────────
    model.freeze_backbone()
    opt1 = torch.optim.AdamW(model.head.parameters(), lr=HEAD_LR,
                              weight_decay=WEIGHT_DECAY)
    print(f"\n  Phase 1 — head warmup ({PHASE1_EPOCHS} epochs)")
    for ep in range(1, PHASE1_EPOCHS + 1):
        tr_l_ = run_epoch(model, tr_dl, opt1, crit, device, scaler)
        va_acc, va_l_ = validate(model, va_dl, device)
        print(f"  Ep {ep:2d}/{PHASE1_EPOCHS} | tr_loss={tr_l_:.4f} | "
              f"val_acc={va_acc:.4f} | val_loss={va_l_:.4f}")

    # ── Phase 2: full fine-tune ───────────────────────────────────────────────
    model.unfreeze_backbone()
    opt2 = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
        {"params": model.head.parameters(),     "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt2,
        max_lr=[BACKBONE_LR * 10, HEAD_LR * 5],
        epochs=PHASE2_EPOCHS,
        steps_per_epoch=len(tr_dl),
        pct_start=0.2,
        anneal_strategy="cos",
    )

    best_acc   = 0.0
    best_state = None
    no_imp     = 0

    print(f"\n  Phase 2 — full fine-tune ({PHASE2_EPOCHS} epochs, patience={PATIENCE})")
    t0 = time.time()
    for ep in range(1, PHASE2_EPOCHS + 1):
        tr_l_ = run_epoch(model, tr_dl, opt2, crit, device, scaler)
        sched.step()
        va_acc, va_l_ = validate(model, va_dl, device)

        improved = va_acc > best_acc + 1e-4
        if improved:
            best_acc   = va_acc
            best_state = deepcopy(model.state_dict())
            no_imp     = 0
            tag        = " +"
        else:
            no_imp += 1
            tag     = ""

        eta = (time.time() - t0) / ep * (PHASE2_EPOCHS - ep)
        print(f"  Ep {ep:3d}/{PHASE2_EPOCHS} | tr_loss={tr_l_:.4f} | "
              f"val_acc={va_acc:.4f} | val_loss={va_l_:.4f} | "
              f"ETA {eta/60:.1f}m{tag}")

        if no_imp >= PATIENCE:
            print(f"  >> Early stopping at epoch {ep}")
            break

    # ── Save ─────────────────────────────────────────────────────────────────
    out = MODELS_DIR / "domain_classifier.pth"
    torch.save({
        "model_state_dict": best_state,
        "class_to_idx":     c2i,
        "idx_to_class":     i2c,
        "num_classes":      n_cls,
        "best_val_acc":     best_acc,
        "backbone":         BACKBONE,
        "img_size":         IMG_SIZE,
    }, out)
    print(f"\n  [SAVED] {out}")
    print(f"  Best val accuracy : {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
