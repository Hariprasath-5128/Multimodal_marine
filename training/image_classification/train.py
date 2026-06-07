"""
train.py — Optimized Top-3 Marine Species Classifier
=====================================================
Backbone : ConvNeXt-Small (ImageNet-22k → 1k fine-tuned)
           50M params | 224px | batch=16 | fits in 4GB VRAM
           Much faster than EfficientNetV2-L, better 22k pre-training

Key techniques:
  - Focal Loss (γ=2) + class weights  → handles rare species
  - CutMix + MixUp                   → better generalisation
  - OneCycleLR                        → converges in 35-40 epochs
  - 2-phase (head warmup → full FT)  → stable training
  - Multi-seed ensemble (3 seeds)    → robust top-3 predictions
  - num_workers=0 on Windows          → avoids cuDNN stream mismatch

Training time: ~5-15 min per domain per seed on RTX 2050
Total (6 domains × 3 seeds): ~1.5-2.5 hours

Usage:
  python train.py                        # all domains, all seeds
  python train.py --domain dolphin       # one domain, all seeds
  python train.py --domain whale --seed 42
"""

import os
import sys
import argparse
import random
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms
from PIL import Image

try:
    import timm
except ImportError:
    raise ImportError("Please install timm: pip install timm")

# ─── Paths ────────────────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).parent.resolve()
PROJECT_ROOT = THIS_DIR.parent.parent
DATA_ROOT    = PROJECT_ROOT / "datasets" / "image_dataset"
MODELS_DIR   = THIS_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
DOMAINS = ["dolphin", "whale", "seal", "sealion", "porpoise", "manatee"]
SEEDS   = [42]

# ConvNeXtV2-Base: 89M params, GRN layer (crucial for tiny datasets)
BACKBONE       = "convnextv2_base"
IMG_SIZE       = 288      # increased from 224; captures finer details, still fits 4GB VRAM
BATCH_SIZE     = 8        # 4GB VRAM → batch 8 for convnext_base
ACCUMULATION_STEPS = 2    # Effective batch = 16
PHASE1_EPOCHS  = 3        # head-only warmup (fast)
PHASE2_EPOCHS  = 45       # full fine-tune with OneCycleLR
PATIENCE       = 12       # early stopping patience
VAL_SPLIT      = 0.2

BACKBONE_LR   = 2e-5      # gentle LR for the backbone
HEAD_LR       = 2e-4      # aggressive LR for the new head
WEIGHT_DECAY  = 1e-4

# Windows: num_workers > 0 causes cuDNN stream mismatch with CUDA AMP
NUM_WORKERS = 0 if sys.platform.startswith("win") else 4

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ─── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Focal Loss ───────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, smoothing: float = 0.1,
                 weight: torch.Tensor = None):
        super().__init__()
        self.gamma     = gamma
        self.smoothing = smoothing
        self.weight    = weight   # class weights for imbalance

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        C = logits.size(1)
        with torch.no_grad():
            eps     = self.smoothing / C
            one_hot = torch.full_like(logits, eps)
            one_hot.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing + eps)

        log_p   = F.log_softmax(logits, dim=1)
        pt      = log_p.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_w = (1.0 - pt) ** self.gamma
        ce_loss = -(one_hot * log_p).sum(dim=1)

        if self.weight is not None:
            ce_loss = ce_loss * self.weight.to(logits.device)[targets]

        return (focal_w * ce_loss).mean()


# ─── Augmentation Helpers ─────────────────────────────────────────────────────
def cutmix(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    lam = float(np.random.beta(alpha, alpha))
    B, _, H, W = x.shape
    r  = torch.randperm(B, device=x.device)
    cr = (1.0 - lam) ** 0.5
    ch, cw = int(H * cr), int(W * cr)
    cx = random.randint(0, W);  cy = random.randint(0, H)
    x1 = max(cx - cw // 2, 0); x2 = min(cx + cw // 2, W)
    y1 = max(cy - ch // 2, 0); y2 = min(cy + ch // 2, H)
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
    xm  = x.clone(); xm[:, :, y1:y2, x1:x2] = x[r, :, y1:y2, x1:x2]
    return xm, y, y[r], lam

def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    r   = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[r], y, y[r], lam

def mixed_loss(crit, out, ya, yb, lam):
    return lam * crit(out, ya) + (1.0 - lam) * crit(out, yb)


# ─── Transforms ───────────────────────────────────────────────────────────────
def train_tfm():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.45, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.07),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandAugment(num_ops=2, magnitude=5),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    ])

def val_tfm():
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


# ─── Model ────────────────────────────────────────────────────────────────────
class MarineNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(
            BACKBONE, pretrained=True,
            num_classes=0, global_pool="avg",
        )
        dim = self.backbone.num_features   # 768 for ConvNeXt-Small

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(0.4),
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes),
        )
        # initialise head weights
        nn.init.trunc_normal_(self.head[2].weight, std=0.02)
        nn.init.zeros_(self.head[2].bias)
        nn.init.trunc_normal_(self.head[5].weight, std=0.02)
        nn.init.zeros_(self.head[5].bias)

    def forward(self, x):
        return self.head(self.backbone(x))

    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False

    def unfreeze_backbone(self):
        for name, p in self.backbone.named_parameters():
            # Freeze low-level features to prevent overfitting on tiny dataset
            if name.startswith("stem") or name.startswith("stages.0"):
                p.requires_grad = False
            else:
                p.requires_grad = True


# ─── Dataset ──────────────────────────────────────────────────────────────────
class DomainDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):  return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), int(self.labels[i])


def load_domain(domain: str, seed: int):
    domain_path = DATA_ROOT / domain
    species_dirs = sorted([d for d in domain_path.iterdir() if d.is_dir()])
    c2i = {sp.name: i for i, sp in enumerate(species_dirs)}
    i2c = {i: sp.name for i, sp in enumerate(species_dirs)}

    paths, labels = [], []
    for sp in species_dirs:
        imgs = sorted(f for f in sp.iterdir()
                      if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for img in imgs:
            paths.append(str(img)); labels.append(c2i[sp.name])

    paths  = np.array(paths)
    labels = np.array(labels)
    rng    = np.random.RandomState(seed)

    tr_idx, va_idx = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]; rng.shuffle(idx)
        n_val = max(1, int(len(idx) * VAL_SPLIT))
        va_idx.extend(idx[:n_val].tolist())
        tr_idx.extend(idx[n_val:].tolist())

    return (paths[tr_idx], labels[tr_idx]), (paths[va_idx], labels[va_idx]), c2i, i2c


def weighted_sampler(labels):
    counts  = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(torch.from_numpy(weights).float(),
                                 num_samples=len(labels), replacement=True)

def class_weights(labels, n, device):
    c = np.bincount(labels, minlength=n).astype(float)
    c = np.where(c == 0, 1, c)
    w = 1.0 / np.sqrt(c); w = w / w.sum() * n
    return torch.tensor(w, dtype=torch.float32, device=device)


# ─── Training helpers ─────────────────────────────────────────────────────────
def run_epoch(model, loader, opt, crit, device, scaler, do_mix: bool, accum_steps: int = 1):
    model.train()
    tot_loss = 0.0
    opt.zero_grad(set_to_none=True)

    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        apply_mix = do_mix and random.random() < 0.5
        if apply_mix:
            if random.random() < 0.5:
                imgs, ya, yb, lam = cutmix(imgs, labels)
            else:
                imgs, ya, yb, lam = mixup(imgs, labels)

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            out  = model(imgs)
            loss = (mixed_loss(crit, out, ya, yb, lam)
                    if apply_mix else crit(out, labels))
            loss = loss / accum_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            if scaler:
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            opt.zero_grad(set_to_none=True)

        tot_loss += loss.item() * accum_steps * len(imgs)

    return tot_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct = total = 0
    tot_loss = 0.0
    crit = nn.CrossEntropyLoss()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        tot_loss += crit(out, labels).item() * len(imgs)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += len(imgs)
    return correct / total, tot_loss / max(total, 1)


# ─── Train one domain+seed ────────────────────────────────────────────────────
def train_one(domain: str, seed: int, device: torch.device) -> float:
    print(f"\n{'='*60}")
    print(f"  [{domain.upper()}]  seed={seed}")
    print(f"{'='*60}")
    set_seed(seed)

    (tr_p, tr_l), (va_p, va_l), c2i, i2c = load_domain(domain, seed)
    n_cls = len(c2i)
    print(f"  Classes={n_cls}  Train={len(tr_p)}  Val={len(va_p)}")

    tr_ds = DomainDataset(tr_p, tr_l, train_tfm())
    va_ds = DomainDataset(va_p, va_l, val_tfm())

    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, sampler=weighted_sampler(tr_l),
                       num_workers=NUM_WORKERS, pin_memory=False)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=False)

    model  = MarineNet(n_cls).to(device)
    cw     = class_weights(tr_l, n_cls, device)
    crit   = FocalLoss(gamma=2.0, smoothing=0.1, weight=cw)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ── Phase 1: train head only (3 epochs, fast warmup) ─────────────────────
    model.freeze_backbone()
    opt1 = torch.optim.AdamW(model.head.parameters(), lr=HEAD_LR,
                              weight_decay=WEIGHT_DECAY)
    print(f"\n  Phase 1 — head warmup ({PHASE1_EPOCHS} epochs)")
    for ep in range(1, PHASE1_EPOCHS + 1):
        tr_l_ = run_epoch(model, tr_dl, opt1, crit, device, scaler, do_mix=False, accum_steps=1)
        va_acc, va_l_ = validate(model, va_dl, device)
        print(f"  Ep {ep:2d}/{PHASE1_EPOCHS} | tr={tr_l_:.3f} | "
              f"va_acc={va_acc:.3f} | va_loss={va_l_:.3f}")

    # ── Phase 2: full fine-tune with OneCycleLR ────────────────────────────────
    model.unfreeze_backbone()
    opt2 = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
        {"params": model.head.parameters(),     "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    # OneCycleLR: ramps up then decays — best convergence for small datasets
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt2,
        max_lr=[BACKBONE_LR * 10, HEAD_LR * 5],
        epochs=PHASE2_EPOCHS,
        steps_per_epoch=len(tr_dl),
        pct_start=0.2,
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=100,
    )

    best_loss  = float("inf")
    best_acc   = 0.0
    best_state = None
    no_imp     = 0

    print(f"\n  Phase 2 — full fine-tune ({PHASE2_EPOCHS} epochs, patience={PATIENCE})")
    t0 = time.time()
    for ep in range(1, PHASE2_EPOCHS + 1):
        tr_l_ = run_epoch(model, tr_dl, opt2, crit, device, scaler, do_mix=True, accum_steps=ACCUMULATION_STEPS)
        sched.step()
        va_acc, va_l_ = validate(model, va_dl, device)

        improved = va_l_ < best_loss - 1e-4
        if improved:
            best_loss = va_l_; best_acc = va_acc
            best_state = deepcopy(model.state_dict()); no_imp = 0; tag = " ✓"
        else:
            no_imp += 1; tag = ""

        elapsed = time.time() - t0
        eta     = elapsed / ep * (PHASE2_EPOCHS - ep)
        print(f"  Ep {ep:3d}/{PHASE2_EPOCHS} | tr={tr_l_:.3f} | "
              f"va_acc={va_acc:.3f} | va_loss={va_l_:.3f} | "
              f"ETA {eta/60:.1f}m{tag}")

        if no_imp >= PATIENCE:
            print(f"  ↳ Early stopping at epoch {ep}")
            break

    # ── Save checkpoint ───────────────────────────────────────────────────────
    out = MODELS_DIR / f"{domain}_seed{seed}.pth"
    torch.save({
        "model_state_dict": best_state,
        "class_to_idx": c2i,
        "idx_to_class": i2c,
        "num_classes":  n_cls,
        "best_val_acc": best_acc,
        "backbone":     BACKBONE,
        "img_size":     IMG_SIZE,
        "domain":       domain,
        "seed":         seed,
    }, out)
    print(f"\n  ✅ Saved → {out}")
    print(f"     Best val acc = {best_acc:.4f}")
    return best_acc


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="all",
                    help=f"Domain or 'all'. Choices: {', '.join(DOMAINS)}")
    ap.add_argument("--seed", type=int, default=None,
                    help="Single seed (default: all 3 seeds)")
    args = ap.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domains = DOMAINS if args.domain == "all" else [args.domain]
    seeds   = SEEDS   if args.seed is None    else [args.seed]

    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Backbone: {BACKBONE}  |  {IMG_SIZE}px  |  batch={BATCH_SIZE}")
    print(f"Domains : {domains}")
    print(f"Seeds   : {seeds}")

    # Download backbone once before training loop
    print("\nLoading backbone (downloads on first run) …")
    timm.create_model(BACKBONE, pretrained=True, num_classes=0)
    print("Backbone ready.\n")

    summary = {}
    t_start = time.time()
    for dom in domains:
        summary[dom] = {}
        for s in seeds:
            acc = train_one(dom, s, device)
            summary[dom][s] = acc

    total_min = (time.time() - t_start) / 60
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE  ({total_min:.1f} min total)")
    print(f"{'='*60}")
    for dom in domains:
        accs = [summary[dom][s] for s in seeds if s in summary[dom]]
        avg  = np.mean(accs)
        print(f"  {dom:<12} | seeds={[f'{a:.3f}' for a in accs]} | avg={avg:.3f}")
    print(f"{'='*60}")
    print("Run:  python evaluate.py")


if __name__ == "__main__":
    main()
