"""
pipeline.py  — Two-Stage Marine Species Pipeline
==================================================
Stage 1: Main domain classifier  (models/main/domain_classifier.pth)
         Predicts which of 6 domains an image belongs to.

Stage 2: Domain-specific species classifier  (models/<domain>_seed42.pth)
         Within that domain, predicts the top-3 species.

This module is imported by both evaluate.py and any inference script.
It never modifies any dataset directory.

Usage (inference):
    from pipeline import MarinePipeline
    pipe = MarinePipeline()
    pipe.load(device)
    results = pipe.predict(image_path)
    # results = [("spinner dolphin", 0.92), ("striped dolphin", 0.05), ...]
"""

import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

try:
    import timm
except ImportError:
    raise ImportError("pip install timm")

# ─── Paths ────────────────────────────────────────────────────────────────────
THIS_DIR   = Path(__file__).parent.resolve()
MODELS_DIR = THIS_DIR / "models"
MAIN_CKPT  = MODELS_DIR / "main" / "domain_classifier.pth"

# ─── Shared constants ─────────────────────────────────────────────────────────
DOMAINS = ["dolphin", "whale", "seal", "sealion", "porpoise", "manatee"]
MEAN    = [0.485, 0.456, 0.406]
STD     = [0.229, 0.224, 0.225]

# ─── Name normalisation ───────────────────────────────────────────────────────
def canonical(name: str) -> str:
    """Normalise a species/domain name to a comparable key."""
    s = name.lower()
    s = s.replace("'", "").replace("`", "")
    s = re.sub(r"[_\-().,]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    aliases = {
        "killer whale": "orca",
        "white sided dolphin": "atlantic white sided dolphin",
        "beluga": "beluga whale",
    }
    return aliases.get(s, s)

# ─── Domain routing heuristic (fallback when main model absent) ───────────────
def heuristic_domain(folder_name: str) -> str | None:
    n = folder_name.lower().replace("-", " ").replace("_", " ")
    if n == "vaquita":                                        return "dolphin"
    if "dolphin" in n:                                        return "dolphin"
    if "whale" in n or "orca" in n or "narwhal" in n or "beluga" in n:
        return "whale"
    if "porpoise" in n:                                       return "porpoise"
    if "sea lion" in n or "sealion" in n:                     return "sealion"
    if "seal" in n:                                           return "seal"
    if "manatee" in n:                                        return "manatee"
    return None

# ─── Model definitions ────────────────────────────────────────────────────────
class DomainNet(nn.Module):
    """Stage-1: domain classifier (matches train_main.py)."""
    def __init__(self, num_classes: int, backbone_name: str):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False,
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

    def forward(self, x):
        return self.head(self.backbone(x))


class SpeciesNet(nn.Module):
    """Stage-2: domain-specific species classifier (matches train.py)."""
    def __init__(self, num_classes: int, backbone_name: str):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False,
            num_classes=0, global_pool="avg",
        )
        dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(0.4),
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ─── TTA helper ───────────────────────────────────────────────────────────────
def build_tta(size: int) -> list:
    sl = int(size * 1.12)
    return [
        transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
        transforms.Compose([
            transforms.Resize((sl, sl)), transforms.CenterCrop(size),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
        transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
        transforms.Compose([
            transforms.Resize((sl, sl)), transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
        transforms.Compose([
            transforms.Resize((int(size * 1.05), int(size * 1.05))),
            transforms.CenterCrop(size),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    ]


# ─── Pipeline ─────────────────────────────────────────────────────────────────
class MarinePipeline:
    """
    Two-stage inference pipeline.

    Stage 1 – main domain classifier decides which domain bucket.
    Stage 2 – domain-specific model produces top-3 species.

    If the main checkpoint is absent, falls back to the keyword heuristic
    (identical behaviour to old evaluate.py).
    """

    def __init__(self):
        self.device         = None
        self.main_model     = None      # DomainNet | None
        self.main_i2c       = None      # int -> domain name
        self.main_img_size  = 224
        self.main_tta       = None

        self.species_models = {}        # domain -> list[(SpeciesNet, i2c)]
        self.species_tta    = None
        self.species_img_size = 288     # will be read from checkpoint

    # ── Load ──────────────────────────────────────────────────────────────────
    def load(self, device: torch.device, domains: list = None):
        """Load main model + all domain species models."""
        self.device  = device
        domains = domains or DOMAINS

        # Stage 1 — main domain classifier
        if MAIN_CKPT.exists():
            ck = torch.load(MAIN_CKPT, map_location=device, weights_only=False)
            self.main_i2c      = ck["idx_to_class"]
            self.main_img_size = ck["img_size"]
            m = DomainNet(ck["num_classes"], ck["backbone"]).to(device)
            m.load_state_dict(ck["model_state_dict"])
            m.eval()
            self.main_model = m
            self.main_tta   = build_tta(self.main_img_size)
            print(f"  [Stage-1] Loaded domain classifier  "
                  f"({ck['backbone']}, {self.main_img_size}px, "
                  f"val_acc={ck['best_val_acc']*100:.1f}%)")
        else:
            print(f"  [Stage-1] WARN: main model not found at {MAIN_CKPT}")
            print(f"            Using keyword heuristic for domain routing.")

        # Stage 2 — species classifiers
        for dom in domains:
            ckpts = sorted(MODELS_DIR.glob(f"{dom}_seed*.pth"))
            if not ckpts:
                print(f"  [Stage-2] WARN: no checkpoint for domain [{dom}]")
                continue
            loaded = []
            for ckpt_path in ckpts:
                ck   = torch.load(ckpt_path, map_location=device, weights_only=False)
                sz   = ck.get("img_size", 288)
                self.species_img_size = sz      # assume uniform across domains
                bbn  = ck.get("backbone", "convnextv2_base")
                m    = SpeciesNet(ck["num_classes"], bbn).to(device)
                m.load_state_dict(ck["model_state_dict"])
                m.eval()
                loaded.append((m, ck["idx_to_class"]))
            self.species_models[dom] = loaded
            print(f"  [Stage-2] [{dom}] {len(loaded)} model(s), "
                  f"{len(loaded[0][1])} species")

        self.species_tta = build_tta(self.species_img_size)

    # ── Predict domain ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_domain(self, img: "PIL.Image") -> str | None:
        """Run Stage-1. Returns domain string or None if main model absent."""
        if self.main_model is None:
            return None     # caller must fall back to heuristic

        combined = None
        for tfm in self.main_tta:
            t   = tfm(img).unsqueeze(0).to(self.device)
            out = self.main_model(t)
            combined = out if combined is None else combined + out

        idx    = combined.argmax(1).item()
        domain = self.main_i2c[idx]
        return domain

    # ── Predict species top-3 ─────────────────────────────────────────────────
    @torch.no_grad()
    def predict_species(self, img: "PIL.Image", domain: str) -> list[tuple]:
        """
        Run Stage-2 for the given domain.
        Returns [(species_name, score), ...] top-3, or [] if domain missing.
        """
        ensemble = self.species_models.get(domain, [])
        if not ensemble:
            return []

        combined = None
        for (model, _) in ensemble:
            for tfm in self.species_tta:
                t   = tfm(img).unsqueeze(0).to(self.device)
                out = model(t)
                combined = out if combined is None else combined + out

        k            = min(3, combined.size(1))
        vals, idxs   = torch.topk(combined, k=k, dim=1)
        idxs         = idxs.squeeze(0).tolist()
        vals         = torch.softmax(vals.squeeze(0), dim=0).tolist()
        i2c          = ensemble[0][1]
        return [(canonical(i2c.get(int(i), "?")), v)
                for i, v in zip(idxs, vals)]

    # ── Full pipeline predict ─────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, img_path: str,
                hint_domain: str = None) -> dict:
        """
        Full 2-stage prediction for one image.

        Args:
            img_path:    Path to image file.
            hint_domain: If provided (e.g. from folder name heuristic),
                         used when main model is absent.

        Returns:
            {
              "domain":   predicted domain string,
              "top3":     [(species, confidence), ...],
              "stage1_confident": bool  (False when heuristic used)
            }
        """
        img    = Image.open(img_path).convert("RGB")
        domain = self.predict_domain(img)
        confident = True

        if domain is None:
            # Fall back to heuristic hint
            domain    = hint_domain
            confident = False

        top3 = self.predict_species(img, domain) if domain else []
        return {
            "domain":           domain,
            "top3":             top3,
            "stage1_confident": confident,
        }
