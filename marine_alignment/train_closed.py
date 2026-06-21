"""
train.py — Joint Multimodal Contrastive Alignment for Marine Species
====================================================================
This script trains the image_head, text_head, and audio_head jointly
using CLIP-style Symmetric Supervised Contrastive Loss (SupCon).

By training all three heads simultaneously:
  1. The projection heads co-adapt to structure a single shared latent space.
  2. It eliminates the phase-transition misalignment that happens in two-phase training.
  3. Image <-> Text and Image <-> Audio are aligned directly to reach >75% R@1.

Usage:
  python train.py
  python train.py --epochs 150 --batch_size 32
"""

import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEVICE, WEIGHT_DECAY, MIN_VALID_SAMPLES, CHECKPOINT_PATH,
    RECALL_WEIGHT_R1, RECALL_WEIGHT_R5, RECALL_WEIGHT_R10,
    HARD_NEG_COUNT
)
from dataset import MarineFeatureDataset, SpeciesBalancedSampler, make_splits
from models  import MarineImageBindPipeline
from loss    import supervised_contrastive_loss, check_loss_finite

# == Training hyperparameters ===================================================
EPOCHS          = 250
BATCH_SIZE      = 128
K_SAMPLES       = 2          # 64 species per batch, 2 samples each (max diversity)
MEMORY_BANK_SIZE = 256       # Memory bank size for retrieval targets
TEMPERATURE_TXT = 0.07       # Contrastive temperature
LEARNING_RATE   = 5e-4       # Base learning rate


# == Joint Training Loop ========================================================

def train_epoch_joint(pipeline, dataloader, optimizer, device, memory_bank):
    pipeline.train()
    
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        image_tensor = batch["image_tensor"].to(device)
        text_emb  = batch["text_emb"].to(device)
        audio_emb = batch["audio_emb"].to(device)
        has_text  = batch["has_text"].to(device)
        has_audio = batch["has_audio"].to(device)
        labels    = batch["species_id"].to(device)
        
        proj_img = pipeline.project_image(image_tensor)
        
        losses = []
        
        # 1. Image <-> Text Symmetric SupCon
        if has_text.sum() >= MIN_VALID_SAMPLES:
            valid_img = proj_img[has_text]
            valid_txt = pipeline.text_head(text_emb[has_text])
            valid_lbl = labels[has_text]
            
            # Text memory bank
            if "text_feats" in memory_bank and memory_bank["text_feats"].size(0) > 0:
                all_txt = torch.cat([valid_txt, memory_bank["text_feats"]], dim=0)
                all_lbl_txt = torch.cat([valid_lbl, memory_bank["labels_txt"]], dim=0)
            else:
                all_txt = valid_txt
                all_lbl_txt = valid_lbl
                
            l_i2t = supervised_contrastive_loss(
                valid_img, all_txt, valid_lbl, all_lbl_txt,
                TEMPERATURE_TXT, HARD_NEG_COUNT,
                exclude_diagonal=False,
            )
            
            # Image memory bank for symmetric Text->Image
            if "image_feats" in memory_bank and memory_bank["image_feats"].size(0) > 0:
                all_img = torch.cat([valid_img, memory_bank["image_feats"]], dim=0)
                all_lbl_img = torch.cat([valid_lbl, memory_bank["labels_img"]], dim=0)
            else:
                all_img = valid_img
                all_lbl_img = valid_lbl
                
            l_t2i = supervised_contrastive_loss(
                valid_txt, all_img, valid_lbl, all_lbl_img,
                TEMPERATURE_TXT, HARD_NEG_COUNT,
                exclude_diagonal=False,
            )
            
            losses.append(0.5 * l_i2t + 0.5 * l_t2i)
            
            # Update memory banks
            memory_bank["text_feats"] = torch.cat([valid_txt.detach(), memory_bank.get("text_feats", torch.zeros(0, valid_txt.size(1), device=device))], dim=0)[:MEMORY_BANK_SIZE]
            memory_bank["labels_txt"] = torch.cat([valid_lbl.detach(), memory_bank.get("labels_txt", torch.zeros(0, dtype=torch.long, device=device))], dim=0)[:MEMORY_BANK_SIZE]
            
            memory_bank["image_feats"] = torch.cat([valid_img.detach(), memory_bank.get("image_feats", torch.zeros(0, valid_img.size(1), device=device))], dim=0)[:MEMORY_BANK_SIZE]
            memory_bank["labels_img"] = torch.cat([valid_lbl.detach(), memory_bank.get("labels_img", torch.zeros(0, dtype=torch.long, device=device))], dim=0)[:MEMORY_BANK_SIZE]
            
        # 2. Image <-> Audio Symmetric SupCon
        if has_audio.sum() >= MIN_VALID_SAMPLES:
            valid_img = proj_img[has_audio]
            valid_aud = pipeline.audio_head(audio_emb[has_audio])
            valid_lbl = labels[has_audio]
            
            # Audio memory bank
            if "audio_feats" in memory_bank and memory_bank["audio_feats"].size(0) > 0:
                all_aud = torch.cat([valid_aud, memory_bank["audio_feats"]], dim=0)
                all_lbl_aud = torch.cat([valid_lbl, memory_bank["labels_aud"]], dim=0)
            else:
                all_aud = valid_aud
                all_lbl_aud = valid_lbl
                
            l_i2a = supervised_contrastive_loss(
                valid_img, all_aud, valid_lbl, all_lbl_aud,
                TEMPERATURE_TXT, HARD_NEG_COUNT,
                exclude_diagonal=False,
            )
            
            l_a2i = supervised_contrastive_loss(
                valid_aud, all_img if "image_feats" in memory_bank else valid_img,
                valid_lbl, all_lbl_img if "image_feats" in memory_bank else valid_lbl,
                TEMPERATURE_TXT, HARD_NEG_COUNT,
                exclude_diagonal=False,
            )
            
            losses.append(0.5 * l_i2a + 0.5 * l_a2i)
            
            # Update memory banks
            memory_bank["audio_feats"] = torch.cat([valid_aud.detach(), memory_bank.get("audio_feats", torch.zeros(0, valid_aud.size(1), device=device))], dim=0)[:MEMORY_BANK_SIZE]
            memory_bank["labels_aud"] = torch.cat([valid_lbl.detach(), memory_bank.get("labels_aud", torch.zeros(0, dtype=torch.long, device=device))], dim=0)[:MEMORY_BANK_SIZE]

        # 3. Text <-> Audio Symmetric SupCon
        has_both = has_text & has_audio
        if has_both.sum() >= MIN_VALID_SAMPLES:
            valid_txt = pipeline.text_head(text_emb[has_both])
            valid_aud = pipeline.audio_head(audio_emb[has_both])
            valid_lbl = labels[has_both]
            
            l_t2a = supervised_contrastive_loss(
                valid_txt, valid_aud, valid_lbl, valid_lbl,
                TEMPERATURE_TXT, HARD_NEG_COUNT,
                exclude_diagonal=False,
            )
            l_a2t = supervised_contrastive_loss(
                valid_aud, valid_txt, valid_lbl, valid_lbl,
                TEMPERATURE_TXT, HARD_NEG_COUNT,
                exclude_diagonal=False,
            )
            losses.append(0.5 * l_t2a + 0.5 * l_a2t)

        if losses:
            loss = sum(losses) / len(losses)
            check_loss_finite(loss, "joint")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipeline.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
        n_batches += 1
        
    safe = lambda a, b: a / b if b > 0 else 0.0
    return safe(total_loss, n_batches)


# == Recall Evaluation =========================================================

@torch.no_grad()
def calculate_cross_modal_recall(pipeline, dataloader, device, ks=(1, 5, 10)):
    pipeline.eval()

    all_img, all_txt, all_aud = [], [], []
    all_lbl_img, all_lbl_txt, all_lbl_aud = [], [], []

    for batch in dataloader:
        image_tensor = batch["image_tensor"].to(device)
        txt_emb   = batch["text_emb"].to(device)
        aud_emb   = batch["audio_emb"].to(device)
        has_text  = batch["has_text"].to(device)
        has_audio = batch["has_audio"].to(device)
        labels    = batch["species_id"].to(device)

        proj_img = pipeline.project_image(image_tensor)
        all_img.append(proj_img)
        all_lbl_img.append(labels)

        if has_text.sum() > 0:
            all_txt.append(pipeline.text_head(txt_emb[has_text]))
            all_lbl_txt.append(labels[has_text])

        if has_audio.sum() > 0:
            all_aud.append(pipeline.audio_head(aud_emb[has_audio]))
            all_lbl_aud.append(labels[has_audio])

    all_img     = torch.cat(all_img,     dim=0)
    all_lbl_img = torch.cat(all_lbl_img, dim=0)
    metrics = {}

    def _recall(queries, q_labels, gallery, g_labels):
        sim = torch.matmul(queries, gallery.T)
        out = {}
        for k in ks:
            topk_idx = sim.topk(min(k, sim.size(1)), dim=1).indices
            topk_lbl = g_labels[topk_idx]
            correct  = (topk_lbl == q_labels.unsqueeze(1)).any(dim=1)
            out[f"R@{k}"] = correct.float().mean().item()
        return out

    if all_txt:
        all_txt     = torch.cat(all_txt,     dim=0)
        all_lbl_txt = torch.cat(all_lbl_txt, dim=0)
        r = _recall(all_img, all_lbl_img, all_txt, all_lbl_txt)
        for k, v in r.items():
            metrics[f"img2txt_{k}"] = v
    else:
        for k in ks:
            metrics[f"img2txt_R@{k}"] = 0.0

    if all_aud:
        all_aud     = torch.cat(all_aud,     dim=0)
        all_lbl_aud = torch.cat(all_lbl_aud, dim=0)
        aud_species_ids = set(all_lbl_aud.tolist()) if all_lbl_aud.numel() > 0 else set()
        if aud_species_ids:
            aud_query_mask = torch.tensor([lbl.item() in aud_species_ids for lbl in all_lbl_img])
            img_embs_aud   = all_img[aud_query_mask]
            img_labels_aud = all_lbl_img[aud_query_mask]
        else:
            img_embs_aud, img_labels_aud = all_img, all_lbl_img

        r = _recall(img_embs_aud, img_labels_aud, all_aud, all_lbl_aud)
        for k, v in r.items():
            metrics[f"img2aud_{k}"] = v
    else:
        for k in ks:
            metrics[f"img2aud_R@{k}"] = 0.0

    return metrics


def composite_score(metrics):
    scores = []
    for d in ["img2txt", "img2aud"]:
        r1  = metrics.get(f"{d}_R@1",  0.0)
        r5  = metrics.get(f"{d}_R@5",  0.0)
        r10 = metrics.get(f"{d}_R@10", 0.0)
        scores.append(
            RECALL_WEIGHT_R1  * r1 +
            RECALL_WEIGHT_R5  * r5 +
            RECALL_WEIGHT_R10 * r10
        )
    return sum(scores) / len(scores)


def save_checkpoint(pipeline, epoch, score, metrics, path=CHECKPOINT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":           epoch,
        "composite_score": score,
        "metrics":         metrics,
        "model_state":     pipeline.state_dict(),
    }, path)
    print(f"   [OK] Checkpoint saved -> {os.path.basename(path)}"
          f"  (composite={score:.4f})")


# == Main ======================================================================

def main(args):
    device     = args.device
    epochs     = args.epochs
    batch_size = args.batch_size

    print("=" * 64)
    print("  Marine Multimodal Alignment - Joint Training")
    print("=" * 64)
    print(f"  Device         : {device}")
    print(f"  Epochs         : {epochs}")
    print(f"  Batch size     : {batch_size}  ({batch_size//K_SAMPLES} species x {K_SAMPLES} per batch)")
    print()

    print("Building datasets ...")
    train_files, val_files = make_splits()
    print(f"  Train : {len(train_files)}  |  Val : {len(val_files)}")

    train_ds = MarineFeatureDataset(train_files)
    val_ds   = MarineFeatureDataset(val_files)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=SpeciesBalancedSampler(
            train_ds, batch_size=batch_size, k_samples=K_SAMPLES
        ),
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    print("\nInitialising model ...")
    pipeline = MarineImageBindPipeline().to(device)
    best_score = -1.0

    pipeline.param_summary()

    opt = optim.AdamW(
        [
            {"params": pipeline.image_head.parameters(), "lr": LEARNING_RATE},
            {"params": pipeline.audio_head.parameters(), "lr": LEARNING_RATE},
            {"params": pipeline.text_head.parameters(),  "lr": LEARNING_RATE * 0.1},
        ],
        weight_decay=args.weight_decay,
    )
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=1e-6
    )

    memory_bank = {}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss = train_epoch_joint(pipeline, train_loader, opt, device, memory_bank)
        vr = calculate_cross_modal_recall(pipeline, val_loader, device)
        sc = composite_score(vr)
        sched.step()

        print(
            f"Epoch {epoch:03d}/{epochs:03d}  |  "
            f"loss={loss:.4f}  |  "
            f"I->T R@1={vr.get('img2txt_R@1', 0):.3f}  "
            f"I->A R@1={vr.get('img2aud_R@1', 0):.3f}  |  "
            f"composite={sc:.4f}  |  {time.time()-t0:.1f}s"
        )
        if sc > best_score:
            best_score = sc
            save_checkpoint(pipeline, epoch, sc, vr)

    print(f"\n{'='*64}")
    print(f"  Training complete.  Best composite: {best_score:.4f}")
    print(f"  Checkpoint    = {CHECKPOINT_PATH}")
    print(f"{'='*64}")


# == CLI =======================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Joint Marine Multimodal Contrastive Alignment Training"
    )
    p.add_argument("--epochs",       type=int,   default=EPOCHS)
    p.add_argument("--batch_size",   type=int,   default=BATCH_SIZE)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--device",       type=str,   default=DEVICE)
    # Ignored legacy flags to keep CLI backward compatible
    p.add_argument("--p1", type=int, default=None)
    p.add_argument("--p2", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
