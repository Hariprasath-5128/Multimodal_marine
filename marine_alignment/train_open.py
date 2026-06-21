"""
train.py ΓÇö Two-Phase Training for Marine Multimodal Alignment
=============================================================
Phase 1 (PHASE1_EPOCHS epochs):
    Train image_head + text_head with SupCon (tau=0.07).
    Audio head is FROZEN - no audio losses at all.
    Batch: 64 samples = 32 species x 2 samples each.
    Goal: push I->T R@1 above 80%.

Phase 2 (PHASE2_EPOCHS epochs):
    FREEZE image_head + text_head completely.
    Train audio_head ONLY with bridge loss:
        l_bridge = mean(1 - cosine_sim(audio_proj, frozen_text_proj))
    No direct loss, no SupCon - eliminates tug-of-war gradient conflict.
    Goal: cos_sim(audio, text) -> 0.998+, making I->A approx I->T.

Why two phases?
    Single-phase training creates gradient conflict: bridge loss pulls
    audio toward text_proj, direct loss pulls audio toward image_proj.
    Since image_proj != text_proj (I->T is only 72%), the two forces
    oppose each other and both plateau. Freezing text_head in phase 2
    gives audio_head a STABLE, FIXED target that it can fully minimize.

Usage
-----
    python train.py
    python train.py --p1 100 --p2 100
    python train.py --device cpu
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
    DEVICE, LEARNING_RATE, WEIGHT_DECAY, TEMPERATURE, EPOCHS,
    BATCH_SIZE, MIN_VALID_SAMPLES, CHECKPOINT_PATH,
    RECALL_WEIGHT_R1, RECALL_WEIGHT_R5, RECALL_WEIGHT_R10,
    HARD_NEG_COUNT
)
from dataset import MarineFeatureDataset, SpeciesBalancedSampler, make_splits
from models  import MarineImageBindPipeline
from loss    import supervised_contrastive_loss, check_loss_finite

# == Phase configuration ======================================================
PHASE1_EPOCHS   = 100        # image <-> text SupCon only; audio head frozen
PHASE2_EPOCHS   = 100        # audio bridge only; image+text heads frozen

# == Batch construction =======================================================
# 64 samples = 32 species x K=2.  More species per batch = harder negatives
# for SupCon = stronger gradient = better I->T separation.
P1_BATCH_SIZE   = 64
P1_K_SAMPLES    = 2          # 32 species per batch

# == Loss hyperparameters =====================================================
TEMPERATURE_TXT = 0.07       # tight tau for image-text SupCon

# == Learning rates ===========================================================
P1_LR           = 1e-4      # Phase 1: image+text heads
P2_AUDIO_LR     = 5e-4      # Phase 2: audio head (stable target, can be high)


def _phase_bar(epoch, total, phase):
    return f"[P{phase}] Epoch {epoch:03d}/{total:03d}"


# == Phase-1 training step ====================================================

def train_epoch_phase1(pipeline, dataloader, optimizer, device):
    """
    Image <-> Text SupCon only. Audio head is frozen.
    """
    pipeline.train()

    total_loss    = 0.0
    text_loss_sum = 0.0
    n_batches     = 0
    n_txt         = 0

    for batch in dataloader:
        optimizer.zero_grad()

        image_emb = batch["image_emb"].to(device)
        text_emb  = batch["text_emb"].to(device)
        has_text  = batch["has_text"].to(device)
        labels    = batch["species_id"].to(device)

        if n_batches == 0:
            unique, counts = torch.unique(labels, return_counts=True)
            print("  [DEBUG] batch species dist: " +
                  str(list(zip(unique.tolist(), counts.tolist()))))

        proj_img = pipeline.image_head(image_emb)

        loss_sum = None
        if has_text.sum() >= MIN_VALID_SAMPLES:
            valid_img = proj_img[has_text]
            valid_txt = pipeline.text_head(text_emb[has_text])
            valid_lbl = labels[has_text]

            l_txt = supervised_contrastive_loss(
                valid_img, valid_txt, valid_lbl,
                TEMPERATURE_TXT, HARD_NEG_COUNT,
                exclude_diagonal=False,
            )
            check_loss_finite(l_txt, "image-text")
            loss_sum      = l_txt
            text_loss_sum += l_txt.item()
            n_txt         += 1

        if loss_sum is not None:
            loss_sum.backward()
            torch.nn.utils.clip_grad_norm_(
                list(pipeline.image_head.parameters()) +
                list(pipeline.text_head.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            total_loss += loss_sum.item()

        n_batches += 1

    safe = lambda a, b: a / b if b > 0 else 0.0
    return {
        "loss_total":   safe(total_loss,    n_batches),
        "loss_text":    safe(text_loss_sum, n_txt),
        "batches_text": n_txt,
    }


# == Phase-2 training step ====================================================

def train_epoch_phase2(pipeline, dataloader, optimizer, device):
    """
    Audio bridge loss ONLY. Image and text heads are frozen.
    Bridge: l = mean(1 - cosine_sim(audio_proj, frozen_text_proj)).
    """
    pipeline.train()
    # Keep frozen heads in eval mode (important for BatchNorm/LayerNorm)
    pipeline.image_head.eval()
    pipeline.text_head.eval()

    total_loss      = 0.0
    bridge_loss_sum = 0.0
    n_batches       = 0

    for batch in dataloader:
        optimizer.zero_grad()

        text_emb  = batch["text_emb"].to(device)
        audio_emb = batch["audio_emb"].to(device)
        labels    = batch["species_id"].to(device)

        if n_batches == 0:
            unique, counts = torch.unique(labels, return_counts=True)
            print("  [DEBUG] batch species dist: " +
                  str(list(zip(unique.tolist(), counts.tolist()))))

        # Frozen text target (no gradient propagated into text_head)
        with torch.no_grad():
            txt_anchor = pipeline.text_head(text_emb)   # [B, D]

        # Audio projection - ONLY trainable part in phase 2
        proj_aud = pipeline.audio_head(audio_emb)       # [B, D]

        cos_sim  = F.cosine_similarity(proj_aud, txt_anchor, dim=-1)
        l_bridge = (1.0 - cos_sim).mean()
        check_loss_finite(l_bridge, "bridge")

        l_bridge.backward()
        torch.nn.utils.clip_grad_norm_(
            pipeline.audio_head.parameters(), max_norm=1.0
        )
        optimizer.step()

        total_loss      += l_bridge.item()
        bridge_loss_sum += l_bridge.item()
        n_batches       += 1

    safe = lambda a, b: a / b if b > 0 else 0.0
    return {
        "loss_total":     safe(total_loss,      n_batches),
        "loss_bridge":    safe(bridge_loss_sum, n_batches),
        "batches_bridge": n_batches,
    }


# == Recall Evaluation =========================================================

@torch.no_grad()
def calculate_cross_modal_recall(pipeline, dataloader, device, ks=(1, 5, 10)):
    pipeline.eval()

    all_img, all_txt, all_aud = [], [], []
    all_lbl_img, all_lbl_txt, all_lbl_aud = [], [], []

    for batch in dataloader:
        img_emb   = batch["image_emb"].to(device)
        txt_emb   = batch["text_emb"].to(device)
        aud_emb   = batch["audio_emb"].to(device)
        has_text  = batch["has_text"].to(device)
        has_audio = batch["has_audio"].to(device)
        labels    = batch["species_id"].to(device)

        proj_img = pipeline.image_head(img_emb)
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
    device    = args.device
    p1_epochs = args.p1
    p2_epochs = args.p2
    batch_size = args.batch_size

    print("=" * 64)
    print("  Marine Multimodal Alignment - Two-Phase Training")
    print("=" * 64)
    print(f"  Device         : {device}")
    print(f"  Phase-1 epochs : {p1_epochs}  (image<->text SupCon, audio frozen)")
    print(f"  Phase-2 epochs : {p2_epochs}  (audio bridge, image+text frozen)")
    print(f"  Batch size     : {batch_size}  ({batch_size//P1_K_SAMPLES} species x {P1_K_SAMPLES} per batch)")
    print()

    print("Building datasets ...")
    train_files, val_files = make_splits()
    print(f"  Train : {len(train_files)}  |  Val : {len(val_files)}")

    train_ds = MarineFeatureDataset(train_files)
    val_ds   = MarineFeatureDataset(val_files)

    p1_loader = DataLoader(
        train_ds,
        batch_sampler=SpeciesBalancedSampler(
            train_ds, batch_size=batch_size, k_samples=P1_K_SAMPLES
        ),
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    p2_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
        drop_last=False,
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
    
    # Load backup checkpoint if we are resuming Phase 2
    ckpt_path = os.path.join(os.path.dirname(CHECKPOINT_PATH), "best_multimodal_pipeline_backup.pth")
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path} to resume training...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        pipeline.load_state_dict(ckpt["model_state"])
        best_score = ckpt.get("composite_score", -1.0)
    else:
        best_score = -1.0

    pipeline.param_summary()
    vr = {}

    # ========================================================================
    # PHASE 1 - Image <-> Text SupCon  (audio head frozen)
    # ========================================================================
    print(f"\n{'-'*64}")
    print(f"  PHASE 1 - Image<->Text SupCon  ({p1_epochs} epochs)")
    print(f"  Audio head FROZEN.  Batch={batch_size}, K={P1_K_SAMPLES}")
    print(f"{'-'*64}\n")

    for p in pipeline.audio_head.parameters():
        p.requires_grad_(False)

    opt_p1 = optim.AdamW(
        [
            {"params": pipeline.image_head.parameters()},
            {"params": pipeline.text_head.parameters()},
        ],
        lr=P1_LR,
        weight_decay=args.weight_decay,
    )
    sched_p1 = optim.lr_scheduler.CosineAnnealingLR(
        opt_p1, T_max=p1_epochs, eta_min=1e-6
    )

    for epoch in range(1, p1_epochs + 1):
        t0 = time.time()
        m  = train_epoch_phase1(pipeline, p1_loader, opt_p1, device)
        vr = calculate_cross_modal_recall(pipeline, val_loader, device)
        sc = composite_score(vr)
        sched_p1.step()

        print(
            f"{_phase_bar(epoch, p1_epochs, 1)}  |  "
            f"txt={m['loss_text']:.4f}  |  "
            f"I->T R@1={vr.get('img2txt_R@1', 0):.3f}  "
            f"I->A R@1={vr.get('img2aud_R@1', 0):.3f}  |  "
            f"composite={sc:.4f}  |  {time.time()-t0:.1f}s"
        )
        if sc > best_score:
            best_score = sc
            save_checkpoint(pipeline, epoch, sc, vr)

    print(f"\n  Phase 1 complete.  Best composite: {best_score:.4f}")
    p1_it = vr.get('img2txt_R@1', 0)
    print(f"  Final I->T R@1 = {p1_it:.3f}\n")

    # ========================================================================
    # PHASE 2 - Audio Bridge  (image+text heads frozen)
    # ========================================================================
    print(f"{'-'*64}")
    print(f"  PHASE 2 - Audio Bridge  ({p2_epochs} epochs)")
    print(f"  Image+Text heads FROZEN.  Audio LR={P2_AUDIO_LR}")
    print(f"{'-'*64}\n")

    for p in pipeline.image_head.parameters():
        p.requires_grad_(False)
    for p in pipeline.text_head.parameters():
        p.requires_grad_(False)
    for p in pipeline.audio_head.parameters():
        p.requires_grad_(True)

    opt_p2 = optim.AdamW(
        pipeline.audio_head.parameters(),
        lr=P2_AUDIO_LR,
        weight_decay=args.weight_decay,
    )
    sched_p2 = optim.lr_scheduler.CosineAnnealingLR(
        opt_p2, T_max=p2_epochs, eta_min=1e-6
    )

    for epoch in range(1, p2_epochs + 1):
        t0 = time.time()
        m  = train_epoch_phase2(pipeline, p2_loader, opt_p2, device)
        vr = calculate_cross_modal_recall(pipeline, val_loader, device)
        sc = composite_score(vr)
        sched_p2.step()

        print(
            f"{_phase_bar(epoch, p2_epochs, 2)}  |  "
            f"brg={m['loss_bridge']:.4f}  "
            f"(cos={1-m['loss_bridge']:.4f})  |  "
            f"I->T R@1={vr.get('img2txt_R@1', 0):.3f}  "
            f"I->A R@1={vr.get('img2aud_R@1', 0):.3f}  |  "
            f"composite={sc:.4f}  |  {time.time()-t0:.1f}s"
        )
        if sc > best_score:
            best_score = sc
            save_checkpoint(pipeline, epoch, sc, vr)

    print(f"\n{'='*64}")
    print(f"  Training complete.  Best composite: {best_score:.4f}")
    print(f"  Final I->T R@1 = {vr.get('img2txt_R@1', 0):.3f}")
    print(f"  Final I->A R@1 = {vr.get('img2aud_R@1', 0):.3f}")
    print(f"  Checkpoint    = {CHECKPOINT_PATH}")
    print(f"{'='*64}")


# == CLI =======================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Two-Phase Marine Multimodal Alignment Training"
    )
    p.add_argument("--p1",           type=int,   default=PHASE1_EPOCHS,
                   help="Phase-1 epochs (image+text SupCon)")
    p.add_argument("--p2",           type=int,   default=PHASE2_EPOCHS,
                   help="Phase-2 epochs (audio bridge)")
    p.add_argument("--batch_size",   type=int,   default=P1_BATCH_SIZE)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--device",       type=str,   default=DEVICE)
    # Legacy flags - accepted but ignored
    p.add_argument("--epochs", type=int,   default=None)
    p.add_argument("--lr",     type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
