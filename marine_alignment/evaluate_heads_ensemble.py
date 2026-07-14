import os
import sys
import json
import argparse
from collections import defaultdict
import torch
import torch.nn.functional as F
from tqdm import tqdm
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CHECKPOINT_PATH_CLOSED, CHECKPOINT_PATH_OPEN, DEVICE,
    IMAGE_MODEL_DIR, TEXT_MODEL_DIR, AUDIO_MODEL_PATH, AST_PRETRAINED_ID,
    AUDIO_SAMPLE_RATE, AUDIO_TARGET_SECONDS, SHARED_DIM,
)

from dataset import (
    get_test_image_split,
    get_test_text_split,
    get_test_audio_split,
)


# ─────────────────────────────────────────────────────────────────────────────
# Encoders & Raw Feature Loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_image_encoder(device: str):
    import timm
    from torchvision import transforms
    from pathlib import Path

    ckpts = sorted(Path(IMAGE_MODEL_DIR).glob("*_seed42.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No *_seed42.pth found in {IMAGE_MODEL_DIR}")
    ckpt = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    bb_name = ckpt.get("backbone", "convnextv2_base")
    img_size = ckpt.get("img_size", 288)

    model = timm.create_model(bb_name, pretrained=False, num_classes=0, global_pool="avg")
    bb_sd = {k[len("backbone."):]: v for k, v in ckpt["model_state_dict"].items() if k.startswith("backbone.")}
    model.load_state_dict(bb_sd, strict=True)
    model.eval().to(device)

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return model, tfm


@torch.no_grad()
def _encode_raw_image(bb, tfm, img_path: str, device: str) -> torch.Tensor:
    from PIL import Image
    import pillow_avif
    img = Image.open(img_path).convert("RGB")
    t = tfm(img).unsqueeze(0).to(device)
    return bb(t).squeeze(0).cpu()


def _load_text_encoder(device: str):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(TEXT_MODEL_DIR, device=device)
    m.eval()
    return m


def _encode_raw_texts(text_model, file_paths: list[str]) -> list[torch.Tensor]:
    """Read and encode text documents for one species."""
    texts = []
    for fp in file_paths:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                c = f.read().strip()
            if len(c.split()) > 5:
                texts.append(c)
        except Exception:
            pass
    if not texts:
        return []
    embs = text_model.encode(
        texts, convert_to_tensor=True,
        show_progress_bar=False, normalize_embeddings=True
    )
    return [e.cpu() for e in embs]


def _load_audio_encoder(device: str):
    from transformers import ASTForAudioClassification, ASTFeatureExtractor
    fe = ASTFeatureExtractor.from_pretrained(AST_PRETRAINED_ID)
    sd = torch.load(AUDIO_MODEL_PATH, map_location="cpu", weights_only=False)
    clf_key = next(
        (k for k in sd if "classifier" in k and "weight" in k and sd[k].ndim == 2),
        None
    )
    num_labels = sd[clf_key].shape[0] if clf_key else 32
    model = ASTForAudioClassification.from_pretrained(
        AST_PRETRAINED_ID, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    hook = {"hidden": None}
    def _fwd_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        hook["hidden"] = h.mean(dim=1)
    model.audio_spectrogram_transformer.encoder.layer[-1].register_forward_hook(_fwd_hook)

    return model, fe, hook


@torch.no_grad()
def _encode_raw_audio_file(model, fe, hook, wav_path: str, device: str) -> torch.Tensor | None:
    import torchaudio
    try:
        wav, sr = torchaudio.load(wav_path)
        wav = wav.mean(dim=0, keepdim=True)
        if sr != AUDIO_SAMPLE_RATE:
            wav = torchaudio.transforms.Resample(sr, AUDIO_SAMPLE_RATE)(wav)
        target = AUDIO_TARGET_SECONDS * AUDIO_SAMPLE_RATE
        if wav.shape[1] > target:
            wav = wav[:, :target]
        else:
            wav = F.pad(wav, (0, target - wav.shape[1]))
        inp = fe(wav.squeeze().numpy(), sampling_rate=AUDIO_SAMPLE_RATE, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        hook["hidden"] = None
        model(**inp)
        feat = hook["hidden"]
        return feat.squeeze(0).cpu() if feat is not None else None
    except Exception:
        return None


def _encode_raw_audio_species(model, fe, hook, wav_paths: list[str], device: str) -> list[torch.Tensor]:
    """Encode all wav clips for one species."""
    embs = [_encode_raw_audio_file(model, fe, hook, w, device) for w in wav_paths]
    embs = [e for e in embs if e is not None]
    return embs


# ─────────────────────────────────────────────────────────────────────────────
# Prototype Construction from Training Set
# ─────────────────────────────────────────────────────────────────────────────

def compute_train_prototypes(pipeline, model_type: str, device: str, sp2id: dict):
    from config import EMBEDDING_DIR
    import glob
    pt_files = glob.glob(os.path.join(EMBEDDING_DIR, "*.pt"))
    
    raw_images_by_sp = defaultdict(list)
    raw_texts_by_sp = defaultdict(list)
    raw_audios_by_sp = defaultdict(list)
    
    for fname in tqdm(pt_files, desc="Loading train features for prototypes"):
        data = torch.load(fname, map_location="cpu", weights_only=True)
        sp = data["species_name"]
        if sp not in sp2id:
            continue
        sp_id = sp2id[sp]
        
        if "image_emb" in data and data["image_emb"] is not None:
            raw_images_by_sp[sp_id].append(data["image_emb"].float().squeeze())
            
        if "text_embs" in data and data["text_embs"] is not None:
            t = data["text_embs"].float()
            if t.dim() == 1:
                raw_texts_by_sp[sp_id].append(t)
            else:
                for row in t:
                    raw_texts_by_sp[sp_id].append(row)
                    
        if "audio_embs" in data and data["audio_embs"] is not None:
            a = data["audio_embs"].float()
            if a.dim() == 1:
                raw_audios_by_sp[sp_id].append(a)
            else:
                for row in a:
                    raw_audios_by_sp[sp_id].append(row)

    G_img = {}
    G_txt = {}
    G_aud = {}
    
    with torch.no_grad():
        for sp_id in range(len(sp2id)):
            # Image
            if sp_id in raw_images_by_sp:
                imgs = torch.stack(raw_images_by_sp[sp_id]).to(device)
                if model_type == "closed":
                    projs = pipeline.project_image(imgs)
                else:
                    projs = pipeline.image_head(imgs)
                G_img[sp_id] = F.normalize(projs.mean(dim=0), p=2, dim=-1).cpu()
            else:
                G_img[sp_id] = torch.zeros(SHARED_DIM)
                
            # Text
            if sp_id in raw_texts_by_sp:
                txts = torch.stack(raw_texts_by_sp[sp_id]).to(device)
                projs = pipeline.text_head(txts)
                G_txt[sp_id] = F.normalize(projs.mean(dim=0), p=2, dim=-1).cpu()
            else:
                G_txt[sp_id] = torch.zeros(SHARED_DIM)
                
            # Audio
            if sp_id in raw_audios_by_sp:
                auds = torch.stack(raw_audios_by_sp[sp_id]).to(device)
                projs = pipeline.audio_head(auds)
                G_aud[sp_id] = F.normalize(projs.mean(dim=0), p=2, dim=-1).cpu()
            else:
                G_aud[sp_id] = torch.zeros(SHARED_DIM)
                
    return G_img, G_txt, G_aud


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble Evaluation Function
# ─────────────────────────────────────────────────────────────────────────────

def run_ensemble_evaluation(model_type: str, raw_data: dict, device: str, use_train_gallery: bool):
    KS = (1, 5, 10)
    
    # 1. Load pipeline
    if model_type == "closed":
        import models_closed
        pipeline = models_closed.MarineImageBindPipeline().to(device)
        ckpt_path = CHECKPOINT_PATH_CLOSED
    else:
        import models_open
        import models_closed  # fallback to same path loading logic
        pipeline = models_open.MarineImageBindPipeline().to(device)
        ckpt_path = CHECKPOINT_PATH_OPEN
        
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pipeline.load_state_dict(ckpt["model_state"], strict=False)
    pipeline.eval()
    
    sp2id = raw_data["sp2id"]
    
    # 2. Compute prototypes / galleries
    G_img, G_txt, G_aud = compute_train_prototypes(pipeline, model_type, device, sp2id)

    # 3. Create the combined dataset of specific triplets (Image, Text, Audio)
    # We create a list of test tuples: (label, img_feat, txt_feat, aud_feat, group_name)
    test_dataset = []
    
    # Pre-project features to speed up evaluation
    projected_images = defaultdict(list)
    for sp_id, raw_feats in raw_data["images"].items():
        if raw_feats:
            with torch.no_grad():
                st = torch.stack(raw_feats).to(device)
                if model_type == "closed":
                    pr = pipeline.project_image(st)
                else:
                    pr = pipeline.image_head(st)
                for p in pr.cpu():
                    projected_images[sp_id].append(p)
                    
    projected_texts = defaultdict(list)
    for sp_id, raw_feats in raw_data["texts"].items():
        if raw_feats:
            with torch.no_grad():
                st = torch.stack(raw_feats).to(device)
                pr = pipeline.text_head(st)
                for p in pr.cpu():
                    projected_texts[sp_id].append(p)
                    
    projected_audios = defaultdict(list)
    for sp_id, raw_feats in raw_data["audios"].items():
        if raw_feats:
            with torch.no_grad():
                st = torch.stack(raw_feats).to(device)
                pr = pipeline.audio_head(st)
                for p in pr.cpu():
                    projected_audios[sp_id].append(p)

    for sp_id in range(len(sp2id)):
        imgs = projected_images[sp_id]
        txts = projected_texts[sp_id]
        auds = projected_audios[sp_id]
        
        # Determine combinations to create a combined dataset
        if len(imgs) > 0:
            if len(txts) > 0 and len(auds) > 0:
                for i, t, a in itertools.product(imgs, txts, auds):
                    test_dataset.append((sp_id, i, t, a, "3_modalities (I+T+A)"))
            elif len(txts) > 0:
                for i, t in itertools.product(imgs, txts):
                    test_dataset.append((sp_id, i, t, None, "2_modalities (I+T)"))
            elif len(auds) > 0:
                for i, a in itertools.product(imgs, auds):
                    test_dataset.append((sp_id, i, None, a, "2_modalities (I+A)"))
            else:
                for i in imgs:
                    test_dataset.append((sp_id, i, None, None, "1_modality (I)"))

    # 4. Run ensemble retrieval for each test query tuple
    correct_k = {k: 0 for k in KS}
    total_queries = 0
    modality_counts = defaultdict(int)
    
    for sp_id, P_I, P_T, P_A, group in test_dataset:
        modality_counts[group] += 1
        
        if P_T is not None and P_A is not None:
            w_img, w_txt, w_aud = 0.40, 0.33, 0.27
        elif P_T is not None:
            w_img, w_txt, w_aud = 0.50, 0.50, 0.00
        elif P_A is not None:
            w_img, w_txt, w_aud = 0.50, 0.00, 0.50
        else:
            w_img, w_txt, w_aud = 1.00, 0.00, 0.00
            
        # Compute combined similarity to all gallery species
        sims = torch.zeros(len(sp2id))
        for C in range(len(sp2id)):
            sim_img = torch.dot(P_I, G_img[C]).item()
            sim_txt = torch.dot(P_T, G_txt[C]).item() if P_T is not None else 0.0
            sim_aud = torch.dot(P_A, G_aud[C]).item() if P_A is not None else 0.0
            
            sims[C] = w_img * sim_img + w_txt * sim_txt + w_aud * sim_aud
            
        # Check Top-K
        for k in KS:
            topk_idx = sims.topk(min(k, len(sims))).indices
            if sp_id in topk_idx.tolist():
                correct_k[k] += 1
        total_queries += 1
        
    recalls = {k: correct_k[k] / total_queries for k in KS}
    return recalls, modality_counts


# ─────────────────────────────────────────────────────────────────────────────
# Main Program
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Ensemble Evaluation of projection heads.")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device to use for model forward passes")
    args = parser.parse_args()

    print("=" * 64)
    print("  Marine Multimodal Alignment -- Multi-Modal Ensemble Evaluation")
    print("=" * 64)

    # 1. Resolve test splits using dataset.py
    print("\nResolving test splits from dataset.py ...")
    img_samples, img_meta = get_test_image_split()
    txt_files, txt_meta = get_test_text_split()
    aud_files, aud_meta = get_test_audio_split()

    print(f"  Image  : {img_meta['n_images']:>5} images, {img_meta['n_species']:>3} species")
    print(f"  Text   : {txt_meta['n_docs']:>5} docs,   {txt_meta['n_species']:>3} species")
    print(f"  Audio  : {len(aud_files):>5} species with audio")

    # Build unique species label maps
    all_species = sorted(
        set(sp for sp, _ in img_samples)
        | set(txt_files.keys())
        | set(aud_files.keys())
    )
    sp2id = {sp: i for i, sp in enumerate(all_species)}

    # 2. Load frozen encoders
    print("\nLoading frozen encoders ...")
    img_enc, img_tfm = _load_image_encoder(args.device)
    print("  [Image] Loaded ConvNeXtV2 model and transforms")
    
    txt_enc = _load_text_encoder(args.device)
    print("  [Text]  Loaded SentenceTransformer")
    
    aud_enc, aud_fe, aud_hook = _load_audio_encoder(args.device)
    print("  [Audio] Loaded AST model")

    # 3. Extract Raw Features (Heavy computation done ONCE)
    print("\n" + "-" * 64)
    print("  STEP 1: Extracting raw features (heavy computation)...")
    print("-" * 64)
    
    # Image
    print("Encoding images...")
    raw_images = defaultdict(list)
    for sp, img_path in tqdm(img_samples, desc="Images", unit="img"):
        try:
            raw_feat = _encode_raw_image(img_enc, img_tfm, img_path, args.device)
            raw_images[sp2id[sp]].append(raw_feat)
        except Exception as e:
            print(f"  [WARN] Skipped {img_path}: {e}")

    # Text
    print("Encoding text docs...")
    raw_texts = defaultdict(list)
    for sp, fpaths in tqdm(txt_files.items(), desc="Text", unit="species"):
        if sp in sp2id:
            feats = _encode_raw_texts(txt_enc, fpaths)
            if feats:
                raw_texts[sp2id[sp]].extend(feats)

    # Audio
    print("Encoding audio files...")
    raw_audios = defaultdict(list)
    for sp, wav_paths in tqdm(aud_files.items(), desc="Audio", unit="species"):
        if sp in sp2id:
            feats = _encode_raw_audio_species(aud_enc, aud_fe, aud_hook, wav_paths, args.device)
            if feats:
                raw_audios[sp2id[sp]].extend(feats)

    # Package raw data
    raw_data = {
        "sp2id": sp2id,
        "images": dict(raw_images),
        "texts": dict(raw_texts),
        "audios": dict(raw_audios)
    }

    # Free memory occupied by frozen encoders
    del img_enc, txt_enc, aud_enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Run evaluations for both Closed and Open models, under both gallery setups
    print("\n" + "-" * 64)
    print("  STEP 2: Running Ensemble Evaluations...")
    print("-" * 64)
    
    results = {}
    
    for gallery_name, use_train_gallery in [("Train Prototypes Gallery", True)]:
        results[gallery_name] = {}
        for model in ["closed", "open"]:
            recalls, counts = run_ensemble_evaluation(model, raw_data, args.device, use_train_gallery)
            results[gallery_name][model] = {
                "recalls": recalls,
                "counts": dict(counts)
            }

    # 5. Print Results
    print("\n" + "=" * 80)
    print("  MULTI-MODAL ENSEMBLE EVALUATION COMPARISON REPORT")
    print("=" * 80)
    
    for gallery_name in results:
        print(f"\n  Gallery Setup: {gallery_name}")
        print("-" * 80)
        print(f"  {'Metric':<25} | {'Closed Head':<24} | {'Open Head':<24}")
        print("-" * 80)
        
        counts = results[gallery_name]["closed"]["counts"]
        print(f"  Query modality distribution:")
        for k, v in sorted(counts.items()):
            print(f"    - {k:<25}: {v:>4} test combinations")
        print("-" * 80)
        
        for k in [1, 5, 10]:
            closed_r = results[gallery_name]["closed"]["recalls"][k]
            open_r = results[gallery_name]["open"]["recalls"][k]
            diff = open_r - closed_r
            print(f"  Recall@{k:<22} | {closed_r:.4%}({closed_r*100:.1f}%)         | {open_r:.4%}({open_r*100:.1f}%)        ({diff:+.1%})")
            
        print("-" * 80)

    # 6. Save JSON report
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_evaluation_report_ensemble.json"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to: {report_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
