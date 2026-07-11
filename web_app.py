"""
web_app.py — FastAPI Backend for Multimodal Marine Species Alignment
=====================================================================
Serves a cross-modal retrieval and zero-shot discovery dashboard.
"""

import os
import sys
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sklearn.svm import OneClassSVM

# Add marine_alignment folder to path
ALIGNMENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "marine_alignment")
sys.path.insert(0, ALIGNMENT_DIR)

from config import (
    CHECKPOINT_PATH_CLOSED, CHECKPOINT_PATH_OPEN, DEVICE,
    TEXT_MODEL_DIR,
)
from dataset import get_test_text_split, make_splits, EMBEDDING_DIR
import models_closed
import models_open

# Global definitions
app = FastAPI(title="Multimodal Marine Alignment Web Interface")

# Create static folder if not exists
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

class ModelManager:
    """Manages lazy-loading and memory caching of models."""
    def __init__(self):
        self.device = DEVICE
        self.closed_pipeline = None
        self.open_pipeline = None
        
        self.img_enc = None
        self.img_tfm = None
        
        self.txt_enc = None
        self.ast_model = None
        self.ast_extractor = None
        self.ast_hook = None

    def get_pipeline(self, model_type: str):
        if model_type == "closed":
            if self.closed_pipeline is None:
                print("  [Backend] Loading CLOSED pipeline...")
                pipeline = models_closed.MarineImageBindPipeline().to(self.device)
                ckpt = torch.load(CHECKPOINT_PATH_CLOSED, map_location=self.device, weights_only=False)
                pipeline.load_state_dict(ckpt["model_state"], strict=False)
                pipeline.eval()
                self.closed_pipeline = pipeline
            return self.closed_pipeline
        else:
            if self.open_pipeline is None:
                print("  [Backend] Loading OPEN pipeline...")
                pipeline = models_open.MarineImageBindPipeline().to(self.device)
                ckpt = torch.load(CHECKPOINT_PATH_OPEN, map_location=self.device, weights_only=False)
                pipeline.load_state_dict(ckpt["model_state"], strict=False)
                pipeline.eval()
                self.open_pipeline = pipeline
            return self.open_pipeline

    def get_image_encoder(self):
        if self.img_enc is None:
            print("  [Backend] Loading ConvNeXt Image encoder...")
            from feature_extractor import _load_image_encoder
            self.img_enc, self.img_tfm, _ = _load_image_encoder(self.device)
        return self.img_enc, self.img_tfm

    def get_text_encoder(self):
        if self.txt_enc is None:
            print("  [Backend] Loading SentenceTransformer Text encoder...")
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(TEXT_MODEL_DIR, device=self.device)
            m.eval()
            self.txt_enc = m
        return self.txt_enc

    def get_audio_encoder(self):
        if self.ast_model is None:
            print("  [Backend] Loading AST Audio encoder...")
            from feature_extractor import _load_audio_encoder
            m, ext, hk = _load_audio_encoder(self.device)
            self.ast_model = m
            self.ast_extractor = ext
            self.ast_hook = hk
        return self.ast_model, self.ast_extractor, self.ast_hook

manager = ModelManager()

# Pre-computed Galleries & SVMs (cached by model type)
galleries = {
    "closed": {"text": None, "image": None, "species_list": []},
    "open": {"text": None, "image": None, "species_list": []}
}

svms = {
    "closed": {"image": None, "text": None, "audio": None},
    "open": {"image": None, "text": None, "audio": None}
}

def precompute_galleries_and_svms():
    """Builds species listings, pre-computes SVMs, and targets."""
    print("\n" + "=" * 64)
    print("  Pre-computing database galleries and SVMs...")
    print("=" * 64)
    
    train_files, val_files = make_splits()
    
    # 1. Collect species list
    species_set = set()
    for fname in val_files:
        path = os.path.join(EMBEDDING_DIR, fname)
        data = torch.load(path, map_location="cpu", weights_only=True)
        species_set.add(data["species_name"])
    species_list = sorted(list(species_set))
    print(f"Loaded {len(species_list)} distinct marine mammal species in catalog.")

    # Load raw training embeddings
    raw_img_train = []
    raw_txt_train = []
    raw_aud_train = []
    for fname in train_files:
        path = os.path.join(EMBEDDING_DIR, fname)
        data = torch.load(path, map_location="cpu", weights_only=True)
        if "image_emb" in data and data["image_emb"] is not None:
            raw_img_train.append(data["image_emb"].float().squeeze())
        if "text_embs" in data and data["text_embs"] is not None:
            raw_txt_train.append(data["text_embs"].float())
        if "audio_embs" in data and data["audio_embs"] is not None:
            feats = data["audio_embs"].float()
            if feats.ndim == 1:
                raw_aud_train.append(feats)
            else:
                for idx in range(feats.size(0)):
                    raw_aud_train.append(feats[idx])

    raw_img_train = torch.stack(raw_img_train).to(DEVICE)
    raw_txt_train = torch.cat(raw_txt_train, dim=0).to(DEVICE)
    raw_aud_train = torch.stack(raw_aud_train).to(DEVICE)

    # Pre-compute text encoder gallery raw embeddings
    print("Encoding text prototypes...")
    txt_enc = manager.get_text_encoder()
    txt_files, _ = get_test_text_split()
    txt_sp_list = []
    raw_txt_list = []
    
    def _encode_texts_helper(file_paths):
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
            return None
        embs = txt_enc.encode(texts, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
        return F.normalize(embs.mean(dim=0), p=2, dim=0).cpu()

    for sp in species_list:
        if sp in txt_files:
            raw = _encode_texts_helper(txt_files[sp])
            if raw is not None:
                txt_sp_list.append(sp)
                raw_txt_list.append(raw)
    raw_txt_tensor = torch.stack(raw_txt_list).to(DEVICE)

    # Pre-compute image gallery validation raw embeddings
    species_imgs = defaultdict(list)
    for fname in val_files:
        path = os.path.join(EMBEDDING_DIR, fname)
        data = torch.load(path, map_location="cpu", weights_only=True)
        if "image_emb" in data and data["image_emb"] is not None:
            species_imgs[data["species_name"]].append(data["image_emb"].float().squeeze())

    for mtype in ["closed", "open"]:
        pipeline = manager.get_pipeline(mtype)
        
        # Project text prototypes to create Text Gallery
        with torch.no_grad():
            txt_embs = pipeline.text_head(raw_txt_tensor).cpu()
        galleries[mtype]["text"] = (txt_sp_list, txt_embs)
        galleries[mtype]["species_list"] = species_list

        # Project validation image centroids to create Image Gallery
        gallery_species = []
        gallery_embs = []
        for sp in species_list:
            if sp in species_imgs:
                img_stack = torch.stack(species_imgs[sp]).to(DEVICE)
                with torch.no_grad():
                    if mtype == "closed":
                        proj = pipeline.project_image(img_stack).mean(dim=0).cpu()
                    else:
                        proj = pipeline.image_head(img_stack).mean(dim=0).cpu()
                gallery_species.append(sp)
                gallery_embs.append(F.normalize(proj, p=2, dim=0))
        galleries[mtype]["image"] = (gallery_species, torch.stack(gallery_embs))

        # Train One-Class SVMs
        with torch.no_grad():
            # Image SVM
            if mtype == "closed":
                proj_img = pipeline.project_image(raw_img_train).cpu().numpy()
            else:
                proj_img = pipeline.image_head(raw_img_train).cpu().numpy()
            svm_img = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(proj_img)
            svms[mtype]["image"] = svm_img

            # Text SVM
            proj_txt = pipeline.text_head(raw_txt_train).cpu().numpy()
            svm_txt = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(proj_txt)
            svms[mtype]["text"] = svm_txt

            # Audio SVM
            if mtype == "closed":
                proj_aud = pipeline.project_audio(raw_aud_train).cpu().numpy()
            else:
                proj_aud = pipeline.audio_head(raw_aud_train).cpu().numpy()
            svm_aud = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(proj_aud)
            svms[mtype]["audio"] = svm_aud
            
    print("Pre-computation completed successfully.")
    print("=" * 64 + "\n")

# Run precomputation on startup
@app.on_event("startup")
def startup_event():
    precompute_galleries_and_svms()

@app.get("/api/species")
def get_species_catalog():
    # Returns sorted list of 75 species names
    return {"species": galleries["closed"]["species_list"]}

@app.post("/api/query")
async def handle_query(
    modality: str = Form(...),
    model_type: str = Form(...),
    text_input: str = Form(None),
    threshold: float = Form(None),
    margin_threshold: float = Form(None),
    file: UploadFile = File(None)
):
    pipeline = manager.get_pipeline(model_type)
    
    # Set default similarity and margin thresholds based on optimal sweeps
    if threshold is None or threshold < 0:
        if modality == "image":
            threshold = 0.56 if model_type == "closed" else 0.36
        elif modality == "text":
            threshold = 0.38 if model_type == "closed" else 0.30
        elif modality == "audio":
            threshold = 0.55 if model_type == "closed" else 0.55
            
    if margin_threshold is None or margin_threshold < 0:
        if modality == "audio":
            margin_threshold = 0.015 if model_type == "closed" else 0.015
        else:
            margin_threshold = 0.02 if model_type == "closed" else 0.02
        
    decision = "accept"
    scores_out = []
    svm_status = "INLIER"
    max_sim = 0.0
    margin = 0.0

    try:
        # Modality A: Text Description
        if modality == "text":
            if not text_input:
                raise HTTPException(status_code=400, detail="Text input is required for text query.")
            
            txt_enc = manager.get_text_encoder()
            embs = txt_enc.encode([text_input], convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
            raw_txt = F.normalize(embs.mean(dim=0), p=2, dim=0)
            
            with torch.no_grad():
                proj_q = pipeline.text_head(raw_txt.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
            
            # Compare against Image Gallery
            gallery_names, gallery_tensor = galleries[model_type]["image"]
            sims = torch.matmul(proj_q, gallery_tensor.T)
            topk = sims.topk(min(5, sims.size(0)))
            
            scores = topk.values.tolist()
            indices = topk.indices.tolist()
            max_sim = scores[0]
            margin = max_sim - np.mean(scores[1:])
            
            # OC-SVM Check
            svm_pred = svms[model_type]["text"].predict(proj_q.numpy().reshape(1, -1))[0]
            svm_status = "OUTLIER" if svm_pred == -1 else "INLIER"
            
            is_confident = (max_sim >= threshold) and (margin >= margin_threshold)
            if not is_confident and (svm_pred == -1 or max_sim < threshold or margin < margin_threshold):
                decision = "reject"
                
            for idx, sc in zip(indices, scores):
                scores_out.append({"species": gallery_names[idx], "similarity": float(sc)})

        # Modality B: Image Upload
        elif modality == "image":
            if not file:
                raise HTTPException(status_code=400, detail="Image file upload is required.")
                
            from PIL import Image
            img_enc, img_tfm = manager.get_image_encoder()
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
                
            try:
                img = Image.open(tmp_path).convert("RGB")
                t_img = img_tfm(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    raw_feat = img_enc(t_img)
                    if model_type == "closed":
                        proj_q = pipeline.project_image(raw_feat).squeeze(0).cpu()
                    else:
                        proj_q = pipeline.image_head(raw_feat).squeeze(0).cpu()
            finally:
                os.remove(tmp_path)
                
            # Compare against Text Gallery
            gallery_names, gallery_tensor = galleries[model_type]["text"]
            sims = torch.matmul(proj_q, gallery_tensor.T)
            topk = sims.topk(min(5, sims.size(0)))
            
            scores = topk.values.tolist()
            indices = topk.indices.tolist()
            max_sim = scores[0]
            margin = max_sim - np.mean(scores[1:])
            
            # OC-SVM Check
            svm_pred = svms[model_type]["image"].predict(proj_q.numpy().reshape(1, -1))[0]
            svm_status = "OUTLIER" if svm_pred == -1 else "INLIER"
            
            is_confident = (max_sim >= threshold) and (margin >= margin_threshold)
            if not is_confident and (svm_pred == -1 or max_sim < threshold or margin < margin_threshold):
                decision = "reject"
                
            for idx, sc in zip(indices, scores):
                scores_out.append({"species": gallery_names[idx], "similarity": float(sc)})

        # Modality C: Audio Upload
        elif modality == "audio":
            if not file:
                raise HTTPException(status_code=400, detail="Audio file upload is required.")
                
            ast_model, ast_extractor, ast_hook = manager.get_audio_encoder()
            from feature_extractor import extract_audio_embedding_from_file
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
                
            try:
                raw_aud = extract_audio_embedding_from_file(ast_model, ast_extractor, ast_hook, tmp_path, DEVICE)
                if raw_aud is None:
                    raise HTTPException(status_code=400, detail="Audio feature extraction failed.")
                with torch.no_grad():
                    if model_type == "closed":
                        proj_q = pipeline.project_audio(raw_aud.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
                    else:
                        proj_q = pipeline.audio_head(raw_aud.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
            finally:
                os.remove(tmp_path)
                
            # Compare against Text Gallery
            gallery_names, gallery_tensor = galleries[model_type]["text"]
            sims = torch.matmul(proj_q, gallery_tensor.T)
            topk = sims.topk(min(5, sims.size(0)))
            
            scores = topk.values.tolist()
            indices = topk.indices.tolist()
            max_sim = scores[0]
            margin = max_sim - np.mean(scores[1:])
            
            # OC-SVM Check
            svm_pred = svms[model_type]["audio"].predict(proj_q.numpy().reshape(1, -1))[0]
            svm_status = "OUTLIER" if svm_pred == -1 else "INLIER"
            
            is_confident = (max_sim >= threshold) and (margin >= margin_threshold)
            if not is_confident and (svm_pred == -1 or max_sim < threshold or margin < margin_threshold):
                decision = "reject"
                
            for idx, sc in zip(indices, scores):
                scores_out.append({"species": gallery_names[idx], "similarity": float(sc)})

        # Modality D: Name Lookup
        elif modality == "name":
            if not text_input:
                raise HTTPException(status_code=400, detail="Species name is required.")
            
            query_name = text_input.lower().strip().replace(" ", "_")
            species_list = galleries[model_type]["species_list"]
            
            # Find closest match or exact match
            matched_sp = None
            for sp in species_list:
                if query_name == sp.lower() or query_name in sp.lower():
                    matched_sp = sp
                    break
                    
            if not matched_sp:
                decision = "reject"
                svm_status = "OUTLIER"
                max_sim = 0.0
                margin = 0.0
            else:
                decision = "accept"
                svm_status = "INLIER"
                max_sim = 1.0
                margin = 1.0
                scores_out = []

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported modality: {modality}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return {
        "decision": decision,
        "svm_status": svm_status,
        "similarity": float(max_sim),
        "margin": float(margin),
        "threshold": float(threshold),
        "margin_threshold": float(margin_threshold),
        "matches": scores_out,
        "is_lookup": (modality == "name"),
        "matched_species": matched_sp if (modality == "name" and decision == "accept") else None
    }

# Serve static resources (images, text descriptions, and audios) from the datasets
from fastapi.responses import FileResponse
from dataset import get_test_image_split, get_test_text_split, get_test_audio_split, canonical

@app.get("/api/species/image/{species_name}")
def get_species_image(species_name: str):
    sp = canonical(species_name)
    try:
        samples, _ = get_test_image_split()
        img_paths = [p for s, p in samples if s == sp]
        if img_paths:
            return FileResponse(img_paths[0])
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="Image not found for this species.")

@app.get("/api/species/image/{species_name}/{img_idx}")
def get_species_image_by_index(species_name: str, img_idx: int):
    sp = canonical(species_name)
    try:
        samples, _ = get_test_image_split()
        img_paths = [p for s, p in samples if s == sp]
        if img_paths and 0 <= img_idx < len(img_paths):
            return FileResponse(img_paths[img_idx])
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="Image index not found.")

@app.get("/api/species/text/{species_name}")
def get_species_text(species_name: str):
    sp = canonical(species_name)
    try:
        texts, _ = get_test_text_split()
        if sp in texts and texts[sp]:
            with open(texts[sp][0], "r", encoding="utf-8") as f:
                return {"text": f.read().strip()}
    except Exception:
        pass
    return {"text": "No text description profile available for this species."}

@app.get("/api/species/audio/{species_name}")
def get_species_audio(species_name: str):
    sp = canonical(species_name)
    try:
        audios, _ = get_test_audio_split()
        if sp in audios and audios[sp]:
            return FileResponse(audios[sp][0])
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="Audio vocalization not found for this species.")

@app.get("/api/species/audio/{species_name}/{aud_idx}")
def get_species_audio_by_index(species_name: str, aud_idx: int):
    sp = canonical(species_name)
    try:
        audios, _ = get_test_audio_split()
        if sp in audios and audios[sp] and 0 <= aud_idx < len(audios[sp]):
            return FileResponse(audios[sp][aud_idx])
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="Audio index not found.")

@app.get("/api/species/assets/{species_name}")
def get_species_assets(species_name: str):
    sp = canonical(species_name)
    num_images = 0
    num_audios = 0
    text_content = "No description profile available for this species."
    
    try:
        samples, _ = get_test_image_split()
        img_paths = [p for s, p in samples if s == sp]
        num_images = len(img_paths)
    except Exception:
        pass
        
    try:
        audios, _ = get_test_audio_split()
        if sp in audios:
            num_audios = len(audios[sp])
    except Exception:
        pass

    try:
        texts, _ = get_test_text_split()
        if sp in texts and texts[sp]:
            with open(texts[sp][0], "r", encoding="utf-8") as f:
                text_content = f.read().strip()
    except Exception:
        pass

    return {
        "species": species_name,
        "num_images": num_images,
        "num_audios": num_audios,
        "text": text_content
    }

# Serve the static webpage
@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return "<h3>Index.html not found. Run web_app.py setup first.</h3>"

# Mount static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    print("\n[Backend] Starting Web Application Server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
