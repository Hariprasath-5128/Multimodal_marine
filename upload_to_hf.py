import os
import shutil
from huggingface_hub import HfApi, login

# 1. Login using your saved token from env/.env
token = None
try:
    with open(r"C:\Projects\marine\env\.env", "r") as f:
        token = f.read().strip()
    login(token)
except Exception as e:
    print("Could not load valid token from .env:", e)

if not token or "Invalid" in str(e) if 'e' in locals() else False:
    print("\n[WARNING] It looks like your Hugging Face token is invalid.")
    print("GitHub likely revoked it automatically for your safety when we accidentally tried to commit it earlier!")
    print("Please go to https://huggingface.co/settings/tokens, create a new WRITE token, and paste it below.")
    token = input("Enter new HF Token: ").strip()
    login(token)

api = HfApi()

# 2. Define your repository details
REPO_ID = "Hariprasath5128/marine-multimodel"
LOCAL_DIR = r"C:\Projects\marine"
REPO_TYPE = "model"

print(f"\nTarget repository: {REPO_ID} on Hugging Face")

# ==========================================
# EXPLICIT MODEL DIRECTORY UPLOADS
# ==========================================
print("\n--- UPLOADING MODELS EXPLICITLY ---")
print("Bypassing the root .gitignore by targeting the model folders directly...")

def zip_and_upload(folder_path, repo_dir):
    if not os.path.exists(folder_path):
        print(f"Skipping {folder_path} because it doesn't exist.")
        return
        
    folder_name = os.path.basename(folder_path)
    zip_path = folder_path + ".zip"
    
    print(f"\nZipping {folder_name}...")
    shutil.make_archive(folder_path, 'zip', folder_path)
    
    zip_name = folder_name + ".zip"
    remote_path = f"{repo_dir}/{zip_name}"
    
    print(f"Uploading {zip_name} to {remote_path}...")
    api.upload_file(
        path_or_fileobj=zip_path,
        path_in_repo=remote_path,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )
    
    print(f"Deleting local zip {zip_path}...")
    os.remove(zip_path)

# 1. Audio Classification Model
zip_and_upload(
    r"C:\Projects\marine\training\audio_classification\marine_audio_classification_model",
    "training/audio_classification"
)

# 2. Image Classification Models
zip_and_upload(
    r"C:\Projects\marine\training\image_classification\models",
    "training/image_classification"
)

# 3. Multimodal Checkpoints
zip_and_upload(
    r"C:\Projects\marine\marine_alignment\checkpoints",
    "marine_alignment"
)
# 4. Multimodal Extracted Features Zip
print("\nUploading Extracted Features Zip...")
if os.path.exists(r"C:\Projects\marine\marine_alignment\extracted_features.zip"):
    api.upload_file(
        path_or_fileobj=r"C:\Projects\marine\marine_alignment\extracted_features.zip",
        path_in_repo="marine_alignment/extracted_features.zip",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )

# 5. Rest of the Codebase (Scripts, Markdown, etc)
print("\nUploading Codebase Scripts...")
api.upload_folder(
    folder_path=r"C:\Projects\marine",
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    ignore_patterns=[".git", ".git/**", "__pycache__", "*.log", "env/**", "datasets/**", "marine_alignment/extracted_features/**", "*.pth", "*.pt", "*.bin", "*.safetensors"], 
)

print(f"\nAll Uploads Complete! View your project here: https://huggingface.co/{REPO_ID}")
