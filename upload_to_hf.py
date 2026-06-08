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
# PHASE 1: COMPRESS AND UPLOAD DATASET
# ==========================================
DATASET_DIR = os.path.join(LOCAL_DIR, "datasets")
ZIP_PATH = os.path.join(LOCAL_DIR, "marine_datasets_archive.zip")

if not os.path.exists(ZIP_PATH):
    print("\n--- PHASE 1: COMPRESSING DATASETS ---")
    print("Compressing datasets into a single ZIP file to bypass the 1,000 file rate limit...")
    print("This may take a few minutes depending on the dataset size...")
    shutil.make_archive(r"C:\Projects\marine\marine_datasets_archive", 'zip', DATASET_DIR)
    print(f"Successfully created {ZIP_PATH}")
else:
    print(f"\n--- PHASE 1: DATASETS ALREADY COMPRESSED ---")
    print(f"Found existing zip file: {ZIP_PATH}")

print("Uploading compressed datasets...")
api.upload_file(
    path_or_fileobj=ZIP_PATH,
    path_in_repo="marine_datasets_archive.zip",
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
)
print("Dataset upload complete!")

# ==========================================
# PHASE 2: UPLOAD CODE AND MODELS
# ==========================================
print("\n--- PHASE 2: UPLOADING CODEBASE AND MODELS ---")
print("Using upload_large_folder() to handle heavy model weights...")

# Uploading folder (We ignore the raw datasets here because we already uploaded the ZIP!)
api.upload_large_folder(
    folder_path=LOCAL_DIR,
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    ignore_patterns=[".git", "__pycache__", "*.log", "eval_output.txt", "datasets/*"], 
)

print(f"\nAll Uploads Complete! View your entire project here: https://huggingface.co/{REPO_ID}")
