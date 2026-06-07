from huggingface_hub import HfApi, login
import os

# 1. Login using your token (Get this from https://huggingface.co/settings/tokens)
# You can uncomment the line below and paste your token, or run `huggingface-cli login` in the terminal beforehand.
# login("hf_YOUR_TOKEN_HERE")

api = HfApi()

# 2. Define your repository details
REPO_ID = "your_username/marine-multimodal-project" # Change this to your HF username and desired repo name
LOCAL_DIR = r"C:\Projects\marine"
REPO_TYPE = "model" # Options: "model", "dataset", or "space"

print(f"Creating repository: {REPO_ID} on Hugging Face...")
try:
    api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False, exist_ok=True)
    print("Repository is ready!")
except Exception as e:
    print(f"Error creating repo: {e}")

# 3. Upload the entire folder
print(f"Uploading entire directory ({LOCAL_DIR}) to Hugging Face. This may take a while depending on your internet speed...")

# Uploading folder
api.upload_folder(
    folder_path=LOCAL_DIR,
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    ignore_patterns=[".git", "__pycache__", "*.log", "eval_output.txt"], # Files/folders to ignore
)

print(f"Upload Complete! View your files here: https://huggingface.co/{REPO_ID}")
