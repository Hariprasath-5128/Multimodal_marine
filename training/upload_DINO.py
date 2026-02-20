from huggingface_hub import hf_hub_download, upload_file

upload_file(
    path_or_fileobj="best_research_model.pth",
    path_in_repo="best_research_model.pth",
    repo_id="Hariprasath5128/Multimodal_marine",
    repo_type="model"
)

print("Renamed repo upload complete")