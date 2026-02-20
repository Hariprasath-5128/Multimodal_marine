from datasets import load_dataset
import pandas as pd

# Load both datasets from Hugging Face
print("Loading datasets...")
audio_ds = load_dataset("ardavey/marine_ocean_mammal_sound", split="train")
image_ds = load_dataset("yeyimilk/LLM-Vision-Marine-Animals", split="train")

# Convert to pandas DataFrames
audio_df = audio_ds.to_pandas()
image_df = image_ds.to_pandas()

print(f"\nAudio dataset shape: {audio_df.shape}")
print(f"Image dataset shape: {image_df.shape}")

# Function to normalize species names for matching
def normalize_name(name):
    """Normalize species names by removing special characters and standardizing format"""
    if pd.isna(name):
        return ""
    name = str(name).strip()
    # Replace underscores with spaces, remove commas, convert to lowercase
    name = name.replace("_", " ").replace(",", "").lower()
    # Remove extra spaces
    name = " ".join(name.split())
    return name

# Normalize species names in both datasets
audio_df["species_normalized"] = audio_df["species"].apply(normalize_name)
image_df["animal_name_normalized"] = image_df["animal_name"].apply(normalize_name)

# Get unique species from both datasets
audio_species = set(audio_df["species_normalized"].unique())
image_species = set(image_df["animal_name_normalized"].unique())

# Remove empty strings if any
audio_species.discard("")
image_species.discard("")

# Find matching species
matched_species = audio_species.intersection(image_species)
unmatched_audio_species = audio_species - image_species

# Calculate mapping percentage
num_audio_species = len(audio_species)
num_matched_species = len(matched_species)
mapping_percentage = (num_matched_species / num_audio_species * 100) if num_audio_species > 0 else 0

# Display results
print("\n" + "="*70)
print("MAPPING ANALYSIS RESULTS")
print("="*70)
print(f"\nTotal unique species in audio dataset: {num_audio_species}")
print(f"Total unique species in image dataset: {len(image_species)}")
print(f"Matched species (audio → image): {num_matched_species}")
print(f"\n*** MAPPING PERCENTAGE: {mapping_percentage:.2f}% ***")
print(f"    ({num_matched_species} out of {num_audio_species} audio species have images)")
print("="*70)

# Show matched species
print("\n✓ MATCHED SPECIES (present in both datasets):")
print("-" * 70)
for i, species in enumerate(sorted(matched_species), 1):
    # Get original names from audio dataset
    original_audio = audio_df[audio_df["species_normalized"] == species]["species"].iloc[0]
    original_image = image_df[image_df["animal_name_normalized"] == species]["animal_name"].iloc[0]
    print(f"{i:2d}. {original_audio:40s} → {original_image}")

# Show unmatched species
print(f"\n✗ UNMATCHED SPECIES (only in audio dataset):")
print("-" * 70)
for i, species in enumerate(sorted(unmatched_audio_species), 1):
    original_audio = audio_df[audio_df["species_normalized"] == species]["species"].iloc[0]
    print(f"{i:2d}. {original_audio}")

# Create summary dataframe
summary_data = {
    'Metric': [
        'Audio Species Count',
        'Image Species Count',
        'Matched Species',
        'Unmatched Audio Species',
        'Mapping Percentage'
    ],
    'Value': [
        num_audio_species,
        len(image_species),
        num_matched_species,
        len(unmatched_audio_species),
        f"{mapping_percentage:.2f}%"
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(summary_df.to_string(index=False))
print("="*70)

# Optional: Save results to CSV
summary_df.to_csv("mapping_summary.csv", index=False)

matched_list = pd.DataFrame({
    'Audio_Species': sorted([audio_df[audio_df["species_normalized"] == s]["species"].iloc[0] 
                             for s in matched_species]),
    'Image_Species': sorted([image_df[image_df["animal_name_normalized"] == s]["animal_name"].iloc[0] 
                             for s in matched_species])
})
matched_list.to_csv("matched_species.csv", index=False)

unmatched_list = pd.DataFrame({
    'Unmatched_Audio_Species': sorted([audio_df[audio_df["species_normalized"] == s]["species"].iloc[0] 
                                       for s in unmatched_audio_species])
})
unmatched_list.to_csv("unmatched_species.csv", index=False)

print("\n✓ Results saved to: mapping_summary.csv, matched_species.csv, unmatched_species.csv")
