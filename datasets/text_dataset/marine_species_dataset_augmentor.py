import os
import requests
import time
import numpy as np
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util

# Configuration
INPUT_DIR = r"C:\Projects\marine\datasets\text_dataset\train_dataset"
OUTPUT_DIR = r"C:\Projects\marine\datasets\text_dataset\expanded_train_dataset"
QA_REPORT_PATH = os.path.join(OUTPUT_DIR, "_Dataset_QA_Report.txt")

MODEL = "llama3:8b"
NUM_VARIATIONS = 15
MAX_TOKENS = 380 
SIMILARITY_THRESHOLD = 0.93
MAX_ATTEMPTS = 45 

print("Loading MPNet Tokenizer & Embedder...")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize the master QA report file
with open(QA_REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write("========================================\n")
    f.write("MARINE DATASET TEXT EXPANSION QA REPORT\n")
    f.write("========================================\n\n")

STYLES = [
    "Wikipedia article", "Scientific paper", "Field guide", 
    "Educational textbook", "Museum exhibit", "Wildlife blog article", 
    "Wildlife reference manual", "Educational science article",
    "Research summary", "Species profile", "Nature magazine feature",
    "Taxonomic description", "Marine mammal handbook", 
    "Documentary narration", "Geographic distribution report"
]

def generate_single_variation(species_name, original_text, style, iteration):
    prompt = f"""
    You are an expert marine biologist.
    
    Original Data for {species_name}:
    {original_text}
    
    Task: Write exactly one paragraph describing this species in the style of a {style}.
    
    Rules:
    - Preserve all facts perfectly. DO NOT add facts not present in the source.
    - DO NOT infer missing information or mention conservation status unless explicitly given.
    - Emphasize distinctive traits explicitly mentioned in the source text.
    - Rewrite using completely different sentence structures.
    - This is variation #{iteration}. Ensure it is structurally and logically distinct.
    - Length: Approximately 120-220 words.
    - DO NOT use bullet points, markdown, or lists. Output ONLY the paragraph.
    """

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.75}
    }

    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            output = response.json().get('response', '').strip()
            
            if output and 100 <= len(output) <= 2500:
                token_count = len(tokenizer.encode(output, truncation=False))
                if token_count <= MAX_TOKENS:
                    return output, token_count
                else:
                    print(f"  [!] Output too long ({token_count} tokens). Retrying...")
            else:
                print(f"  [!] Output length invalid ({len(output)} chars). Retrying...")
                
        except requests.exceptions.RequestException as e:
            print(f"  [!] Request Error: {e}")
            time.sleep(2)
            
    return None, 0

def process_dataset():
    global_token_counts = []
    global_duplicate_rejections = 0
    global_total_attempts = 0
    
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".txt"):
            base_name = filename.replace(".txt", "")
            species_name = base_name.replace("_", " ").title()
            input_path = os.path.join(INPUT_DIR, filename)

            with open(input_path, 'r', encoding='utf-8') as file:
                original_text = file.read()

            print(f"\n{'='*40}\nProcessing: {species_name}\n{'='*40}")
            
            successful_generations = 0
            total_attempts = 0
            
            species_tokens = []
            original_emb = embedder.encode(original_text, convert_to_tensor=True)
            generated_embeddings = [original_emb] 
            
            while successful_generations < NUM_VARIATIONS and total_attempts < MAX_ATTEMPTS:
                total_attempts += 1
                global_total_attempts += 1
                
                # CYCLING FIX: Use total_attempts to bypass failing styles
                style = STYLES[(total_attempts - 1) % len(STYLES)]
                print(f"  Attempt {total_attempts} | Generating valid sample {successful_generations + 1}/{NUM_VARIATIONS} (Style: {style})...")
                
                variation_text, token_count = generate_single_variation(species_name, original_text, style, successful_generations + 1)

                if variation_text:
                    new_emb = embedder.encode(variation_text, convert_to_tensor=True)
                    
                    is_duplicate = False
                    for old_emb in generated_embeddings:
                        sim = util.cos_sim(new_emb, old_emb).item()
                        if sim > SIMILARITY_THRESHOLD:
                            is_duplicate = True
                            print(f"  [!] Semantic duplicate detected (Similarity: {sim:.2f}). Rejecting.")
                            global_duplicate_rejections += 1
                            break
                            
                    if not is_duplicate:
                        generated_embeddings.append(new_emb)
                        species_tokens.append(token_count)
                        global_token_counts.append(token_count)
                        
                        clean_style = "".join(word.capitalize() for word in style.split())
                        new_filename = f"{base_name}_{successful_generations + 1}_{clean_style}.txt"
                        output_path = os.path.join(OUTPUT_DIR, new_filename)
                        
                        with open(output_path, 'w', encoding='utf-8') as out_file:
                            out_file.write(variation_text)
                            
                        successful_generations += 1

            # --- Write Per-Species Report to File ---
            report_chunk = f"--- {species_name} ---\n"
            report_chunk += f"Count: {successful_generations} files generated in {total_attempts} attempts\n"
            
            if species_tokens:
                report_chunk += f"Average tokens: {int(np.mean(species_tokens))}\n"
                report_chunk += f"Min tokens: {np.min(species_tokens)}\n"
                report_chunk += f"Max tokens: {np.max(species_tokens)}\n"
            
            # TENSOR FIX: Native PyTorch stacking for the similarity matrix
            if len(generated_embeddings) > 2:
                gen_tensors = torch.stack(generated_embeddings[1:])
                sim_matrix = util.cos_sim(gen_tensors, gen_tensors).cpu().numpy()
                
                upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
                sim_values = sim_matrix[upper_tri_indices]
                
                report_chunk += f"Mean similarity: {np.mean(sim_values):.2f}\n"
                report_chunk += f"Max similarity:  {np.max(sim_values):.2f}\n"
                report_chunk += f"Min similarity:  {np.min(sim_values):.2f}\n"
            
            report_chunk += "\n"
            
            with open(QA_REPORT_PATH, 'a', encoding='utf-8') as f:
                f.write(report_chunk)
                
    # --- Write Global QA Report to File ---
    if global_token_counts:
        global_chunk = f"{'='*40}\nGLOBAL DATASET QA REPORT\n{'='*40}\n"
        global_chunk += f"Total generation attempts: {global_total_attempts}\n"
        global_chunk += f"Accepted files: {len(global_token_counts)}\n"
        global_chunk += f"Rejected duplicates: {global_duplicate_rejections}\n"
        global_chunk += f"Average tokens: {int(np.mean(global_token_counts))}\n"
        global_chunk += f"Median tokens:  {int(np.median(global_token_counts))}\n"
        global_chunk += f"Minimum tokens: {np.min(global_token_counts)}\n"
        global_chunk += f"Maximum tokens: {np.max(global_token_counts)}\n"
        global_chunk += f"Files > 300 tokens: {sum(1 for x in global_token_counts if x > 300)}\n"
        global_chunk += f"Files > 384 tokens: {sum(1 for x in global_token_counts if x > 384)}\n"
        global_chunk += f"Files < 80 tokens:  {sum(1 for x in global_token_counts if x < 80)}\n"
        
        with open(QA_REPORT_PATH, 'a', encoding='utf-8') as f:
            f.write(global_chunk)
            
        print(f"\n[SUCCESS] Full QA report written to {QA_REPORT_PATH}")

if __name__ == "__main__":
    process_dataset()