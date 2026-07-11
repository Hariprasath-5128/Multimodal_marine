import os
import requests
import time
import numpy as np
from transformers import AutoTokenizer

# Configuration for Test Data
INPUT_DIR = r"C:\Projects\marine\datasets\text_dataset\train_dataset"
OUTPUT_DIR = r"C:\Projects\marine\datasets\text_dataset\expanded_test_dataset"
QA_REPORT_PATH = os.path.join(OUTPUT_DIR, "_Test_Dataset_QA_Report.txt")

MODEL = "llama3:8b"
MAX_TOKENS = 380 

# Expand this dictionary with known aliases for maximum strictness
COMMON_NAME_ALIASES = {
    "amazon_river_dolphin": ["boto", "pink river dolphin", "bufeo"],
    "dugong": ["sea cow", "seacow"],
    "killer_whale": ["orca", "blackfish"],
    # Add other species as needed...
}

print("Loading MPNet Tokenizer for validation...")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(QA_REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write("========================================\n")
    f.write("MARINE DATASET TEST EXPANSION QA REPORT\n")
    f.write("========================================\n\n")

TEST_CASES = [
    {
        "name": "FeatureOnly",
        "prompt": "Write a 3-sentence description focusing strictly on physical appearance and diet. Do NOT mention geographic location or taxonomy.",
        "naming_rule": "CRITICAL: Replace the species name entirely with the exact string '[MASK]'. Do not use common or scientific names."
    },
    {
        "name": "BehaviorFirst",
        "prompt": "Write a paragraph starting with behavior and feeding habits, then discuss physical traits. Avoid mentioning taxonomy.",
        "naming_rule": "Use only the common everyday name of the species."
    },
    {
        "name": "HardNegative",
        "prompt": "Write a paragraph that briefly contrasts this species with a typical marine mammal, noting how this species has adapted differently. The primary subject must remain this species.",
        "naming_rule": "Use only the common everyday name of the species."
    },
    {
        "name": "ScientificOnly",
        "prompt": "Write a formal, taxonomic description of this species.",
        "naming_rule": "CRITICAL: Use ONLY the formal scientific (Latin) name. Do not include the common name, local aliases, and do not use phrases like 'commonly known as'."
    },
    {
        "name": "ShortFragment",
        "prompt": """
        Write EXACTLY ONE sentence.

        Maximum 15 words.

        No commas.
        No semicolons.
        No additional explanations.

        Focus only on the single most distinctive trait.
        """,
        "naming_rule": "Use the common name."
    }
]

def generate_test_variation(species_name, base_name, original_text, case_config):
    prompt = f"""
    You are an expert data synthesizer for a machine learning pipeline.
    
    Original Data for {species_name}:
    {original_text}
    
    Task: {case_config['prompt']}
    
    Naming Constraint: {case_config['naming_rule']}
    
    Rules:
    - Preserve the core factual traits perfectly.
    - DO NOT use bullet points, markdown, or lists. 
    - Output ONLY the requested text. Do not include introductions or conversational notes.
    """

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.70}
    }

    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            output = response.json().get('response', '').strip()
            
            if output and len(output) >= 15:
                token_count = len(tokenizer.encode(output, truncation=False))
                word_count = len(output.split())
                
                if token_count <= MAX_TOKENS:
                    
                    # 1. FeatureOnly Mask Validation
                    if case_config['name'] == "FeatureOnly" and output.count("[MASK]") == 0:
                        print("  [!] FeatureOnly failed: '[MASK]' token missing. Retrying...")
                        continue
                        
                    # 2. ShortFragment Word Count Validation
                    if case_config['name'] == "ShortFragment" and word_count > 20:
                        print(f"  [!] ShortFragment failed: Output too long ({word_count} words). Retrying...")
                        continue
                        
                    # 3. ScientificOnly Leakage Validation (Now with dynamic alias dictionary)
                    if case_config['name'] == "ScientificOnly":
                        output_lower = output.lower()
                        leak_phrases = ["commonly known", "also known", "called the", "english name", species_name.lower()]
                        
                        # Dynamically inject known aliases for this specific file
                        specific_aliases = COMMON_NAME_ALIASES.get(base_name, [])
                        leak_phrases.extend([alias.lower() for alias in specific_aliases])
                        
                        if any(phrase in output_lower for phrase in leak_phrases):
                            print("  [!] ScientificOnly failed: Common name or leakage phrase detected. Retrying...")
                            continue
                        
                    return output, token_count
                else:
                    print(f"  [!] Output too long ({token_count} tokens). Retrying...")
            else:
                print("  [!] Output length invalid. Retrying...")
                
        except requests.exceptions.RequestException as e:
            print(f"  [!] Request Error: {e}")
            time.sleep(2)
            
    return None, 0

def process_dataset():
    global_token_counts = []
    global_total_attempts = 0
    
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".txt"):
            base_name = filename.replace(".txt", "")
            species_name = base_name.replace("_", " ").title()
            input_path = os.path.join(INPUT_DIR, filename)

            with open(input_path, 'r', encoding='utf-8') as file:
                original_text = file.read()

            print(f"\n{'='*40}\nProcessing Test Set: {species_name}\n{'='*40}")
            
            successful_generations = 0
            total_attempts = 0
            species_tokens = []
            failed_cases = []
            
            for index, test_case in enumerate(TEST_CASES):
                case_name = test_case['name']
                attempts_for_case = 0
                success = False
                
                while not success and attempts_for_case < 3:
                    total_attempts += 1
                    global_total_attempts += 1
                    attempts_for_case += 1
                    
                    print(f"  Attempt {attempts_for_case} | Generating test case {index + 1}/5 ({case_name})...")
                    
                    # Pass base_name into the generator to fetch aliases
                    variation_text, token_count = generate_test_variation(species_name, base_name, original_text, test_case)

                    if variation_text:
                        species_tokens.append(token_count)
                        global_token_counts.append(token_count)
                        
                        new_filename = f"{base_name}_test_{index + 1}_{case_name}.txt"
                        output_path = os.path.join(OUTPUT_DIR, new_filename)
                        
                        with open(output_path, 'w', encoding='utf-8') as out_file:
                            out_file.write(variation_text)
                            
                        successful_generations += 1
                        success = True
                
                if not success:
                    failed_cases.append(case_name)

            # Write Per-Species Report
            report_chunk = f"--- {species_name} (TEST DATA) ---\n"
            report_chunk += f"Count: {successful_generations}/5 test cases generated in {total_attempts} total API calls\n"
            
            if failed_cases:
                report_chunk += f"FAILED CASES: {', '.join(failed_cases)}\n"
            
            if species_tokens:
                report_chunk += f"Average tokens: {int(np.mean(species_tokens))}\n"
                report_chunk += f"Min tokens: {np.min(species_tokens)}\n"
                report_chunk += f"Max tokens: {np.max(species_tokens)}\n"
            
            report_chunk += "\n"
            with open(QA_REPORT_PATH, 'a', encoding='utf-8') as f:
                f.write(report_chunk)
                
    # Write Global QA Report
    if global_token_counts:
        global_chunk = f"{'='*40}\nTEST DATASET QA REPORT\n{'='*40}\n"
        global_chunk += f"Total API calls: {global_total_attempts}\n"
        global_chunk += f"Total test files generated: {len(global_token_counts)}\n"
        global_chunk += f"Average tokens: {int(np.mean(global_token_counts))}\n"
        global_chunk += f"Minimum tokens: {np.min(global_token_counts)}\n"
        global_chunk += f"Maximum tokens: {np.max(global_token_counts)}\n"
        
        with open(QA_REPORT_PATH, 'a', encoding='utf-8') as f:
            f.write(global_chunk)
            
        print(f"\n[SUCCESS] Full Test QA report written to {QA_REPORT_PATH}")

if __name__ == "__main__":
    process_dataset()