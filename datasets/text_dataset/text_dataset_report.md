# Marine Dataset Report: Text Descriptions

## 1. Dataset Overview
**Dataset Name:** Marine Text Dataset  
**Version Number:** 1.0  
**Date Created:** June 2026  
**Purpose of the dataset:** Text-based encyclopedia and contextual descriptions used for Large Language Model (LLM) fine-tuning, RAG (Retrieval-Augmented Generation), and multi-modal text-image pairing.  
**Target Application:** Semantic search, automated species querying, and descriptive marine biology applications.

## 2. Dataset Statistics
| Metric | Value |
|--------|-------|
| **Total samples** | 1,586 text files |
| **Training samples** | 1,141 |
| **Test samples** | 445 |
| **Number of classes** | 75 distinct species |
| **Average text length** | 141.9 words |
| **Min text length** | 7 words |
| **Max text length** | 347 words |
| **Vocabulary size** | 9,197 unique words |

## 3. Data Collection Process
**Source of data:** Text was aggregated from multiple informative domains including Wikipedia articles, scientific papers, museum exhibit descriptions, field guides, and documentary narrations.  
**Collection method:** Automated querying and parsing of knowledge bases. Files are explicitly named according to their source (e.g., `amazon_river_dolphin_1_WikipediaArticle.txt`).  
**Quality Assurance:** The dataset contains a `_Dataset_QA_Report.txt` detailing the formatting validation and consistency checks performed on the texts.

## 4. Dataset Structure
The text files are stored natively as `.txt` files categorized into training and testing directories. The filenames themselves encode the species label, the index, and the source of the text.

```text
datasets/
├── text_dataset/
│   ├── train/
│   │   ├── train_dataset/
│   │   │   ├── amazon_river_dolphin.txt
│   │   │   └── ...
│   │   └── expanded_train_dataset/
│   │       ├── amazon_river_dolphin_1_WikipediaArticle.txt
│   │       ├── amazon_river_dolphin_2_MuseumExhibit.txt
│   │       └── ...
│   └── test/
│       ├── amazon_river_dolphin_test_1.txt
│       └── ...
```

## 5. Class Distribution
The dataset contains 75 unique species, matching the Image Classification dataset schema.

### Sample Distribution (Per Species)
The text dataset is highly balanced by design. During the expansion and aggregation process, a consistent number of documents was generated for almost all species.

| Class | Train Samples | Test Samples | Total |
|-------|---------------|--------------|-------|
| **Average Species** (e.g., Bottlenose Dolphin, Orca, Harp Seal, etc.) | ~16 | 5 | ~21 |
| **Minority Species** (e.g., Leopard Seal, Indo-Pacific Finless Porpoise) | ~9 - 10 | 4 - 5 | ~14 |

### Imbalance Discussion
Unlike the audio and image datasets, the text dataset exhibits **excellent class balance**. Because text can be synthetically expanded or sourced easily from the internet (via varying sources like blogs, scientific papers, and field guides), almost every species has precisely 16 training documents and 5 testing documents. This ensures unbiased training for Language Models reading these texts.
