# Marine Text Classification Model Report

## 1. Model Overview
**Model Name:** Marine Text Reasoning Model V4  
**Task:** Semantic Text Classification / Retrieval  
**Number of Classes:** 75  
**Framework:** PyTorch (SentenceTransformers)  
**Base Model:** `all-MiniLM-L6-v2` + MultipleNegativesRankingLoss  

## 2. Problem Statement
**Input:** A text query, paragraph, or encyclopedia excerpt regarding a marine species.  
**Output:** The predicted scientific or common name (out of 75 species).  
**Objective:** Encode the text into a dense semantic vector space and classify it by computing the highest cosine similarity against a database of reference species documents.  

## 3. Dataset Summary
| Metric | Value |
|--------|-------|
| **Total Samples** | 1,586 |
| **Training** | 1,141 |
| **Test** | 445 |
| **Classes** | 75 |

**Dataset Source:** Extracted from Wikipedia, scientific papers, and museum exhibits.  
**Class Distribution:** Highly balanced. Most species possess 16 training documents and 5 test documents.

## 4. Model Architecture
The model leverages a bi-encoder architecture optimized for asymmetric search.

```text
Input Text
    ↓
Tokenizer (WordPiece)
    ↓
Transformer (MiniLM-L6-v2)  <-- 6 Hidden Layers, GELU Activation
    ↓
Mean Pooling Layer
    ↓
Embedding Layer (384 dimensions)
    ↓
Cosine Similarity (against 75 class embeddings)
    ↓
Predicted Species Class
```

## 5. Training Configuration
| Parameter | Value |
|-----------|-------|
| **Epochs** | 10 |
| **Batch Size** | 8 |
| **Learning Rate** | 2e-5 (Cosine Warmup) |
| **Optimizer** | AdamW |
| **Loss Function** | MultipleNegativesRankingLoss |
| **Hardware** | GPU (CUDA) |

## 6. Evaluation Metrics
*Evaluated on the 445-document test set across 6 difficulty types (BehaviorFirst, FeatureOnly, HardNegative, etc.)*

| Metric | Value |
|--------|-------|
| **Accuracy** | 78.65% |
| **Precision (Macro)** | 82.00% |
| **Recall (Macro)** | 79.00% |
| **F1 Score (Macro)** | 79.00% |

## 7. Class-wise Performance
Below is a selection of class-wise performance highlights. The model performs perfectly on highly distinct species but struggles with visually or behaviorally identical sub-species.

| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| **Amazon River Dolphin** | 1.00 | 1.00 | 1.00 |
| **Killer Whale (Orca)** | 0.88 | 0.70 | 0.78 |
| **Bottlenose Dolphin** | 0.80 | 0.67 | 0.73 |
| **Steller Sea Lion** | 0.50 | 0.40 | 0.44 |
| **White-sided Dolphin** | 0.75 | 0.50 | 0.60 |

## 8. Confusion Matrix
The confusion matrix heatmap illustrates where semantic overlaps occur.

![Confusion Matrix](confusion_matrix.png)

**Observations:** 
- *Steller Sea Lions* are frequently confused with *California Sea Lions* due to overlapping geographic and behavioral textual descriptions.
- *HardNegative* queries (where texts explicitly mention "Not an X, but a Y") caused the most misclassifications due to the model's limited logical negation capabilities.

## 9. Training Curves
The loss converged smoothly, with MultipleNegativesRankingLoss stabilizing around Epoch 8.

![Training Curves](training_curves.png)

## 10. Model Efficiency
| Metric | Value |
|--------|-------|
| **Model Size** | 86 MB |
| **Embedding Dimension** | 384 |
| **Avg Inference Time** | 12 ms (CPU) / 2 ms (GPU) |
| **Throughput** | ~800 queries/sec |
| **Memory Usage** | 450 MB (VRAM) |
