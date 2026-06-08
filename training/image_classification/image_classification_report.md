# Marine Image Classification Model Report

## 1. Model Overview
**Model Name:** Hierarchical Marine Image Classifier V1  
**Task:** Multi-class image classification (Domain & Species)  
**Number of Classes:** 71 species across 6 domains  
**Framework:** PyTorch (timm)  
**Base Model:** `convnext_small` (Domain) + `convnextv2_base` (Species)

## 2. Problem Statement
**Input:** An image of a marine animal (`.jpg`, `.png`).  
**Output:** The predicted parent domain (e.g., *whale*) and the top-3 predicted species (e.g., *Orca, False Killer Whale, Humpback*).  
**Objective:** Accurately classify images using a 2-stage pipeline. Stage 1 identifies the domain, routing the image to an expert Stage 2 species classifier.

## 3. Dataset Summary
| Metric | Value |
|--------|-------|
| **Total Samples** | 1909 images |
| **Training** | 1518 images |
| **Test** | 391 images |
| **Classes** | 71 |

**Class Distribution:** Severe class imbalance, handled via square-root Weighted Random Sampling.  
**Dataset Source:** Subset of the `yeyimilk/LLM-Vision-Marine-Animals` HuggingFace dataset.

## 4. Model Architecture
The architecture utilizes a two-stage hierarchical routing system.

```text
Input Image (224x224 RGB)
    ↓
[Stage-1 Model] ConvNeXt Small (12k pre-trained)
    ↓
Predicted Domain Bucket (dolphin, whale, seal, sealion, porpoise, manatee)
    ↓
[Stage-2 Model] ConvNeXt V2 Base (Specific to predicted domain)
    ↓
5-View Test-Time Augmentation (TTA)
    ↓
Softmax / Logits
    ↓
Top-3 Predicted Species
```

## 5. Training Configuration
| Parameter | Value |
|-----------|-------|
| **Epochs** | 20 |
| **Batch Size** | 16 |
| **Learning Rate** | 3e-5 (Cosine Annealing) |
| **Optimizer** | AdamW |
| **Loss Function** | CrossEntropyLoss (with Label Smoothing) |
| **Augmentations** | RandomResizedCrop, RandAugment, RandomErasing |

## 6. Evaluation Metrics
*Evaluated on the 362-image test split using Top-1, Top-2 (0.35), and Top-3 (0.15) weighting.*

| Metric | Value |
|--------|-------|
| **Accuracy (Top-1)** | 84.0% |
| **Precision (Macro)** | 84.0% |
| **Recall (Macro)** | 78.0% |
| **F1 Score (Macro)** | 80.0% |
| **Overall Top-3 Weighted Acc** | 86.8% [PASS] |

### Domain-Level Accuracy
The hierarchical model's performance on the 6 domain buckets:

| Domain | N | Top-1 | Top-2 | Top-3 | Miss | DomErr | Coverage | WgtAcc | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| dolphin | 138 | 121 | 7 | 1 | 9 | 1 | 93.5% | **89.6%** | [PASS] |
| whale | 81 | 70 | 4 | 1 | 6 | 3 | 92.6% | **88.3%** | [PASS] |
| seal | 73 | 60 | 5 | 2 | 6 | 1 | 91.8% | **85.0%** | [PASS] |
| sealion | 29 | 22 | 3 | 1 | 3 | 2 | 89.7% | **80.0%** | [PASS] |
| porpoise | 17 | 13 | 1 | 0 | 3 | 3 | 82.4% | **78.5%** | [PASS] |
| manatee | 17 | 13 | 4 | 0 | 0 | 0 | 100.0% | **84.7%** | [PASS] |
| **TOTAL** | 355 | | | | | | | **86.8%** | [PASS] |

## 7. Class-wise Performance
*Top-performing and under-performing species examples:*

| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| **Killer Whale (Orca)** | 0.95 | 0.92 | 0.93 |
| **Humpback Whale** | 0.88 | 0.90 | 0.89 |
| **Bottlenose Dolphin** | 0.85 | 0.83 | 0.84 |
| **Amazon River Dolphin** | 0.60 | 0.50 | 0.55 |

*Note: Classes like the Amazon River Dolphin suffer due to low training examples (fewer than 2 in some cases).*

## 8. Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

**Observations:** 
- Stage-1 Domain errors immediately result in a miss (e.g., misclassifying a Porpoise as a Dolphin).
- Misclassification primarily occurs between closely related sub-species within the same expert model (e.g., *Atlantic Spotted Dolphin* vs *Pantropical Spotted Dolphin*).

## 9. Training Curves
The ConvNeXt architectures converged rapidly due to strong pre-training weights, stabilizing around Epoch 15.

![Training Curves](training_curves.png)

## 10. Model Efficiency
| Metric | Value |
|--------|-------|
| **Model Size (Stage 1 + Stage 2)** | ~190 MB + ~340 MB (per domain) |
| **Avg Inference Time** | 120 ms (CPU) / 18 ms (GPU) |
| **Throughput** | ~55 images/sec (TTA enabled) |
| **Memory Usage** | 2.8 GB (VRAM for full pipeline) |
