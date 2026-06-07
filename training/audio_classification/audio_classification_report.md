# Marine Audio Classification Model Report

## 1. Model Overview
**Model Name:** Marine Audio Classifier V1 (AST-Optimized)  
**Task:** Multi-class audio/spectrogram classification  
**Number of Classes:** 32 marine species  
**Framework:** PyTorch (HuggingFace Transformers)  
**Base Model:** `MIT/ast-finetuned-audioset-10-10-0.4593` (Audio Spectrogram Transformer)

## 2. Problem Statement
**Input:** A raw underwater audio recording (`.wav`) of varying length containing marine mammal vocalizations (clicks, whistles, songs).  
**Output:** The predicted species responsible for the vocalization.  
**Objective:** Resample audio to 16kHz, extract spectrogram features, and classify the acoustic signature into one of 32 species categories using a transformer architecture.

## 3. Dataset Summary
| Metric | Value |
|--------|-------|
| **Total Samples** | 1,697 |
| **Training** | 1,348 |
| **Validation** | 349 |
| **Classes** | 32 |

**Class Distribution:** Severe class imbalance (e.g., *Spinner Dolphin*: 114 samples vs. *Weddell Seal*: 2 samples).  
**Dataset Source:** Aggregated from open-source marine bioacoustic databases.

## 4. Model Architecture
The model uses an Audio Spectrogram Transformer (AST) that processes audio as a sequence of spectrogram patches.

```text
Input Audio (.wav)
    ↓
Resampler (16kHz) & Padding/Trimming (10 seconds)
    ↓
AST Feature Extractor (Mel-spectrogram patches)
    ↓
Audio Spectrogram Transformer (12 layers, 768 hidden dimension)
    ↓
Mean Pooling
    ↓
Dense Classifier Layer
    ↓
Predicted Species Class (32 units)
```

## 5. Training Configuration
| Parameter | Value |
|-----------|-------|
| **Epochs** | 40 |
| **Batch Size** | 4 (Effective batch size: 8 via accumulation) |
| **Learning Rate** | 1e-5 (Cosine Warmup Schedule) |
| **Optimizer** | AdamW |
| **Loss Function** | Focal Loss (Gamma = 2) |
| **Hardware** | Optimized for RTX 2050 (4GB VRAM) via Mixed Precision |

## 6. Evaluation Metrics
*Evaluated on the 349-file validation split.*

| Metric | Value |
|--------|-------|
| **Accuracy** | 91.00% |
| **Precision (Macro)** | 90.00% |
| **Recall (Macro)** | 89.00% |
| **F1 Score (Macro)** | 89.00% |

## 7. Class-wise Performance
Highlights from the classification report, demonstrating the model's robustness despite dataset imbalance.

| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| **Bottlenose Dolphin** | 1.00 | 1.00 | 1.00 |
| **Fin Whale** | 1.00 | 1.00 | 1.00 |
| **Sperm Whale** | 0.80 | 0.80 | 0.80 |
| **Clymene Dolphin** | 0.73 | 0.85 | 0.79 |
| **Weddell Seal** | 0.00 | 0.00 | 0.00 |

*Note: The Weddell Seal failed completely due to having only 1 validation sample and 1 training sample, providing insufficient data for the transformer to generalize.*

## 8. Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

**Observations:** 
- The model performs exceptionally well on distinct whale songs and clicks.
- *Clymene Dolphins* are occasionally confused with *Common Dolphins* due to similar whistle frequencies.
- Classes with < 5 samples suffer significantly.

## 9. Training Curves
The Focal Loss allowed the model to converge effectively despite the imbalance. Validation accuracy peaked around Epoch 35.

![Training Curves](training_curves.png)

## 10. Model Efficiency
| Metric | Value |
|--------|-------|
| **Model Size** | ~330 MB |
| **Parameter Count** | ~85 Million |
| **Avg Inference Time** | 45 ms (GPU) |
| **Throughput** | ~22 audio clips/sec |
| **Memory Usage** | 1.4 GB (VRAM) |
