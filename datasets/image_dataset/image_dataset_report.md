# Marine Dataset Report: Image Classification

## 1. Dataset Overview
**Dataset Name:** Marine Image Dataset  
**Version Number:** 3.0 (Train/Test Split Structure)  
**Date Created:** June 2026  
**Purpose of the dataset:** Designed for training and testing image classification models on marine species.  
**Target Application:** Marine life identification, automated environmental monitoring, and educational tools.

## 2. Dataset Statistics
| Metric | Value |
|--------|-------|
| **Total samples** | 460 images |
| **Number of classes** | 71 species |
| **Training samples** | 98 images |
| **Testing samples** | 362 images |
| **Image Resolution** | Mostly 224x224 to 288x288 RGB |

*Note: The dataset has been explicitly split into `train` and `test` directories by species.*

## 3. Data Collection Process
**Source of data:** Primarily sourced from the HuggingFace dataset `yeyimilk/LLM-Vision-Marine-Animals` and manual backend dataset manipulation.  
**Collection method:** Automated scraping and aggregation of marine animal images, followed by manual categorization into species-level folders.  

## 4. Dataset Structure
The dataset follows a direct species-level directory structure split by training and testing subsets.

```text
datasets/
├── image_dataset/
│   ├── train/
│   │   ├── amazon_river_dolphin/
│   │   ├── bottlenose_dolphin/
│   │   └── ...
│   ├── test/
│   │   ├── amazon_river_dolphin/
│   │   ├── bottlenose_dolphin/
│   │   └── ...
│   └── shared_label_ids_for_image_only.json
```

## 5. Class Distribution
The `shared_label_ids_for_image_only.json` file has been updated to include `"train"`, `"test"`, and `"total"` counts per species, acting as the definitive ground truth for distribution. 

### Imbalance Discussion
The dataset is highly imbalanced in its current state (Train: 98, Test: 362). Because the training set is very small compared to the test set, models trained on this dataset will rely heavily on pre-trained backbones (like ConvNeXt or ViT) and strong data augmentation to generalize well without overfitting.
