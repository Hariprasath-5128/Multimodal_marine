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
| **Total samples** | 1909 images |
| **Number of classes** | 71 species |
| **Training samples** | 1518 images |
| **Testing samples** | 391 images |
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
The dataset previously had a very small training set compared to the test set, but it has since been expanded. Currently, there are 1518 training images across 10 classes, and 391 testing images spread across 71 classes. While the overall size is better, there is still a significant class imbalance since many species in the test set do not have corresponding training data, and the classes that do have training data are unevenly distributed. Models will likely still need strong augmentation or pre-trained backbones.
