# Marine Dataset Report: Audio Classification

## 1. Dataset Overview
**Dataset Name:** Marine Audio Dataset  
**Version Number:** 1.0  
**Date Created:** June 2026  
**Purpose of the dataset:** Designed for training acoustic classification and semantic models to identify marine mammal vocalizations (clicks, whistles, songs).  
**Target Application:** Underwater bioacoustics monitoring, passive acoustic monitoring (PAM) systems, and multi-modal marine research.

## 2. Dataset Statistics
| Metric | Value |
|--------|-------|
| **Total samples** | 1,697 audio files |
| **Training samples** | 1,348 |
| **Validation samples** | 349 |
| **Number of classes** | 32 species |
| **Split Ratio** | ~80% Train / 20% Val |

## 3. Data Collection Process
**Source of data:** Derived from multi-modal marine repositories and open-source bioacoustic databases (e.g., Watkins Marine Mammal Sound Database, HuggingFace repositories).  
**Collection method:** Automated audio extraction, segmentation into uniform lengths, and normalization.  
**Manual validation:** Audio clips were likely vetted against verified species calls to ensure label accuracy.

## 4. Dataset Structure
The audio dataset is split explicitly into training and validation directories on the filesystem, bypassing the domain hierarchy in favor of direct species classification.

```text
datasets/
├── audio_dataset/
│   ├── audio_split/
│   │   ├── train/
│   │   │   ├── bottlenose_dolphin/
│   │   │   ├── humpback_whale/
│   │   │   └── killer_whale/
│   │   └── val/
│   │       ├── bottlenose_dolphin/
│   │       └── ...
```

## 5. Class Distribution
The dataset encompasses 32 marine species. Below is the total sample count across the entire dataset (Train + Val).

| Class | Samples |
|-------|---------|
| Spinner Dolphin | 114 |
| Frasers Dolphin | 87 |
| Striped Dolphin | 81 |
| Sperm Whale | 75 |
| Long-finned Pilot Whale | 70 |
| Grampus (Rissos) Dolphin | 67 |
| Short-finned Pilot Whale | 67 |
| Pantropical Spotted Dolphin | 66 |
| Humpback Whale | 64 |
| Clymene Dolphin | 63 |
| Melon-headed Whale | 63 |
| Bowhead Whale | 60 |
| False Killer Whale | 59 |
| Atlantic Spotted Dolphin | 58 |
| White-beaked Dolphin | 57 |
| White-sided Dolphin | 55 |
| Northern Right Whale | 54 |
| Common Dolphin | 52 |
| Narwhal | 50 |
| Fin Whale | 50 |
| Beluga Whale | 50 |
| Ross Seal | 50 |
| Rough-toothed Dolphin | 50 |
| Harp Seal | 47 |
| Walrus | 38 |
| Bearded Seal | 37 |
| Killer Whale (Orca) | 35 |
| Southern Right Whale | 25 |
| Bottlenose Dolphin | 24 |
| Minke Whale | 17 |
| Leopard Seal | 10 |
| Weddell Seal | 2 |

### Imbalance Discussion
The dataset shows **severe class imbalance**. While the most highly-represented class (Spinner Dolphin) has 114 samples, the least represented class (Weddell Seal) has only 2 samples, and Leopard Seals have just 10. Training models on this dataset requires heavy augmentation (e.g., time stretching, pitch shifting, background noise injection) and strong weighted loss functions to ensure the model does not become heavily biased towards Dolphin and Whale calls over Seal vocalizations.
