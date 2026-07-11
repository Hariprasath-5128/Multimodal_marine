# Marine Multimodal Alignment Report

## Overview
The `marine_alignment` module is responsible for binding three distinct data modalities—**Images**, **Text (Species Descriptions)**, and **Audio (Vocalizations)**—into a single, unified embedding space. By aligning these modalities, the system can perform cross-modal retrieval, such as identifying a marine species from a photograph using a textual query, or retrieving an image of an animal based on its recorded sound.

This report summarizes the workflow, methods, and evaluation results based on our **final architecture iteration**, where the model was trained using **Joint Multimodal Contrastive Alignment** (end-to-end) rather than the previous two-phase frozen approach.

---

## 1. Workflow

1. **Feature Extraction**
   Raw images, text, and audio files are pre-processed and fed into modality-specific backbone networks to extract high-dimensional dense embeddings.
   - **Images**: ConvNeXt-Base
   - **Text**: DistilRoBERTa
   - **Audio**: AST (Audio Spectrogram Transformer)

2. **Projection into a Joint Space**
   The extracted backbone embeddings are passed through modality-specific **Projection Heads** (Multi-Layer Perceptrons). These heads learn to linearly and non-linearly transform the raw embeddings into a shared, normalized 512-dimensional vector space.

3. **Joint End-to-End Training (SupCon)**
   In the final iteration, the system uses a **Symmetric Supervised Contrastive Loss (SupCon)** to train the `image_head`, `text_head`, and `audio_head` jointly.
   - By training all three heads simultaneously, the projection heads co-adapt to structure a single shared latent space without suffering from the phase-transition misalignment that plagued earlier two-phase training.
   - Image ↔ Text and Image ↔ Audio relationships are aligned directly and concurrently.

4. **Test-Set Evaluation**
   The trained pipeline is evaluated on a completely unseen test split containing images, text, and audio. The metric used is **Recall@K** (R@1, R@5, R@10), which measures the percentage of queries where the correct cross-modal match is found within the top-K retrieved results.

---

## 2. Methods (Joint Alignment Architecture)

In this final implementation, we employed an **End-to-End Joint Alignment** strategy:
- We transitioned away from the strict "Frozen Backbone" two-phase curriculum.
- All three modalities (Image, Text, Audio) are projected into the 512-dimensional shared space concurrently using a unified training loop.
- The `DoubleNormProjectionHead` networks are optimized to pull samples of the same species together (regardless of modality) while pushing different species apart in the latent space.
- **Advantages:** Eliminates gradient tug-of-war between phases, creates a much tighter 3-way alignment (especially improving Audio ↔ Image direct relationships), and achieves superior performance metrics across the board.

---

## 3. Results (Final Model)

The final model was evaluated against the unseen Test split (391 image queries across 71 species). The cross-modal retrieval performance is as follows:

```text
  Image -> Text Retrieval:
    R@1  : 0.7059  (70.6%)
    R@5  : 0.8338  (83.4%)
    R@10 : 0.8696  (87.0%)
    Composite : 0.7606

  Image -> Audio Retrieval:
    R@1  : 0.7222  (72.2%)
    R@5  : 0.8472  (84.7%)
    R@10 : 0.9236  (92.4%)
    Composite : 0.7799

  Overall System Composite: 0.7702
```

### Conclusion
The **Joint Multimodal Contrastive Alignment** approach successfully clustered the disparate modalities into a shared space. By shifting to joint training, we achieved a highly balanced system, specifically seeing a massive leap in **Image-to-Audio retrieval** which reached **72.2% Top-1 Accuracy** and an exceptional **92.4% Top-10 Accuracy**. The unified space securely binds visual phenotypes, semantic text descriptions, and acoustic vocalizations into a single, highly performant zero-shot retrieval engine.
