# Marine Multimodal Alignment Report

## Overview
The `marine_alignment` module is responsible for binding three distinct data modalities—**Images**, **Text (Species Descriptions)**, and **Audio (Vocalizations)**—into a single, unified embedding space. By aligning these modalities, the system can perform cross-modal retrieval, such as identifying a marine species from a photograph using a textual query, or retrieving an image of an animal based on its recorded sound.

This report summarizes the workflow, methods, and evaluation results based on our **first architecture iteration**, where the deep neural network backbones were kept frozen and only lightweight projection heads were trained.

---

## 1. Workflow

1. **Feature Extraction**
   Raw images, text, and audio files are pre-processed and fed into modality-specific backbone networks to extract high-dimensional dense embeddings.
   - **Images**: ConvNeXt-Base
   - **Text**: DistilRoBERTa
   - **Audio**: AST (Audio Spectrogram Transformer)

2. **Projection into a Joint Space**
   The extracted backbone embeddings are passed through modality-specific **Projection Heads** (Multi-Layer Perceptrons). These heads learn to linearly and non-linearly transform the raw embeddings into a shared, normalized 512-dimensional vector space.

3. **Two-Phase Training**
   Because the three modalities have highly misaligned initial distributions, a single-phase end-to-end training suffers from gradient tug-of-war. We solve this using a decoupled two-phase curriculum:
   - **Phase 1 (Image ↔ Text)**: The Image and Text projection heads are trained using a Supervised Contrastive Loss algorithm, treating the Image projection as the anchor and pulling the correct Text projection closer while pushing away non-matching species.
   - **Phase 2 (Audio ↔ Text Bridge)**: The Image and Text heads are frozen. The Audio projection head is trained using a simple Cosine Similarity Loss to match the (now frozen and stable) Text projections of the same species. Since Image ↔ Text are already aligned, aligning Audio ↔ Text inherently aligns Audio ↔ Image.

4. **Test-Set Evaluation**
   The trained pipeline is evaluated on a completely unseen test split containing images, text, and audio. The metric used is **Recall@K** (R@1, R@5, R@10), which measures the percentage of queries where the correct cross-modal match is found within the top-K retrieved results.

---

## 2. Methods (Frozen Backbone Architecture)

In this specific implementation (the first backup model), we employed a **Frozen Backbone** strategy:
- The heavy weights of the ConvNeXt, DistilRoBERTa, and AST encoders were strictly frozen and not updated during training.
- Only the lightweight `DoubleNormProjectionHead` networks (consisting of two Linear layers with GELU activation and LayerNorm) were updated.
- **Advantages:** Highly memory-efficient, fast to train, and avoids catastrophic forgetting of the general-purpose knowledge stored in the pre-trained backbones.
- **Limitations:** The projection heads can only linearly shift the embeddings; if two species were inextricably tangled in the original ConvNeXt space, the projection head lacks the capacity to perfectly untangle them without the backbone's help.

---

## 3. Results

The model was evaluated against the unseen Test split (391 image queries). The cross-modal retrieval performance of the frozen-backbone projection heads is as follows:

```text
  Image -> Text Retrieval:
    R@1  : 0.7161  (71.6%)
    R@5  : 0.8338  (83.4%)
    R@10 : 0.8926  (89.3%)
    Composite : 0.7691

  Image -> Audio Retrieval:
    R@1  : 0.6736  (67.4%)
    R@5  : 0.7778  (77.8%)
    R@10 : 0.8403  (84.0%)
    Composite : 0.7215
```

### Conclusion
The frozen-backbone approach successfully clustered the disparate modalities into a shared space, achieving **~71.6% Top-1 accuracy** for Image-to-Text retrieval and **~67.4% Top-1 accuracy** for Image-to-Audio retrieval. While highly effective as a baseline, reaching higher performance (e.g., >80% R@1) requires unlocking the backbones (like ConvNeXt) so the model can learn domain-specific marine visual features directly.
