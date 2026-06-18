"""
marine_alignment — Multi-modal alignment pipeline for marine species.

Quick start
-----------
    # Step 1: Extract features from frozen encoders
    python feature_extractor.py

    # Step 2: Verify extracted features
    python verify_features.py

    # Step 3: Train projection heads
    python train.py --epochs 15

    # Step 4: Build FAISS index (from your indexing script)
    from indexer import MarineVectorDB
"""
