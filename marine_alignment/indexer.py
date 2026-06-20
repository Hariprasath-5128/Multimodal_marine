"""
indexer.py — Metadata-Coupled FAISS Vector Database
=====================================================
Wraps a faiss.IndexFlatIP index with a Python metadata registry so
every indexed vector is linked to its species name, modality, and
source file.

IndexFlatIP is chosen because:
  • All projected embeddings lie on the unit hyper-sphere.
  • inner_product(u, v) == cosine_similarity(u, v) when ||u||=||v||=1.
  • Exact (non-approximate) retrieval is fast enough for the expected
    corpus size (≤ 50 k vectors).

Persistence:
  save() serialises the FAISS index (via faiss.write_index) and the
  metadata registry (via pickle) together in a single .faissdb file
  (which is a ZIP archive internally).
  load() reverses this.

Usage
-----
    from indexer import MarineVectorDB
    from models  import MarineImageBindPipeline

    db       = MarineVectorDB()
    pipeline = MarineImageBindPipeline()
    pipeline.load_state_dict(torch.load("trained_projection_heads/best_multimodal_pipeline.pth")["model_state"])
    pipeline.eval()

    # ── Index all val embeddings ──────────────────────────────────────────
    with torch.no_grad():
        for batch in val_loader:
            proj = pipeline.image_head(batch["image_emb"])
            for i in range(proj.size(0)):
                db.add_record(proj[i], batch["species_id"][i].item(),
                              modality="image", file_name=batch["file_name"][i])

    db.save("marine_index.faissdb")

    # ── Query ──────────────────────────────────────────────────────────────
    results = db.search_query(query_tensor, k=5)
    for r in results:
        print(r["rank"], r["species"], r["confidence_score"])
"""

import os
import io
import pickle
import zipfile
import numpy as np
import torch

try:
    import faiss
except ImportError:
    raise ImportError(
        "faiss-cpu is required: pip install faiss-cpu\n"
        "For GPU support use:   pip install faiss-gpu"
    )

from config import SHARED_DIM
from dataset import get_idx_to_species


class MarineVectorDB:
    """
    Metadata-coupled FAISS inner-product index for marine species retrieval.

    Parameters
    ----------
    dimension : int   — embedding dimension (must match projection head output)
    """

    def __init__(self, dimension: int = SHARED_DIM):
        # IndexFlatIP = exact inner-product search (== cosine on unit sphere)
        self.index            = faiss.IndexFlatIP(dimension)
        self.dimension        = dimension
        # int registry_id → {"species_name", "modality", "source_file"}
        self.metadata_registry: dict[int, dict] = {}
        self.current_id       = 0

    # ── Indexing ──────────────────────────────────────────────────────────────

    def add_record(
        self,
        projected_tensor: torch.Tensor,
        species_id:       int,
        modality:         str,
        file_name:        str,
        species_name:     str | None = None,
    ) -> int:
        """
        Add one 512-D projected embedding to the FAISS index.

        Parameters
        ----------
        projected_tensor : FloatTensor [SHARED_DIM] or [1, SHARED_DIM]
                           — must be L2-normalised (unit sphere)
        species_id       : integer species index
        modality         : "image" | "text" | "audio"
        file_name        : source .pt filename for provenance
        species_name     : optional str override — if provided, used directly
                           instead of resolving from the dataset registry.
                           Useful for standalone / testing contexts.

        Returns
        -------
        registry_id : int  — position assigned in the index
        """
        # Convert to contiguous float32 numpy row vector
        vec = projected_tensor.detach().cpu().float()
        if vec.dim() == 1:
            vec = vec.unsqueeze(0)                   # [1, D]
        vec_np = vec.numpy().astype(np.float32)

        if vec_np.shape[1] != self.dimension:
            raise ValueError(
                f"Expected vector dimension {self.dimension}, "
                f"got {vec_np.shape[1]}."
            )

        # Sanity-check: warn if not approximately unit-norm
        norm = float(np.linalg.norm(vec_np))
        if abs(norm - 1.0) > 0.05:
            import warnings
            warnings.warn(
                f"Vector norm is {norm:.4f} (expected ~1.0). "
                "Embeddings should be L2-normalised before indexing.",
                stacklevel=2,
            )

        self.index.add(vec_np)

        # Resolve species name — prefer explicit string, else use registry
        if species_name is None:
            try:
                idx_to_species = get_idx_to_species()
                species_name   = idx_to_species.get(int(species_id), f"unknown_{species_id}")
            except FileNotFoundError:
                species_name = f"species_{species_id}"

        self.metadata_registry[self.current_id] = {
            "species_name": species_name,
            "modality":     modality,
            "source_file":  file_name,
        }
        assigned_id     = self.current_id
        self.current_id += 1
        return assigned_id

    def add_batch(
        self,
        projected_tensors: torch.Tensor,
        species_ids:       torch.Tensor,
        modality:          str,
        file_names:        list[str],
    ) -> list[int]:
        """
        Vectorised batch add.  Returns list of assigned registry IDs.
        """
        ids = []
        for i in range(projected_tensors.size(0)):
            rid = self.add_record(
                projected_tensors[i],
                int(species_ids[i].item()),
                modality,
                file_names[i],
            )
            ids.append(rid)
        return ids

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def search_query(
        self,
        query_tensor: torch.Tensor,
        k:            int = 5,
    ) -> list[dict]:
        """
        Retrieve the k nearest neighbours for a query embedding.

        Parameters
        ----------
        query_tensor : FloatTensor [SHARED_DIM] or [1, SHARED_DIM]
                       — L2-normalised projected embedding
        k            : number of top results to return

        Returns
        -------
        list of dicts, one per result, sorted by descending cosine score:
            {
                "rank"            : 1-indexed rank,
                "species"         : str,
                "modality"        : str,
                "file"            : str,
                "confidence_score": float  (inner product ≈ cosine similarity)
            }
        """
        if self.index.ntotal == 0:
            return []

        query_np = (
            query_tensor.detach().cpu().float().numpy()
            .astype(np.float32)
            .reshape(1, -1)
        )
        effective_k   = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_np, effective_k)

        results = []
        for rank, (idx, dist) in enumerate(
            zip(indices[0], distances[0]), start=1
        ):
            if idx < 0:          # FAISS returns -1 for padding
                continue
            meta = self.metadata_registry.get(int(idx), {})
            results.append(
                {
                    "rank":             rank,
                    "species":          meta.get("species_name", "unknown"),
                    "modality":         meta.get("modality",     "unknown"),
                    "file":             meta.get("source_file",  "unknown"),
                    "confidence_score": float(dist),
                }
            )
        return results

    def search_batch(
        self,
        query_tensors: torch.Tensor,
        k:             int = 5,
    ) -> list[list[dict]]:
        """Batch version of search_query.  Returns one result list per query."""
        return [self.search_query(query_tensors[i], k=k)
                for i in range(query_tensors.size(0))]

    # ── Statistics ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.index.ntotal

    def stats(self) -> dict:
        """Return summary statistics about the indexed corpus."""
        from collections import Counter
        modalities = [m["modality"]     for m in self.metadata_registry.values()]
        species    = [m["species_name"] for m in self.metadata_registry.values()]
        return {
            "total_vectors":    self.index.ntotal,
            "dimension":        self.dimension,
            "modality_counts":  dict(Counter(modalities)),
            "num_species":      len(set(species)),
            "species_list":     sorted(set(species)),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Serialise the FAISS index + metadata registry to a single file.
        Uses ZIP format: index stored as 'faiss.index', metadata as 'meta.pkl'.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Serialise FAISS index to bytes
        buf = io.BytesIO()
        faiss.write_index(self.index, faiss.PyCallbackIOWriter(buf.write))
        index_bytes = buf.getvalue()

        # Serialise metadata
        meta_bytes = pickle.dumps(
            {
                "metadata_registry": self.metadata_registry,
                "current_id":        self.current_id,
                "dimension":         self.dimension,
            }
        )

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("faiss.index", index_bytes)
            zf.writestr("meta.pkl",    meta_bytes)

        print(f"  MarineVectorDB saved -> {path}  ({self.index.ntotal} vectors)")

    @classmethod
    def load(cls, path: str) -> "MarineVectorDB":
        """
        Load a previously saved MarineVectorDB from disk.

        Parameters
        ----------
        path : str — path written by save()

        Returns
        -------
        MarineVectorDB instance with index and metadata restored.
        """
        with zipfile.ZipFile(path, "r") as zf:
            index_bytes = zf.read("faiss.index")
            meta_bytes  = zf.read("meta.pkl")

        # Restore FAISS index
        buf   = io.BytesIO(index_bytes)
        index = faiss.read_index(faiss.PyCallbackIOReader(buf.read))

        # Restore metadata
        meta_dict = pickle.loads(meta_bytes)

        db = cls(dimension=meta_dict["dimension"])
        db.index             = index
        db.metadata_registry = meta_dict["metadata_registry"]
        db.current_id        = meta_dict["current_id"]

        print(f"  MarineVectorDB loaded <- {path}  ({db.index.ntotal} vectors)")
        return db


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, torch

    FAKE_SPECIES = ["humpback_whale", "spinner_dolphin", "harp_seal"]

    db = MarineVectorDB(dimension=512)
    print("Empty DB:", len(db), "vectors")

    # Add 10 random unit-sphere vectors for 3 fake species
    # Pass species_name directly to bypass the dataset registry
    for i in range(10):
        v = torch.randn(512)
        v = v / v.norm()
        db.add_record(
            projected_tensor=v,
            species_id=i % 3,
            modality="image",
            file_name=f"sample_{i:03d}.pt",
            species_name=FAKE_SPECIES[i % 3],
        )

    print("After adds:", len(db), "vectors")
    print("Stats:", db.stats())

    # Query
    q = torch.randn(512); q = q / q.norm()
    results = db.search_query(q, k=3)
    print("\nTop-3 results:")
    for r in results:
        print(f"  {r['rank']}. {r['species']:30s}  score={r['confidence_score']:.4f}")

    # Save / Load round-trip
    with tempfile.NamedTemporaryFile(suffix=".faissdb", delete=False) as f:
        tmp_path = f.name
    db.save(tmp_path)
    db2 = MarineVectorDB.load(tmp_path)
    assert len(db2) == len(db), "Round-trip vector count mismatch"
    print("\nSave/load round-trip [OK]")
    os.unlink(tmp_path)
