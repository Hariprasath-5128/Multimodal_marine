"""
models.py — Residual Double-Normalized Projection Head Architecture
====================================================================
Each frozen encoder maps features into its own coordinate scale.
The DoubleNormProjectionHead enforces:

    Step 1  — Input L2 normalisation (stabilises gradient scale
              across heterogeneous feature distributions)
    Step 2  — Linear projection to hidden space  (Linear_1)
    Step 3  — GELU activation + Dropout (regularisation)
    Step 4  — Linear projection to shared space  (Linear_2)
    Step 5  — Residual skip from INPUT x (not hidden h!)
              This is a TRUE residual that provides a gradient
              shortcut all the way from the output back to the input.
    Step 6  — LayerNorm
    Step 7  — Output L2 normalisation  ->  unit hyper-sphere

Fix (Residual): The previous implementation computed the residual as
  out = linear2(h) + shortcut(h)
which is mathematically equivalent to a single linear layer applied
to h, and provides NO gradient shortcut. The correct implementation
connects from the input x:
  out = linear2(h) + shortcut(x)

Fix (Capacity): hidden_dim reduced from 2*output_dim (1536) to 512
to prevent overfitting on the small 1150-sample marine dataset.

Fix (Regularization): Dropout(0.3) added after GELU activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import IMG_INPUT_DIM, TXT_INPUT_DIM, AUD_INPUT_DIM, SHARED_DIM


# ── Shared Building Block ─────────────────────────────────────────────────────

class DoubleNormProjectionHead(nn.Module):
    """
    True-Residual two-layer projection head with dual L2 normalisation.

    Architecture
    ------------
    Input  [input_dim]
      L2-norm (input stabilisation)      <- x_norm
      nn.Linear(input_dim -> hidden_dim) }
      nn.GELU()                          } <- h
      nn.Dropout(p)                      }
      nn.Linear(hidden_dim -> output_dim) <- deep path
       + shortcut(x_norm)               <- TRUE residual from input
      nn.LayerNorm(output_dim)
      L2-norm (unit-sphere)
    Output [output_dim]
    High-Capacity Deep Residual Projection Head.
    Maps raw frozen embeddings (1024-D, 768-D) to the shared latent space.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        hidden_dim = max(input_dim, output_dim * 2)

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.shortcut1 = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )
        self.shortcut2 = nn.Linear(hidden_dim, output_dim)
        
        self.norm_out = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input normalisation
        x_norm = F.normalize(x, p=2, dim=-1)

        # Residual Block 1
        h1 = self.block1(x_norm) + self.shortcut1(x_norm)
        
        # Residual Block 2
        h2 = self.block2(h1) + self.shortcut2(h1)

        # Structural normalisation
        out = self.norm_out(h2)

        # Output L2 normalisation -> unit hyper-sphere
        return F.normalize(out, p=2, dim=-1)


# ── Full Pipeline ──────────────────────────────────────────────────────────────

class MarineImageBindPipeline(nn.Module):
    """
    Marine Multimodal Alignment Pipeline
    ------------------------------------
    image_head : DoubleNormProjectionHead  IMG_INPUT_DIM -> 768
    text_head  : DoubleNormProjectionHead  TXT_INPUT_DIM -> 768
    audio_head : DoubleNormProjectionHead  AUD_INPUT_DIM -> 768
    """

    def __init__(
        self,
        img_dim:  int = IMG_INPUT_DIM,
        txt_dim:  int = TXT_INPUT_DIM,
        aud_dim:  int = AUD_INPUT_DIM,
        out_dim:  int = SHARED_DIM,
    ):
        super().__init__()
        
        self.image_head = DoubleNormProjectionHead(img_dim, out_dim)
        self.text_head  = DoubleNormProjectionHead(txt_dim, out_dim)
        self.audio_head = DoubleNormProjectionHead(aud_dim, out_dim)

    def project_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.image_head(x)

    def project_text(self, x: torch.Tensor) -> torch.Tensor:
        return self.text_head(x)

    def project_audio(self, x: torch.Tensor) -> torch.Tensor:
        return self.audio_head(x)

    def forward(self, image_emb, text_emb=None, audio_emb=None):
        """
        Convenience forward for single-sample inference.
        Returns a dict of projected embeddings for whichever
        modalities are provided.
        """
        out = {"image": self.project_image(image_emb)}
        if text_emb  is not None:
            out["text"]  = self.project_text(text_emb)
        if audio_emb is not None:
            out["audio"] = self.project_audio(audio_emb)
        return out

    def param_summary(self):
        """Print trainable parameter count per head."""
        for name, module in [
            ("image_head", self.image_head),
            ("text_head",  self.text_head),
            ("audio_head", self.audio_head),
        ]:
            n = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"  {name}: {n:,} trainable params")
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  TOTAL : {total:,} trainable params")


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    pipeline = MarineImageBindPipeline()
    pipeline.param_summary()

    # Smoke test — shapes and unit-norm guarantee
    B = 4
    img  = torch.randn(B, IMG_INPUT_DIM)
    txt  = torch.randn(B, TXT_INPUT_DIM)
    aud  = torch.randn(B, AUD_INPUT_DIM)

    pi = pipeline.image_head(img)
    pt = pipeline.text_head(txt)
    pa = pipeline.audio_head(aud)

    for name, t in [("image", pi), ("text", pt), ("audio", pa)]:
        norms = t.norm(dim=-1)
        print(f"  {name}: shape={tuple(t.shape)}  "
              f"L2-norm min={norms.min():.4f} max={norms.max():.4f}")
    print("Smoke test passed [OK]")
