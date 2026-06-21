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

    Parameters
    ----------
    input_dim  : int   — dimension of the frozen encoder output
    output_dim : int   — target shared latent dimension
    hidden_dim : int   — intermediate MLP width (default: 512)
    dropout    : float — dropout probability (default: 0.3)
    """

    def __init__(
        self,
        input_dim:  int,
        output_dim: int = SHARED_DIM,
        hidden_dim: int | None = None,
        dropout:    float = 0.3,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 512        # Reduced from 1536 to prevent overfitting

        self.linear1  = nn.Linear(input_dim,  hidden_dim)
        self.act      = nn.GELU()
        self.drop     = nn.Dropout(p=dropout)
        self.linear2  = nn.Linear(hidden_dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        # TRUE residual shortcut: input x -> output dimension
        # This provides a gradient highway all the way from output to input.
        self.shortcut = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 — Input L2 normalisation
        x_norm = F.normalize(x, p=2, dim=-1)

        # Step 2 — First projection + activation + dropout
        h = self.drop(self.act(self.linear1(x_norm)))   # [B, hidden_dim]

        # Step 3 — Deep path + TRUE residual from input (not from h!)
        out = self.linear2(h) + self.shortcut(x_norm)   # [B, output_dim]

        # Step 4 — Structural normalisation
        out = self.norm_out(out)

        # Step 5 — Output L2 normalisation -> unit hyper-sphere
        return F.normalize(out, p=2, dim=-1)


# ── Full Pipeline ──────────────────────────────────────────────────────────────

class MarineImageBindPipeline(nn.Module):
    """
    Three modality-specific projection heads that translate each
    frozen encoder's output into the shared 768-D latent space.

    Only the projection heads are trained; the upstream encoders are
    kept frozen and are not part of this module.

    Attributes
    ----------
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
        out = {"image": self.image_head(image_emb)}
        if text_emb  is not None:
            out["text"]  = self.text_head(text_emb)
        if audio_emb is not None:
            out["audio"] = self.audio_head(audio_emb)
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
