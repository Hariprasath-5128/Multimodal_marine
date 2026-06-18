"""
loss.py — Image-Anchored Multi-Positive Supervised Contrastive Loss
                     with Hard Negative Mining
====================================================================
Standard InfoNCE treats only the main diagonal as a positive, which
breaks down when a batch contains multiple samples of the same class.

SupCon (Supervised Contrastive, Khosla et al. 2020) extends InfoNCE
to N positives per anchor by:

    1.  Building a binary positive mask from integer class labels.
    2.  Row-normalising the mask so every anchor receives a balanced
        gradient signal regardless of how many positives it has.
    3.  Computing cross-entropy over the full similarity matrix and
        weighting each row entry by the normalised positive mask.

Priority 4 — Hard Negative Mining
-----------------------------------
The original SupCon denominator sums over ALL non-self entries, giving
equal weight to easy negatives (very dissimilar species) and hard
negatives (very similar, e.g., Fin Whale vs Blue Whale).

Hard negative mining restricts the denominator to the top-N most
similar negatives per anchor row, discarding uninformative easy pairs.

Cross-Modal vs Intra-Modal Diagonal Handling
--------------------------------------------
When computing Image -> Audio loss, anchor_i's exact positive target_i
sits on the main diagonal. Removing it (as intra-modal SupCon does to
prevent trivial self-similarity) deletes the BEST and most direct
learning signal.

Solution: `exclude_diagonal=False` (default) keeps the diagonal for
cross-modal losses. Set `exclude_diagonal=True` only for intra-modal.
"""

import torch


def supervised_contrastive_loss(
    anchor_feats:     torch.Tensor,
    target_feats:     torch.Tensor,
    labels:           torch.Tensor,
    temperature:      float = 0.07,
    hard_neg_count:   int   = 20,
    exclude_diagonal: bool  = False,
) -> torch.Tensor:
    """
    Multi-Positive Supervised Contrastive Loss with Hard Negative Mining.

    Parameters
    ----------
    anchor_feats     : FloatTensor [B, D]  — L2-normalised image projections
    target_feats     : FloatTensor [B, D]  — L2-normalised text/audio projections
    labels           : LongTensor  [B]     — integer species IDs
    temperature      : float               — softmax sharpening factor (tau)
    hard_neg_count   : int                 — number of hard negatives to keep per
                                             anchor. If 0, all negatives are used.
    exclude_diagonal : bool                — If True, exclude diagonal from the
                                             positive mask and the denominator
                                             (use for intra-modal SupCon).
                                             If False (default), keep diagonal as
                                             a valid positive (use for cross-modal).

    Returns
    -------
    loss : scalar FloatTensor
    """
    batch_size = anchor_feats.size(0)
    if batch_size < 2:
        raise ValueError(
            f"supervised_contrastive_loss requires batch_size >= 2, got {batch_size}."
        )

    # ── Similarity Logits  [B x B] ────────────────────────────────────────────
    similarity_matrix = torch.matmul(anchor_feats, target_feats.T) / temperature

    # ── Positive Mask  [B x B]  ───────────────────────────────────────────────
    labels_col    = labels.unsqueeze(1)                             # [B, 1]
    label_matches = torch.eq(labels_col, labels_col.T).float()      # [B, B]

    eye = torch.eye(batch_size, device=labels.device)

    if exclude_diagonal:
        # Intra-modal: self-comparisons are trivially perfect (sim=1.0)
        # so we exclude the diagonal entirely from both positives and denom
        positive_mask = label_matches * (1.0 - eye)
        # Mask self-pairs to -inf in the similarity matrix
        sim_base = similarity_matrix.masked_fill(eye.bool(), float("-inf"))
    else:
        # Cross-modal: diagonal is anchor_i paired with target_i which is
        # the EXACT matched sample — the strongest possible positive signal.
        # Keep it in positive_mask; do NOT mask it out of the denominator.
        positive_mask = label_matches
        sim_base = similarity_matrix

    # Negative mask: different-label pairs (diagonal never selected as negative)
    negative_mask = (1.0 - label_matches) * (1.0 - eye)            # [B, B]

    # ── Hard Negative Mining ──────────────────────────────────────────────────
    if hard_neg_count > 0:
        k = min(hard_neg_count, batch_size - 1)

        if k > 0:
            # Mask positives so they aren't selected as hard negatives
            neg_sims = sim_base.masked_fill(positive_mask.bool(), float("-inf"))
            # Also mask the diagonal from negative selection
            neg_sims = neg_sims.masked_fill(eye.bool(), float("-inf"))

            _, topk_neg_idx = neg_sims.topk(k, dim=1, largest=True, sorted=False)
            hard_neg_mask = torch.zeros_like(similarity_matrix)
            hard_neg_mask.scatter_(1, topk_neg_idx, 1.0)
        else:
            hard_neg_mask = negative_mask

        keep_mask = (positive_mask + hard_neg_mask).clamp(max=1.0)  # [B, B]
        sim_masked = sim_base.masked_fill(keep_mask == 0, float("-inf"))
    else:
        sim_masked = sim_base

    # ── Row-Normalised Positive Mask ──────────────────────────────────────────
    row_sums        = positive_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    normalised_mask = positive_mask / row_sums                      # [B, B]

    # ── Numerically-Stable Log-Softmax ────────────────────────────────────────
    log_denom = torch.logsumexp(sim_masked, dim=1, keepdim=True)    # [B, 1]
    log_prob  = similarity_matrix - log_denom                       # [B, B]

    # ── Per-Anchor Mean-Positive Log-Probability ──────────────────────────────
    mean_log_prob_pos = (normalised_mask * log_prob).sum(dim=1)     # [B]

    return -mean_log_prob_pos.mean()


# ── Numerical validation helper ───────────────────────────────────────────────
def check_loss_finite(loss: torch.Tensor, context: str = "") -> None:
    """
    Raise RuntimeError if the loss is NaN or Inf.
    """
    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite loss detected{' (' + context + ')' if context else ''}:"
            f" {loss.item()}"
        )


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    B, D = 8, 768

    a = torch.randn(B, D);  a = a / a.norm(dim=-1, keepdim=True)
    t = torch.randn(B, D);  t = t / t.norm(dim=-1, keepdim=True)

    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    # Cross-modal (exclude_diagonal=False, default)
    loss_cm = supervised_contrastive_loss(a, t, labels, temperature=0.07, hard_neg_count=0, exclude_diagonal=False)
    print(f"Cross-modal SupCon loss  : {loss_cm.item():.4f}")
    check_loss_finite(loss_cm, "cross-modal")

    # Intra-modal (exclude_diagonal=True)
    loss_im = supervised_contrastive_loss(a, t, labels, temperature=0.07, hard_neg_count=0, exclude_diagonal=True)
    print(f"Intra-modal SupCon loss  : {loss_im.item():.4f}")
    check_loss_finite(loss_im, "intra-modal")

    # With HNM
    loss_hnm = supervised_contrastive_loss(a, t, labels, temperature=0.07, hard_neg_count=3, exclude_diagonal=False)
    print(f"Cross-modal + HNM loss   : {loss_hnm.item():.4f}")
    check_loss_finite(loss_hnm, "cross-modal + HNM")

    print("All checks passed [OK]")
