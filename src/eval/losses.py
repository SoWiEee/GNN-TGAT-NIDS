"""Loss functions for NIDS edge classification."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multiclass Focal Loss — down-weights easy examples to focus on hard ones.

    Addresses the high-precision / low-recall imbalance that arises from
    class-weighted CrossEntropy alone.  Recommended γ=2.0 (Lin et al. 2017).

    FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)

    Args:
        weight:  Per-class weights (same semantics as CrossEntropyLoss.weight).
                 Pass the inverse-frequency weights from compute_class_weights.
        gamma:   Focusing exponent.  0 → standard weighted CE.  2 is typical.
        reduction: ``"mean"`` or ``"sum"``.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Upcast to float32: AMP may pass float16 logits whose extreme values
        # cause log_softmax → -inf (float16 underflow), producing NaN loss.
        logits = logits.float()
        weight = self.weight.float() if self.weight is not None else None  # type: ignore[union-attr]
        # Per-sample CE without reduction; weight applies class-level α
        ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
        # p_t = probability assigned to the correct class
        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce

        if self.reduction == "mean":
            return focal.mean()
        return focal.sum()
