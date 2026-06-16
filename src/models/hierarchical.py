"""Two-stage hierarchical classifier for NIDS edge classification.

Stage 1: Binary (Benign vs Attack) — high accuracy since the model already
         distinguishes benign from attack well.
Stage 2: Attack subtype (9 classes) — trained only on attack edges with
         full graph structure preserved for message passing.

Combined inference uses probabilistic decomposition:
    P(class=0) = P(benign | stage1)
    P(class=k) = P(attack | stage1) × P(subtype=k-1 | stage2)  for k=1..9
"""
from __future__ import annotations

import torch
from torch_geometric.data import Data

from src.models.base import BaseNIDSModel


class HierarchicalNIDSModel(BaseNIDSModel):
    """Wraps two trained models into a single 10-class predictor.

    Parameters
    ----------
    stage1 : BaseNIDSModel
        Binary classifier (num_classes=2, 0=Benign 1=Attack).
    stage2 : BaseNIDSModel
        Attack subtype classifier (num_classes=9).
    """

    def __init__(self, stage1: BaseNIDSModel, stage2: BaseNIDSModel) -> None:
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2

    def forward(self, data: Data) -> torch.Tensor:
        """Return (num_edges, 10) logits via probabilistic decomposition."""
        binary_logits = self.stage1(data)
        attack_logits = self.stage2(data)

        binary_proba = torch.softmax(binary_logits, dim=-1)
        attack_proba = torch.softmax(attack_logits, dim=-1)

        p_benign = binary_proba[:, 0:1]
        p_attack = binary_proba[:, 1:2]

        combined = torch.cat([p_benign, p_attack * attack_proba], dim=-1)
        return torch.log(combined.clamp(min=1e-12))
