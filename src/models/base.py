"""
src/models/base.py
Abstract base class for all GARF-NIDS graph neural network models.

All concrete models (GraphSAGE, GAT, TGAT, TGN) must inherit BaseNIDSModel and
implement forward(). This allows eval/comparison.py to call every model through
a single unified interface via Hydra instantiate, without importing concrete classes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseNIDSModel(ABC, nn.Module):
    """Unified interface for all NIDS GNN models.

    Task: edge-level binary / multi-class classification.
    Each directed NetFlow (edge) is classified as benign (0) or one of the
    attack categories defined in NF-UNSW-NB15-v2.

    Subclasses must implement forward(). predict_edges() is provided as a
    convenience wrapper and should not be overridden.
    """

    # ── Abstract interface ────────────────────────────────────────────────

    @abstractmethod
    def forward(self, data) -> torch.Tensor:
        """Compute per-edge logits.

        Args:
            data: PyG ``Data`` (static models) or ``TemporalData`` (temporal
                models). Concrete subclasses type-narrow this argument.

        Returns:
            Tensor of shape ``(num_edges, num_classes)`` containing unnormalised
            logits. Do **not** apply softmax here; the loss function handles it.
        """
        ...

    # ── Provided implementation ───────────────────────────────────────────

    def predict_edges(self, data) -> torch.Tensor:
        """Return predicted edge labels (argmax over logits).

        Args:
            data: Same type accepted by ``forward()``.

        Returns:
            Integer tensor of shape ``(num_edges,)`` with predicted class
            indices (0 = benign).
        """
        with torch.no_grad():
            logits = self.forward(data)
        return logits.argmax(dim=-1)

    def predict_proba(self, data) -> torch.Tensor:
        """Return per-edge class probabilities (softmax of logits).

        Args:
            data: Same type accepted by ``forward()``.

        Returns:
            Float tensor of shape ``(num_edges, num_classes)`` in [0, 1].
        """
        with torch.no_grad():
            logits = self.forward(data)
        return torch.softmax(logits, dim=-1)
