"""DyGFormer for edge-level NIDS classification.

Yu et al., "Towards Better Dynamic Graph Learning:
New Architecture and Unified Library", NeurIPS 2023.

Stateless temporal model with three key innovations over TGAT:
1. **Neighbor co-occurrence encoding** — learnable embedding of how many
   shared neighbors src and dst have, capturing repeated interaction patterns.
2. **Temporal patching** — groups consecutive temporal neighbors into patches
   (analogous to ViT), reducing transformer sequence length while preserving
   temporal locality.
3. **Transformer encoder** — standard multi-head self-attention replaces
   TGAT's custom temporal attention, enabling deeper stacking.

Like TGAT, DyGFormer is stateless (no persistent memory like TGN).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import TemporalData

from src.models.base import BaseNIDSModel
from src.models.tgat import LastNeighborLoader, TimeEncoder


class DyGFormerModel(BaseNIDSModel):
    """DyGFormer for edge-level NIDS classification.

    Parameters
    ----------
    num_nodes:
        Total unique nodes (injected at runtime by train.py).
    raw_msg_dim:
        Dimension of raw edge features (injected at runtime).
    hidden_dim:
        Hidden dimension for transformer and classifier.
    time_dim:
        Dimension of the learnable time encoding.
    num_classes:
        Number of output classes (injected at runtime).
    n_neighbors:
        Number of most-recent temporal neighbours per node.
        Must be divisible by ``patch_size``.
    n_layers:
        Number of transformer encoder layers.
    n_heads:
        Number of attention heads in the transformer.
    patch_size:
        Number of consecutive neighbors per patch.
    dropout:
        Dropout probability.
    max_co_occurrence:
        Maximum co-occurrence count to embed (clamped).
    """

    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        hidden_dim: int = 172,
        time_dim: int = 64,
        num_classes: int = 10,
        n_neighbors: int = 20,
        n_layers: int = 2,
        n_heads: int = 2,
        patch_size: int = 2,
        dropout: float = 0.1,
        max_co_occurrence: int = 32,
    ) -> None:
        super().__init__()
        assert n_neighbors % patch_size == 0, (
            f"n_neighbors ({n_neighbors}) must be divisible by patch_size ({patch_size})"
        )

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.n_neighbors = n_neighbors
        self.patch_size = patch_size
        self.n_patches = n_neighbors // patch_size
        self.max_co_occ = max_co_occurrence

        self.neighbor_loader = LastNeighborLoader(num_nodes, n_neighbors, raw_msg_dim)
        self.time_enc = TimeEncoder(time_dim)
        self.co_occ_enc = nn.Embedding(max_co_occurrence + 1, hidden_dim)
        self.input_proj = nn.Linear(raw_msg_dim + time_dim, hidden_dim)
        self.pos_enc = nn.Embedding(self.n_patches, hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)

        clf_in = 2 * hidden_dim + raw_msg_dim
        self.edge_clf = nn.Sequential(
            nn.Linear(clf_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    # ------------------------------------------------------------------
    # Public interface (duck-typed by train.py)
    # ------------------------------------------------------------------

    def forward(self, data: TemporalData) -> Tensor:
        """Compute per-edge logits for a temporal event batch.

        Returns shape ``(batch_size, num_classes)``.
        """
        src, dst, t, msg = data.src, data.dst, data.t, data.msg
        device = src.device

        nbr_s = self._query(src, device)
        nbr_d = self._query(dst, device)

        co_s = self._co_occurrence(nbr_s[0], nbr_d[0], nbr_s[3], nbr_d[3])
        co_d = self._co_occurrence(nbr_d[0], nbr_s[0], nbr_d[3], nbr_s[3])

        z_src = self._embed(nbr_s[1], nbr_s[2], nbr_s[3], co_s, t, device)
        z_dst = self._embed(nbr_d[1], nbr_d[2], nbr_d[3], co_d, t, device)

        return self.edge_clf(torch.cat([z_src, z_dst, msg], dim=-1))

    def update_state(
        self, src: Tensor, dst: Tensor, t: Tensor, msg: Tensor,
    ) -> None:
        """Update neighbour loader after backward pass."""
        self.neighbor_loader.insert(src, dst, t, msg)

    def reset_memory(self) -> None:
        """Reset neighbour history."""
        self.neighbor_loader.reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query(
        self, node_ids: Tensor, device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Query neighbor loader, return (ids, timestamps, messages, valid)."""
        ids, ts, msgs = self.neighbor_loader.query(node_ids)
        ids, ts, msgs = ids.to(device), ts.to(device), msgs.to(device)
        valid = ids >= 0
        return ids, ts, msgs, valid

    def _co_occurrence(
        self,
        ids_a: Tensor,   # (N, k)
        ids_b: Tensor,   # (N, k)
        va: Tensor,       # (N, k) bool
        vb: Tensor,       # (N, k) bool
    ) -> Tensor:
        """Count how many of A's neighbors also appear in B's neighbor list."""
        matches = (ids_a.unsqueeze(2) == ids_b.unsqueeze(1)) & va.unsqueeze(2) & vb.unsqueeze(1)
        return matches.sum(dim=2).clamp(max=self.max_co_occ)

    def _embed(
        self,
        nbr_t: Tensor,     # (N, k)
        nbr_msg: Tensor,   # (N, k, msg_dim)
        valid: Tensor,      # (N, k) bool
        co_occ: Tensor,     # (N, k)
        t_ref: Tensor,      # (N,)
        device: torch.device,
    ) -> Tensor:
        """Encode → sort → patch → transformer → pool."""
        N, k = nbr_msg.shape[:2]
        P, S = self.n_patches, self.patch_size

        # Sort by time (oldest first, invalid pushed to end)
        sort_key = nbr_t.clone()
        sort_key[~valid] = float("inf")
        idx = sort_key.argsort(dim=1)

        nbr_msg = nbr_msg.gather(1, idx.unsqueeze(-1).expand_as(nbr_msg))
        nbr_t = nbr_t.gather(1, idx)
        valid = valid.gather(1, idx)
        co_occ = co_occ.gather(1, idx)

        # Encode each neighbor interaction
        dt = (t_ref.unsqueeze(1) - nbr_t).clamp(min=0.0)
        t_feat = self.time_enc(dt.reshape(-1)).reshape(N, k, -1)
        x = self.input_proj(torch.cat([nbr_msg, t_feat], dim=-1))
        x = x + self.co_occ_enc(co_occ)
        x = x * valid.float().unsqueeze(-1)

        # Patch: mean-pool groups of consecutive neighbors
        x = x.view(N, P, S, self.hidden_dim)
        v = valid.view(N, P, S)
        v_count = v.float().sum(dim=2, keepdim=True).clamp(min=1)
        x = (x * v.float().unsqueeze(-1)).sum(dim=2) / v_count  # (N, P, H)

        # Add position encoding
        x = x + self.pos_enc(torch.arange(P, device=device))

        # Transformer with padding mask for fully-invalid patches
        pad_mask = ~v.any(dim=2)  # (N, P)
        x = self.transformer(x, src_key_padding_mask=pad_mask)

        # Mean-pool valid patches
        pv = (~pad_mask).float().unsqueeze(-1)
        z = (x * pv).sum(dim=1) / pv.sum(dim=1).clamp(min=1)

        return self.output_norm(z)
