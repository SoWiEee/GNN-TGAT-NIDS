"""Temporal Graph Attention Network (TGAT) for edge-level NIDS classification.

Implements the TGAT architecture from:
  Xu et al., "Inductive Representation Learning on Temporal Graphs"
  ICLR 2020

Architecture
------------
* **Functional time encoding** — learnable Time2Vec: ``cos(W·Δt + b)``
* **Multi-head temporal attention** — for each node, attends over its
  k most-recent temporal neighbours. Query = node's own features + time
  encoding of the current timestamp; Key/Value = neighbour features +
  time encoding of the interaction delta + edge message.
* **Stacked layers** — L layers of temporal attention produce
  increasingly abstract node embeddings.
* **Edge classifier** — 2-layer MLP over ``concat(z_src, z_dst, msg)``.

Unlike TGN, TGAT is **stateless** — it has no persistent per-node memory.
Instead it recomputes node embeddings from scratch at each forward pass
using temporal neighbourhood sampling.

Usage
-----
    model = TGATModel(num_nodes=5000, raw_msg_dim=40, hidden_dim=172, heads=2)
    logits = model(temporal_batch)    # (batch_size, num_classes)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import TemporalData

from src.models.base import BaseNIDSModel


class TimeEncoder(nn.Module):
    """Learnable time encoding: ``cos(W·t + b)``."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.lin = nn.Linear(1, out_channels)

    def forward(self, t: Tensor) -> Tensor:
        return self.lin(t.unsqueeze(-1)).cos()


class LastNeighborLoader:
    """CPU ring buffer tracking the k most recent interactions per node.

    Identical to the one in tgn.py — kept as a standalone copy so TGAT
    has no import dependency on TGN.
    """

    def __init__(self, num_nodes: int, size: int, msg_dim: int) -> None:
        self.size = size
        self.msg_dim = msg_dim
        self.neighbors = torch.full((num_nodes, size), -1, dtype=torch.long)
        self.timestamps = torch.zeros(num_nodes, size, dtype=torch.float32)
        self.messages = torch.zeros(num_nodes, size, msg_dim, dtype=torch.float32)
        self._pos = torch.zeros(num_nodes, dtype=torch.long)

    @torch.no_grad()
    def insert(self, src: Tensor, dst: Tensor, t: Tensor, msg: Tensor) -> None:
        src_cpu = src.cpu()
        dst_cpu = dst.cpu()
        t_cpu = t.cpu().float()
        msg_cpu = msg.detach().cpu().float()

        for node, nbr in [(src_cpu, dst_cpu), (dst_cpu, src_cpu)]:
            sort_idx = node.argsort(stable=True)
            s_node = node[sort_idx]
            s_nbr = nbr[sort_idx]
            s_t = t_cpu[sort_idx]
            s_msg = msg_cpu[sort_idx]

            new_group = torch.cat([
                torch.ones(1, dtype=torch.bool),
                s_node[1:] != s_node[:-1],
            ])
            group_id = new_group.cumsum(0) - 1
            group_start_pos = new_group.nonzero(as_tuple=True)[0]
            local_rank = torch.arange(len(s_node)) - group_start_pos[group_id]

            pos = (self._pos[s_node] + local_rank) % self.size

            self.neighbors[s_node, pos] = s_nbr
            self.timestamps[s_node, pos] = s_t
            self.messages[s_node, pos] = s_msg

            ones = torch.ones(len(s_node), dtype=self._pos.dtype)
            self._pos.scatter_add_(0, s_node, ones)

    def query(self, n_id: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        n_cpu = n_id.cpu()
        return (
            self.neighbors[n_cpu],
            self.timestamps[n_cpu],
            self.messages[n_cpu],
        )

    def reset(self) -> None:
        self.neighbors.fill_(-1)
        self.timestamps.zero_()
        self.messages.zero_()
        self._pos.zero_()


class TemporalMultiHeadAttention(nn.Module):
    """Multi-head temporal graph attention (TGAT-style).

    For each target node u at reference time t:
      - Q = W_q( [feat_u ; time_enc(0)] )
      - K = W_k( [feat_nbr ; time_enc(Δt) ; msg] )
      - V = W_v( [feat_nbr ; time_enc(Δt) ; msg] )
      - output = W_out( multi_head_attention(Q, K, V) )
    """

    def __init__(
        self,
        feat_dim: int,
        time_dim: int,
        msg_dim: int,
        out_dim: int,
        heads: int = 2,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = out_dim // heads
        assert out_dim % heads == 0

        self.time_enc = TimeEncoder(time_dim)

        q_in = feat_dim + time_dim
        kv_in = feat_dim + time_dim + msg_dim

        self.W_q = nn.Linear(q_in, out_dim)
        self.W_k = nn.Linear(kv_in, out_dim)
        self.W_v = nn.Linear(kv_in, out_dim)
        self.W_out = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        feat_u: Tensor,      # (N, feat_dim)
        t_ref: Tensor,        # (N,)
        nbr_feat: Tensor,     # (N, k, feat_dim)
        nbr_t: Tensor,        # (N, k)
        nbr_msg: Tensor,      # (N, k, msg_dim)
        valid: Tensor,        # (N, k) bool
    ) -> Tensor:
        N, k = nbr_feat.shape[:2]
        device = feat_u.device
        H = self.heads
        D = self.head_dim

        dt = (t_ref.unsqueeze(1) - nbr_t).clamp(min=0.0)
        time_neigh = self.time_enc(dt.reshape(-1)).reshape(N, k, -1)
        time_zero = self.time_enc(torch.zeros(N, device=device))

        Q = self.W_q(torch.cat([feat_u, time_zero], dim=-1))
        Q = Q.view(N, 1, H, D).transpose(1, 2)

        mask_f = valid.float().unsqueeze(-1)
        kv_in = torch.cat([nbr_feat, time_neigh, nbr_msg], dim=-1) * mask_f
        K = self.W_k(kv_in).view(N, k, H, D).transpose(1, 2)
        V = self.W_v(kv_in).view(N, k, H, D).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(D)
        attn_mask = valid.unsqueeze(1).expand(-1, H, -1)
        scores = scores.squeeze(2).masked_fill(~attn_mask, float("-inf"))

        all_invalid = ~valid.any(dim=1)
        if all_invalid.any():
            scores[all_invalid] = 0.0

        attn = torch.softmax(scores, dim=-1) * attn_mask.float()
        z = (attn.unsqueeze(2) @ V).squeeze(2)
        z = z.transpose(1, 2).contiguous().view(N, -1)

        return self.norm(self.W_out(z))


class TGATModel(BaseNIDSModel):
    """Temporal Graph Attention Network for edge-level NIDS classification.

    Parameters
    ----------
    num_nodes:
        Total number of unique nodes in the dataset.
    raw_msg_dim:
        Dimension of raw edge features (NetFlow features).
    hidden_dim:
        Hidden dimension for attention layers and classifier.
    time_dim:
        Dimension of the learnable time encoding.
    heads:
        Number of attention heads.
    num_classes:
        Number of output classes.
    n_neighbors:
        Number of most-recent temporal neighbours per node.
    n_layers:
        Number of stacked temporal attention layers.
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        hidden_dim: int = 172,
        time_dim: int = 64,
        heads: int = 2,
        num_classes: int = 10,
        n_neighbors: int = 20,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.neighbor_loader = LastNeighborLoader(num_nodes, n_neighbors, raw_msg_dim)

        self.node_proj = nn.Linear(raw_msg_dim, hidden_dim)

        self.attn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.attn_layers.append(
                TemporalMultiHeadAttention(
                    feat_dim=hidden_dim,
                    time_dim=time_dim,
                    msg_dim=raw_msg_dim,
                    out_dim=hidden_dim,
                    heads=heads,
                )
            )

        self.dropout = nn.Dropout(dropout)

        clf_in = 2 * hidden_dim + raw_msg_dim
        self.edge_clf = nn.Sequential(
            nn.Linear(clf_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._node_feats = nn.Parameter(
            torch.zeros(num_nodes, raw_msg_dim), requires_grad=False
        )
        self._node_counts = nn.Parameter(
            torch.zeros(num_nodes), requires_grad=False
        )

    def forward(self, data: TemporalData) -> Tensor:
        """Compute per-edge logits for a temporal event batch.

        Parameters
        ----------
        data:
            Batch from TemporalDataLoader with src, dst, t, msg.

        Returns
        -------
        Tensor
            Shape ``(batch_size, num_classes)``.
        """
        src, dst, t, msg = data.src, data.dst, data.t, data.msg
        device = src.device

        z_src = self._compute_embedding(src, t, device)
        z_dst = self._compute_embedding(dst, t, device)

        return self.edge_clf(torch.cat([z_src, z_dst, msg], dim=-1))

    def _compute_embedding(
        self, node_ids: Tensor, t_ref: Tensor, device: torch.device
    ) -> Tensor:
        """Compute temporal embedding for a set of nodes."""
        feat = self._get_node_features(node_ids, device)
        h = self.node_proj(feat)

        nbr_ids, nbr_t, nbr_msg = self.neighbor_loader.query(node_ids)
        nbr_ids = nbr_ids.to(device)
        nbr_t = nbr_t.to(device)
        nbr_msg = nbr_msg.to(device)
        valid = nbr_ids >= 0

        nbr_feat = self._get_node_features(nbr_ids.clamp(min=0).reshape(-1), device)
        E, k = nbr_ids.shape
        nbr_feat = nbr_feat.reshape(E, k, -1) * valid.float().unsqueeze(-1)
        nbr_h = self.node_proj(nbr_feat)

        for attn_layer in self.attn_layers:
            h = attn_layer(h, t_ref, nbr_h, nbr_t, nbr_msg, valid)
            h = self.dropout(h)

        return h

    def _get_node_features(self, node_ids: Tensor, device: torch.device) -> Tensor:
        """Return running-average features for nodes."""
        counts = self._node_counts[node_ids.cpu()].to(device).clamp(min=1).unsqueeze(-1)
        feats = self._node_feats[node_ids.cpu()].to(device)
        return feats / counts

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, msg: Tensor) -> None:
        """Update neighbour loader and running node feature averages."""
        self.neighbor_loader.insert(src, dst, t, msg)

        src_cpu = src.cpu()
        dst_cpu = dst.cpu()
        msg_cpu = msg.detach().cpu().float()

        self._node_feats.data[src_cpu] += msg_cpu
        self._node_feats.data[dst_cpu] += msg_cpu
        self._node_counts.data[src_cpu] += 1
        self._node_counts.data[dst_cpu] += 1

    def reset_memory(self) -> None:
        """Reset neighbour history and node features."""
        self.neighbor_loader.reset()
        self._node_feats.data.zero_()
        self._node_counts.data.zero_()
