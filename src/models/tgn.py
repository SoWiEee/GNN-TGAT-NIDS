"""Temporal Graph Networks (TGN) for edge-level NIDS classification.

Implements the TGN architecture from:
  Rossi et al., "Temporal Graph Networks for Deep Learning on Dynamic Graphs"
  arXiv 2006.10637

Architecture
------------
* **Memory module** — per-node GRU memory updated after each interaction batch
  (uses PyG's TGNMemory with IdentityMessage + LastAggregator).
* **Embedding module** — either ``"identity"`` (memory vector only) or
  ``"graph_attention"`` (1-hop temporal attention over recent neighbors, TGAT-
  style, conditioned on time-encoded deltas).
* **Edge classifier** — 2-layer MLP over concatenated src/dst embeddings and
  the raw edge feature vector.

Usage
-----
The model is stateful: call :meth:`update_state` after each training batch
and :meth:`reset_memory` between experiments / adversarial evaluations.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import TemporalData
from torch_geometric.nn.models import TGNMemory
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator

from src.models.base import BaseNIDSModel

# ── Time encoding ─────────────────────────────────────────────────────────────

class TimeEncoder(nn.Module):
    """Cosine time encoding: ``cos(W·t + b)`` — same as PyG's TimeEncoder."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.lin = nn.Linear(1, out_channels)

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()

    def forward(self, t: Tensor) -> Tensor:
        # t: arbitrary shape; output shape = (*t.shape, out_channels)
        return self.lin(t.unsqueeze(-1)).cos()


# ── Temporal neighbor ring buffer ─────────────────────────────────────────────

class LastNeighborLoader:
    """CPU-based ring buffer tracking the *k* most recent interactions per node.

    Stores neighbor IDs, timestamps, and edge features.  Kept on CPU to avoid
    GPU memory pressure from the O(num_nodes × k) message buffers; tensors are
    moved to GPU inside :class:`TGNModel`.
    """

    def __init__(self, num_nodes: int, size: int, msg_dim: int) -> None:
        self.size = size
        self.msg_dim = msg_dim
        # Ring buffers (CPU)
        self.neighbors  = torch.full((num_nodes, size), -1, dtype=torch.long)
        self.timestamps = torch.zeros(num_nodes, size, dtype=torch.float32)
        self.messages   = torch.zeros(num_nodes, size, msg_dim, dtype=torch.float32)
        self._pos       = torch.zeros(num_nodes, dtype=torch.long)

    @torch.no_grad()
    def insert(self, src: Tensor, dst: Tensor, t: Tensor, msg: Tensor) -> None:
        """Insert a batch of interactions (both directions)."""
        src_cpu = src.cpu()
        dst_cpu = dst.cpu()
        t_cpu   = t.cpu().float()
        msg_cpu = msg.detach().cpu().float()

        for node, nbr in [(src_cpu, dst_cpu), (dst_cpu, src_cpu)]:
            for i in range(len(node)):
                n = int(node[i])
                pos = int(self._pos[n]) % self.size
                self.neighbors[n, pos]   = int(nbr[i])
                self.timestamps[n, pos]  = float(t_cpu[i])
                self.messages[n, pos]    = msg_cpu[i]
                self._pos[n] += 1

    def query(self, n_id: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return (neighbors, timestamps, messages) for nodes *n_id* (CPU tensors)."""
        n_cpu = n_id.cpu()
        return (
            self.neighbors[n_cpu],   # (N, k)
            self.timestamps[n_cpu],  # (N, k)
            self.messages[n_cpu],    # (N, k, msg_dim)
        )

    def reset(self) -> None:
        self.neighbors.fill_(-1)
        self.timestamps.zero_()
        self.messages.zero_()
        self._pos.zero_()


# ── Temporal attention embedding ──────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """Single-head temporal graph attention (1-hop TGAT-style embedding).

    For each node u at time t:
      - Query:  Q = W_q([memory_u ; time_enc(0)])
      - Key/V:  constructed from each neighbor's memory, time delta, and msg
      - Output: residual = W_out([memory_u ; attention_result])
    """

    def __init__(
        self,
        memory_dim: int,
        time_dim: int,
        msg_dim: int,
        attn_dim: int,
    ) -> None:
        super().__init__()
        self.attn_dim = attn_dim
        self.time_enc = TimeEncoder(time_dim)
        self.W_q = nn.Linear(memory_dim + time_dim, attn_dim)
        kv_in = memory_dim + time_dim + msg_dim
        self.W_k = nn.Linear(kv_in, attn_dim)
        self.W_v = nn.Linear(kv_in, attn_dim)
        self.W_out = nn.Linear(memory_dim + attn_dim, attn_dim)
        self.norm = nn.LayerNorm(attn_dim)

    def forward(
        self,
        memory_u: Tensor,   # (N, memory_dim)
        t_ref: Tensor,       # (N,)
        nbr_mem: Tensor,     # (N, k, memory_dim)
        nbr_t: Tensor,       # (N, k)
        nbr_msg: Tensor,     # (N, k, msg_dim)
        valid: Tensor,       # (N, k) bool — False for padding slots
    ) -> Tensor:
        N, k = nbr_mem.shape[:2]
        device = memory_u.device

        # Time encodings
        dt = (t_ref.unsqueeze(1) - nbr_t).clamp(min=0.0)          # (N, k)
        time_neigh = self.time_enc(dt.reshape(-1)).reshape(N, k, -1)   # (N, k, D)
        time_zero  = self.time_enc(torch.zeros(N, device=device))       # (N, D)

        # Query (from current node)
        Q = self.W_q(torch.cat([memory_u, time_zero], dim=-1)).unsqueeze(1)  # (N, 1, A)

        # Key / Value (from neighbors) — zero out padding
        mask_f = valid.float().unsqueeze(-1)                               # (N, k, 1)
        kv_in = torch.cat([nbr_mem, time_neigh, nbr_msg], dim=-1) * mask_f
        K = self.W_k(kv_in)   # (N, k, A)
        V = self.W_v(kv_in)   # (N, k, A)

        # Scaled dot-product attention
        scores = (Q @ K.transpose(1, 2)).squeeze(1) / math.sqrt(self.attn_dim)  # (N, k)
        scores = scores.masked_fill(~valid, float("-inf"))
        # Rows where all neighbors are padding → uniform weights (result zeroed by mask_f)
        scores[~valid.any(dim=1)] = 0.0
        attn = torch.softmax(scores, dim=1) * valid.float()   # (N, k)

        z = (attn.unsqueeze(1) @ V).squeeze(1)                # (N, A)
        out = self.norm(self.W_out(torch.cat([memory_u, z], dim=-1)))
        return out   # (N, attn_dim)


# ── TGN model ─────────────────────────────────────────────────────────────────

class TGNModel(BaseNIDSModel):
    """Temporal Graph Network for edge-level NIDS classification.

    Parameters
    ----------
    num_nodes:
        Total number of unique nodes (IPs) in the dataset.
    raw_msg_dim:
        Dimension of raw edge features (NetFlow features).
    memory_dim:
        Dimension of the per-node GRU memory state.
    time_dim:
        Dimension of the time encoding.
    hidden_dim:
        Hidden dimension of the edge classifier MLP and attention.
    num_classes:
        Number of output classes (1 benign + N attack types).
    num_neighbors:
        Number of most-recent temporal neighbors to use in attention.
    embedding_module:
        ``"graph_attention"`` (full TGAT-style) or ``"identity"``
        (memory vector only).
    dropout:
        Dropout probability in the edge classifier MLP.
    """

    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int = 172,
        time_dim: int = 64,
        hidden_dim: int = 256,
        num_classes: int = 10,
        num_neighbors: int = 20,
        embedding_module: str = "graph_attention",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.embedding_module = embedding_module

        # ── Memory module ────────────────────────────────────────────────────
        msg_module  = IdentityMessage(raw_msg_dim, memory_dim, time_dim)
        aggr_module = LastAggregator()
        self.memory = TGNMemory(
            num_nodes, raw_msg_dim, memory_dim, time_dim,
            msg_module, aggr_module,
        )

        # ── Neighbor loader (CPU ring buffer) ─────────────────────────────────
        self.neighbor_loader = LastNeighborLoader(num_nodes, num_neighbors, raw_msg_dim)

        # ── Embedding module ──────────────────────────────────────────────────
        if embedding_module == "graph_attention":
            self.attn = TemporalAttention(memory_dim, time_dim, raw_msg_dim, hidden_dim)
            emb_dim = hidden_dim
        else:
            self.attn = None
            emb_dim = memory_dim

        # ── Edge classifier ───────────────────────────────────────────────────
        clf_in = 2 * emb_dim + raw_msg_dim
        self.edge_clf = nn.Sequential(
            nn.Linear(clf_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    # ── BaseNIDSModel interface ───────────────────────────────────────────────

    def forward(self, data: TemporalData) -> Tensor:
        """Compute per-edge logits for a batch of temporal events.

        Parameters
        ----------
        data:
            A batch produced by :class:`~torch_geometric.loader.TemporalDataLoader`.

        Returns
        -------
        Tensor
            Shape ``(batch_size, num_classes)`` — unnormalised logits.
        """
        src, dst, t, msg = data.src, data.dst, data.t, data.msg
        device = src.device

        if self.embedding_module == "graph_attention" and self.attn is not None:
            z_src, z_dst = self._graph_attention_embedding(src, dst, t, msg, device)
        else:
            z_src, z_dst = self._identity_embedding(src, dst, device)

        return self.edge_clf(torch.cat([z_src, z_dst, msg], dim=-1))

    # ── Stateful memory management ────────────────────────────────────────────

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, msg: Tensor) -> None:
        """Update memory and neighbor loader after processing a batch.

        Must be called **after** the backward pass.  Detaches memory from the
        computational graph to prevent backprop through the entire history.
        """
        self.memory.update_state(src, dst, t, msg)
        self.neighbor_loader.insert(src, dst, t, msg)
        self.memory.detach()

    def reset_memory(self) -> None:
        """Reset memory and neighbor history to initial state."""
        self.memory.reset_state()
        self.neighbor_loader.reset()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _identity_embedding(
        self, src: Tensor, dst: Tensor, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        n_id = torch.cat([src, dst]).unique()
        memory, _ = self.memory(n_id)
        assoc = src.new_full((self.num_nodes,), -1)
        assoc[n_id] = torch.arange(len(n_id), device=device)
        return memory[assoc[src]], memory[assoc[dst]]

    def _graph_attention_embedding(
        self,
        src: Tensor,
        dst: Tensor,
        t: Tensor,
        msg: Tensor,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        # Query neighbor loader (CPU → GPU)
        src_nbr, src_nbr_t, src_nbr_msg = self.neighbor_loader.query(src)
        dst_nbr, dst_nbr_t, dst_nbr_msg = self.neighbor_loader.query(dst)

        src_nbr = src_nbr.to(device)
        dst_nbr = dst_nbr.to(device)
        src_nbr_t   = src_nbr_t.to(device)
        dst_nbr_t   = dst_nbr_t.to(device)
        src_nbr_msg = src_nbr_msg.to(device)
        dst_nbr_msg = dst_nbr_msg.to(device)

        # Collect all unique node IDs (src, dst, valid neighbors)
        valid_src_nbrs = src_nbr[src_nbr >= 0]
        valid_dst_nbrs = dst_nbr[dst_nbr >= 0]
        n_id = torch.cat([src, dst, valid_src_nbrs, valid_dst_nbrs]).unique()

        memory, _ = self.memory(n_id)

        # Build lookup: global node id → row in memory
        assoc = src.new_full((self.num_nodes,), 0)  # default → row 0 (zeroed below)
        assoc[n_id] = torch.arange(len(n_id), device=device)

        # Gather src/dst memories
        src_mem = memory[assoc[src]]   # (E, memory_dim)
        dst_mem = memory[assoc[dst]]   # (E, memory_dim)

        # Gather neighbor memories; invalid (-1) slots map to row 0 then are masked
        E, k = src_nbr.shape
        src_flat = src_nbr.clamp(min=0).reshape(-1)
        dst_flat = dst_nbr.clamp(min=0).reshape(-1)
        src_nbr_mem = memory[assoc[src_flat]].reshape(E, k, self.memory_dim)
        dst_nbr_mem = memory[assoc[dst_flat]].reshape(E, k, self.memory_dim)

        # Mask for valid neighbor slots
        src_valid = src_nbr >= 0   # (E, k)
        dst_valid = dst_nbr >= 0   # (E, k)

        # Zero out invalid neighbor memory (cleaner than masking inside attention)
        src_nbr_mem = src_nbr_mem * src_valid.float().unsqueeze(-1)
        dst_nbr_mem = dst_nbr_mem * dst_valid.float().unsqueeze(-1)

        z_src = self.attn(src_mem, t, src_nbr_mem, src_nbr_t, src_nbr_msg, src_valid)
        z_dst = self.attn(dst_mem, t, dst_nbr_mem, dst_nbr_t, dst_nbr_msg, dst_valid)

        return z_src, z_dst
