"""GAT-based edge classifier for static snapshot graphs.

Architecture:
    3 × GATConv — first (num_layers-1) layers: num_heads heads with concat=True
                — final layer: 1 head with concat=False (returns hidden_dim)
    Edge classifier MLP: concat(h_src, h_dst, edge_attr) → hidden_dim → num_classes
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from src.models.base import BaseNIDSModel


class GATModel(BaseNIDSModel):
    """GAT edge classifier.

    The hidden dimension per head is ``hidden_dim // num_heads`` so that the
    concatenated output of intermediate layers is always ``hidden_dim``.
    The final layer uses a single head (no concatenation) and outputs ``hidden_dim``.

    Parameters
    ----------
    in_node_channels:
        Dimension of input node features.
    in_edge_channels:
        Dimension of input edge features.
    hidden_dim:
        Total hidden dimension.  Must be divisible by ``num_heads``.
    num_classes:
        Number of output classes.
    num_layers:
        Number of GATConv layers (≥ 2).
    num_heads:
        Number of attention heads in all but the final layer.
    dropout:
        Dropout probability applied after each conv layer and in attention.
    """

    def __init__(
        self,
        in_node_channels: int,
        in_edge_channels: int,
        hidden_dim: int = 256,
        num_classes: int = 10,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.dropout = dropout
        self.num_layers = num_layers

        head_dim = hidden_dim // num_heads  # per-head output dimension

        # Input projection
        self.node_proj = nn.Linear(in_node_channels, hidden_dim)

        # GATConv layers
        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            is_last = layer_idx == num_layers - 1
            if is_last:
                # Final layer: single head, out_channels = hidden_dim
                self.convs.append(
                    GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
                )
            else:
                # num_heads heads, concat → output = head_dim * num_heads = hidden_dim
                self.convs.append(
                    GATConv(hidden_dim, head_dim, heads=num_heads, concat=True, dropout=dropout)
                )

        # Edge classifier MLP
        edge_in_dim = hidden_dim * 2 + in_edge_channels
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Compute per-edge logits.

        Parameters
        ----------
        data:
            PyG ``Data`` with ``x``, ``edge_index``, and ``edge_attr``.

        Returns
        -------
        torch.Tensor
            Shape ``(num_edges, num_classes)``.
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Node feature projection
        h = self.node_proj(x)
        h = F.elu(h)

        # Message passing with attention
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < self.num_layers - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        # Edge representation
        src, dst = edge_index[0], edge_index[1]
        edge_repr = torch.cat([h[src], h[dst], edge_attr], dim=-1)

        return self.edge_mlp(edge_repr)
