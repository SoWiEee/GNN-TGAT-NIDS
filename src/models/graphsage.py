"""GraphSAGE-based edge classifier for static snapshot graphs.

Architecture:
    3 × SAGEConv(hidden_dim, hidden_dim, aggr='mean')
    Edge classifier MLP: concat(h_src, h_dst, edge_attr) → hidden_dim → num_classes
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from src.models.base import BaseNIDSModel


class GraphSAGEModel(BaseNIDSModel):
    """GraphSAGE edge classifier.

    Parameters
    ----------
    in_node_channels:
        Dimension of input node features (``data.x.shape[1]``).
    in_edge_channels:
        Dimension of input edge features (``data.edge_attr.shape[1]``).
    hidden_dim:
        Hidden dimension for all SAGEConv layers and the edge MLP.
    num_classes:
        Number of output classes (e.g., 10 for NF-UNSW-NB15-v2).
    num_layers:
        Number of SAGEConv layers.
    dropout:
        Dropout probability applied after each conv layer.
    aggregation:
        Aggregation scheme for SAGEConv (``"mean"``, ``"max"``, ``"lstm"``).
    """

    def __init__(
        self,
        in_node_channels: int,
        in_edge_channels: int,
        hidden_dim: int = 256,
        num_classes: int = 10,
        num_layers: int = 3,
        dropout: float = 0.3,
        aggregation: str = "mean",
    ) -> None:
        super().__init__()

        self.dropout = dropout
        self.num_layers = num_layers

        # Input projection for node features
        self.node_proj = nn.Linear(in_node_channels, hidden_dim)

        # SAGEConv layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregation))

        # Edge classifier MLP: concat(h_src, h_dst, edge_attr) → hidden_dim → num_classes
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
        h = F.relu(h)

        # Message passing
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Edge representation: concat source and dest node embeddings + edge attrs
        src, dst = edge_index[0], edge_index[1]
        edge_repr = torch.cat([h[src], h[dst], edge_attr], dim=-1)

        return self.edge_mlp(edge_repr)
