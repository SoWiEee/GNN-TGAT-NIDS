"""E-GraphSAGE: edge-feature-aware GraphSAGE for NIDS.

Standard GraphSAGE aggregates only node features during message passing;
edge attributes (flow-level features like bytes, packets, TCP flags) are
only seen by the final classifier MLP.  E-GraphSAGE injects edge features
into every message so the GNN layers can learn from them directly.

Message:  m_{u→v} = W · concat(h_u, e_{uv})
Aggregation: h_v' = combine(h_v, AGG({m_{u→v} : u ∈ N(v)}))

Reference: Lo et al., "E-GraphSAGE: A Graph Neural Network based Intrusion
Detection System for IoT" (2022).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from src.models.base import BaseNIDSModel


class EGraphSAGEConv(MessagePassing):
    """Single E-GraphSAGE convolution layer.

    Each neighbour message is formed by concatenating the neighbour's node
    embedding with the connecting edge feature, then projecting through a
    linear layer.  Messages are aggregated (mean by default) and combined
    with the target node's own embedding via a second linear layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_channels: int,
        aggr: str = "mean",
    ) -> None:
        super().__init__(aggr=aggr)
        self.lin_msg = nn.Linear(in_channels + edge_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
    ) -> Tensor:
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.lin_self(x) + agg
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return self.lin_msg(torch.cat([x_j, edge_attr], dim=-1))


class EGraphSAGEModel(BaseNIDSModel):
    """E-GraphSAGE edge classifier.

    Parameters
    ----------
    in_node_channels:
        Dimension of input node features (``data.x.shape[1]``).
    in_edge_channels:
        Dimension of input edge features (``data.edge_attr.shape[1]``).
    hidden_dim:
        Hidden dimension for all conv layers and the edge MLP.
    num_classes:
        Number of output classes.
    num_layers:
        Number of E-GraphSAGE convolution layers.
    dropout:
        Dropout probability applied after each conv layer.
    aggregation:
        Aggregation scheme (``"mean"``, ``"max"``, ``"add"``).
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

        self.node_proj = nn.Linear(in_node_channels, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                EGraphSAGEConv(hidden_dim, hidden_dim, in_edge_channels, aggr=aggregation)
            )

        edge_in_dim = hidden_dim * 2 + in_edge_channels
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h = self.node_proj(x)
        h = F.relu(h)

        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        src, dst = edge_index[0], edge_index[1]
        edge_repr = torch.cat([h[src], h[dst], edge_attr], dim=-1)

        return self.edge_mlp(edge_repr)
