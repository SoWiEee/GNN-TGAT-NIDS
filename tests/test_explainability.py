"""Tests for GNN Explainability module."""
from __future__ import annotations

import torch
from torch_geometric.data import Data

from src.explain.gnn_explainer import explain_flow, explain_top_alerts


class _SimpleEdgeClassifier(torch.nn.Module):
    """Minimal edge classifier compatible with the explainer adapter."""

    def __init__(self, n_feat: int, n_classes: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(n_feat * 2 + n_feat, n_classes)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()
            self.linear.bias[1] = 5.0

    def forward(self, data: Data) -> torch.Tensor:
        src = data.x[data.edge_index[0]]
        dst = data.x[data.edge_index[1]]
        edge_repr = torch.cat([src, dst, data.edge_attr], dim=-1)
        return self.linear(edge_repr)


def _make_data(n_nodes: int = 6, n_edges: int = 10, n_feat: int = 4) -> Data:
    return Data(
        x=torch.randn(n_nodes, n_feat),
        edge_index=torch.randint(0, n_nodes, (2, n_edges)),
        edge_attr=torch.randn(n_edges, n_feat),
    )


def test_explain_flow_returns_expected_keys():
    model = _SimpleEdgeClassifier(n_feat=4, n_classes=3)
    data = _make_data()
    result = explain_flow(model, data, edge_idx=0, epochs=10)

    assert "edge_idx" in result
    assert "predicted_class" in result
    assert "confidence" in result
    assert "node_feature_importance" in result
    assert "top_features" in result
    assert result["edge_idx"] == 0


def test_explain_flow_feature_importance_not_empty():
    model = _SimpleEdgeClassifier(n_feat=4, n_classes=3)
    data = _make_data()
    result = explain_flow(model, data, edge_idx=0, epochs=10)

    assert len(result["top_features"]) > 0
    for feat in result["top_features"]:
        assert "name" in feat
        assert "importance" in feat


def test_explain_top_alerts_returns_list():
    model = _SimpleEdgeClassifier(n_feat=4, n_classes=3)
    data = _make_data()
    results = explain_top_alerts(model, data, top_k=2, epochs=10)

    assert isinstance(results, list)
    for r in results:
        assert "rank" in r
        assert "predicted_class" in r
        assert r["predicted_class"] > 0


def test_explain_top_alerts_empty_when_no_attacks():
    model = _SimpleEdgeClassifier(n_feat=4, n_classes=3)
    with torch.no_grad():
        model.linear.bias.zero_()
        model.linear.bias[0] = 10.0

    data = _make_data()
    results = explain_top_alerts(model, data, top_k=5, epochs=10)

    assert results == []
