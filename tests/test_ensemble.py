"""Tests for the ensemble model."""
from __future__ import annotations

import torch
from torch_geometric.data import Data

from src.models.ensemble import EnsembleModel


def _make_dummy_model(n_features: int, n_classes: int, bias_class: int = 0) -> torch.nn.Module:
    """Create a simple linear model that biases predictions toward a class."""
    model = torch.nn.Linear(n_features * 2 + n_features, n_classes)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
        model.bias[bias_class] = 5.0
    return model


class _WrapperModel(torch.nn.Module):
    """Wraps a linear layer to accept PyG Data like BaseNIDSModel."""

    def __init__(self, linear: torch.nn.Module):
        super().__init__()
        self.linear = linear

    def forward(self, data: Data) -> torch.Tensor:
        src = data.x[data.edge_index[0]]
        dst = data.x[data.edge_index[1]]
        edge_repr = torch.cat([src, dst, data.edge_attr], dim=-1)
        return self.linear(edge_repr)


def _make_data(n_nodes: int = 4, n_edges: int = 6, n_feat: int = 3) -> Data:
    return Data(
        x=torch.randn(n_nodes, n_feat),
        edge_index=torch.randint(0, n_nodes, (2, n_edges)),
        edge_attr=torch.randn(n_edges, n_feat),
    )


def test_soft_vote_averages_probabilities():
    n_feat, n_cls = 3, 4
    m1 = _WrapperModel(_make_dummy_model(n_feat, n_cls, bias_class=0))
    m2 = _WrapperModel(_make_dummy_model(n_feat, n_cls, bias_class=1))
    ensemble = EnsembleModel({"m1": m1, "m2": m2}, strategy="soft_vote")

    data = _make_data(n_feat=n_feat)
    result = ensemble(data)

    assert result.shape == (data.edge_index.shape[1], n_cls)
    assert torch.allclose(result.sum(dim=-1), torch.ones(result.shape[0]), atol=1e-5)


def test_hard_vote_returns_vote_proportions():
    n_feat, n_cls = 3, 4
    m1 = _WrapperModel(_make_dummy_model(n_feat, n_cls, bias_class=2))
    m2 = _WrapperModel(_make_dummy_model(n_feat, n_cls, bias_class=2))
    m3 = _WrapperModel(_make_dummy_model(n_feat, n_cls, bias_class=1))
    ensemble = EnsembleModel({"m1": m1, "m2": m2, "m3": m3}, strategy="hard_vote")

    data = _make_data(n_feat=n_feat)
    result = ensemble(data)
    preds = result.argmax(dim=-1)

    assert (preds == 2).all()


def test_weighted_vote_respects_weights():
    n_feat, n_cls = 3, 4
    m1 = _WrapperModel(_make_dummy_model(n_feat, n_cls, bias_class=0))
    m2 = _WrapperModel(_make_dummy_model(n_feat, n_cls, bias_class=1))
    ensemble = EnsembleModel(
        {"m1": m1, "m2": m2},
        strategy="weighted",
        weights={"m1": 0.9, "m2": 0.1},
    )

    data = _make_data(n_feat=n_feat)
    result = ensemble(data)
    preds = result.argmax(dim=-1)

    assert (preds == 0).all()


def test_predict_returns_per_model_breakdown():
    n_feat, n_cls = 3, 4
    m1 = _WrapperModel(_make_dummy_model(n_feat, n_cls, bias_class=0))
    m2 = _WrapperModel(_make_dummy_model(n_feat, n_cls, bias_class=1))
    ensemble = EnsembleModel({"m1": m1, "m2": m2})

    data = _make_data(n_feat=n_feat)
    result = ensemble.predict(data)

    assert "ensemble_predictions" in result
    assert "per_model" in result
    assert "m1" in result["per_model"]
    assert "m2" in result["per_model"]
    assert len(result["ensemble_predictions"]) == data.edge_index.shape[1]


def test_raises_on_empty_models():
    try:
        EnsembleModel({})
        assert False, "Should have raised"
    except ValueError:
        pass
