"""Tests for validation-weighted ensemble."""
import torch
from torch_geometric.data import Data

from src.models.ensemble import EnsembleModel


class _DummyModel(torch.nn.Module):
    def __init__(self, n_classes: int, bias: int = 0):
        super().__init__()
        self._n_classes = n_classes
        self._bias = bias
        self._linear = torch.nn.Linear(1, 1)

    def forward(self, data):
        n_edges = data.edge_index.shape[1]
        logits = torch.randn(n_edges, self._n_classes)
        logits[:, self._bias] += 5.0
        return logits


def _make_data(n_edges=20, n_classes=3) -> Data:
    return Data(
        x=torch.randn(5, 4),
        edge_index=torch.randint(0, 5, (2, n_edges)),
        edge_attr=torch.randn(n_edges, 4),
        y=torch.zeros(n_edges, dtype=torch.long),
        y_multi=torch.randint(0, n_classes, (n_edges,)),
    )


class TestFromValidation:
    def test_produces_weighted_strategy(self):
        models = {
            "a": _DummyModel(3, bias=0),
            "b": _DummyModel(3, bias=1),
        }
        loader = [_make_data()]
        ensemble = EnsembleModel.from_validation(models, loader)
        assert ensemble.strategy == "weighted"

    def test_weights_are_normalized(self):
        models = {
            "a": _DummyModel(3, bias=0),
            "b": _DummyModel(3, bias=1),
        }
        loader = [_make_data()]
        ensemble = EnsembleModel.from_validation(models, loader)
        total = sum(ensemble._weights.values())
        assert abs(total - 1.0) < 1e-5

    def test_forward_produces_output(self):
        models = {
            "a": _DummyModel(3, bias=0),
            "b": _DummyModel(3, bias=1),
        }
        data = _make_data()
        ensemble = EnsembleModel.from_validation(models, [data])
        result = ensemble(data)
        assert result.shape == (20, 3)

    def test_better_model_gets_higher_weight(self):
        good = _DummyModel(3, bias=0)
        bad = _DummyModel(3, bias=2)
        data = _make_data(n_edges=50, n_classes=3)
        data.y_multi = torch.zeros(50, dtype=torch.long)

        ensemble = EnsembleModel.from_validation({"good": good, "bad": bad}, [data])
        assert ensemble._weights["good"] > ensemble._weights["bad"]
