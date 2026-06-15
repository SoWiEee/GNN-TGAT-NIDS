"""Tests for temporal model explainability."""
import torch
from torch_geometric.data import TemporalData

from src.explain.temporal_explainer import (
    explain_temporal_flow,
    explain_temporal_top_alerts,
)


class _DummyTemporalModel(torch.nn.Module):
    def __init__(self, msg_dim: int, n_classes: int = 3):
        super().__init__()
        self.linear = torch.nn.Linear(msg_dim, n_classes)

    def forward(self, batch):
        return self.linear(batch.msg)

    def reset_memory(self):
        pass

    def update_state(self, src, dst, t, msg):
        pass


def _make_batch(n_events=20, msg_dim=8, n_classes=3) -> TemporalData:
    y = torch.zeros(n_events, dtype=torch.long)
    y[5:10] = 1
    y[15:18] = 2
    return TemporalData(
        src=torch.randint(0, 10, (n_events,)),
        dst=torch.randint(0, 10, (n_events,)),
        t=torch.arange(n_events, dtype=torch.float),
        msg=torch.randn(n_events, msg_dim),
        y=y,
    )


class TestExplainTemporalFlow:
    def test_returns_expected_keys(self):
        model = _DummyTemporalModel(8)
        batch = _make_batch()
        result = explain_temporal_flow(model, batch, edge_idx=5, n_steps=3)
        assert "predicted_class" in result
        assert "confidence" in result
        assert "top_features" in result
        assert "feature_attribution" in result
        assert result["method"] == "integrated_gradients_approx"

    def test_attribution_sums_to_one(self):
        model = _DummyTemporalModel(8)
        batch = _make_batch()
        result = explain_temporal_flow(model, batch, edge_idx=5, n_steps=5)
        total = sum(result["feature_attribution"])
        assert abs(total - 1.0) < 0.01

    def test_custom_feature_names(self):
        model = _DummyTemporalModel(8)
        batch = _make_batch()
        names = [f"feat_{i}" for i in range(8)]
        result = explain_temporal_flow(
            model, batch, edge_idx=5, n_steps=3, feature_names=names
        )
        for feat in result["top_features"]:
            assert feat["name"].startswith("feat_")

    def test_confidence_in_range(self):
        model = _DummyTemporalModel(8)
        batch = _make_batch()
        result = explain_temporal_flow(model, batch, edge_idx=5, n_steps=3)
        assert 0.0 <= result["confidence"] <= 1.0


class TestExplainTemporalTopAlerts:
    def test_returns_list(self):
        model = _DummyTemporalModel(8)
        batch = _make_batch()
        results = explain_temporal_top_alerts(model, batch, top_k=3, n_steps=3)
        assert isinstance(results, list)

    def test_respects_top_k(self):
        model = _DummyTemporalModel(8)
        batch = _make_batch()
        results = explain_temporal_top_alerts(model, batch, top_k=2, n_steps=3)
        assert len(results) <= 2

    def test_has_rank_field(self):
        model = _DummyTemporalModel(8)
        batch = _make_batch()
        results = explain_temporal_top_alerts(model, batch, top_k=3, n_steps=3)
        for r in results:
            assert "rank" in r
