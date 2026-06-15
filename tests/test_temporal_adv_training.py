"""Tests for temporal adversarial training."""
import torch
from torch_geometric.data import TemporalData

from src.defense.adversarial_training import (
    _fast_pgd_temporal_batch,
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


def _make_batch(n_events=30, msg_dim=8, n_classes=3) -> TemporalData:
    return TemporalData(
        src=torch.randint(0, 10, (n_events,)),
        dst=torch.randint(0, 10, (n_events,)),
        t=torch.arange(n_events, dtype=torch.float),
        msg=torch.randn(n_events, msg_dim),
        y=torch.randint(0, n_classes, (n_events,)),
    )


class TestFastPGDTemporalBatch:
    def test_returns_temporal_data(self):
        model = _DummyTemporalModel(8)
        batch = _make_batch()
        result = _fast_pgd_temporal_batch(model, batch, epsilon=0.1, steps=5)
        assert hasattr(result, "msg")
        assert result.msg.shape == batch.msg.shape

    def test_perturbation_bounded(self):
        model = _DummyTemporalModel(8)
        batch = _make_batch()
        batch.msg = batch.msg.clamp(-1.0, 1.0)
        result = _fast_pgd_temporal_batch(
            model, batch, epsilon=0.1, steps=5, clip_min=-3.0, clip_max=3.0,
        )
        delta = (result.msg - batch.msg).abs()
        assert delta.max() <= 0.1 + 1e-5

    def test_does_not_modify_original(self):
        model = _DummyTemporalModel(8)
        batch = _make_batch()
        original_msg = batch.msg.clone()
        _fast_pgd_temporal_batch(model, batch, epsilon=0.1, steps=5)
        assert torch.allclose(batch.msg, original_msg)

    def test_clipping_respected(self):
        model = _DummyTemporalModel(8)
        batch = _make_batch()
        result = _fast_pgd_temporal_batch(
            model, batch, epsilon=0.5, steps=10, clip_min=-1.0, clip_max=1.0,
        )
        assert result.msg.min() >= -1.0 - 1e-5
        assert result.msg.max() <= 1.0 + 1e-5
