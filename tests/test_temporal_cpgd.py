"""Tests for constrained temporal C-PGD."""
from __future__ import annotations

import torch
from torch_geometric.data import TemporalData

from src.attack.base import BaseAttack
from src.attack.temporal_cpgd import ConstrainedTemporalCPGDAttack
from src.models.tgn import TGNModel


def _batch() -> TemporalData:
    torch.manual_seed(7)
    return TemporalData(
        src=torch.randint(0, 12, (10,)),
        dst=torch.randint(0, 12, (10,)),
        t=torch.arange(10, dtype=torch.float32),
        msg=torch.randn(10, 8).clamp(-2.0, 2.0),
        y=torch.randint(0, 4, (10,)),
    )


def _model() -> TGNModel:
    return TGNModel(
        num_nodes=12,
        raw_msg_dim=8,
        memory_dim=16,
        hidden_dim=24,
        num_classes=4,
        num_neighbors=3,
    )


def test_inherits_base():
    assert isinstance(ConstrainedTemporalCPGDAttack(), BaseAttack)


def test_generate_preserves_shape_and_bounds():
    model = _model()
    data = _batch()
    attack = ConstrainedTemporalCPGDAttack(epsilon=0.1, steps=2, clip_min=-2.0, clip_max=2.0)
    adv = attack.generate(model, data)
    assert adv.msg.shape == data.msg.shape
    assert float((adv.msg - data.msg).abs().max()) <= 0.10001
    assert bool((adv.msg >= -2.0).all())
    assert bool((adv.msg <= 2.0).all())


def test_attack_success_rate_schema():
    model = _model()
    data = _batch()
    attack = ConstrainedTemporalCPGDAttack(epsilon=0.1, steps=2)
    result = attack.attack_success_rate(model, data)
    assert "asr" in result
    assert "n_attack_edges" in result
    assert "n_evaded" in result
    assert "csr" in result
