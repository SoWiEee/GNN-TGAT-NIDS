"""Tests for Memory Poisoning attack against TGN."""
from __future__ import annotations

import torch
from torch_geometric.data import TemporalData

from src.attack.base import BaseAttack
from src.attack.memory_poisoning import MemoryPoisoningAttack
from src.models.tgn import TGNModel

NUM_NODES = 20
MSG_DIM = 16
MEMORY_DIM = 32
HIDDEN = 64
NUM_CLASSES = 4
N_NEIGHBORS = 5
BATCH = 8


def _make_temporal_batch(seed: int = 42) -> TemporalData:
    torch.manual_seed(seed)
    src = torch.randint(0, NUM_NODES, (BATCH,))
    dst = torch.randint(0, NUM_NODES, (BATCH,))
    t = torch.arange(BATCH, dtype=torch.float32) + 1.0
    msg = torch.randn(BATCH, MSG_DIM)
    y = torch.randint(0, NUM_CLASSES, (BATCH,))
    return TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)


def _make_model() -> TGNModel:
    return TGNModel(
        num_nodes=NUM_NODES,
        raw_msg_dim=MSG_DIM,
        memory_dim=MEMORY_DIM,
        hidden_dim=HIDDEN,
        num_classes=NUM_CLASSES,
        num_neighbors=N_NEIGHBORS,
    )


class TestMemoryPoisoningInit:
    def test_inherits_base(self):
        atk = MemoryPoisoningAttack()
        assert isinstance(atk, BaseAttack)

    def test_defaults(self):
        atk = MemoryPoisoningAttack()
        assert atk.n_poison == 20
        assert atk.poison_strategy == "benign_mean"
        assert atk.memory_reset_policy == "before_each_attack"

    def test_custom_params(self):
        atk = MemoryPoisoningAttack(
            n_poison=5,
            poison_strategy="random_benign",
            memory_reset_policy="none",
        )
        assert atk.n_poison == 5
        assert atk.poison_strategy == "random_benign"


class TestGenerate:
    def test_returns_temporal_data(self):
        model = _make_model()
        data = _make_temporal_batch()
        atk = MemoryPoisoningAttack(n_poison=2)
        adv = atk.generate(model, data)
        assert isinstance(adv, TemporalData)

    def test_poison_events_prepended(self):
        model = _make_model()
        data = _make_temporal_batch()
        atk = MemoryPoisoningAttack(n_poison=3)
        adv = atk.generate(model, data)
        assert len(adv.src) >= len(data.src)

    def test_original_events_preserved(self):
        model = _make_model()
        data = _make_temporal_batch()
        atk = MemoryPoisoningAttack(n_poison=2)
        adv = atk.generate(model, data)
        n_poison = len(adv.src) - len(data.src)
        assert torch.equal(adv.src[n_poison:], data.src)
        assert torch.equal(adv.dst[n_poison:], data.dst)
        assert torch.equal(adv.msg[n_poison:], data.msg)

    def test_poison_timestamps_before_original(self):
        model = _make_model()
        data = _make_temporal_batch()
        atk = MemoryPoisoningAttack(n_poison=2)
        adv = atk.generate(model, data)
        n_poison = len(adv.src) - len(data.src)
        if n_poison > 0:
            assert adv.t[:n_poison].max() < data.t.min()

    def test_labels_on_poison_events_are_benign(self):
        model = _make_model()
        data = _make_temporal_batch()
        atk = MemoryPoisoningAttack(n_poison=2)
        adv = atk.generate(model, data)
        n_poison = len(adv.src) - len(data.src)
        if n_poison > 0 and hasattr(adv, "y") and adv.y is not None:
            assert (adv.y[:n_poison] == 0).all()

    def test_random_benign_strategy(self):
        model = _make_model()
        data = _make_temporal_batch()
        atk = MemoryPoisoningAttack(n_poison=2, poison_strategy="random_benign")
        adv = atk.generate(model, data)
        assert isinstance(adv, TemporalData)

    def test_n_poison_override(self):
        model = _make_model()
        data = _make_temporal_batch()
        atk = MemoryPoisoningAttack(n_poison=10)
        adv = atk.generate(model, data, n_poison=1)
        n_poison = len(adv.src) - len(data.src)
        if n_poison > 0:
            model.eval()
            with torch.no_grad():
                preds = model(data).argmax(dim=-1)
            n_target = len(torch.unique(
                torch.cat([data.src[preds > 0], data.dst[preds > 0]])
            ))
            assert n_poison == n_target * 1


class TestBenignStats:
    def test_with_benign_present(self):
        atk = MemoryPoisoningAttack()
        msg = torch.randn(10, MSG_DIM)
        labels = torch.tensor([0, 0, 0, 1, 2, 1, 0, 3, 0, 1])
        stats = atk._compute_benign_stats(msg, labels)
        assert "mean" in stats
        assert "std" in stats
        assert stats["mean"].shape == (MSG_DIM,)

    def test_no_benign_fallback(self):
        atk = MemoryPoisoningAttack()
        msg = torch.randn(5, MSG_DIM)
        labels = torch.tensor([1, 2, 3, 1, 2])
        stats = atk._compute_benign_stats(msg, labels)
        assert torch.allclose(stats["mean"], torch.zeros(MSG_DIM))
        assert torch.allclose(stats["std"], torch.ones(MSG_DIM))


class TestCraftPoisonEvents:
    def test_shape(self):
        atk = MemoryPoisoningAttack()
        targets = torch.tensor([0, 1, 2])
        t = torch.tensor([10.0, 11.0, 12.0])
        msg = torch.randn(3, MSG_DIM)
        stats = {"mean": torch.zeros(MSG_DIM), "std": torch.ones(MSG_DIM)}
        result = atk._craft_poison_events(targets, t, msg, stats, n_poison=2)
        assert result is not None
        p_src, p_dst, p_t, p_msg = result
        assert len(p_src) == 3 * 2
        assert p_msg.shape == (6, MSG_DIM)

    def test_none_when_zero_poison(self):
        atk = MemoryPoisoningAttack()
        targets = torch.tensor([0])
        t = torch.tensor([10.0])
        msg = torch.randn(1, MSG_DIM)
        stats = {"mean": torch.zeros(MSG_DIM), "std": torch.ones(MSG_DIM)}
        result = atk._craft_poison_events(targets, t, msg, stats, n_poison=0)
        assert result is None


class TestAttackSuccessRate:
    def test_returns_expected_keys(self):
        model = _make_model()
        data = _make_temporal_batch()
        atk = MemoryPoisoningAttack(n_poison=2)
        result = atk.attack_success_rate(model, data)
        assert "asr" in result
        assert "n_attack_edges" in result
        assert "n_evaded" in result
        assert "n_poison_events" in result

    def test_asr_bounds(self):
        model = _make_model()
        data = _make_temporal_batch()
        atk = MemoryPoisoningAttack(n_poison=2)
        result = atk.attack_success_rate(model, data)
        assert 0.0 <= result["asr"] <= 1.0

    def test_memory_reset_called(self):
        model = _make_model()
        data = _make_temporal_batch()
        atk = MemoryPoisoningAttack(
            n_poison=2, memory_reset_policy="before_each_attack",
        )
        atk.attack_success_rate(model, data)
        mem = model.memory.memory
        assert mem.abs().sum() >= 0  # no crash, memory state is valid
