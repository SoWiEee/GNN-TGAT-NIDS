"""Tests for C-PGD adversarial attack."""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from src.attack.base import BaseAttack
from src.attack.constraints import ConstraintSet
from src.attack.cpgd import CPGDAttack
from src.models.graphsage import GraphSAGEModel

NUM_NODES = 6
EDGE_DIM = 8
HIDDEN = 32
NUM_CLASSES = 4


def _make_graph() -> Data:
    torch.manual_seed(42)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long,
    )
    x = torch.randn(NUM_NODES, EDGE_DIM)
    edge_attr = torch.randn(edge_index.shape[1], EDGE_DIM)
    y_multi = torch.tensor([0, 1, 2, 0, 3])
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y_multi=y_multi)


def _make_model() -> GraphSAGEModel:
    return GraphSAGEModel(
        in_node_channels=EDGE_DIM,
        in_edge_channels=EDGE_DIM,
        hidden_dim=HIDDEN,
        num_classes=NUM_CLASSES,
    )


class TestCPGDInit:
    def test_inherits_base(self):
        atk = CPGDAttack(epsilon=0.1, steps=5)
        assert isinstance(atk, BaseAttack)

    def test_default_alpha(self):
        atk = CPGDAttack(epsilon=0.2, steps=10)
        expected = 0.2 / 10 * 2.5
        assert abs(atk.alpha - expected) < 1e-6

    def test_custom_alpha(self):
        atk = CPGDAttack(epsilon=0.1, steps=5, alpha=0.05)
        assert atk.alpha == 0.05

    def test_constraint_set_passthrough(self):
        cs = ConstraintSet()
        atk = CPGDAttack(constraint_set=cs)
        assert atk.cs is cs


class TestCPGDGenerate:
    def test_returns_data(self):
        model = _make_model()
        data = _make_graph()
        atk = CPGDAttack(epsilon=0.1, steps=2)
        adv = atk.generate(model, data)
        assert isinstance(adv, Data)
        assert adv.edge_attr.shape == data.edge_attr.shape

    def test_benign_edges_unchanged(self):
        model = _make_model()
        data = _make_graph()
        atk = CPGDAttack(epsilon=0.1, steps=2)

        model.eval()
        with torch.no_grad():
            preds = model(data).argmax(dim=-1)
        benign_mask = preds == 0

        adv = atk.generate(model, data)
        if benign_mask.any():
            orig_benign = data.edge_attr[benign_mask]
            adv_benign = adv.edge_attr[benign_mask]
            assert torch.allclose(orig_benign, adv_benign)

    def test_linf_bound(self):
        model = _make_model()
        data = _make_graph()
        eps = 0.15
        atk = CPGDAttack(epsilon=eps, steps=5)
        adv = atk.generate(model, data)
        diff = (adv.edge_attr - data.edge_attr).abs()
        assert diff.max() <= eps + 1e-5

    def test_kwargs_override(self):
        model = _make_model()
        data = _make_graph()
        atk = CPGDAttack(epsilon=0.5, steps=10)
        adv = atk.generate(model, data, epsilon=0.01, steps=1)
        diff = (adv.edge_attr - data.edge_attr).abs()
        assert diff.max() <= 0.01 + 1e-5

    def test_noop_when_no_attacks(self):
        """If all predictions are benign, generate returns original data."""
        model = _make_model()
        data = _make_graph()
        atk = CPGDAttack(epsilon=0.1, steps=2)
        model.eval()
        with torch.no_grad():
            preds = model(data).argmax(dim=-1)

        if (preds == 0).all():
            adv = atk.generate(model, data)
            assert torch.equal(adv.edge_attr, data.edge_attr)


class TestConstraintCheck:
    def test_empty_cs_passes(self):
        cs = ConstraintSet(feature_names=[])
        atk = CPGDAttack(constraint_set=cs)
        assert atk.constraint_check(np.zeros(EDGE_DIM))

    def test_delegates_to_constraint_set(self):
        cs = ConstraintSet(feature_names=[])
        atk = CPGDAttack(constraint_set=cs)
        result = atk.constraint_check(np.ones(EDGE_DIM), attack_label=1)
        assert isinstance(result, bool)


class TestAttackSuccessRate:
    def test_returns_expected_keys(self):
        model = _make_model()
        data = _make_graph()
        atk = CPGDAttack(epsilon=0.1, steps=2)
        result = atk.attack_success_rate(model, data)
        assert "asr" in result
        assert "n_attack_edges" in result
        assert "n_evaded" in result
        assert "csr" in result

    def test_asr_bounds(self):
        model = _make_model()
        data = _make_graph()
        atk = CPGDAttack(epsilon=0.1, steps=2)
        result = atk.attack_success_rate(model, data)
        assert 0.0 <= result["asr"] <= 1.0
        assert 0.0 <= result["csr"] <= 1.0


class TestBatchCSR:
    def test_empty_batch(self):
        atk = CPGDAttack()
        assert atk.batch_csr([]) == 1.0

    def test_all_pass(self):
        cs = ConstraintSet(feature_names=[])
        atk = CPGDAttack(constraint_set=cs)
        vectors = [np.zeros(EDGE_DIM) for _ in range(5)]
        assert atk.batch_csr(vectors) == 1.0
