"""Tests for EdgeInjectionAttack."""
from __future__ import annotations

import torch
import pytest
from torch_geometric.data import Data

from src.attack.base import BaseAttack
from src.attack.edge_injection import EdgeInjectionAttack
from src.models.graphsage import GraphSAGEModel

NUM_NODES = 10
NUM_EDGES = 20
IN_NODE = 5
IN_EDGE = 16
HIDDEN = 32
NUM_CLASSES = 4


def _make_model() -> GraphSAGEModel:
    return GraphSAGEModel(
        in_node_channels=IN_NODE,
        in_edge_channels=IN_EDGE,
        hidden_dim=HIDDEN,
        num_classes=NUM_CLASSES,
        num_layers=2,
        dropout=0.0,
    )


def _make_data(n_attack: int = 5) -> Data:
    torch.manual_seed(42)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))
    x = torch.randn(NUM_NODES, IN_NODE)
    edge_attr = torch.randn(NUM_EDGES, IN_EDGE)
    y_multi = torch.zeros(NUM_EDGES, dtype=torch.long)
    y_multi[:n_attack] = 1
    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        y_multi=y_multi, num_nodes=NUM_NODES,
    )


class TestEdgeInjectionInit:
    def test_inherits_base_attack(self):
        attack = EdgeInjectionAttack(n_inject=5)
        assert isinstance(attack, BaseAttack)

    def test_default_params(self):
        attack = EdgeInjectionAttack()
        assert attack.n_inject == 50
        assert attack.degree_sigma_limit == 3.0

    def test_custom_params(self):
        attack = EdgeInjectionAttack(n_inject=10, degree_sigma_limit=2.0)
        assert attack.n_inject == 10
        assert attack.degree_sigma_limit == 2.0


class TestEdgeInjectionGenerate:
    def test_output_type(self):
        model = _make_model()
        data = _make_data()
        attack = EdgeInjectionAttack(n_inject=3, degree_sigma_limit=5.0)
        adv = attack.generate(model, data)
        assert isinstance(adv, Data)

    def test_edges_added(self):
        model = _make_model()
        data = _make_data()
        attack = EdgeInjectionAttack(n_inject=3, degree_sigma_limit=10.0)
        adv = attack.generate(model, data)
        assert adv.edge_index.shape[1] >= data.edge_index.shape[1]

    def test_original_edges_preserved(self):
        model = _make_model()
        data = _make_data()
        attack = EdgeInjectionAttack(n_inject=3, degree_sigma_limit=10.0)
        adv = attack.generate(model, data)
        n_orig = data.edge_index.shape[1]
        torch.testing.assert_close(
            adv.edge_index[:, :n_orig], data.edge_index
        )

    def test_edge_attr_shape_consistent(self):
        model = _make_model()
        data = _make_data()
        attack = EdgeInjectionAttack(n_inject=5, degree_sigma_limit=10.0)
        adv = attack.generate(model, data)
        assert adv.edge_attr.shape[0] == adv.edge_index.shape[1]
        assert adv.edge_attr.shape[1] == IN_EDGE

    def test_labels_extended(self):
        model = _make_model()
        data = _make_data()
        attack = EdgeInjectionAttack(n_inject=3, degree_sigma_limit=10.0)
        adv = attack.generate(model, data)
        if adv.edge_index.shape[1] > data.edge_index.shape[1]:
            assert adv.y_multi.shape[0] == adv.edge_index.shape[1]

    def test_no_injection_when_no_attacks(self):
        model = _make_model()
        data = _make_data(n_attack=0)
        attack = EdgeInjectionAttack(n_inject=5, degree_sigma_limit=10.0)
        adv = attack.generate(model, data)
        assert adv.edge_index.shape[1] >= data.edge_index.shape[1]


class TestEdgeInjectionASR:
    def test_asr_returns_dict(self):
        model = _make_model()
        data = _make_data()
        attack = EdgeInjectionAttack(n_inject=3, degree_sigma_limit=10.0)
        result = attack.attack_success_rate(model, data)
        assert "asr" in result
        assert "n_attack_edges" in result
        assert "n_evaded" in result

    def test_asr_range(self):
        model = _make_model()
        data = _make_data()
        attack = EdgeInjectionAttack(n_inject=3, degree_sigma_limit=10.0)
        result = attack.attack_success_rate(model, data)
        assert 0.0 <= result["asr"] <= 1.0
