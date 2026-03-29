"""Tests for GraphSAGE and GAT edge classifiers."""

from __future__ import annotations

import torch
import pytest
from torch_geometric.data import Data

from src.models.graphsage import GraphSAGEModel
from src.models.gat import GATModel
from src.models.base import BaseNIDSModel


# ── Fixtures ──────────────────────────────────────────────────────────────────

NUM_NODES = 8
NUM_EDGES = 12
IN_NODE = 5
IN_EDGE = 16
HIDDEN = 64
NUM_CLASSES = 4


def _make_data(n_nodes: int = NUM_NODES, n_edges: int = NUM_EDGES) -> Data:
    """Create a random PyG Data object for testing."""
    torch.manual_seed(0)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    x = torch.randn(n_nodes, IN_NODE)
    edge_attr = torch.randn(n_edges, IN_EDGE)
    y_multi = torch.randint(0, NUM_CLASSES, (n_edges,))
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y_multi=y_multi, num_nodes=n_nodes)


@pytest.fixture
def sage_model():
    return GraphSAGEModel(
        in_node_channels=IN_NODE,
        in_edge_channels=IN_EDGE,
        hidden_dim=HIDDEN,
        num_classes=NUM_CLASSES,
        num_layers=3,
        dropout=0.0,
    )


@pytest.fixture
def gat_model():
    return GATModel(
        in_node_channels=IN_NODE,
        in_edge_channels=IN_EDGE,
        hidden_dim=HIDDEN,
        num_classes=NUM_CLASSES,
        num_layers=3,
        num_heads=4,
        dropout=0.0,
    )


# ── Inheritance ───────────────────────────────────────────────────────────────

class TestInheritance:
    def test_graphsage_is_base_nids_model(self, sage_model):
        assert isinstance(sage_model, BaseNIDSModel)

    def test_gat_is_base_nids_model(self, gat_model):
        assert isinstance(gat_model, BaseNIDSModel)


# ── Output shape ──────────────────────────────────────────────────────────────

class TestOutputShape:
    def test_graphsage_logits_shape(self, sage_model):
        data = _make_data()
        logits = sage_model(data)
        assert logits.shape == (NUM_EDGES, NUM_CLASSES)

    def test_gat_logits_shape(self, gat_model):
        data = _make_data()
        logits = gat_model(data)
        assert logits.shape == (NUM_EDGES, NUM_CLASSES)

    def test_graphsage_single_edge(self, sage_model):
        data = _make_data(n_nodes=2, n_edges=1)
        logits = sage_model(data)
        assert logits.shape == (1, NUM_CLASSES)

    def test_gat_single_edge(self, gat_model):
        data = _make_data(n_nodes=2, n_edges=1)
        logits = gat_model(data)
        assert logits.shape == (1, NUM_CLASSES)


# ── predict_edges / predict_proba ─────────────────────────────────────────────

class TestPredictHelpers:
    def test_predict_edges_shape(self, sage_model):
        data = _make_data()
        preds = sage_model.predict_edges(data)
        assert preds.shape == (NUM_EDGES,)
        assert preds.dtype == torch.long

    def test_predict_proba_sums_to_one(self, sage_model):
        data = _make_data()
        proba = sage_model.predict_proba(data)
        assert proba.shape == (NUM_EDGES, NUM_CLASSES)
        torch.testing.assert_close(proba.sum(dim=-1), torch.ones(NUM_EDGES))

    def test_gat_predict_edges_shape(self, gat_model):
        data = _make_data()
        preds = gat_model.predict_edges(data)
        assert preds.shape == (NUM_EDGES,)

    def test_predict_edges_no_grad(self, sage_model):
        """predict_edges must not accumulate gradients."""
        data = _make_data()
        preds = sage_model.predict_edges(data)
        assert not preds.requires_grad


# ── Gradient flow ─────────────────────────────────────────────────────────────

class TestGradientFlow:
    def test_graphsage_backward(self, sage_model):
        data = _make_data()
        logits = sage_model(data)
        loss = logits.sum()
        loss.backward()
        # At least one parameter should have a non-None gradient
        has_grad = any(p.grad is not None for p in sage_model.parameters())
        assert has_grad

    def test_gat_backward(self, gat_model):
        data = _make_data()
        logits = gat_model(data)
        loss = logits.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in gat_model.parameters())
        assert has_grad


# ── GAT configuration constraints ─────────────────────────────────────────────

class TestGATConfig:
    def test_hidden_not_divisible_by_heads_raises(self):
        with pytest.raises(ValueError, match="divisible by num_heads"):
            GATModel(
                in_node_channels=IN_NODE,
                in_edge_channels=IN_EDGE,
                hidden_dim=65,  # not divisible by 4
                num_heads=4,
            )

    def test_different_head_counts(self):
        for heads in (1, 2, 8):
            model = GATModel(
                in_node_channels=IN_NODE,
                in_edge_channels=IN_EDGE,
                hidden_dim=64,
                num_heads=heads,
                num_classes=NUM_CLASSES,
            )
            data = _make_data()
            out = model(data)
            assert out.shape == (NUM_EDGES, NUM_CLASSES)


# ── Train / eval mode ─────────────────────────────────────────────────────────

class TestTrainEvalMode:
    def test_graphsage_deterministic_in_eval_mode(self, sage_model):
        sage_model.eval()
        data = _make_data()
        out1 = sage_model(data)
        out2 = sage_model(data)
        torch.testing.assert_close(out1, out2)

    def test_gat_deterministic_in_eval_mode(self, gat_model):
        gat_model.eval()
        data = _make_data()
        out1 = gat_model(data)
        out2 = gat_model(data)
        torch.testing.assert_close(out1, out2)
