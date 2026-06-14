"""Tests for TGAT model and supporting components."""
from __future__ import annotations

import torch
from torch_geometric.data import TemporalData

from src.models.base import BaseNIDSModel
from src.models.tgat import (
    LastNeighborLoader,
    TemporalMultiHeadAttention,
    TGATModel,
    TimeEncoder,
)

NUM_NODES = 20
MSG_DIM = 16
HIDDEN = 64
TIME_DIM = 32
HEADS = 2
NUM_CLASSES = 4
N_NEIGHBORS = 5
BATCH = 8


def _make_temporal_batch() -> TemporalData:
    torch.manual_seed(42)
    src = torch.randint(0, NUM_NODES, (BATCH,))
    dst = torch.randint(0, NUM_NODES, (BATCH,))
    t = torch.arange(BATCH, dtype=torch.float32)
    msg = torch.randn(BATCH, MSG_DIM)
    y = torch.randint(0, NUM_CLASSES, (BATCH,))
    return TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)


class TestTimeEncoder:
    def test_output_shape(self):
        enc = TimeEncoder(TIME_DIM)
        t = torch.randn(10)
        out = enc(t)
        assert out.shape == (10, TIME_DIM)

    def test_output_bounded(self):
        enc = TimeEncoder(TIME_DIM)
        t = torch.randn(100)
        out = enc(t)
        assert out.abs().max() <= 1.0 + 1e-6

    def test_zero_input(self):
        enc = TimeEncoder(TIME_DIM)
        t = torch.zeros(5)
        out = enc(t)
        assert out.shape == (5, TIME_DIM)


class TestLastNeighborLoader:
    def test_insert_and_query(self):
        loader = LastNeighborLoader(NUM_NODES, size=3, msg_dim=MSG_DIM)
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([3, 4, 5])
        t = torch.tensor([1.0, 2.0, 3.0])
        msg = torch.randn(3, MSG_DIM)

        loader.insert(src, dst, t, msg)

        nbr_ids, nbr_t, nbr_msg = loader.query(torch.tensor([0]))
        assert nbr_ids.shape == (1, 3)
        assert (nbr_ids[0] >= 0).any()

    def test_ring_buffer_wraps(self):
        loader = LastNeighborLoader(NUM_NODES, size=2, msg_dim=MSG_DIM)
        for i in range(5):
            loader.insert(
                torch.tensor([0]), torch.tensor([i + 1]),
                torch.tensor([float(i)]), torch.randn(1, MSG_DIM),
            )
        nbr_ids, _, _ = loader.query(torch.tensor([0]))
        assert nbr_ids.shape == (1, 2)
        assert (nbr_ids[0] >= 0).all()

    def test_reset_clears(self):
        loader = LastNeighborLoader(NUM_NODES, size=3, msg_dim=MSG_DIM)
        loader.insert(
            torch.tensor([0]), torch.tensor([1]),
            torch.tensor([1.0]), torch.randn(1, MSG_DIM),
        )
        loader.reset()
        nbr_ids, _, _ = loader.query(torch.tensor([0]))
        assert (nbr_ids == -1).all()


class TestTemporalMultiHeadAttention:
    def test_output_shape(self):
        attn = TemporalMultiHeadAttention(
            feat_dim=HIDDEN, time_dim=TIME_DIM, msg_dim=MSG_DIM,
            out_dim=HIDDEN, heads=HEADS,
        )
        N, k = 5, N_NEIGHBORS
        feat_u = torch.randn(N, HIDDEN)
        t_ref = torch.randn(N)
        nbr_feat = torch.randn(N, k, HIDDEN)
        nbr_t = torch.randn(N, k)
        nbr_msg = torch.randn(N, k, MSG_DIM)
        valid = torch.ones(N, k, dtype=torch.bool)

        out = attn(feat_u, t_ref, nbr_feat, nbr_t, nbr_msg, valid)
        assert out.shape == (N, HIDDEN)

    def test_handles_all_invalid_neighbors(self):
        attn = TemporalMultiHeadAttention(
            feat_dim=HIDDEN, time_dim=TIME_DIM, msg_dim=MSG_DIM,
            out_dim=HIDDEN, heads=HEADS,
        )
        N, k = 3, N_NEIGHBORS
        feat_u = torch.randn(N, HIDDEN)
        t_ref = torch.randn(N)
        nbr_feat = torch.zeros(N, k, HIDDEN)
        nbr_t = torch.zeros(N, k)
        nbr_msg = torch.zeros(N, k, MSG_DIM)
        valid = torch.zeros(N, k, dtype=torch.bool)

        out = attn(feat_u, t_ref, nbr_feat, nbr_t, nbr_msg, valid)
        assert out.shape == (N, HIDDEN)
        assert torch.isfinite(out).all()


class TestTGATModel:
    def test_inherits_base(self):
        model = TGATModel(
            num_nodes=NUM_NODES, raw_msg_dim=MSG_DIM,
            hidden_dim=HIDDEN, n_neighbors=N_NEIGHBORS,
            num_classes=NUM_CLASSES,
        )
        assert isinstance(model, BaseNIDSModel)

    def test_forward_shape(self):
        model = TGATModel(
            num_nodes=NUM_NODES, raw_msg_dim=MSG_DIM,
            hidden_dim=HIDDEN, n_neighbors=N_NEIGHBORS,
            num_classes=NUM_CLASSES, heads=HEADS,
        )
        batch = _make_temporal_batch()
        logits = model(batch)
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_backward(self):
        model = TGATModel(
            num_nodes=NUM_NODES, raw_msg_dim=MSG_DIM,
            hidden_dim=HIDDEN, n_neighbors=N_NEIGHBORS,
            num_classes=NUM_CLASSES,
        )
        batch = _make_temporal_batch()
        logits = model(batch)
        logits.sum().backward()
        has_grad = any(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )
        assert has_grad

    def test_update_state(self):
        model = TGATModel(
            num_nodes=NUM_NODES, raw_msg_dim=MSG_DIM,
            hidden_dim=HIDDEN, n_neighbors=N_NEIGHBORS,
            num_classes=NUM_CLASSES,
        )
        batch = _make_temporal_batch()
        model.update_state(batch.src, batch.dst, batch.t, batch.msg)

        nbr_ids, _, _ = model.neighbor_loader.query(batch.src[:1])
        assert (nbr_ids >= 0).any()

    def test_reset_memory(self):
        model = TGATModel(
            num_nodes=NUM_NODES, raw_msg_dim=MSG_DIM,
            hidden_dim=HIDDEN, n_neighbors=N_NEIGHBORS,
            num_classes=NUM_CLASSES,
        )
        batch = _make_temporal_batch()
        model.update_state(batch.src, batch.dst, batch.t, batch.msg)
        model.reset_memory()

        nbr_ids, _, _ = model.neighbor_loader.query(batch.src[:1])
        assert (nbr_ids == -1).all()
        assert model._node_counts.sum() == 0

    def test_eval_deterministic(self):
        model = TGATModel(
            num_nodes=NUM_NODES, raw_msg_dim=MSG_DIM,
            hidden_dim=HIDDEN, n_neighbors=N_NEIGHBORS,
            num_classes=NUM_CLASSES, dropout=0.0,
        )
        model.eval()
        batch = _make_temporal_batch()
        out1 = model(batch)
        out2 = model(batch)
        torch.testing.assert_close(out1, out2)
