"""Tests for TGN model and supporting components."""
from __future__ import annotations

import torch
from torch_geometric.data import TemporalData

from src.models.base import BaseNIDSModel
from src.models.tgn import (
    LastNeighborLoader,
    TemporalAttention,
    TGNModel,
    TimeEncoder,
)

NUM_NODES = 20
MSG_DIM = 16
MEMORY_DIM = 32
HIDDEN = 64
TIME_DIM = 16
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
        out = enc(torch.randn(10))
        assert out.shape == (10, TIME_DIM)

    def test_cosine_bounded(self):
        enc = TimeEncoder(TIME_DIM)
        out = enc(torch.randn(100))
        assert out.abs().max() <= 1.0 + 1e-6


class TestLastNeighborLoader:
    def test_insert_and_query(self):
        loader = LastNeighborLoader(NUM_NODES, size=3, msg_dim=MSG_DIM)
        src = torch.tensor([0, 1])
        dst = torch.tensor([2, 3])
        t = torch.tensor([1.0, 2.0])
        msg = torch.randn(2, MSG_DIM)
        loader.insert(src, dst, t, msg)

        nbr_ids, nbr_t, nbr_msg = loader.query(torch.tensor([0]))
        assert nbr_ids.shape == (1, 3)
        assert (nbr_ids[0] >= 0).any()

    def test_reset(self):
        loader = LastNeighborLoader(NUM_NODES, size=3, msg_dim=MSG_DIM)
        loader.insert(
            torch.tensor([0]), torch.tensor([1]),
            torch.tensor([1.0]), torch.randn(1, MSG_DIM),
        )
        loader.reset()
        nbr_ids, _, _ = loader.query(torch.tensor([0]))
        assert (nbr_ids == -1).all()


class TestTemporalAttention:
    def test_output_shape(self):
        attn = TemporalAttention(
            memory_dim=MEMORY_DIM, time_dim=TIME_DIM,
            msg_dim=MSG_DIM, attn_dim=HIDDEN,
        )
        N, k = 5, N_NEIGHBORS
        out = attn(
            memory_u=torch.randn(N, MEMORY_DIM),
            t_ref=torch.randn(N),
            nbr_mem=torch.randn(N, k, MEMORY_DIM),
            nbr_t=torch.randn(N, k),
            nbr_msg=torch.randn(N, k, MSG_DIM),
            valid=torch.ones(N, k, dtype=torch.bool),
        )
        assert out.shape == (N, HIDDEN)

    def test_all_invalid_neighbors(self):
        attn = TemporalAttention(
            memory_dim=MEMORY_DIM, time_dim=TIME_DIM,
            msg_dim=MSG_DIM, attn_dim=HIDDEN,
        )
        N, k = 3, N_NEIGHBORS
        out = attn(
            memory_u=torch.randn(N, MEMORY_DIM),
            t_ref=torch.randn(N),
            nbr_mem=torch.zeros(N, k, MEMORY_DIM),
            nbr_t=torch.zeros(N, k),
            nbr_msg=torch.zeros(N, k, MSG_DIM),
            valid=torch.zeros(N, k, dtype=torch.bool),
        )
        assert out.shape == (N, HIDDEN)
        assert torch.isfinite(out).all()


class TestTGNModel:
    def test_inherits_base(self):
        model = TGNModel(
            num_nodes=NUM_NODES, raw_msg_dim=MSG_DIM,
            memory_dim=MEMORY_DIM, hidden_dim=HIDDEN,
            num_classes=NUM_CLASSES, num_neighbors=N_NEIGHBORS,
        )
        assert isinstance(model, BaseNIDSModel)

    def test_forward_shape(self):
        model = TGNModel(
            num_nodes=NUM_NODES, raw_msg_dim=MSG_DIM,
            memory_dim=MEMORY_DIM, hidden_dim=HIDDEN,
            num_classes=NUM_CLASSES, num_neighbors=N_NEIGHBORS,
        )
        batch = _make_temporal_batch()
        logits = model(batch)
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_identity_embedding(self):
        model = TGNModel(
            num_nodes=NUM_NODES, raw_msg_dim=MSG_DIM,
            memory_dim=MEMORY_DIM, hidden_dim=HIDDEN,
            num_classes=NUM_CLASSES, num_neighbors=N_NEIGHBORS,
            embedding_module="identity",
        )
        batch = _make_temporal_batch()
        logits = model(batch)
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_update_and_reset(self):
        model = TGNModel(
            num_nodes=NUM_NODES, raw_msg_dim=MSG_DIM,
            memory_dim=MEMORY_DIM, hidden_dim=HIDDEN,
            num_classes=NUM_CLASSES, num_neighbors=N_NEIGHBORS,
        )
        batch = _make_temporal_batch()
        model(batch)
        model.update_state(batch.src, batch.dst, batch.t, batch.msg)

        nbr_ids, _, _ = model.neighbor_loader.query(batch.src[:1])
        assert (nbr_ids >= 0).any()

        model.reset_memory()
        nbr_ids, _, _ = model.neighbor_loader.query(batch.src[:1])
        assert (nbr_ids == -1).all()

    def test_backward(self):
        model = TGNModel(
            num_nodes=NUM_NODES, raw_msg_dim=MSG_DIM,
            memory_dim=MEMORY_DIM, hidden_dim=HIDDEN,
            num_classes=NUM_CLASSES, num_neighbors=N_NEIGHBORS,
        )
        batch = _make_temporal_batch()
        logits = model(batch)
        logits.sum().backward()
        has_grad = any(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )
        assert has_grad
