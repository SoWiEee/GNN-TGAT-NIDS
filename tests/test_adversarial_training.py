"""Tests for adversarial training defense module."""
from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.defense.adversarial_training import AdvTrainingConfig, adversarial_train_epoch
from src.models.graphsage import GraphSAGEModel

NUM_NODES = 8
NUM_EDGES = 12
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


def _make_data_list(n: int = 3) -> list[Data]:
    torch.manual_seed(42)
    graphs = []
    for _ in range(n):
        edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))
        x = torch.randn(NUM_NODES, IN_NODE)
        edge_attr = torch.randn(NUM_EDGES, IN_EDGE)
        y_multi = torch.randint(0, NUM_CLASSES, (NUM_EDGES,))
        graphs.append(Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            y_multi=y_multi, num_nodes=NUM_NODES,
        ))
    return graphs


class TestAdvTrainingConfig:
    def test_defaults(self):
        cfg = AdvTrainingConfig()
        assert cfg.epsilon == 0.1
        assert cfg.steps == 10
        assert cfg.ratio == 0.3
        assert cfg.alpha is None

    def test_custom(self):
        cfg = AdvTrainingConfig(epsilon=0.05, steps=5, ratio=0.5)
        assert cfg.epsilon == 0.05
        assert cfg.ratio == 0.5


class TestAdversarialTrainEpoch:
    def test_returns_loss(self):
        model = _make_model()
        loader = DataLoader(_make_data_list(), batch_size=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        cfg = AdvTrainingConfig(epsilon=0.05, steps=2, ratio=0.3)

        loss = adversarial_train_epoch(
            model, loader, optimizer, criterion,
            device=torch.device("cpu"), adv_cfg=cfg,
        )
        assert isinstance(loss, float)
        assert loss > 0.0

    def test_zero_ratio_is_clean_training(self):
        model = _make_model()
        loader = DataLoader(_make_data_list(), batch_size=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        cfg = AdvTrainingConfig(ratio=0.0)

        loss = adversarial_train_epoch(
            model, loader, optimizer, criterion,
            device=torch.device("cpu"), adv_cfg=cfg,
        )
        assert isinstance(loss, float)

    def test_model_params_updated(self):
        model = _make_model()
        loader = DataLoader(_make_data_list(1), batch_size=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        cfg = AdvTrainingConfig(epsilon=0.05, steps=2, ratio=0.5)

        params_before = {
            name: p.clone() for name, p in model.named_parameters() if p.requires_grad
        }

        adversarial_train_epoch(
            model, loader, optimizer, criterion,
            device=torch.device("cpu"), adv_cfg=cfg,
        )

        any_changed = any(
            not torch.equal(params_before[name], p)
            for name, p in model.named_parameters() if p.requires_grad
        )
        assert any_changed
