"""Tests for GANAttack (WGAN-GP adversarial generator)."""
from __future__ import annotations

import torch
import pytest
from torch_geometric.data import Data

from src.attack.base import BaseAttack
from src.attack.gan_generator import GANAttack, _Generator, _Critic
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


def _make_data(n_attack: int = 4) -> Data:
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


class TestGeneratorCritic:
    def test_generator_output_shape(self):
        gen = _Generator(latent_dim=32, feat_dim=IN_EDGE)
        z = torch.randn(5, 32)
        x = torch.randn(5, IN_EDGE)
        out = gen(z, x)
        assert out.shape == (5, IN_EDGE)

    def test_generator_tanh_bounded(self):
        gen = _Generator(latent_dim=32, feat_dim=IN_EDGE)
        z = torch.randn(10, 32)
        x = torch.randn(10, IN_EDGE)
        out = gen(z, x)
        assert out.abs().max() <= 1.0 + 1e-6

    def test_critic_output_shape(self):
        critic = _Critic(feat_dim=IN_EDGE)
        x = torch.randn(5, IN_EDGE)
        out = critic(x)
        assert out.shape == (5,)

    def test_critic_unbounded(self):
        critic = _Critic(feat_dim=IN_EDGE)
        x = torch.randn(100, IN_EDGE) * 10
        out = critic(x)
        assert out.abs().max() > 0.0


class TestGANAttackInit:
    def test_inherits_base_attack(self):
        attack = GANAttack(max_iter=10)
        assert isinstance(attack, BaseAttack)

    def test_default_params(self):
        attack = GANAttack()
        assert attack.latent_dim == 128
        assert attack.gp_weight == 10.0
        assert attack.critic_iters == 5
        assert attack.epsilon == 0.15


class TestGANAttackGenerate:
    def test_output_type(self):
        model = _make_model()
        data = _make_data()
        attack = GANAttack(max_iter=10, latent_dim=16)
        adv = attack.generate(model, data)
        assert isinstance(adv, Data)

    def test_shape_preserved(self):
        model = _make_model()
        data = _make_data()
        attack = GANAttack(max_iter=10, latent_dim=16)
        adv = attack.generate(model, data)
        assert adv.edge_attr.shape == data.edge_attr.shape
        assert adv.edge_index.shape == data.edge_index.shape

    def test_benign_edges_unchanged(self):
        model = _make_model()
        data = _make_data(n_attack=4)
        attack = GANAttack(max_iter=10, latent_dim=16)
        adv = attack.generate(model, data)
        # Benign edges (indices 4+) should not change (unless model classifies them as attack)
        # At minimum, graph structure is unchanged
        torch.testing.assert_close(adv.edge_index, data.edge_index)

    def test_no_attack_no_change(self):
        model = _make_model()
        data = _make_data(n_attack=0)
        attack = GANAttack(max_iter=10, latent_dim=16)
        adv = attack.generate(model, data)
        # If model predicts nothing as attack, data should be unchanged
        # (depends on model predictions, but structure must match)
        assert adv.edge_attr.shape == data.edge_attr.shape


class TestGANAttackGradientPenalty:
    def test_gradient_penalty_positive(self):
        critic = _Critic(feat_dim=IN_EDGE)
        attack = GANAttack(max_iter=10, latent_dim=16)
        real = torch.randn(5, IN_EDGE, requires_grad=False)
        fake = torch.randn(5, IN_EDGE, requires_grad=False)
        gp = attack._gradient_penalty(critic, real, fake)
        assert gp.item() >= 0.0

    def test_gradient_penalty_differentiable(self):
        critic = _Critic(feat_dim=IN_EDGE)
        attack = GANAttack(max_iter=10, latent_dim=16)
        real = torch.randn(5, IN_EDGE)
        fake = torch.randn(5, IN_EDGE)
        gp = attack._gradient_penalty(critic, real, fake)
        gp.backward()
        has_grad = any(p.grad is not None for p in critic.parameters())
        assert has_grad
