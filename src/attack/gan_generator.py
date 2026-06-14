"""WGAN-GP adversarial flow generator for NIDS evasion.

Trains a generator to produce constraint-satisfying adversarial flows that
evade the target GNN classifier. Uses Wasserstein loss with gradient penalty
(WGAN-GP) for stable training, with early stopping for instability detection.

The generator learns to map latent vectors to flow feature perturbations
that (1) cause misclassification, (2) satisfy all protocol constraints.

Usage:
    attack = GANAttack(latent_dim=128, scaler_path="data/processed/static/scaler.json")
    adv_data = attack.generate(model, data)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.attack.base import BaseAttack
from src.attack.constraints import ConstraintSet
from src.models.base import BaseNIDSModel

logger = logging.getLogger(__name__)

_BENIGN_CLASS = 0


class _Generator(nn.Module):
    """Maps latent z + original features to adversarial perturbation delta."""

    def __init__(self, latent_dim: int, feat_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + feat_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, feat_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, x_orig], dim=-1))


class _Critic(nn.Module):
    """WGAN-GP critic — outputs unbounded scalar (no sigmoid)."""

    def __init__(self, feat_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GANAttack(BaseAttack):
    """WGAN-GP based adversarial flow generator.

    Parameters
    ----------
    latent_dim:
        Dimension of the latent noise vector.
    gp_weight:
        Gradient penalty coefficient for WGAN-GP.
    critic_iters:
        Number of critic updates per generator update.
    max_iter:
        Maximum training iterations.
    epsilon:
        Maximum perturbation magnitude (scales tanh output).
    early_stop:
        Early stopping config dict with keys: critic_loss_range,
        min_csr, cosine_similarity, patience.
    scaler_path:
        Path to scaler for constraint checking.
    constraint_set:
        Pre-built ConstraintSet.
    target_split:
        Data split to attack.
    memory_reset_policy:
        Memory reset policy for temporal models.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        gp_weight: float = 10.0,
        critic_iters: int = 5,
        max_iter: int = 50000,
        epsilon: float = 0.15,
        early_stop: dict | None = None,
        scaler_path: str | Path | None = None,
        constraint_set: ConstraintSet | None = None,
        target_split: str = "test",
        memory_reset_policy: str = "before_each_attack",
        **kwargs,
    ) -> None:
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight
        self.critic_iters = critic_iters
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.target_split = target_split
        self.memory_reset_policy = memory_reset_policy

        self.early_stop_cfg = early_stop or {
            "critic_loss_range": 10.0,
            "min_csr": 0.30,
            "cosine_similarity": 0.95,
            "patience": 1000,
        }

        if constraint_set is not None:
            self.cs = constraint_set
        elif scaler_path is not None:
            self.cs = ConstraintSet.from_scaler(scaler_path)
        else:
            self.cs = ConstraintSet()

        self._scaler = None
        if scaler_path is not None:
            self._scaler = self._load_scaler(Path(scaler_path))

    def generate(self, model: BaseNIDSModel, data: Data, **kwargs) -> Data:
        """Train a GAN to generate adversarial perturbations, then apply them."""
        model.eval()
        edge_attr = data.edge_attr
        n_features = edge_attr.shape[1]

        with torch.no_grad():
            preds = model(data).argmax(dim=-1)

        attack_mask = preds > 0
        if not attack_mask.any():
            return data

        attack_feats = edge_attr[attack_mask].detach()
        n_attack = attack_feats.shape[0]

        generator = _Generator(self.latent_dim, n_features)
        critic = _Critic(n_features)

        opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
        opt_c = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))

        critic_losses: list[float] = []
        low_csr_count = 0

        for iteration in range(self.max_iter):
            # Train critic
            for _ in range(self.critic_iters):
                z = torch.randn(n_attack, self.latent_dim)
                with torch.no_grad():
                    delta = generator(z, attack_feats) * self.epsilon
                x_fake = attack_feats + delta

                c_real = critic(attack_feats).mean()
                c_fake = critic(x_fake).mean()

                gp = self._gradient_penalty(critic, attack_feats, x_fake)
                c_loss = c_fake - c_real + self.gp_weight * gp

                opt_c.zero_grad()
                c_loss.backward()
                opt_c.step()

            critic_losses.append(c_loss.item())

            # Train generator
            z = torch.randn(n_attack, self.latent_dim)
            delta = generator(z, attack_feats) * self.epsilon
            x_fake = attack_feats + delta

            g_loss = -critic(x_fake).mean()

            # Evasion loss: encourage misclassification to benign
            adv_data = data.clone()
            adv_attr = edge_attr.clone()
            adv_attr[attack_mask] = x_fake
            adv_data.edge_attr = adv_attr
            logits = model(adv_data)
            attack_logits = logits[attack_mask]
            target = torch.full((n_attack,), _BENIGN_CLASS, dtype=torch.long)
            evasion_loss = nn.functional.cross_entropy(attack_logits, target)

            total_g_loss = g_loss + evasion_loss

            opt_g.zero_grad()
            total_g_loss.backward()
            opt_g.step()

            # Early stopping checks
            if self._should_early_stop(
                iteration, critic_losses, x_fake.detach(), low_csr_count
            ):
                logger.info("GAN early stopping at iteration %d", iteration)
                break

            if (iteration + 1) % 5000 == 0:
                logger.info(
                    "GAN iter %d/%d  c_loss=%.4f  g_loss=%.4f  evasion=%.4f",
                    iteration + 1, self.max_iter, c_loss.item(),
                    g_loss.item(), evasion_loss.item(),
                )

        # Generate final adversarial examples
        generator.eval()
        with torch.no_grad():
            z = torch.randn(n_attack, self.latent_dim)
            delta = generator(z, attack_feats) * self.epsilon
            x_adv = attack_feats + delta

        # Constraint projection
        if self._scaler is not None:
            x_adv_np = x_adv.numpy()
            for i in range(len(x_adv_np)):
                x_raw = self._inverse_transform(x_adv_np[i:i+1])
                x_raw = self.cs.project(x_raw.squeeze())
                x_adv_np[i] = self._transform(x_raw.reshape(1, -1)).squeeze()
            x_adv = torch.from_numpy(x_adv_np).float()

        adv_data = data.clone()
        adv_attr = edge_attr.clone()
        adv_attr[attack_mask] = x_adv
        adv_data.edge_attr = adv_attr
        return adv_data

    def constraint_check(self, x_adv: np.ndarray, attack_label: int | None = None) -> bool:
        return self.cs.check(x_adv, attack_label=attack_label)

    def _gradient_penalty(
        self, critic: _Critic, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        alpha = torch.rand(real.shape[0], 1)
        interpolated = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
        c_interp = critic(interpolated)
        grad = torch.autograd.grad(
            outputs=c_interp,
            inputs=interpolated,
            grad_outputs=torch.ones_like(c_interp),
            create_graph=True,
            retain_graph=True,
        )[0]
        return ((grad.norm(2, dim=1) - 1) ** 2).mean()

    def _should_early_stop(
        self,
        iteration: int,
        critic_losses: list[float],
        x_fake: torch.Tensor,
        low_csr_count: int,
    ) -> bool:
        patience = self.early_stop_cfg.get("patience", 1000)
        if iteration < patience:
            return False

        recent = critic_losses[-500:] if len(critic_losses) >= 500 else critic_losses
        loss_range = max(recent) - min(recent)
        if loss_range > self.early_stop_cfg.get("critic_loss_range", 10.0):
            logger.warning("Critic loss range %.2f exceeds threshold", loss_range)
            return True

        if x_fake.shape[0] > 1:
            normed = nn.functional.normalize(x_fake, dim=1)
            sim_matrix = normed @ normed.T
            mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool)
            mean_sim = sim_matrix[mask].mean().item()
            if mean_sim > self.early_stop_cfg.get("cosine_similarity", 0.95):
                logger.warning("Mode collapse detected (mean cosine sim=%.4f)", mean_sim)
                return True

        return False

    @staticmethod
    def _load_scaler(path: Path):
        from sklearn.preprocessing import StandardScaler

        json_path = path if path.suffix == ".json" else path.parent / "scaler.json"
        pkl_path = path if path.suffix == ".pkl" else path.parent / "scaler.pkl"

        if json_path.exists():
            import json
            params = json.loads(json_path.read_text())
            scaler = StandardScaler()
            scaler.mean_ = np.array(params["mean_"])
            scaler.scale_ = np.array(params["scale_"])
            scaler.n_features_in_ = len(scaler.mean_)
            return scaler

        if pkl_path.exists():
            import pickle
            with open(pkl_path, "rb") as f:
                return pickle.load(f)

        raise FileNotFoundError(f"Scaler not found at {path}")

    def _inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (x * self._scaler.scale_) + self._scaler.mean_

    def _transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self._scaler.mean_) / self._scaler.scale_

    def attack_success_rate(
        self, model: BaseNIDSModel, data: Data, **kwargs
    ) -> dict[str, float]:
        """Compute ASR post-GAN generation."""
        with torch.no_grad():
            orig_preds = model(data).argmax(dim=-1)

        adv_data = self.generate(model, data, **kwargs)

        with torch.no_grad():
            adv_preds = model(adv_data).argmax(dim=-1)

        attack_mask = orig_preds > 0
        n_attack = int(attack_mask.sum())
        if n_attack == 0:
            return {"asr": 0.0, "n_attack_edges": 0, "n_evaded": 0}

        evaded = int(((orig_preds > 0) & (adv_preds == 0)).sum())
        return {
            "asr": round(evaded / n_attack, 4),
            "n_attack_edges": n_attack,
            "n_evaded": evaded,
        }
