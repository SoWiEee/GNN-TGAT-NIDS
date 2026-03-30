"""Constrained PGD (C-PGD) adversarial attack for static NIDS GNNs.

Implements the algorithm from spec.md §3.3.1:

    For t = 1 to T:
        g     ← ∇_x  CrossEntropy(model(x_adv), target=benign)
        x_adv ← x_adv + α · g / (‖g‖₂ + ε_num)   # normalised gradient step
        x_raw ← scaler.inverse_transform(x_adv)     # back to raw scale
        x_raw ← ConstraintSet.project(x_raw)        # enforce protocol constraints
        x_adv ← scaler.transform(x_raw)             # re-normalise
        x_adv ← clip(x_adv, x − ε, x + ε)          # ℓ∞ ball projection

    Return x_adv if ConstraintSet.check(x_adv) else None (CSR = 1.0 gate)

Only adversarial examples with CSR = 1.0 are returned. Flows where the
constraint set cannot be satisfied within the budget are dropped (returned
as None in per-sample mode, excluded from batch output).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from src.attack.base import BaseAttack
from src.attack.constraints import ConstraintSet
from src.models.base import BaseNIDSModel

logger = logging.getLogger(__name__)

_BENIGN_CLASS = 0  # index 0 is always "Benign" in NF-UNSW-NB15-v2 label encoding


class CPGDAttack(BaseAttack):
    """Constrained Projected Gradient Descent attack.

    Operates on **edge feature vectors** in normalised space.
    After each gradient step, features are inverse-transformed to raw scale,
    projected onto the constraint set, then re-normalised before the next step.

    Parameters
    ----------
    epsilon:
        Maximum ℓ∞ perturbation budget in normalised space.
    steps:
        Number of PGD update steps.
    alpha:
        Step size. Defaults to ``epsilon / steps * 2.5`` (rule-of-thumb that
        gives roughly 60 % of the budget per pass).
    scaler_path:
        Path to ``scaler.json`` or ``scaler.pkl`` produced by
        :func:`src.data.static_builder.build_static_graphs`.
        If None, constraint projection operates in normalised space only.
    constraint_set:
        Pre-built :class:`ConstraintSet`.  Takes priority over
        ``scaler_path`` when provided.
    random_init:
        Whether to initialise x_adv with uniform noise in [-ε, ε].
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        steps: int = 40,
        alpha: float | None = None,
        scaler_path: str | Path | None = None,
        constraint_set: ConstraintSet | None = None,
        random_init: bool = True,
    ) -> None:
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha if alpha is not None else epsilon / max(steps, 1) * 2.5
        self.random_init = random_init

        # Build or accept constraint set
        if constraint_set is not None:
            self.cs = constraint_set
        elif scaler_path is not None:
            self.cs = ConstraintSet.from_scaler(scaler_path)
        else:
            self.cs = ConstraintSet()  # default bounds (unconstrained in practice)

        # Load scaler for inverse_transform / transform cycle
        self._scaler = None
        if scaler_path is not None:
            self._scaler = self._load_scaler(Path(scaler_path))

    # ── BaseAttack interface ──────────────────────────────────────────────────

    def generate(self, model: BaseNIDSModel, data: Data, **kwargs) -> Data:
        """Generate adversarial examples for all attack-class edges in ``data``.

        Only edges predicted as attacks (class > 0) and for which a valid
        adversarial perturbation is found (CSR = 1.0) are perturbed.
        Benign-predicted edges are left unchanged.

        Args:
            model: Target static GNN (GraphSAGE or GAT).
            data: PyG ``Data`` with ``edge_attr`` and ``y_multi``.
            **kwargs: Runtime overrides — ``epsilon``, ``steps``, ``alpha``.

        Returns:
            A copy of ``data`` with perturbed ``edge_attr`` for eligible edges.
        """
        epsilon = float(kwargs.get("epsilon", self.epsilon))
        steps = int(kwargs.get("steps", self.steps))
        alpha = float(kwargs.get("alpha", self.alpha))

        model.eval()
        x_orig = data.edge_attr.clone()  # [E, F]
        x_adv = x_orig.clone()

        # Identify attack-predicted edges (these are our targets to perturb)
        with torch.no_grad():
            logits_orig = model(data)
            preds_orig = logits_orig.argmax(dim=-1)  # [E]
        attack_mask = preds_orig > 0  # True for edges predicted as attacks

        if not attack_mask.any():
            return data  # nothing to perturb

        # Perturb each attack edge independently
        for edge_idx in attack_mask.nonzero(as_tuple=True)[0]:
            x_e = x_orig[edge_idx].unsqueeze(0)  # [1, F]
            x_adv_e = self._perturb_single(
                model, data, edge_idx, x_e, epsilon, steps, alpha
            )
            if x_adv_e is not None:
                x_adv[edge_idx] = x_adv_e.squeeze(0)

        adv_data = data.clone()
        adv_data.edge_attr = x_adv
        return adv_data

    def constraint_check(self, x_adv: np.ndarray, attack_label: int | None = None) -> bool:
        """Return True if all constraints are satisfied for a raw-scale vector."""
        return self.cs.check(x_adv, attack_label=attack_label)

    # ── Core PGD loop ─────────────────────────────────────────────────────────

    def _perturb_single(
        self,
        model: BaseNIDSModel,
        data: Data,
        edge_idx: int,
        x_e: torch.Tensor,  # [1, F] normalised
        epsilon: float,
        steps: int,
        alpha: float,
    ) -> torch.Tensor | None:
        """Run C-PGD on a single edge. Returns perturbed vector or None."""
        target = torch.tensor([_BENIGN_CLASS], dtype=torch.long)

        # Random initialisation within ε-ball
        x_adv = x_e.clone()
        if self.random_init:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
            x_adv = x_adv.clamp(x_e - epsilon, x_e + epsilon)

        for _ in range(steps):
            x_adv = x_adv.detach().requires_grad_(True)

            # Swap in the perturbed edge feature, keep graph structure
            edge_attr_mod = data.edge_attr.clone()
            edge_attr_mod = edge_attr_mod.detach()
            edge_attr_mod[edge_idx] = x_adv.squeeze(0)

            mod_data = data.clone()
            mod_data.edge_attr = edge_attr_mod

            logits = model(mod_data)  # [E, C]
            edge_logits = logits[edge_idx].unsqueeze(0)  # [1, C]
            loss = F.cross_entropy(edge_logits, target)
            loss.backward()

            grad = x_adv.grad
            if grad is None:
                break

            # Normalised gradient step
            norm = grad.norm(2) + 1e-8
            x_adv_new = (x_adv + alpha * grad / norm).detach()

            # Constraint projection in raw space (if scaler available)
            if self._scaler is not None:
                x_raw = self._inverse_transform(x_adv_new.numpy())
                x_raw = self.cs.project(x_raw)
                x_adv_new = torch.from_numpy(self._transform(x_raw)).float()

            # ℓ∞ projection back into ε-ball
            x_adv = x_adv_new.clamp(x_e - epsilon, x_e + epsilon)

        # CSR gate: only return if all constraints satisfied
        if self._scaler is not None:
            x_raw_final = self._inverse_transform(x_adv.detach().numpy())
            # check() expects 1-D; squeeze batch dim added in generate()
            if not self.cs.check(x_raw_final.squeeze()):
                return None

        return x_adv.detach()

    # ── Scaler helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _load_scaler(path: Path):
        """Load StandardScaler from JSON (preferred) or pickle (fallback)."""
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
        """Inverse z-score: normalised → raw scale."""
        return (x * self._scaler.scale_) + self._scaler.mean_  # type: ignore[union-attr]

    def _transform(self, x: np.ndarray) -> np.ndarray:
        """Z-score: raw → normalised scale."""
        return (x - self._scaler.mean_) / self._scaler.scale_  # type: ignore[union-attr]

    # ── Batch evaluation helper ───────────────────────────────────────────────

    def attack_success_rate(
        self,
        model: BaseNIDSModel,
        data: Data,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute ASR on a graph: fraction of attack edges evaded post-perturbation.

        Returns dict with keys: ``asr``, ``n_attack_edges``, ``n_evaded``, ``csr``.
        """
        adv_data = self.generate(model, data, **kwargs)

        with torch.no_grad():
            orig_preds = model(data).argmax(dim=-1)
            adv_preds = model(adv_data).argmax(dim=-1)

        attack_mask = orig_preds > 0
        n_attack = int(attack_mask.sum())
        if n_attack == 0:
            return {"asr": 0.0, "n_attack_edges": 0, "n_evaded": 0, "csr": 1.0}

        # Evaded = was attack, now predicted benign
        evaded = ((orig_preds > 0) & (adv_preds == 0)).sum().item()
        asr = evaded / n_attack

        # CSR over perturbed edges
        if self._scaler is not None and adv_data.edge_attr is not None:
            raw = self._inverse_transform(adv_data.edge_attr[attack_mask].numpy())
            csr = float(sum(self.cs.check(row) for row in raw)) / n_attack
        else:
            csr = 1.0

        return {
            "asr": round(asr, 4),
            "n_attack_edges": n_attack,
            "n_evaded": int(evaded),
            "csr": round(csr, 4),
        }
