"""Constrained temporal C-PGD attack for TGAT/TGN message features."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch_geometric.data import TemporalData

from src.attack.base import BaseAttack
from src.models.base import BaseNIDSModel

_BENIGN_CLASS = 0


class ConstrainedTemporalCPGDAttack(BaseAttack):
    """Targeted C-PGD over temporal edge messages.

    The temporal pipeline stores edge features as z-scored and clipped message
    vectors. This attack keeps every perturbation inside both the original
    L-infinity ball and the normalized feature range used by preprocessing.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        steps: int = 40,
        alpha: float | None = None,
        clip_min: float = -3.0,
        clip_max: float = 3.0,
        random_init: bool = True,
    ) -> None:
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha if alpha is not None else epsilon / max(steps, 1) * 2.5
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.random_init = random_init

    def generate(self, model: BaseNIDSModel, data: TemporalData, **kwargs: Any) -> TemporalData:
        epsilon = float(kwargs.get("epsilon", self.epsilon))
        steps = int(kwargs.get("steps", self.steps))
        alpha = float(kwargs.get("alpha", self.alpha))
        clip_min = float(kwargs.get("clip_min", self.clip_min))
        clip_max = float(kwargs.get("clip_max", self.clip_max))

        model.eval()
        with torch.no_grad():
            orig_preds = model(data).argmax(dim=-1)

        attack_idx = (orig_preds > _BENIGN_CLASS).nonzero(as_tuple=True)[0]
        if len(attack_idx) == 0:
            return data

        msg_orig = data.msg.detach()
        selected_orig = msg_orig[attack_idx].clone()
        selected_adv = selected_orig.clone()
        if self.random_init:
            selected_adv = selected_adv + torch.empty_like(selected_adv).uniform_(
                -epsilon, epsilon
            )
            selected_adv = selected_adv.clamp(selected_orig - epsilon, selected_orig + epsilon)
            selected_adv = selected_adv.clamp(clip_min, clip_max)

        target = torch.zeros(len(attack_idx), dtype=torch.long, device=msg_orig.device)
        for _ in range(steps):
            selected_adv = selected_adv.detach().requires_grad_(True)
            msg_mod = msg_orig.clone()
            msg_mod[attack_idx] = selected_adv
            batch = data.clone()
            batch.msg = msg_mod

            logits = model(batch)[attack_idx]
            loss = F.cross_entropy(logits, target)
            grad = torch.autograd.grad(
                loss,
                selected_adv,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if grad is None:
                break

            grad_norm = grad.flatten(1).norm(p=2, dim=1).clamp_min(1e-8).unsqueeze(1)
            next_adv = (selected_adv - alpha * grad / grad_norm).detach()
            next_adv = next_adv.clamp(selected_orig - epsilon, selected_orig + epsilon)
            selected_adv = next_adv.clamp(clip_min, clip_max)

        adv = data.clone()
        adv_msg = msg_orig.clone()
        adv_msg[attack_idx] = selected_adv.detach()
        adv.msg = adv_msg
        return adv

    def constraint_check(self, x_adv, attack_label: int | None = None) -> bool:
        x = torch.as_tensor(x_adv)
        in_bounds = (x >= self.clip_min).all() and (x <= self.clip_max).all()
        return bool(torch.isfinite(x).all() and in_bounds)

    def attack_success_rate(
        self,
        model: BaseNIDSModel,
        data: TemporalData,
        **kwargs: Any,
    ) -> dict[str, float | int]:
        with torch.no_grad():
            orig_preds = model(data).argmax(dim=-1)
        adv_data = self.generate(model, data, **kwargs)
        with torch.no_grad():
            adv_preds = model(adv_data).argmax(dim=-1)

        attack_mask = orig_preds > _BENIGN_CLASS
        n_attack = int(attack_mask.sum())
        if n_attack == 0:
            return {"asr": 0.0, "n_attack_edges": 0, "n_evaded": 0, "csr": 1.0}
        evaded = int(((orig_preds > _BENIGN_CLASS) & (adv_preds == _BENIGN_CLASS)).sum())
        return {
            "asr": round(evaded / n_attack, 4),
            "n_attack_edges": n_attack,
            "n_evaded": evaded,
            "csr": 1.0,
        }
