"""Memory Poisoning Attack for temporal GNN models (TGN).

Crafts adversarial temporal events that corrupt TGN's persistent per-node
memory, degrading classification of subsequent legitimate flows.

Strategy:
    1. Identify target attack nodes whose flows are currently detected.
    2. Inject synthetic "poisoning" events *before* the target batch, carrying
       benign-looking features that shift the target node's GRU memory toward
       a benign representation.
    3. Re-run inference on the target batch with corrupted memory.

Unlike feature-space attacks (C-PGD) or structure attacks (EdgeInjection),
this attack exploits TGN's *stateful* design: memory written by batch t
affects classification at batch t+1.

Usage:
    attack = MemoryPoisoningAttack(n_poison=20)
    result = attack.generate(model, data)
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from torch_geometric.data import TemporalData

from src.attack.base import BaseAttack
from src.attack.constraints import ConstraintSet
from src.models.base import BaseNIDSModel

logger = logging.getLogger(__name__)

_BENIGN_CLASS = 0


class MemoryPoisoningAttack(BaseAttack):
    """Adversarial attack that poisons TGN memory with synthetic events.

    Parameters
    ----------
    n_poison:
        Number of poisoning events to inject per target node.
    poison_strategy:
        How to craft poison features:
        - ``"benign_mean"`` — use mean of benign edge features
        - ``"random_benign"`` — sample from benign feature distribution
    scaler_path:
        Path to scaler for constraint checking.
    constraint_set:
        Pre-built ConstraintSet.
    target_split:
        Data split to attack.
    memory_reset_policy:
        When to reset memory:
        - ``"before_each_attack"`` — reset before each attack run
        - ``"none"`` — attack with existing memory state
    """

    def __init__(
        self,
        n_poison: int = 20,
        poison_strategy: str = "benign_mean",
        scaler_path: str | None = None,
        constraint_set: ConstraintSet | None = None,
        target_split: str = "test",
        memory_reset_policy: str = "before_each_attack",
        **kwargs,
    ) -> None:
        self.n_poison = n_poison
        self.poison_strategy = poison_strategy
        self.target_split = target_split
        self.memory_reset_policy = memory_reset_policy

        if constraint_set is not None:
            self.cs = constraint_set
        elif scaler_path is not None:
            self.cs = ConstraintSet.from_scaler(scaler_path)
        else:
            self.cs = ConstraintSet()

    def generate(self, model: BaseNIDSModel, data: TemporalData, **kwargs) -> TemporalData:
        """Inject poison events into the temporal data stream.

        Poison events are prepended to the batch with timestamps slightly
        before the earliest event, targeting nodes involved in attack flows.

        Returns a new TemporalData with poison events prepended.
        """
        n_poison = int(kwargs.get("n_poison", self.n_poison))
        model.eval()

        src, dst, t, msg = data.src, data.dst, data.t, data.msg
        y = data.y if hasattr(data, "y") and data.y is not None else None

        with torch.no_grad():
            logits = model(data)
            preds = logits.argmax(dim=-1)

        attack_mask = preds > 0
        if not attack_mask.any():
            return data

        target_nodes = torch.unique(
            torch.cat([src[attack_mask], dst[attack_mask]])
        )

        benign_stats = self._compute_benign_stats(msg, y if y is not None else preds)

        poison_events = self._craft_poison_events(
            target_nodes, t, msg, benign_stats, n_poison,
        )

        if poison_events is None:
            return data

        p_src, p_dst, p_t, p_msg = poison_events

        adv_data = TemporalData(
            src=torch.cat([p_src, src]),
            dst=torch.cat([p_dst, dst]),
            t=torch.cat([p_t, t]),
            msg=torch.cat([p_msg, msg]),
        )
        if y is not None:
            p_y = torch.zeros(len(p_src), dtype=y.dtype)
            adv_data.y = torch.cat([p_y, y])

        return adv_data

    def constraint_check(self, x_adv: np.ndarray, attack_label: int | None = None) -> bool:
        return self.cs.check(x_adv, attack_label=attack_label)

    def _compute_benign_stats(
        self, msg: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        benign_mask = labels == _BENIGN_CLASS
        if not benign_mask.any():
            return {
                "mean": torch.zeros(msg.shape[1]),
                "std": torch.ones(msg.shape[1]),
            }
        benign_feats = msg[benign_mask]
        return {
            "mean": benign_feats.mean(dim=0),
            "std": benign_feats.std(dim=0) + 1e-8,
        }

    def _craft_poison_events(
        self,
        target_nodes: torch.Tensor,
        t: torch.Tensor,
        msg: torch.Tensor,
        benign_stats: dict[str, torch.Tensor],
        n_poison: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Create synthetic poison events targeting specific nodes."""
        t_min = t.min().item()
        feat_dim = msg.shape[1]
        num_targets = len(target_nodes)

        total_poison = num_targets * n_poison
        if total_poison == 0:
            return None

        p_src_list = []
        p_dst_list = []
        p_t_list = []

        for node_id in target_nodes:
            for i in range(n_poison):
                p_src_list.append(node_id.item())
                p_dst_list.append(node_id.item())
                p_t_list.append(t_min - (n_poison - i) * 0.001)

        p_src = torch.tensor(p_src_list, dtype=torch.long)
        p_dst = torch.tensor(p_dst_list, dtype=torch.long)
        p_t = torch.tensor(p_t_list, dtype=torch.float32)

        if self.poison_strategy == "benign_mean":
            p_msg = benign_stats["mean"].unsqueeze(0).expand(total_poison, -1).clone()
        else:
            noise = torch.randn(total_poison, feat_dim) * 0.3
            p_msg = benign_stats["mean"].unsqueeze(0) + noise * benign_stats["std"].unsqueeze(0)

        return p_src, p_dst, p_t, p_msg

    def attack_success_rate(
        self, model: BaseNIDSModel, data: TemporalData, **kwargs
    ) -> dict[str, float]:
        """Compute ASR: fraction of attack events evaded after memory poisoning."""
        if hasattr(model, "reset_memory") and self.memory_reset_policy == "before_each_attack":
            model.reset_memory()

        with torch.no_grad():
            orig_preds = model(data).argmax(dim=-1)

        if hasattr(model, "reset_memory") and self.memory_reset_policy == "before_each_attack":
            model.reset_memory()

        adv_data = self.generate(model, data, **kwargs)
        n_orig = len(data.src)
        n_poison = len(adv_data.src) - n_orig

        if hasattr(model, "update_state") and n_poison > 0:
            model.update_state(
                adv_data.src[:n_poison],
                adv_data.dst[:n_poison],
                adv_data.t[:n_poison],
                adv_data.msg[:n_poison],
            )

        orig_batch = TemporalData(
            src=adv_data.src[n_poison:],
            dst=adv_data.dst[n_poison:],
            t=adv_data.t[n_poison:],
            msg=adv_data.msg[n_poison:],
        )

        with torch.no_grad():
            adv_preds = model(orig_batch).argmax(dim=-1)

        attack_mask = orig_preds > 0
        n_attack = int(attack_mask.sum())
        if n_attack == 0:
            return {"asr": 0.0, "n_attack_edges": 0, "n_evaded": 0, "n_poison_events": n_poison}

        evaded = int(((orig_preds > 0) & (adv_preds == 0)).sum())
        return {
            "asr": round(evaded / n_attack, 4),
            "n_attack_edges": n_attack,
            "n_evaded": evaded,
            "n_poison_events": n_poison,
        }
