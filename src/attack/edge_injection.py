"""Edge Injection Attack for static and temporal NIDS GNNs.

Injects synthetic edges into the graph to evade detection. Each injected edge
connects an existing attack node to a benign-looking node, with features
sampled from benign-class statistics and validated through the constraint set.

The attack targets the graph structure rather than individual edge features
(complementing C-PGD which perturbs features on existing edges).

Usage:
    attack = EdgeInjectionAttack(n_inject=50, scaler_path="data/processed/static/scaler.json")
    adv_data = attack.generate(model, data)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from src.attack.base import BaseAttack
from src.attack.constraints import ConstraintSet
from src.models.base import BaseNIDSModel

logger = logging.getLogger(__name__)

_BENIGN_CLASS = 0


class EdgeInjectionAttack(BaseAttack):
    """Adversarial attack via synthetic edge injection.

    Strategy: for each attack-classified edge, inject ``n_inject`` new edges
    that connect the source/destination nodes to other nodes in the graph.
    Injected edge features are sampled from the benign feature distribution
    (mean ± noise) to dilute the attack signal in the GNN's neighbourhood
    aggregation.

    Parameters
    ----------
    n_inject:
        Number of synthetic edges to inject per target node.
    degree_sigma_limit:
        Maximum standard deviations above training mean degree allowed
        for any node after injection.
    scaler_path:
        Path to ``scaler.json`` or ``scaler.pkl`` for constraint checking.
    constraint_set:
        Pre-built ConstraintSet. Takes priority over ``scaler_path``.
    target_split:
        Data split to attack (for CLI use).
    memory_reset_policy:
        Memory reset policy for temporal models (ignored for static).
    """

    def __init__(
        self,
        n_inject: int = 50,
        degree_sigma_limit: float = 3.0,
        scaler_path: str | Path | None = None,
        constraint_set: ConstraintSet | None = None,
        target_split: str = "test",
        memory_reset_policy: str = "before_each_attack",
        **kwargs,
    ) -> None:
        self.n_inject = n_inject
        self.degree_sigma_limit = degree_sigma_limit
        self.target_split = target_split
        self.memory_reset_policy = memory_reset_policy

        if constraint_set is not None:
            self.cs = constraint_set
        elif scaler_path is not None:
            self.cs = ConstraintSet.from_scaler(scaler_path)
        else:
            self.cs = ConstraintSet()

        self._scaler_path = scaler_path
        self._benign_stats: dict[str, np.ndarray] | None = None

    def _compute_benign_stats(self, edge_attr: torch.Tensor, y: torch.Tensor) -> None:
        """Compute mean and std of benign edge features for sampling."""
        benign_mask = y == _BENIGN_CLASS
        if not benign_mask.any():
            self._benign_stats = {
                "mean": torch.zeros(edge_attr.shape[1]).numpy(),
                "std": torch.ones(edge_attr.shape[1]).numpy(),
            }
            return
        benign_feats = edge_attr[benign_mask].numpy()
        self._benign_stats = {
            "mean": benign_feats.mean(axis=0),
            "std": benign_feats.std(axis=0) + 1e-8,
        }

    def _sample_benign_features(self, n: int, n_features: int) -> torch.Tensor:
        """Sample synthetic edge features from the benign distribution."""
        if self._benign_stats is None:
            return torch.randn(n, n_features) * 0.1
        mean = self._benign_stats["mean"]
        std = self._benign_stats["std"]
        noise = np.random.randn(n, n_features).astype(np.float32)
        samples = mean + noise * std * 0.5
        return torch.from_numpy(samples)

    def _compute_degrees(self, edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
        """Compute node degrees from edge_index."""
        degrees = np.zeros(num_nodes, dtype=np.float64)
        src, dst = edge_index[0].numpy(), edge_index[1].numpy()
        np.add.at(degrees, src, 1)
        np.add.at(degrees, dst, 1)
        return degrees

    def generate(self, model: BaseNIDSModel, data: Data, **kwargs) -> Data:
        """Inject synthetic benign-looking edges to dilute attack neighbourhoods.

        Only injects edges around nodes that participate in attack-classified
        flows. Respects degree anomaly limits from the constraint set.
        """
        n_inject = int(kwargs.get("n_inject", self.n_inject))
        model.eval()

        edge_index = data.edge_index
        edge_attr = data.edge_attr
        num_nodes = data.num_nodes or int(edge_index.max().item()) + 1
        n_features = edge_attr.shape[1]

        with torch.no_grad():
            preds = model(data).argmax(dim=-1)

        attack_mask = preds > 0
        if not attack_mask.any():
            return data

        y_labels = data.y_multi if hasattr(data, "y_multi") else preds
        self._compute_benign_stats(edge_attr, y_labels)

        attack_edges = edge_index[:, attack_mask]
        target_nodes = torch.unique(torch.cat([attack_edges[0], attack_edges[1]]))

        original_degrees = self._compute_degrees(edge_index, num_nodes)
        train_mean = float(original_degrees.mean())
        train_std = float(original_degrees.std())
        degree_threshold = train_mean + self.degree_sigma_limit * max(train_std, 1.0)

        all_nodes = torch.arange(num_nodes)
        new_src_list: list[int] = []
        new_dst_list: list[int] = []
        current_degrees = original_degrees.copy()

        for node_id in target_nodes.tolist():
            if current_degrees[node_id] >= degree_threshold:
                continue

            budget = min(
                n_inject,
                int(degree_threshold - current_degrees[node_id]),
            )
            if budget <= 0:
                continue

            candidates = all_nodes[all_nodes != node_id]
            if len(candidates) == 0:
                continue

            n_actual = min(budget, len(candidates))
            chosen_idx = torch.randperm(len(candidates))[:n_actual]
            chosen_nodes = candidates[chosen_idx]

            for dst_node in chosen_nodes.tolist():
                if current_degrees[dst_node] >= degree_threshold:
                    continue
                new_src_list.append(node_id)
                new_dst_list.append(dst_node)
                current_degrees[node_id] += 1
                current_degrees[dst_node] += 1

        if not new_src_list:
            return data

        n_injected = len(new_src_list)
        logger.info("Injecting %d synthetic edges", n_injected)

        new_edge_index = torch.tensor(
            [new_src_list, new_dst_list], dtype=torch.long
        )
        new_edge_attr = self._sample_benign_features(n_injected, n_features)

        adv_data = data.clone()
        adv_data.edge_index = torch.cat([edge_index, new_edge_index], dim=1)
        adv_data.edge_attr = torch.cat([edge_attr, new_edge_attr], dim=0)

        if hasattr(data, "y") and data.y is not None:
            new_y = torch.zeros(n_injected, dtype=data.y.dtype)
            adv_data.y = torch.cat([data.y, new_y])
        if hasattr(data, "y_multi") and data.y_multi is not None:
            new_y_multi = torch.zeros(n_injected, dtype=data.y_multi.dtype)
            adv_data.y_multi = torch.cat([data.y_multi, new_y_multi])

        if not self.cs.check_degree_anomaly(
            original_degrees,
            current_degrees,
            sigma_multiplier=self.degree_sigma_limit,
            train_mean=train_mean,
            train_std=train_std,
        ):
            logger.warning("Degree anomaly check failed after injection")

        return adv_data

    def constraint_check(self, x_adv: np.ndarray, attack_label: int | None = None) -> bool:
        return self.cs.check(x_adv, attack_label=attack_label)

    def attack_success_rate(
        self, model: BaseNIDSModel, data: Data, **kwargs
    ) -> dict[str, float]:
        """Compute ASR: fraction of attack edges no longer detected after injection."""
        with torch.no_grad():
            orig_preds = model(data).argmax(dim=-1)

        n_orig_edges = data.edge_index.shape[1]
        adv_data = self.generate(model, data, **kwargs)

        with torch.no_grad():
            adv_preds = model(adv_data).argmax(dim=-1)

        adv_preds_orig = adv_preds[:n_orig_edges]
        attack_mask = orig_preds > 0
        n_attack = int(attack_mask.sum())
        if n_attack == 0:
            return {"asr": 0.0, "n_attack_edges": 0, "n_evaded": 0}

        evaded = int(((orig_preds > 0) & (adv_preds_orig == 0)).sum())
        return {
            "asr": round(evaded / n_attack, 4),
            "n_attack_edges": n_attack,
            "n_evaded": evaded,
        }
