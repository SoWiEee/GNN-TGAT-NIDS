"""Ensemble inference: combine predictions from multiple GNN models.

Supports three strategies:
    - soft_vote  : average softmax probabilities, take argmax (default)
    - hard_vote  : majority vote on predicted class labels
    - weighted   : weighted average of probabilities using per-model weights

Usage:
    from src.models.ensemble import EnsembleModel
    ensemble = EnsembleModel(models={"graphsage": m1, "gat": m2}, strategy="soft_vote")
    logits = ensemble(data)
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch_geometric.data import Data


class EnsembleModel(nn.Module):
    """Combine predictions from multiple NIDS models."""

    def __init__(
        self,
        models: dict[str, nn.Module],
        strategy: Literal["soft_vote", "hard_vote", "weighted"] = "soft_vote",
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if not models:
            raise ValueError("At least one model is required")

        self._model_names = list(models.keys())
        self._models = nn.ModuleDict(models)
        self._strategy = strategy

        if weights is not None:
            total = sum(weights.values())
            self._weights = {k: v / total for k, v in weights.items()}
        else:
            self._weights = {k: 1.0 / len(models) for k in models}

    @property
    def strategy(self) -> str:
        return self._strategy

    @property
    def model_names(self) -> list[str]:
        return list(self._model_names)

    @torch.inference_mode()
    def forward(self, data: Data) -> torch.Tensor:
        all_logits: list[torch.Tensor] = []
        for name in self._model_names:
            model = self._models[name]
            logits = model(data)
            all_logits.append(logits)

        if self._strategy == "hard_vote":
            return self._hard_vote(all_logits)
        if self._strategy == "weighted":
            return self._weighted_vote(all_logits)
        return self._soft_vote(all_logits)

    def _soft_vote(self, all_logits: list[torch.Tensor]) -> torch.Tensor:
        probas = [torch.softmax(lg, dim=-1) for lg in all_logits]
        avg = torch.stack(probas).mean(dim=0)
        return avg

    def _hard_vote(self, all_logits: list[torch.Tensor]) -> torch.Tensor:
        n_classes = all_logits[0].shape[-1]
        votes = torch.stack([lg.argmax(dim=-1) for lg in all_logits])
        result = torch.zeros_like(all_logits[0])
        for i in range(votes.shape[1]):
            counts = torch.bincount(votes[:, i], minlength=n_classes).float()
            result[i] = counts / counts.sum()
        return result

    def _weighted_vote(self, all_logits: list[torch.Tensor]) -> torch.Tensor:
        result = torch.zeros_like(all_logits[0])
        for name, logits in zip(self._model_names, all_logits):
            proba = torch.softmax(logits, dim=-1)
            result += self._weights[name] * proba
        return result

    @classmethod
    def from_validation(
        cls,
        models: dict[str, nn.Module],
        val_loader,
    ) -> EnsembleModel:
        """Build a weighted ensemble using per-model validation F1 scores."""
        from src.eval.metrics import compute_metrics

        weights: dict[str, float] = {}
        for name, model in models.items():
            model.eval()
            all_true, all_pred = [], []
            with torch.inference_mode():
                for data in val_loader:
                    logits = model(data)
                    pred = logits.argmax(dim=-1)
                    all_true.append(data.y_multi)
                    all_pred.append(pred)
            y_true = torch.cat(all_true)
            y_pred = torch.cat(all_pred)
            f1 = compute_metrics(y_true, y_pred, average="weighted")["f1"]
            weights[name] = max(float(f1), 1e-6)

        return cls(models, strategy="weighted", weights=weights)

    def predict(self, data: Data) -> dict:
        """Return ensemble prediction with per-model breakdown."""
        self.eval()
        proba = self.forward(data)
        preds = proba.argmax(dim=-1)

        per_model: dict[str, dict] = {}
        for name in self._model_names:
            model = self._models[name]
            with torch.inference_mode():
                lg = model(data)
            mp = torch.softmax(lg, dim=-1)
            per_model[name] = {
                "predictions": lg.argmax(dim=-1).cpu().tolist(),
                "confidence": mp.max(dim=-1).values.cpu().tolist(),
            }

        return {
            "ensemble_predictions": preds.cpu().tolist(),
            "ensemble_confidence": proba.max(dim=-1).values.cpu().tolist(),
            "ensemble_probabilities": proba.cpu(),
            "per_model": per_model,
        }
