"""Evaluation metrics for edge-level NIDS classification."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    y_proba: torch.Tensor | np.ndarray | None = None,
    average: str = "weighted",
    binary_positive_class: int = 1,
) -> dict[str, float]:
    """Compute classification metrics for edge-level NIDS evaluation.

    Parameters
    ----------
    y_true:
        Ground-truth class indices, shape ``(N,)``.
    y_pred:
        Predicted class indices, shape ``(N,)``.
    y_proba:
        Class probability estimates, shape ``(N, C)``. Required for ROC-AUC
        and Average Precision.  If None these metrics are skipped.
    average:
        Averaging strategy for multi-class metrics (``"weighted"``,
        ``"macro"``, ``"micro"``).
    binary_positive_class:
        Class index to treat as positive when computing binary ROC-AUC
        (used only when ``C == 2``).

    Returns
    -------
    dict[str, float]
        Keys: ``f1``, ``precision``, ``recall``, and optionally ``roc_auc``,
        ``avg_precision``.
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, float] = {}

    metrics["f1"] = float(
        f1_score(y_true, y_pred, average=average, zero_division=0)
    )
    metrics["precision"] = float(
        precision_score(y_true, y_pred, average=average, zero_division=0)
    )
    metrics["recall"] = float(
        recall_score(y_true, y_pred, average=average, zero_division=0)
    )

    if y_proba is not None:
        if isinstance(y_proba, torch.Tensor):
            y_proba = y_proba.cpu().numpy()
        y_proba = np.asarray(y_proba)

        n_classes = y_proba.shape[1] if y_proba.ndim == 2 else 2

        try:
            if n_classes == 2:
                pos_proba = y_proba[:, binary_positive_class]
                metrics["roc_auc"] = float(roc_auc_score(y_true, pos_proba))
                metrics["avg_precision"] = float(average_precision_score(y_true, pos_proba))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class="ovr",
                        average=average,
                    )
                )
                # Average precision per-class OvR, then average
                ap_scores = []
                for cls in range(n_classes):
                    binary_true = (y_true == cls).astype(int)
                    if binary_true.sum() > 0:
                        ap_scores.append(average_precision_score(binary_true, y_proba[:, cls]))
                metrics["avg_precision"] = float(np.mean(ap_scores)) if ap_scores else 0.0
        except ValueError:
            # Happens when only one class present in y_true
            pass

    return metrics


def compute_class_weights(
    labels: torch.Tensor | np.ndarray,
    n_classes: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Compute inverse-frequency class weights for weighted CrossEntropyLoss.

    weight_c = N / (n_classes * count_c)

    Classes with zero samples receive weight 0 to avoid division by zero.

    Parameters
    ----------
    labels:
        Integer label array, shape ``(N,)``.
    n_classes:
        Total number of classes.
    device:
        Target device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Shape ``(n_classes,)`` of dtype float32.
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    labels = np.asarray(labels)

    n = len(labels)
    weights = np.zeros(n_classes, dtype=np.float32)

    for c in range(n_classes):
        count = (labels == c).sum()
        if count > 0:
            weights[c] = n / (n_classes * count)

    return torch.tensor(weights, dtype=torch.float32, device=device)
