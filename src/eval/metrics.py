"""Evaluation metrics for edge-level NIDS classification."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
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


def compute_per_class_metrics(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    label_names: list[str] | None = None,
) -> dict:
    """Compute per-class precision, recall, F1, and the confusion matrix.

    Returns
    -------
    dict
        ``per_class``: list of dicts with name/precision/recall/f1/support per class.
        ``confusion_matrix``: row=true, col=pred, as nested lists.
        ``class_names``: ordered label names.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if label_names is None:
        label_names = [str(c) for c in classes]

    p_per = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    r_per = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    f_per = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    per_class = []
    for i, cls in enumerate(classes):
        name = label_names[i] if i < len(label_names) else str(cls)
        per_class.append({
            "class_id": int(cls),
            "name": name,
            "precision": round(float(p_per[i]), 4),
            "recall": round(float(r_per[i]), 4),
            "f1": round(float(f_per[i]), 4),
            "support": int((y_true == cls).sum()),
        })

    return {
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": label_names,
    }


def calibrate_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_classes: int,
    n_iter: int = 800,
) -> np.ndarray:
    """Find per-class logit biases that maximise macro F1 via Nelder-Mead.

    Parameters
    ----------
    y_true : shape ``(N,)``
    y_proba : shape ``(N, C)`` — softmax probabilities
    n_classes : number of classes
    n_iter : max optimiser iterations

    Returns
    -------
    np.ndarray
        Shape ``(C,)`` biases to add to log-probabilities before argmax.
    """
    from scipy.optimize import minimize

    log_proba = np.log(np.clip(y_proba, 1e-12, None))

    def _neg_macro_f1(biases: np.ndarray) -> float:
        adjusted = log_proba + biases
        preds = adjusted.argmax(axis=1)
        return -float(f1_score(y_true, preds, average="macro", zero_division=0))

    result = minimize(
        _neg_macro_f1,
        x0=np.zeros(n_classes),
        method="Nelder-Mead",
        options={"maxiter": n_iter, "xatol": 1e-4, "fatol": 1e-6},
    )
    return result.x


def apply_calibrated_prediction(
    y_proba: np.ndarray,
    biases: np.ndarray,
) -> np.ndarray:
    """Apply calibrated biases to produce predictions."""
    log_proba = np.log(np.clip(y_proba, 1e-12, None))
    return (log_proba + biases).argmax(axis=1)


def compute_class_weights(
    labels: torch.Tensor | np.ndarray,
    n_classes: int,
    device: torch.device | str = "cpu",
    strategy: str = "inverse",
) -> torch.Tensor:
    """Compute per-class weights for loss functions.

    Strategies
    ----------
    ``"inverse"``
        Standard inverse-frequency: ``weight_c = N / (n_classes * count_c)``.
    ``"effective"``
        Effective number of samples (Cui et al. 2019):
        ``weight_c = (1 - beta) / (1 - beta^count_c)`` where
        ``beta = (N - 1) / N``.  Provides smoother scaling under
        extreme imbalance — avoids the 5000x weight ratio that
        raw inverse-frequency produces with NF-UNSW-NB15-v2.
    ``"sqrt_inverse"``
        Square-root damped inverse-frequency:
        ``weight_c = sqrt(N / (n_classes * count_c))``.

    Parameters
    ----------
    labels:
        Integer label array, shape ``(N,)``.
    n_classes:
        Total number of classes.
    device:
        Target device for the returned tensor.
    strategy:
        Weighting strategy (``"inverse"``, ``"effective"``, ``"sqrt_inverse"``).

    Returns
    -------
    torch.Tensor
        Shape ``(n_classes,)`` of dtype float32, normalised so ``mean == 1``.
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    labels = np.asarray(labels)

    n = len(labels)
    weights = np.zeros(n_classes, dtype=np.float32)

    if strategy == "effective":
        beta = (n - 1) / n
        for c in range(n_classes):
            count = int((labels == c).sum())
            if count > 0:
                weights[c] = (1.0 - beta) / (1.0 - beta ** count)
    elif strategy == "sqrt_inverse":
        for c in range(n_classes):
            count = (labels == c).sum()
            if count > 0:
                weights[c] = np.sqrt(n / (n_classes * count))
    else:
        for c in range(n_classes):
            count = (labels == c).sum()
            if count > 0:
                weights[c] = n / (n_classes * count)

    if weights.sum() > 0:
        weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32, device=device)
