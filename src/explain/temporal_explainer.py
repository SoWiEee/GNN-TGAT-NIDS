"""Temporal model explainability via gradient-based feature attribution.

PyG's GNNExplainer requires MessagePassing layers and does not work with
TGAT/TGN temporal models. This module provides gradient-based input
attribution (integrated-gradients-lite) for temporal edge predictions.

Usage:
    from src.explain.temporal_explainer import explain_temporal_flow
    result = explain_temporal_flow(model, batch, edge_idx=42)
"""
from __future__ import annotations

import logging

import torch
from torch_geometric.data import TemporalData

logger = logging.getLogger(__name__)


def _gradient_attribution(
    model: torch.nn.Module,
    batch: TemporalData,
    edge_idx: int,
    n_steps: int = 20,
) -> torch.Tensor:
    """Compute input × gradient attribution for a single temporal edge.

    Uses a simple integrated-gradients approximation: interpolate message
    features from a zero baseline to the actual value, accumulate gradients.
    """
    model.eval()
    msg = batch.msg.detach()
    baseline = torch.zeros_like(msg[edge_idx])
    target_msg = msg[edge_idx].clone()

    accumulated = torch.zeros_like(target_msg)

    for step in range(n_steps):
        alpha = (step + 1) / n_steps
        interpolated = baseline + alpha * (target_msg - baseline)

        interp_msg = msg.clone()
        interp_msg[edge_idx] = interpolated
        interp_msg = interp_msg.detach().requires_grad_(True)

        mod_batch = batch.clone()
        mod_batch.msg = interp_msg

        logits = model(mod_batch)
        pred_class = logits[edge_idx].argmax()
        score = logits[edge_idx, pred_class]
        score.backward(retain_graph=False)

        if interp_msg.grad is not None:
            accumulated += interp_msg.grad[edge_idx].detach()
        model.zero_grad()

    attribution = (target_msg - baseline) * accumulated / n_steps
    return attribution


def explain_temporal_flow(
    model: torch.nn.Module,
    batch: TemporalData,
    edge_idx: int,
    n_steps: int = 20,
    feature_names: list[str] | None = None,
) -> dict:
    """Explain a single temporal edge prediction via gradient attribution."""
    model.eval()

    with torch.no_grad():
        logits = model(batch)
        proba = torch.softmax(logits, dim=-1)
        pred_class = int(logits[edge_idx].argmax())
        confidence = float(proba[edge_idx].max())

    attribution = _gradient_attribution(model, batch, edge_idx, n_steps)
    attr_abs = attribution.abs()
    attr_norm = attr_abs / attr_abs.sum().clamp(min=1e-8)

    if feature_names is None:
        feature_names = [f"msg_feat_{i}" for i in range(len(attribution))]

    top_indices = attr_norm.argsort(descending=True)[:10]
    top_features = []
    for i in top_indices:
        idx = int(i)
        top_features.append({
            "name": feature_names[idx] if idx < len(feature_names) else f"feat_{idx}",
            "importance": round(float(attr_norm[idx]), 6),
            "attribution": round(float(attribution[idx]), 6),
        })

    return {
        "edge_idx": edge_idx,
        "src": int(batch.src[edge_idx]),
        "dst": int(batch.dst[edge_idx]),
        "timestamp": float(batch.t[edge_idx]),
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "feature_attribution": attr_norm.cpu().tolist(),
        "top_features": top_features,
        "method": "integrated_gradients_approx",
    }


def explain_temporal_top_alerts(
    model: torch.nn.Module,
    batch: TemporalData,
    top_k: int = 5,
    n_steps: int = 20,
    feature_names: list[str] | None = None,
) -> list[dict]:
    """Explain the top-K most confident attack predictions in a temporal batch."""
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        proba = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

    attack_mask = preds > 0
    if not attack_mask.any():
        return []

    attack_indices = attack_mask.nonzero(as_tuple=True)[0]
    confidences = proba[attack_indices].max(dim=-1).values
    top_indices = confidences.argsort(descending=True)[:top_k]

    results = []
    for rank, idx in enumerate(top_indices):
        edge_idx = int(attack_indices[idx])
        logger.info("Explaining temporal edge %d (rank %d/%d)", edge_idx, rank + 1, top_k)
        result = explain_temporal_flow(model, batch, edge_idx, n_steps, feature_names)
        result["rank"] = rank + 1
        results.append(result)

    return results
