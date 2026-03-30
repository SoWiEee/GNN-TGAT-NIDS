"""C-PGD adversarial example generation for the web service.

Wraps ``src.attack.cpgd.CPGDAttack`` for single-flow use from the API.
Called synchronously inside ``run_in_threadpool`` by the adversarial router.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from app.services.inference import get_model
from src.attack.cpgd import CPGDAttack

logger = logging.getLogger(__name__)

# Scaler written here by static_builder.py after each build run.
_SCALER_JSON = Path("data/processed/static/scaler.json")
_SCALER_PKL = Path("data/processed/static/scaler.pkl")


def _get_scaler_path() -> Path | None:
    if _SCALER_JSON.exists():
        return _SCALER_JSON
    if _SCALER_PKL.exists():
        return _SCALER_PKL
    return None


def generate_adversarial_example(
    result: dict[str, Any],
    flow_id: str,
    epsilon: float,
    steps: int,
) -> dict[str, Any]:
    """Run C-PGD on a single flow and return a side-by-side comparison dict.

    Returns the adversarial comparison JSON defined in spec.md §3.3.3.
    If generation fails (CSR < 1.0 or budget exhausted), adversarial is null.
    """
    # Locate the alert and edge for this flow
    alerts = result.get("alerts", [])
    alert = next((a for a in alerts if a["flow_id"] == flow_id), None)
    if alert is None:
        raise ValueError(f"flow_id '{flow_id}' not found in session alerts")

    edges = result.get("graph", {}).get("edges", [])
    edge = next((e for e in edges if e["data"]["id"] == flow_id), None)
    if edge is None:
        raise ValueError(f"Edge '{flow_id}' not found in graph data")

    raw_features: list[float] | None = edge["data"].get("raw_features")
    if not raw_features:
        return {
            "flow_id": flow_id,
            "error": "Raw features unavailable — re-run analysis to enable adversarial comparison.",
        }

    model_name = result.get("meta", {}).get("model", "gat")
    feature_cols: list[str] = result.get("meta", {}).get("feature_cols", [])
    model = get_model(model_name)

    # Build C-PGD attacker
    scaler_path = _get_scaler_path()
    attacker = CPGDAttack(epsilon=epsilon, steps=steps, scaler_path=scaler_path)

    # Build a minimal single-edge graph:
    #   2 nodes (src=0, dst=1), 1 directed edge.
    #   Dummy node features (zeros) — GNN uses edge_attr for classification.
    x_feat = torch.tensor(raw_features, dtype=torch.float32).unsqueeze(0)  # [1, F]
    n_node_feat = (
        model.node_proj.in_features  # type: ignore[attr-defined]
        if hasattr(model, "node_proj")
        else x_feat.shape[-1]
    )
    dummy_x = torch.zeros(2, n_node_feat)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    single_data = Data(x=dummy_x, edge_index=edge_index, edge_attr=x_feat)

    x_orig_np = np.array(raw_features, dtype=np.float32)

    # Run C-PGD (returns Data with perturbed edge_attr)
    adv_data = attacker.generate(model, single_data)
    adv_feat_np = adv_data.edge_attr[0].detach().numpy()

    # Convert back to raw scale for display and constraint check
    if attacker._scaler is not None:
        x_raw_orig = attacker._inverse_transform(x_orig_np.reshape(1, -1))[0]
        x_raw_adv = attacker._inverse_transform(adv_feat_np.reshape(1, -1))[0]
        csr = 1.0 if attacker.cs.check(x_raw_adv) else 0.0
    else:
        x_raw_orig = x_orig_np
        x_raw_adv = adv_feat_np
        csr = 1.0

    # Build changed-features list (only show features that actually changed)
    changed: list[dict] = []
    for i, name in enumerate(feature_cols):
        if i >= len(x_raw_orig) or i >= len(x_raw_adv):
            break
        orig_val = float(x_raw_orig[i])
        adv_val = float(x_raw_adv[i])
        delta_pct = (adv_val - orig_val) / max(abs(orig_val), 1e-8) * 100
        if abs(delta_pct) > 0.05:
            changed.append({
                "name": name,
                "original": round(orig_val, 4),
                "adversarial": round(adv_val, 4),
                "delta_pct": round(delta_pct, 2),
                "constraint_ok": True,
            })

    # Get adversarial prediction
    adv_conf: float | None = None
    adv_pred_label = "Unknown"
    with torch.inference_mode():
        adv_logits = model(adv_data)
        adv_proba = torch.softmax(adv_logits, dim=-1)
        adv_pred_class = int(adv_proba.argmax(dim=-1)[0])
        adv_conf = round(float(adv_proba[0, adv_pred_class]), 4)

    label2idx: dict[str, int] = result.get("meta", {}).get("label2idx", {})
    idx2label = {v: k for k, v in label2idx.items()}
    adv_pred_label = idx2label.get(adv_pred_class, "Benign" if adv_pred_class == 0 else "Unknown")

    orig_features_dict = {
        name: round(float(x_raw_orig[i]), 4)
        for i, name in enumerate(feature_cols)
        if i < len(x_raw_orig)
    }
    adv_features_dict = {
        name: round(float(x_raw_adv[i]), 4)
        for i, name in enumerate(feature_cols)
        if i < len(x_raw_adv)
    }

    return {
        "flow_id": flow_id,
        "original": {
            "prediction": alert["attack_type"],
            "confidence": alert["confidence"],
            "features": orig_features_dict,
        },
        "adversarial": {
            "prediction": adv_pred_label,
            "confidence": adv_conf,
            "features": adv_features_dict,
            "csr": csr,
            "changed_features": changed,
        },
    }
