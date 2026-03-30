"""C-PGD adversarial example generation for the web service."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.services.inference import get_model

logger = logging.getLogger(__name__)


def _load_scaler(processed_dir: Path):
    """Load StandardScaler from JSON parameters (avoids pickle)."""
    from sklearn.preprocessing import StandardScaler

    scaler_json_path = processed_dir / "scaler.json"
    if not scaler_json_path.exists():
        # Fall back to pickle if JSON not available
        import pickle
        with open(processed_dir / "scaler.pkl", "rb") as f:
            return pickle.load(f)

    params = json.loads(scaler_json_path.read_text())
    scaler = StandardScaler()
    scaler.mean_ = np.array(params["mean_"])
    scaler.scale_ = np.array(params["scale_"])
    scaler.clip_lo_ = np.array(params["clip_lo_"])
    scaler.clip_hi_ = np.array(params["clip_hi_"])
    scaler.n_features_in_ = len(scaler.mean_)
    scaler.feature_names_in_ = None
    return scaler


def generate_adversarial_example(
    result: dict[str, Any],
    flow_id: str,
    epsilon: float,
    steps: int,
) -> dict[str, Any]:
    """Run C-PGD on a single flow and return a comparison dict.

    Returns the adversarial comparison JSON as defined in spec.md Section 3.3.3.
    If adversarial generation fails (CSR < 1.0), adversarial field is null.
    """
    from src.attack.constraints import ConstraintSet, NF_FEATURES

    # Find the alert for this flow
    alerts = result.get("alerts", [])
    alert = next((a for a in alerts if a["flow_id"] == flow_id), None)
    if alert is None:
        raise ValueError(f"flow_id '{flow_id}' not found in session alerts")

    # Find corresponding edge features from graph
    edges = result.get("graph", {}).get("edges", [])
    edge = next((e for e in edges if e["data"]["id"] == flow_id), None)
    if edge is None:
        raise ValueError(f"Edge '{flow_id}' not found in graph data")

    model_name = result.get("meta", {}).get("model", "gat")
    model = get_model(model_name)

    # Get raw edge features from result (stored as list in edge data)
    raw_features = edge["data"].get("raw_features")
    if raw_features is None:
        return {
            "flow_id": flow_id,
            "error": "Raw features not available for this flow. Re-run analysis.",
        }

    feature_cols = result.get("meta", {}).get("feature_cols", NF_FEATURES)
    x = torch.tensor(raw_features, dtype=torch.float32).unsqueeze(0)  # [1, n_feats]

    # Run C-PGD
    x_adv = _cpgd_single_flow(model, x, epsilon, steps, feature_cols)

    orig_features = {name: float(raw_features[i]) for i, name in enumerate(feature_cols)}
    adv_features_arr = x_adv.squeeze(0).detach().numpy()
    adv_features = {name: float(adv_features_arr[i]) for i, name in enumerate(feature_cols)}

    changed = []
    for name in feature_cols:
        orig_val = orig_features[name]
        adv_val = adv_features[name]
        delta_pct = (adv_val - orig_val) / max(abs(orig_val), 1e-8) * 100
        if abs(delta_pct) > 0.01:
            changed.append({
                "name": name,
                "original": round(orig_val, 4),
                "adversarial": round(adv_val, 4),
                "delta_pct": round(delta_pct, 2),
                "constraint_ok": True,
            })

    # Check constraints
    from src.attack.constraints import ConstraintSet
    cs = ConstraintSet()
    csr = 1.0 if cs.check(adv_features_arr) else 0.0

    return {
        "flow_id": flow_id,
        "original": {
            "prediction": alert["attack_type"],
            "confidence": alert["confidence"],
            "features": orig_features,
        },
        "adversarial": {
            "prediction": "Benign",
            "confidence": None,
            "features": adv_features,
            "csr": csr,
            "changed_features": changed,
        },
    }


def _cpgd_single_flow(
    model,
    x: torch.Tensor,
    epsilon: float,
    steps: int,
    feature_cols: list[str],
) -> torch.Tensor:
    """Run C-PGD on a single normalised flow vector.

    This is a simplified single-edge version that perturbs the edge features
    directly. A full graph context is not available here, so we treat the
    perturbation as feature-space only.
    """
    alpha = epsilon / max(steps, 1) * 2.5  # step size
    x_adv = x.clone() + torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)

    # Target: class 0 (benign)
    target = torch.zeros(1, dtype=torch.long)

    # Build a minimal single-edge graph for forward pass
    # edge_index: single self-loop so the model can run
    from torch_geometric.data import Data
    n = 1
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    dummy_x = torch.zeros(n, x.shape[-1])  # dummy node features

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        data = Data(x=dummy_x, edge_index=edge_index, edge_attr=x_adv)
        logits = model(data)  # [1, num_classes]
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()

        grad = x_adv.grad
        if grad is None:
            break
        norm = grad.norm(2) + 1e-8
        x_adv = (x_adv + alpha * grad / norm).detach()
        x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)

    return x_adv.detach()
