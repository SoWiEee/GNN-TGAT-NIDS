"""Build Cytoscape.js-compatible graph JSON from PyG inference results.

Node layout is computed server-side with networkx spring_layout so the frontend
can use the cheap 'preset' layout instead of running expensive force-directed
layout calculation in the browser.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np
import torch

# Attack class colours (index → CSS colour)
_CLASS_COLOURS = [
    "#9ca3af",  # 0 Benign — grey
    "#ef4444",  # 1 DoS — red
    "#f97316",  # 2 DDoS — orange
    "#eab308",  # 3 Reconnaissance — yellow
    "#84cc16",  # 4 Backdoor — lime
    "#06b6d4",  # 5 Exploits — cyan
    "#8b5cf6",  # 6 Fuzzers — violet
    "#ec4899",  # 7 Analysis — pink
    "#14b8a6",  # 8 Shellcode — teal
    "#f59e0b",  # 9 Worms — amber
]


def _class_colour(class_idx: int) -> str:
    if 0 <= class_idx < len(_CLASS_COLOURS):
        return _CLASS_COLOURS[class_idx]
    return "#6b7280"


def _risk_colour(risk_score: float) -> str:
    if risk_score < 0.5:
        return "#22c55e"   # green
    if risk_score < 0.8:
        return "#f97316"   # orange
    return "#ef4444"       # red


def build_graph_response(
    all_data: list,
    all_logits: list[torch.Tensor],
    meta: dict,
    csv_path: str,
) -> dict[str, Any]:
    """Convert PyG inference results to frontend-ready JSON.

    Returns a dict with keys: graph, alerts, timeline.
    """
    label2idx: dict[str, int] = meta.get("label2idx", {})
    idx2label: dict[int, str] = {v: k for k, v in label2idx.items()}

    nodes: dict[str, dict] = {}  # node_id → cytoscape node
    edges: list[dict] = []
    alerts: list[dict] = []
    timeline: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    global_edge_idx = 0

    for window_idx, (data, logits) in enumerate(zip(all_data, all_logits)):
        proba = torch.softmax(logits, dim=-1).numpy()
        preds = proba.argmax(axis=-1)
        confidence = proba.max(axis=-1)

        # Build node_id → IP:port string from data
        # data may carry node metadata; fall back to integer indices
        n_nodes = data.num_nodes
        node_ids = [f"n{i}" for i in range(n_nodes)]

        # Accumulate node risk scores
        node_risk: dict[str, float] = defaultdict(float)

        edge_index = data.edge_index.numpy()
        for edge_pos in range(edge_index.shape[1]):
            src_i = int(edge_index[0, edge_pos])
            dst_i = int(edge_index[1, edge_pos])
            src_id = node_ids[src_i]
            dst_id = node_ids[dst_i]
            pred_class = int(preds[edge_pos])
            conf = float(confidence[edge_pos])
            attack_label = idx2label.get(pred_class, "Unknown")
            is_attack = pred_class > 0
            edge_id = f"e{global_edge_idx}"
            global_edge_idx += 1

            # Update node risk
            if is_attack:
                node_risk[src_id] = max(node_risk[src_id], conf)
                node_risk[dst_id] = max(node_risk[dst_id], conf)

            # Register nodes
            for nid in (src_id, dst_id):
                if nid not in nodes:
                    nodes[nid] = {
                        "data": {"id": nid, "ip": nid, "riskScore": 0.0, "x": 0.0, "y": 0.0}
                    }

            # Store raw (normalised) edge features so the adversarial module
            # can retrieve them without re-running the data pipeline.
            raw_feats = (
                data.edge_attr[edge_pos].tolist()
                if data.edge_attr is not None
                else []
            )

            edges.append({
                "data": {
                    "id": edge_id,
                    "source": src_id,
                    "target": dst_id,
                    "prediction": attack_label,
                    "confidence": round(conf, 4),
                    "flowId": edge_id,
                    "colour": _class_colour(pred_class),
                    "width": max(1, int(conf * 4)),
                    "window": window_idx,
                    "raw_features": raw_feats,
                }
            })

            if is_attack:
                # Top-3 feature deviations by absolute value
                if data.edge_attr is not None:
                    feat = data.edge_attr[edge_pos].numpy()
                    top3 = np.argsort(np.abs(feat))[-3:][::-1].tolist()
                    feat_names = meta.get("feature_cols", [])
                    top_features = [
                        {"name": feat_names[fi] if fi < len(feat_names) else f"f{fi}",
                         "value": round(float(feat[fi]), 4)}
                        for fi in top3
                    ]
                else:
                    top_features = []

                alerts.append({
                    "flow_id": edge_id,
                    "src": src_id,
                    "dst": dst_id,
                    "attack_type": attack_label,
                    "confidence": round(conf, 4),
                    "top_features": top_features,
                    "window": window_idx,
                })

            # Timeline count
            timeline[window_idx][attack_label] += 1

        # Finalize node risk colours
        for nid in node_risk:
            if nid in nodes:
                nodes[nid]["data"]["riskScore"] = round(node_risk[nid], 4)

    # Compute layout with networkx
    _apply_layout(nodes, edges)

    # Apply risk colour to nodes
    for nid, node in nodes.items():
        risk = node["data"]["riskScore"]
        node["data"]["colour"] = _risk_colour(risk)

    # Build Plotly-compatible timeline
    all_labels = sorted({k for w in timeline.values() for k in w})
    plotly_timeline = {
        "x": list(range(len(timeline))),
        "traces": [
            {
                "name": label,
                "y": [timeline[w].get(label, 0) for w in sorted(timeline)],
                "colour": _class_colour(label2idx.get(label, 0)),
            }
            for label in all_labels
        ],
    }

    return {
        "graph": {"nodes": list(nodes.values()), "edges": edges},
        "alerts": alerts,
        "timeline": plotly_timeline,
        "meta": {
            "total_flows": len(edges),
            "total_alerts": len(alerts),
            "n_windows": len(all_data),
            "csv_path": csv_path,
        },
    }


def _apply_layout(nodes: dict, edges: list) -> None:
    """Compute spring layout positions with networkx and store in node data."""
    if not nodes:
        return
    G = nx.DiGraph()
    G.add_nodes_from(nodes.keys())
    for edge in edges:
        G.add_edge(edge["data"]["source"], edge["data"]["target"])

    k = 2.0 / max(1, math.sqrt(len(nodes)))
    pos = nx.spring_layout(G, seed=42, k=k, iterations=50)

    # Scale positions to [0, 1000] for Cytoscape.js
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)

    for nid, (x, y) in pos.items():
        if nid in nodes:
            nodes[nid]["data"]["x"] = round((x - x_min) / x_range * 1000, 1)
            nodes[nid]["data"]["y"] = round((y - y_min) / y_range * 1000, 1)
