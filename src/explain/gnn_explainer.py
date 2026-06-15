"""GNN Explainability for NIDS edge-level predictions.

Uses PyG's GNNExplainer to identify which node features and graph edges
contribute most to a specific flow classification.

Usage:
    from src.explain.gnn_explainer import explain_flow, explain_top_alerts

    # Explain a single edge
    result = explain_flow(model, data, edge_idx=42)

    # Explain top-K alerts
    results = explain_top_alerts(model, data, top_k=5)
"""
from __future__ import annotations

import logging

import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class _EdgeExplainableWrapper(torch.nn.Module):
    """Adapter that makes a BaseNIDSModel compatible with PyG's Explainer.

    PyG Explainer expects forward(x, edge_index, edge_attr, ...) → logits.
    Our models expect forward(Data) → (E, C) logits. This wrapper bridges
    the gap, returning logits for a specific target edge.
    """

    def __init__(self, model: torch.nn.Module, target_edge_idx: int) -> None:
        super().__init__()
        self.model = model
        self.target_edge_idx = target_edge_idx

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        all_logits = self.model(data)
        return all_logits[self.target_edge_idx].unsqueeze(0)


def _has_message_passing(model: torch.nn.Module) -> bool:
    """Check if model uses PyG MessagePassing layers."""
    from torch_geometric.nn import MessagePassing
    for module in model.modules():
        if isinstance(module, MessagePassing):
            return True
    return False


def explain_flow(
    model: torch.nn.Module,
    data: Data,
    edge_idx: int,
    epochs: int = 200,
) -> dict:
    """Explain a single flow (edge) prediction.

    Returns a dict with:
        - edge_idx: the explained edge index
        - predicted_class: model's prediction for this edge
        - confidence: softmax probability of predicted class
        - node_feature_importance: per-feature importance scores for src/dst nodes
        - edge_importance: importance scores for neighbouring edges
        - top_features: ranked list of most important features
    """
    from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

    model.eval()

    with torch.no_grad():
        all_logits = model(data)
        proba = torch.softmax(all_logits, dim=-1)
        pred_class = int(all_logits[edge_idx].argmax())
        confidence = float(proba[edge_idx].max())

    wrapper = _EdgeExplainableWrapper(model, edge_idx)

    use_edge_mask = _has_message_passing(model)

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="phenomenon",
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="graph",
            return_type="raw",
        ),
        node_mask_type="attributes",
        edge_mask_type="object" if use_edge_mask else None,
    )

    target = torch.tensor([pred_class])
    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        target=target,
    )

    node_mask = explanation.node_mask
    edge_mask = getattr(explanation, "edge_mask", None)

    src_node = int(data.edge_index[0, edge_idx])
    dst_node = int(data.edge_index[1, edge_idx])

    src_importance = node_mask[src_node].cpu().tolist() if node_mask is not None else []
    dst_importance = node_mask[dst_node].cpu().tolist() if node_mask is not None else []

    feature_importance = {}
    if node_mask is not None:
        combined = (node_mask[src_node] + node_mask[dst_node]) / 2.0
        for i, score in enumerate(combined.cpu().tolist()):
            feature_importance[f"node_feat_{i}"] = round(score, 6)

    if edge_mask is not None and data.edge_attr is not None:
        edge_weight = float(edge_mask[edge_idx]) if edge_idx < len(edge_mask) else 0.0
    else:
        edge_weight = 0.0

    top_features = sorted(
        feature_importance.items(), key=lambda kv: abs(kv[1]), reverse=True
    )[:10]

    return {
        "edge_idx": edge_idx,
        "src_node": src_node,
        "dst_node": dst_node,
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
        "node_feature_importance": {
            "src": src_importance,
            "dst": dst_importance,
        },
        "edge_self_importance": round(edge_weight, 6),
        "top_features": [{"name": k, "importance": v} for k, v in top_features],
        "edge_mask": edge_mask.cpu().tolist() if edge_mask is not None else [],
    }


def explain_top_alerts(
    model: torch.nn.Module,
    data: Data,
    top_k: int = 5,
    epochs: int = 200,
) -> list[dict]:
    """Explain the top-K most confident attack predictions."""
    model.eval()
    with torch.no_grad():
        logits = model(data)
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
        logger.info("Explaining edge %d (rank %d/%d) ...", edge_idx, rank + 1, top_k)
        result = explain_flow(model, data, edge_idx, epochs=epochs)
        result["rank"] = rank + 1
        results.append(result)

    return results
