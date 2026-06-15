"""Explainability router: explain GNN predictions for specific flows."""
from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.services.inference import get_model

logger = logging.getLogger(__name__)
router = APIRouter(tags=["explain"])

SESSIONS_DIR = Path("data/sessions")
TEMPORAL_MODELS = {"tgat", "tgn"}


class ExplainRequest(BaseModel):
    model: str = Field(default="graphsage", pattern="^(graphsage|gat|egraphsage|tgat|tgn)$")
    edge_idx: int = Field(ge=0)
    epochs: int = Field(default=200, ge=10, le=1000)


class ExplainTopRequest(BaseModel):
    model: str = Field(default="graphsage", pattern="^(graphsage|gat|egraphsage|tgat|tgn)$")
    top_k: int = Field(default=5, ge=1, le=20)
    epochs: int = Field(default=200, ge=10, le=1000)


def _load_session_data(session_id: UUID):
    """Load PyG data from a completed session's result."""
    import tempfile

    from src.data.static_builder import build_static_graphs
    from src.data.static_dataset import StaticNIDSDataset

    sdir = SESSIONS_DIR / str(session_id)
    csv_path = sdir / "upload.csv"
    if not csv_path.exists():
        raise HTTPException(404, detail="Session not found")

    with tempfile.TemporaryDirectory() as tmpdir:
        build_static_graphs(
            csv_path=str(csv_path),
            output_dir=tmpdir,
            window_size_s=60.0,
            ratios=(1.0, 0.0, 0.0),
        )
        dataset = StaticNIDSDataset(root=tmpdir, split="train")
        all_data = [data for data in dataset]

    return all_data


def _load_session_temporal_data(session_id: UUID):
    """Load temporal data for explainability from pre-built test split."""
    temporal_dir = Path("data/processed/temporal")
    test_path = temporal_dir / "test.pt"
    if not test_path.exists():
        raise HTTPException(
            400,
            detail="Temporal data not available. Run temporal_builder.py first.",
        )

    import torch
    return torch.load(test_path, weights_only=False)


def _sync_explain_flow(session_id: UUID, model_name: str, edge_idx: int, epochs: int) -> dict:
    from src.explain.gnn_explainer import explain_flow

    model = get_model(model_name)
    all_data = _load_session_data(session_id)

    cumulative_edges = 0
    for data in all_data:
        n_edges = data.edge_index.shape[1]
        if edge_idx < cumulative_edges + n_edges:
            local_idx = edge_idx - cumulative_edges
            return explain_flow(model, data, local_idx, epochs=epochs)
        cumulative_edges += n_edges

    raise HTTPException(400, detail=f"edge_idx {edge_idx} out of range (total: {cumulative_edges})")


def _sync_explain_temporal_flow(session_id: UUID, model_name: str, edge_idx: int) -> dict:
    from src.explain.temporal_explainer import explain_temporal_flow

    model = get_model(model_name)
    batch = _load_session_temporal_data(session_id)
    if edge_idx >= len(batch.src):
        total = len(batch.src)
        raise HTTPException(
            400, detail=f"edge_idx {edge_idx} out of range (total: {total})",
        )
    return explain_temporal_flow(model, batch, edge_idx)


def _sync_explain_top(session_id: UUID, model_name: str, top_k: int, epochs: int) -> list:
    from src.explain.gnn_explainer import explain_top_alerts

    model = get_model(model_name)
    all_data = _load_session_data(session_id)

    all_results = []
    for window_idx, data in enumerate(all_data):
        results = explain_top_alerts(model, data, top_k=top_k, epochs=epochs)
        for r in results:
            r["window"] = window_idx
        all_results.extend(results)

    all_results.sort(key=lambda r: r.get("confidence", 0), reverse=True)
    return all_results[:top_k]


def _sync_explain_temporal_top(session_id: UUID, model_name: str, top_k: int) -> list:
    from src.explain.temporal_explainer import explain_temporal_top_alerts

    model = get_model(model_name)
    batch = _load_session_temporal_data(session_id)
    return explain_temporal_top_alerts(model, batch, top_k=top_k)


@router.post("/explain/{session_id}")
async def explain_flow_endpoint(session_id: UUID, req: ExplainRequest):
    """Explain a specific flow prediction."""
    if req.model in TEMPORAL_MODELS:
        return await run_in_threadpool(
            _sync_explain_temporal_flow, session_id, req.model, req.edge_idx
        )
    return await run_in_threadpool(
        _sync_explain_flow, session_id, req.model, req.edge_idx, req.epochs
    )


@router.post("/explain-top/{session_id}")
async def explain_top_endpoint(session_id: UUID, req: ExplainTopRequest):
    """Explain top-K most confident attack predictions."""
    if req.model in TEMPORAL_MODELS:
        return await run_in_threadpool(
            _sync_explain_temporal_top, session_id, req.model, req.top_k
        )
    return await run_in_threadpool(
        _sync_explain_top, session_id, req.model, req.top_k, req.epochs
    )
