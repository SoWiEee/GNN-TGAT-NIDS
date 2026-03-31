"""Analysis router: upload CSV, run GNN inference, serve graph/alerts/timeline."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from app.schemas import AnalyzeRequest, AnalyzeResponse, StatusResponse
from app.services.inference import run_inference

router = APIRouter(tags=["analysis"])

SESSIONS_DIR = Path("data/sessions")
MAX_UPLOAD_BYTES = int(50 * 1024 * 1024)  # 50 MB


def _session_dir(session_id: UUID) -> Path:
    return SESSIONS_DIR / str(session_id)


def _write_status(session_id: UUID, status: str, progress: float = 0.0, message: str = "") -> None:
    path = _session_dir(session_id) / "status.json"
    _atomic_write_json(path, {"status": status, "progress_pct": progress, "message": message})


def _atomic_write_json(path: Path, data: dict) -> None:
    import os
    import tempfile
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as f:
        json.dump(data, f)
        tmp = f.name
    os.replace(tmp, path)


@router.post("/upload", response_model=dict)
async def upload_csv(file: UploadFile = File(...)):
    """Upload a NetFlow CSV and create a session."""
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, detail=f"File too large (max {MAX_UPLOAD_BYTES // 1024 ** 2} MB)")
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(422, detail="Only .csv files are accepted")

    session_id = uuid.uuid4()
    sdir = _session_dir(session_id)
    sdir.mkdir(parents=True, exist_ok=True)

    csv_path = sdir / "upload.csv"
    csv_path.write_bytes(content)
    _write_status(session_id, "idle")

    # Count rows (approximate: count newlines minus header)
    n_flows = max(0, content.count(b"\n") - 1)
    return {"session_id": str(session_id), "n_flows": n_flows, "filename": file.filename}


async def _run_analysis_bg(session_id: UUID, model_name: str) -> None:
    """Background task: run GNN inference and save results."""
    _write_status(session_id, "analyzing", 0.0)
    try:
        sdir = _session_dir(session_id)
        csv_path = sdir / "upload.csv"
        result = await run_inference(str(csv_path), model_name, session_id)
        _atomic_write_json(sdir / "result.json", result)
        _write_status(session_id, "ready", 100.0)
    except Exception as exc:
        _write_status(session_id, "error", 0.0, str(exc))
        raise


@router.post("/analyze/{session_id}", status_code=202)
async def analyze(session_id: UUID, req: AnalyzeRequest, background_tasks: BackgroundTasks):
    """Start GNN inference (async). Poll /status/{session_id} for progress."""
    sdir = _session_dir(session_id)
    if not (sdir / "upload.csv").exists():
        raise HTTPException(404, detail="Session not found or CSV not uploaded")
    background_tasks.add_task(_run_analysis_bg, session_id, req.model)
    return AnalyzeResponse(session_id=session_id, status="analyzing")


@router.get("/status/{session_id}", response_model=StatusResponse)
async def get_status(session_id: UUID):
    status_path = _session_dir(session_id) / "status.json"
    if not status_path.exists():
        raise HTTPException(404, detail="Session not found")
    data = json.loads(status_path.read_text())
    return StatusResponse(session_id=session_id, **data)


def _load_result(session_id: UUID) -> dict:
    result_path = _session_dir(session_id) / "result.json"
    if not result_path.exists():
        raise HTTPException(404, detail="Results not ready. Run /analyze first.")
    return json.loads(result_path.read_text())


@router.get("/graph/{session_id}")
async def get_graph(session_id: UUID, max_edges: int = 2000):
    result = _load_result(session_id)
    graph = result.get("graph", {"nodes": [], "edges": []})
    # Truncate edges by confidence (descending) to max_edges
    edges = sorted(
        graph.get("edges", []), key=lambda e: e["data"].get("confidence", 0), reverse=True
    )
    graph["edges"] = edges[:max_edges]
    return graph


@router.get("/alerts/{session_id}")
async def get_alerts(
    session_id: UUID,
    sort: str = "confidence",
    page: int = 1,
    limit: int = 50,
    attack_type: str = "",
):
    result = _load_result(session_id)
    alerts = result.get("alerts", [])
    if attack_type:
        alerts = [a for a in alerts if a.get("attack_type", "").lower() == attack_type.lower()]
    if sort == "confidence":
        alerts = sorted(alerts, key=lambda a: a.get("confidence", 0), reverse=True)
    total = len(alerts)
    start = (page - 1) * limit
    return {"alerts": alerts[start : start + limit], "total": total}


@router.get("/timeline/{session_id}")
async def get_timeline(session_id: UUID):
    result = _load_result(session_id)
    return result.get("timeline", {})
