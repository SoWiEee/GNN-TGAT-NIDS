"""Report download and model metrics endpoints."""
from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from app.schemas import ReportRequest

router = APIRouter(tags=["report"])

SESSIONS_DIR = Path("data/sessions")
METRICS_PATH = Path("data/metrics/reliability.json")


@router.post("/report/{session_id}")
async def generate_report(session_id: UUID, req: ReportRequest):
    """Generate PDF or HTML report. Accepts base64 graph PNG from frontend."""
    from app.services.report_builder import build_report

    sdir = SESSIONS_DIR / str(session_id)
    result_path = sdir / "result.json"
    if not result_path.exists():
        raise HTTPException(404, detail="Results not ready")

    result = json.loads(result_path.read_text())
    graph_png_b64 = req.graph_png_b64

    await run_in_threadpool(
        build_report, session_id, result, graph_png_b64, sdir
    )
    return {"report_url": f"/api/report/{session_id}/download"}


@router.get("/report/{session_id}/download")
async def download_report(
    session_id: UUID, format: str = Query(default="pdf", pattern="^(pdf|html)$")
):
    sdir = SESSIONS_DIR / str(session_id)
    path = sdir / f"report.{format}"
    if not path.exists():
        raise HTTPException(404, detail=f"Report ({format}) not generated yet.")
    media = "application/pdf" if format == "pdf" else "text/html"
    return FileResponse(path, media_type=media, filename=f"nids_report.{format}")


@router.get("/metrics")
async def get_metrics():
    """Return pre-computed model reliability metrics."""
    if not METRICS_PATH.exists():
        placeholder = {
            "clean_f1": None,
            "dr_under_cpgd_eps01": None,
            "delta_f1_after_adv_training": None,
        }
        return {"graphsage": placeholder, "gat": placeholder}
    return json.loads(METRICS_PATH.read_text())
