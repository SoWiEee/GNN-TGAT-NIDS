"""Adversarial comparison router: C-PGD generation with caching and timeout."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.schemas import AdversarialRequest

router = APIRouter(tags=["adversarial"])

SESSIONS_DIR = Path("data/sessions")
ADV_TIMEOUT_SECONDS = 30.0


def _cache_path(session_id: UUID, flow_id: str, epsilon: float, steps: int) -> Path:
    key = f"{flow_id}_eps{epsilon:.3f}_steps{steps}"
    return SESSIONS_DIR / str(session_id) / "adversarial" / f"{key}.json"


def _sync_run_cpgd(session_id: UUID, flow_id: str, epsilon: float, steps: int) -> dict:
    """Run C-PGD synchronously (called via run_in_threadpool)."""
    # Import here to avoid circular imports and keep startup fast
    from app.services.cpgd_service import generate_adversarial_example

    result_path = SESSIONS_DIR / str(session_id) / "result.json"
    if not result_path.exists():
        raise FileNotFoundError("Results not found for session")
    result = json.loads(result_path.read_text())
    return generate_adversarial_example(result, flow_id, epsilon, steps)


@router.post("/adversarial")
async def generate_adversarial(req: AdversarialRequest):
    """Generate a C-PGD adversarial example for a specific flow.

    Caches result by (flow_id, epsilon, steps) — repeated calls are instant.
    Times out after 30 s; returns null adversarial on failure.
    """
    cache = _cache_path(req.session_id, req.flow_id, req.epsilon, req.steps)

    # Return cached result if available
    if cache.exists():
        return json.loads(cache.read_text())

    sdir = SESSIONS_DIR / str(req.session_id)
    if not sdir.exists():
        raise HTTPException(404, detail="Session not found")
    cache.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = await asyncio.wait_for(
            run_in_threadpool(_sync_run_cpgd, req.session_id, req.flow_id, req.epsilon, req.steps),
            timeout=ADV_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        raise HTTPException(408, detail="Adversarial generation timed out (>30 s).")
    except FileNotFoundError as exc:
        raise HTTPException(404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(500, detail=f"C-PGD failed: {exc}")

    # Persist cache
    cache.write_text(json.dumps(result))
    return result
