"""FastAPI application entry point for GNN-NIDS Analyzer."""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import adversarial, analysis, report
from app.services.inference import load_models

logger = logging.getLogger(__name__)

SESSIONS_DIR = Path("data/sessions")
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
CLEANUP_INTERVAL_SECONDS = 300  # check every 5 minutes


async def _cleanup_sessions() -> None:
    """Background task: remove session directories older than SESSION_TTL_SECONDS."""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        if not SESSIONS_DIR.exists():
            continue
        now = time.time()
        for session_dir in SESSIONS_DIR.iterdir():
            if not session_dir.is_dir():
                continue
            try:
                age = now - session_dir.stat().st_mtime
                if age > SESSION_TTL_SECONDS:
                    shutil.rmtree(session_dir, ignore_errors=True)
                    logger.info("Cleaned up session %s (age %.0fs)", session_dir.name, age)
            except OSError:
                pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    load_models()
    cleanup_task = asyncio.create_task(_cleanup_sessions())
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="GNN-NIDS Analyzer", version="1.0.0", lifespan=lifespan)

origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis.router, prefix="/api")
app.include_router(adversarial.router, prefix="/api")
app.include_router(report.router, prefix="/api")
