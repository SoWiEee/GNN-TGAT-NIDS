"""Pydantic request/response schemas for the GNN-NIDS Analyzer API."""
from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    model: str = Field(default="gat", pattern="^(graphsage|gat|egraphsage|tgat|tgn|ensemble)$")


class AnalyzeResponse(BaseModel):
    session_id: UUID
    status: str = "analyzing"


class StatusResponse(BaseModel):
    session_id: UUID
    status: str  # idle | analyzing | ready | error
    progress_pct: float = 0.0
    message: str = ""


class AdversarialRequest(BaseModel):
    session_id: UUID
    flow_id: str
    epsilon: float = Field(default=0.1, ge=0.0, le=1.0)
    steps: int = Field(default=40, ge=1, le=200)


class ReportRequest(BaseModel):
    session_id: UUID
    graph_png_b64: str = ""  # base64 PNG from Cytoscape.js cy.png()


class MemoryPoisoningRequest(BaseModel):
    model: str = Field(default="tgn", pattern="^(tgn|tgat)$")
    n_poison: int = Field(default=3, ge=0, le=20)
    max_batches: int = Field(default=20, ge=1, le=200)
    batch_size: int = Field(default=200, ge=10, le=2000)
    poison_strategy: str = Field(default="benign_mean", pattern="^(benign_mean|random_benign)$")
