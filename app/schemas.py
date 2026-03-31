"""Pydantic request/response schemas for the GNN-NIDS Analyzer API."""
from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    model: str = Field(default="gat", pattern="^(graphsage|gat)$")


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
