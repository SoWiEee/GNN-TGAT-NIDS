"""API endpoint tests for WebSocket streaming router."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _skip_model_load():
    with patch("app.main.load_models"):
        yield


def test_ws_rejects_invalid_window_seconds():
    with client.websocket_connect("/api/ws/stream?model=graphsage&window_seconds=-1") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "window_seconds" in msg["message"]


def test_ws_rejects_zero_window():
    with client.websocket_connect("/api/ws/stream?model=graphsage&window_seconds=0") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "error"


def test_ws_rejects_huge_window():
    with client.websocket_connect("/api/ws/stream?model=graphsage&window_seconds=9999") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "window_seconds" in msg["message"]


def test_ws_rejects_unknown_model():
    with client.websocket_connect("/api/ws/stream?model=nonexistent&window_seconds=60") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "nonexistent" in msg["message"]
