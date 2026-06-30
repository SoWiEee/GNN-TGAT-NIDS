"""API endpoint tests for analysis router: upload, status, graph, alerts, timeline."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)


DEMO_CSV = Path("data/demo/demo_flows.csv")


@pytest.fixture(autouse=True)
def _skip_model_load():
    with patch("app.main.load_models"):
        yield


@pytest.fixture
def csv_bytes() -> bytes:
    if DEMO_CSV.exists():
        return DEMO_CSV.read_bytes()[:4096]
    header = "L4_SRC_PORT,L4_DST_PORT,PROTOCOL,IN_BYTES,OUT_BYTES,IN_PKTS,OUT_PKTS,TCP_FLAGS\n"
    row = "12345,80,6,500,200,5,3,2\n"
    return (header + row * 10).encode()


# --- Upload ---


def test_upload_csv(csv_bytes: bytes):
    resp = client.post("/api/upload", files={"file": ("test.csv", csv_bytes, "text/csv")})
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert data["n_flows"] >= 1


def test_upload_rejects_non_csv():
    resp = client.post("/api/upload", files={"file": ("test.txt", b"hello", "text/plain")})
    assert resp.status_code == 422


def test_upload_rejects_oversized(csv_bytes: bytes):
    huge = csv_bytes * 20000
    resp = client.post("/api/upload", files={"file": ("big.csv", huge, "text/csv")})
    assert resp.status_code == 413


# --- Status ---


def test_status_404_unknown_session():
    resp = client.get("/api/status/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


def test_status_after_upload(csv_bytes: bytes):
    upload = client.post("/api/upload", files={"file": ("t.csv", csv_bytes, "text/csv")})
    sid = upload.json()["session_id"]
    resp = client.get(f"/api/status/{sid}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "idle"


# --- Graph ---


def test_graph_404_no_result():
    resp = client.get("/api/graph/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


def test_graph_max_edges_validation():
    resp = client.get("/api/graph/00000000-0000-0000-0000-000000000000?max_edges=5")
    assert resp.status_code == 422

    resp = client.get("/api/graph/00000000-0000-0000-0000-000000000000?max_edges=99999")
    assert resp.status_code == 422


# --- Alerts ---


def test_alerts_pagination_validation():
    sid = "00000000-0000-0000-0000-000000000000"
    resp = client.get(f"/api/alerts/{sid}?page=0")
    assert resp.status_code == 422

    resp = client.get(f"/api/alerts/{sid}?limit=0")
    assert resp.status_code == 422

    resp = client.get(f"/api/alerts/{sid}?limit=999")
    assert resp.status_code == 422


def test_alerts_sort_validation():
    sid = "00000000-0000-0000-0000-000000000000"
    resp = client.get(f"/api/alerts/{sid}?sort=malicious_field")
    assert resp.status_code == 422


def test_alerts_valid_sort_fields():
    sid = "00000000-0000-0000-0000-000000000000"
    for field in ("confidence", "timestamp", "attack_type"):
        resp = client.get(f"/api/alerts/{sid}?sort={field}")
        assert resp.status_code in (200, 404)


def test_alerts_with_results(csv_bytes: bytes):
    upload = client.post("/api/upload", files={"file": ("t.csv", csv_bytes, "text/csv")})
    sid = upload.json()["session_id"]
    sdir = Path("data/sessions") / sid
    result = {
        "graph": {"nodes": [], "edges": []},
        "alerts": [
            {"flow_id": "f1", "attack_type": "Fuzzers", "confidence": 0.9, "timestamp": 100},
            {"flow_id": "f2", "attack_type": "DoS", "confidence": 0.5, "timestamp": 200},
        ],
        "timeline": {},
    }
    (sdir / "result.json").write_text(json.dumps(result))

    resp = client.get(f"/api/alerts/{sid}?sort=confidence&page=1&limit=50")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert data["alerts"][0]["confidence"] >= data["alerts"][1]["confidence"]


# --- Timeline ---


def test_timeline_404():
    resp = client.get("/api/timeline/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


# --- Health ---


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# --- Analyze ---


def test_analyze_invalid_model(csv_bytes: bytes):
    upload = client.post("/api/upload", files={"file": ("t.csv", csv_bytes, "text/csv")})
    sid = upload.json()["session_id"]
    resp = client.post(f"/api/analyze/{sid}", json={"model": "invalid_model"})
    assert resp.status_code == 422


def test_analyze_404_no_session():
    resp = client.post(
        "/api/analyze/00000000-0000-0000-0000-000000000000",
        json={"model": "graphsage"},
    )
    assert resp.status_code == 404
