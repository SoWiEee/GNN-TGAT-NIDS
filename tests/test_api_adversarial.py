"""API endpoint tests for adversarial router."""
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


def test_adversarial_404_no_session():
    resp = client.post(
        "/api/adversarial",
        json={
            "session_id": "00000000-0000-0000-0000-000000000000",
            "flow_id": "f0",
            "epsilon": 0.1,
            "steps": 40,
        },
    )
    assert resp.status_code == 404


def test_adversarial_epsilon_validation():
    resp = client.post(
        "/api/adversarial",
        json={
            "session_id": "00000000-0000-0000-0000-000000000000",
            "flow_id": "f0",
            "epsilon": 5.0,
            "steps": 40,
        },
    )
    assert resp.status_code == 422


def test_adversarial_steps_validation():
    resp = client.post(
        "/api/adversarial",
        json={
            "session_id": "00000000-0000-0000-0000-000000000000",
            "flow_id": "f0",
            "epsilon": 0.1,
            "steps": 999,
        },
    )
    assert resp.status_code == 422


def test_adversarial_error_does_not_leak_details():
    """500 responses must not contain exception details."""
    from pathlib import Path

    sid = "00000000-0000-0000-0000-111111111111"
    sdir = Path("data/sessions") / sid
    sdir.mkdir(parents=True, exist_ok=True)

    with patch(
        "app.routers.adversarial._sync_run_cpgd",
        side_effect=RuntimeError("secret internal error: /path/to/file"),
    ):
        resp = client.post(
            "/api/adversarial",
            json={"session_id": sid, "flow_id": "f0", "epsilon": 0.1, "steps": 10},
        )
    assert resp.status_code == 500
    assert "secret" not in resp.json()["detail"]
    assert "path" not in resp.json()["detail"]

    import shutil
    shutil.rmtree(sdir, ignore_errors=True)


def test_adversarial_disabled_when_env_false():
    with patch("app.routers.adversarial.ENABLE_ATTACK_ENDPOINTS", False):
        resp = client.post(
            "/api/adversarial",
            json={
                "session_id": "00000000-0000-0000-0000-000000000000",
                "flow_id": "f0",
                "epsilon": 0.1,
                "steps": 10,
            },
        )
    assert resp.status_code == 403


def test_memory_poisoning_disabled_when_env_false():
    with patch("app.routers.memory_poisoning.ENABLE_ATTACK_ENDPOINTS", False):
        resp = client.post(
            "/api/memory-poisoning",
            json={"model": "tgn", "n_poison": 3, "max_batches": 5, "batch_size": 50},
        )
    assert resp.status_code == 403
