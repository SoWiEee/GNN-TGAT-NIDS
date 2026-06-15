"""Tests for the streaming inference session."""
from __future__ import annotations

import time

from app.routers.streaming import StreamingSession


class _FakeStreamingSession(StreamingSession):
    """Override _run_inference to avoid needing real model/data."""

    def __init__(self, **kwargs):
        super().__init__(model_name="graphsage", **kwargs)
        self.inference_calls: list[list[dict]] = []

    def _run_inference(self, flows: list[dict]) -> dict | None:
        self.inference_calls.append(flows)
        return {
            "type": "alerts",
            "window": self.window_idx,
            "alerts": [],
            "stats": {"total_flows": len(flows), "total_attacks": 0, "detection_rate": 0.0},
        }


def test_window_triggers_on_time_boundary():
    session = _FakeStreamingSession(window_seconds=10.0)
    flows = [
        {"timestamp": 100.0, "col1": 1},
        {"timestamp": 105.0, "col1": 2},
        {"timestamp": 111.0, "col1": 3},
    ]
    session.add_flows(flows)

    assert len(session.inference_calls) == 1
    assert len(session.inference_calls[0]) == 2
    assert len(session.buffer) == 1


def test_flush_processes_remaining():
    session = _FakeStreamingSession(window_seconds=60.0)
    session.add_flows([{"timestamp": 1.0, "val": "a"}, {"timestamp": 2.0, "val": "b"}])

    assert len(session.inference_calls) == 0
    assert len(session.buffer) == 2

    result = session.flush()
    assert result is not None
    assert result["stats"]["total_flows"] == 2
    assert len(session.buffer) == 0


def test_empty_flush_returns_none():
    session = _FakeStreamingSession()
    assert session.flush() is None


def test_buffer_overflow_triggers_inference():
    session = _FakeStreamingSession(window_seconds=9999.0)
    large_batch = [{"timestamp": 1.0} for _ in range(10_001)]
    session.add_flows(large_batch)

    assert len(session.inference_calls) == 1
    assert len(session.inference_calls[0]) == 10_000
    assert len(session.buffer) == 1


def test_flows_to_csv():
    session = _FakeStreamingSession()
    csv = session._flows_to_csv([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    assert csv is not None
    lines = csv.strip().split("\n")
    assert len(lines) == 3  # header + 2 rows
    assert "a,b" in lines[0]


def test_timestamp_fallback():
    session = _FakeStreamingSession(window_seconds=1.0)
    now = time.time()
    session.add_flows([{"no_ts_field": "val"}])
    assert session.window_start is not None
    assert abs(session.window_start - now) < 2.0
