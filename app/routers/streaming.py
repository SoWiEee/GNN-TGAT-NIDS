"""Streaming inference router: WebSocket endpoint for real-time NetFlow analysis.

Clients connect via WebSocket and send NetFlow records as JSON lines.
The server accumulates flows into time windows, runs GNN inference on each
completed window, and pushes alerts back in real-time.

Protocol:
    Client → Server (JSON per message):
        {"flows": [{"col1": val, "col2": val, ...}, ...]}
        {"command": "flush"}   — force inference on accumulated flows
        {"command": "close"}   — close the stream

    Server → Client (JSON per message):
        {"type": "ack", "n_buffered": int, "n_processed": int}
        {"type": "alerts", "window": int, "alerts": [...], "stats": {...}}
        {"type": "error", "message": str}
"""
from __future__ import annotations

import csv
import io
import json
import logging
import time

import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.inference import get_model

logger = logging.getLogger(__name__)
router = APIRouter(tags=["streaming"])

DEFAULT_WINDOW_SECONDS = 60.0
MAX_BUFFER_SIZE = 10_000


class StreamingSession:
    """Accumulates flows and runs inference when a time window completes."""

    def __init__(self, model_name: str, window_seconds: float = DEFAULT_WINDOW_SECONDS) -> None:
        self.model_name = model_name
        self.window_seconds = window_seconds
        self.buffer: list[dict] = []
        self.window_start: float | None = None
        self.n_processed = 0
        self.window_idx = 0

    def add_flows(self, flows: list[dict]) -> list[dict]:
        """Add flows and return completed windows (list of flow lists)."""
        completed: list[list[dict]] = []
        for flow in flows:
            ts = self._extract_timestamp(flow)
            if self.window_start is None:
                self.window_start = ts

            if ts - self.window_start >= self.window_seconds and self.buffer:
                completed.append(list(self.buffer))
                self.buffer.clear()
                self.window_start = ts

            self.buffer.append(flow)
            if len(self.buffer) >= MAX_BUFFER_SIZE:
                completed.append(list(self.buffer))
                self.buffer.clear()
                self.window_start = None

        results = []
        for window_flows in completed:
            result = self._run_inference(window_flows)
            if result is not None:
                results.append(result)
        return results

    def flush(self) -> dict | None:
        """Force inference on remaining buffered flows."""
        if not self.buffer:
            return None
        result = self._run_inference(list(self.buffer))
        self.buffer.clear()
        self.window_start = None
        return result

    def _extract_timestamp(self, flow: dict) -> float:
        for key in ("timestamp", "Timestamp", "TIMESTAMP", "ts"):
            if key in flow:
                try:
                    return float(flow[key])
                except (ValueError, TypeError):
                    pass
        return time.time()

    def _run_inference(self, flows: list[dict]) -> dict | None:
        """Build graph from flows and run model inference."""
        import tempfile

        from src.data.static_builder import build_static_graphs
        from src.data.static_dataset import StaticNIDSDataset

        csv_content = self._flows_to_csv(flows)
        if csv_content is None:
            return None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as f:
                f.write(csv_content)
                csv_path = f.name

            with tempfile.TemporaryDirectory() as tmpdir:
                meta = build_static_graphs(
                    csv_path=csv_path,
                    output_dir=tmpdir,
                    window_size_s=self.window_seconds,
                    ratios=(1.0, 0.0, 0.0),
                    label_col="attack_cat",
                )
                dataset = StaticNIDSDataset(root=tmpdir, split="train")

                model = get_model(self.model_name)
                label2idx = meta.get("label2idx", {})
                idx2label = {v: k for k, v in label2idx.items()}

                alerts: list[dict] = []
                total_flows = 0
                total_attacks = 0

                with torch.inference_mode():
                    for data in dataset:
                        logits = model(data)
                        proba = torch.softmax(logits, dim=-1)
                        preds = proba.argmax(dim=-1)
                        confidence = proba.max(dim=-1).values

                        edge_index = data.edge_index
                        for i in range(preds.shape[0]):
                            total_flows += 1
                            pred_class = int(preds[i])
                            if pred_class > 0:
                                total_attacks += 1
                                alerts.append({
                                    "src": f"n{int(edge_index[0, i])}",
                                    "dst": f"n{int(edge_index[1, i])}",
                                    "attack_type": idx2label.get(pred_class, "Unknown"),
                                    "confidence": round(float(confidence[i]), 4),
                                })

            self.n_processed += total_flows
            window = self.window_idx
            self.window_idx += 1

            return {
                "type": "alerts",
                "window": window,
                "alerts": alerts,
                "stats": {
                    "total_flows": total_flows,
                    "total_attacks": total_attacks,
                    "detection_rate": round(total_attacks / max(total_flows, 1), 4),
                },
            }
        except Exception as exc:
            logger.error("Streaming inference error: %s", exc)
            return {"type": "error", "message": str(exc)}

    def _flows_to_csv(self, flows: list[dict]) -> str | None:
        if not flows:
            return None
        fieldnames = list(flows[0].keys())
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flows)
        return buf.getvalue()


@router.websocket("/ws/stream")
async def websocket_stream(
    websocket: WebSocket,
    model: str = "graphsage",
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
):
    """WebSocket endpoint for real-time NetFlow analysis."""
    await websocket.accept()

    try:
        get_model(model)
    except ValueError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close(code=1008)
        return

    session = StreamingSession(model, window_seconds)
    logger.info("Streaming session started: model=%s, window=%.0fs", model, window_seconds)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            if isinstance(msg, dict) and msg.get("command") == "close":
                result = session.flush()
                if result:
                    await websocket.send_json(result)
                break

            if isinstance(msg, dict) and msg.get("command") == "flush":
                result = session.flush()
                if result:
                    await websocket.send_json(result)
                else:
                    await websocket.send_json({
                        "type": "ack",
                        "n_buffered": 0,
                        "n_processed": session.n_processed,
                    })
                continue

            flows = msg.get("flows", []) if isinstance(msg, dict) else []
            if not flows:
                await websocket.send_json({"type": "error", "message": "No flows in message"})
                continue

            from fastapi.concurrency import run_in_threadpool
            results = await run_in_threadpool(session.add_flows, flows)

            for result in results:
                await websocket.send_json(result)

            await websocket.send_json({
                "type": "ack",
                "n_buffered": len(session.buffer),
                "n_processed": session.n_processed,
            })

    except WebSocketDisconnect:
        logger.info("Streaming client disconnected")
    finally:
        result = session.flush()
        if result:
            logger.info("Final flush: %d alerts", len(result.get("alerts", [])))
