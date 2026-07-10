"""GNN inference service.

Models are loaded once at FastAPI startup (lifespan) and held in _models.
All PyTorch operations run in a thread pool executor to avoid blocking the
asyncio event loop.

Checkpoint format expected at ``checkpoints/{name}_best.pt``:
    A complete model object saved with ``torch.save(model, path)`` by
    ``train.py`` whenever a new best validation F1 is reached.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any
from uuid import UUID

import torch
from fastapi.concurrency import run_in_threadpool

from src.models.base import BaseNIDSModel
from src.models.ensemble import EnsembleModel

logger = logging.getLogger(__name__)

_models: dict[str, BaseNIDSModel] = {}
_ensemble: EnsembleModel | None = None

MAX_CONCURRENT_INFER = int(os.getenv("MAX_CONCURRENT_INFER", "3"))
_infer_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFER)

CHECKPOINTS_DIR = Path("checkpoints")
CHECKPOINT_FILES = {
    "graphsage": CHECKPOINTS_DIR / "graphsage_best.pt",
    "gat": CHECKPOINTS_DIR / "gat_best.pt",
    "egraphsage": CHECKPOINTS_DIR / "egraphsage_best.pt",
    "tgat": CHECKPOINTS_DIR / "tgat_best.pt",
    "tgn": CHECKPOINTS_DIR / "tgn_best.pt",
    "dygformer": CHECKPOINTS_DIR / "dygformer_best.pt",
}

TEMPORAL_MODELS = {"tgat", "tgn", "dygformer"}


def _load_single_model(name: str, path: Path) -> BaseNIDSModel | None:
    """Load a complete model object from disk. Returns None if unavailable."""
    if not path.exists():
        logger.warning("Checkpoint not found: %s — model '%s' unavailable", path, name)
        return None
    try:
        from app.services.torch_load import load_torch_artifact
        model: BaseNIDSModel = load_torch_artifact(path)
        model.eval()
        logger.info("Loaded model '%s' from %s", name, path)
        return model
    except Exception as exc:
        logger.error("Failed to load model '%s': %s", name, exc)
        return None


def load_models() -> None:
    """Load all available model checkpoints at startup."""
    global _ensemble
    for name, path in CHECKPOINT_FILES.items():
        model = _load_single_model(name, path)
        if model is not None:
            _models[name] = model
    if not _models:
        logger.warning(
            "No model checkpoints found in %s — inference will not work until models are trained.",
            CHECKPOINTS_DIR,
        )
        return

    static_models = {k: v for k, v in _models.items() if k not in TEMPORAL_MODELS}
    if len(static_models) >= 2:
        _ensemble = EnsembleModel(static_models, strategy="soft_vote")
        logger.info("Ensemble model built from: %s", list(static_models.keys()))


def get_ensemble() -> EnsembleModel:
    if _ensemble is None:
        raise ValueError("Ensemble not available — need at least 2 static models loaded")
    return _ensemble


def get_model(name: str) -> BaseNIDSModel:
    if name not in _models:
        available = list(_models.keys())
        raise ValueError(f"Model '{name}' not available. Loaded: {available}")
    return _models[name]


def _sync_inference(csv_path: str, model_name: str) -> dict[str, Any]:
    """Run the full inference pipeline synchronously (called via run_in_threadpool)."""
    import tempfile

    from app.services.graph_builder import build_graph_response

    model = get_model(model_name)

    if model_name in TEMPORAL_MODELS:
        return _sync_inference_temporal(csv_path, model_name, model)

    from src.data.static_builder import build_static_graphs
    from src.data.static_dataset import StaticNIDSDataset

    # Build graphs into a temp dir; pass ratios=(1,0,0) so ALL flows land
    # in the "train" split — for inference we want every window, not just test.
    with tempfile.TemporaryDirectory() as tmpdir:
        meta = build_static_graphs(
            csv_path=csv_path,
            output_dir=tmpdir,
            window_size_s=60.0,
            ratios=(1.0, 0.0, 0.0),
            label_col="attack_cat",
        )
        dataset = StaticNIDSDataset(root=tmpdir, split="train")

        all_logits: list[torch.Tensor] = []
        all_data = []
        with torch.inference_mode():
            for data in dataset:
                logits = model(data)
                all_logits.append(logits)
                all_data.append(data)

    meta["model"] = model_name
    return build_graph_response(all_data, all_logits, meta, csv_path)


def _sync_inference_temporal(
    csv_path: str, model_name: str, model: BaseNIDSModel
) -> dict[str, Any]:
    """Inference pipeline for temporal models (TGAT/TGN)."""
    import tempfile

    from app.services.graph_builder import build_graph_response
    from src.data.static_builder import build_static_graphs
    from src.data.static_dataset import StaticNIDSDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        meta = build_static_graphs(
            csv_path=csv_path,
            output_dir=tmpdir,
            window_size_s=60.0,
            ratios=(1.0, 0.0, 0.0),
            label_col="attack_cat",
        )
        dataset = StaticNIDSDataset(root=tmpdir, split="train")

        if hasattr(model, "reset_memory"):
            model.reset_memory()

        all_logits: list[torch.Tensor] = []
        all_data = []
        with torch.inference_mode():
            for data in dataset:
                logits = model(data)
                all_logits.append(logits)
                all_data.append(data)

    meta["model"] = model_name
    return build_graph_response(all_data, all_logits, meta, csv_path)


def _sync_inference_ensemble(csv_path: str) -> dict[str, Any]:
    """Run ensemble inference over all static models."""
    import tempfile

    from app.services.graph_builder import build_graph_response

    ensemble = get_ensemble()

    from src.data.static_builder import build_static_graphs
    from src.data.static_dataset import StaticNIDSDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        meta = build_static_graphs(
            csv_path=csv_path,
            output_dir=tmpdir,
            window_size_s=60.0,
            ratios=(1.0, 0.0, 0.0),
            label_col="attack_cat",
        )
        dataset = StaticNIDSDataset(root=tmpdir, split="train")

        all_logits: list[torch.Tensor] = []
        all_data = []
        with torch.inference_mode():
            for data in dataset:
                proba = ensemble(data)
                all_logits.append(proba)
                all_data.append(data)

    meta["model"] = f"ensemble({', '.join(ensemble.model_names)})"
    return build_graph_response(all_data, all_logits, meta, csv_path)


async def run_inference(csv_path: str, model_name: str, session_id: UUID) -> dict[str, Any]:
    """Async wrapper: runs _sync_inference in thread pool with concurrency limit."""
    async with _infer_semaphore:
        if model_name == "ensemble":
            return await run_in_threadpool(_sync_inference_ensemble, csv_path)
        return await run_in_threadpool(_sync_inference, csv_path, model_name)
