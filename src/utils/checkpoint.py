"""
src/utils/checkpoint.py
Model checkpoint save / load utilities.

Training loops should call ``save_checkpoint`` every ``cfg.train.save_every``
epochs and additionally whenever the validation F1 reaches a new best.
``load_checkpoint`` resumes from any saved state and returns the epoch to
continue from, enabling transparent recovery from interruptions.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

log = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    path: str | Path,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a training checkpoint to disk.

    Args:
        model: Model whose ``state_dict`` is saved.
        optimizer: Optimizer whose ``state_dict`` is saved (preserves
            momentum buffers, adaptive learning rates, etc.).
        epoch: The epoch that just finished (used to resume from epoch+1).
        path: Destination file path. Parent directories are created if absent.
        extra: Optional dict of additional metadata to store (e.g. val F1,
            attack config, dataset split info).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if extra:
        payload["extra"] = extra

    torch.save(payload, path)
    log.info("Checkpoint saved → %s (epoch %d)", path, epoch)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optimizer | None,
    path: str | Path,
    map_location: str | torch.device | None = None,
) -> int:
    """Load a checkpoint and restore model (and optionally optimizer) state.

    Args:
        model: Model to restore in-place.
        optimizer: Optimizer to restore in-place. Pass ``None`` for
            inference-only loading (optimizer state is ignored).
        path: Checkpoint file path.
        map_location: Passed directly to ``torch.load``; use ``"cpu"`` when
            loading a GPU checkpoint on a CPU-only machine.

    Returns:
        The epoch stored in the checkpoint. Resume training from epoch + 1.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(payload["model_state"])

    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])

    epoch: int = payload["epoch"]
    log.info("Checkpoint loaded ← %s (epoch %d)", path, epoch)
    return epoch
