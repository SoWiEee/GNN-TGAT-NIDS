"""Centralised torch.load wrapper.

All app/ code that needs to deserialise .pt files MUST go through this
function so that the security implications of pickle-based loading are
contained in one auditable location.

If the project migrates to state_dict-only checkpoints in the future,
only this module needs to change.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def load_torch_artifact(path: str | Path, *, map_location: str = "cpu"):
    # weights_only=False is required because training saves complete model
    # objects (torch.save(model, ...)), not state_dicts.  Only load files
    # from the local checkpoints/ and data/processed/ directories that are
    # produced by our own training pipeline.
    return torch.load(path, map_location=map_location, weights_only=False)
