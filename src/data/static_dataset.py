"""On-demand PyG Dataset for pre-built static snapshot graphs."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data, Dataset


class StaticNIDSDataset(Dataset):
    """Load pre-built static snapshot graphs on demand.

    Expects the directory layout produced by :func:`src.data.static_builder.build_static_graphs`::

        root/
        ├── meta.json
        ├── scaler.pkl
        ├── label2idx.json
        ├── train/  *.pt
        ├── val/    *.pt
        └── test/   *.pt

    Parameters
    ----------
    root:
        Root directory (parent of train/val/test subdirs).
    split:
        One of ``"train"``, ``"val"``, ``"test"``.
    """

    def __init__(self, root: str | Path, split: str = "train") -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

        self._root = Path(root)
        self._split = split
        self._split_dir = self._root / split

        # Load metadata
        meta_path = self._root / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found in {self._root}")

        with open(meta_path) as f:
            self._meta: dict[str, Any] = json.load(f)

        # Sorted list of .pt file paths
        self._files = sorted(self._split_dir.glob("*.pt"))

        # Lazy-loaded scaler and label map
        self._scaler = None
        self._label2idx: dict[str, int] | None = None

        # Do NOT call super().__init__() with transforms — we manage files manually
        super().__init__(root=str(root))

    # ── PyG Dataset interface ────────────────────────────────────────────────

    def len(self) -> int:
        return len(self._files)

    def get(self, idx: int) -> Data:
        return torch.load(self._files[idx], weights_only=False)

    # ── Metadata helpers ─────────────────────────────────────────────────────

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta

    @property
    def n_classes(self) -> int:
        return self._meta["n_classes"]

    @property
    def n_edge_features(self) -> int:
        return self._meta["n_features"]

    @property
    def label2idx(self) -> dict[str, int]:
        if self._label2idx is None:
            lbl_path = self._root / "label2idx.json"
            with open(lbl_path) as f:
                self._label2idx = json.load(f)
        return self._label2idx

    @property
    def scaler(self):
        """Return the StandardScaler fitted on the train split."""
        if self._scaler is None:
            scaler_path = self._root / "scaler.pkl"
            with open(scaler_path, "rb") as f:
                self._scaler = pickle.load(f)
        return self._scaler

    # ── PyG internal bookkeeping (skip processing/downloading) ──────────────

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return []

    def download(self) -> None:
        pass

    def process(self) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"split={self._split!r}, "
            f"n_graphs={len(self)}, "
            f"n_classes={self.n_classes})"
        )
