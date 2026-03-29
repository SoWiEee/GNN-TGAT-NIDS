"""Build static time-window snapshot graphs from NetFlow CSV files.

Usage (Hydra):
    uv run python src/data/static_builder.py --config-name static_default
    uv run python src/data/static_builder.py --config-name static_default data.window_size_s=120

Output layout::

    data/processed/static/
    ├── meta.json          # dataset metadata (n_windows, splits, label2idx, …)
    ├── scaler.pkl         # StandardScaler fitted on train split only
    ├── label2idx.json     # string-label → int mapping
    ├── train/
    │   ├── 00000.pt       # PyG Data for window 0
    │   └── ...
    ├── val/
    └── test/
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

import hydra

from src.data.loader import (
    chronological_split,
    encode_labels,
    get_feature_columns,
    load_csv,
)

# Node features built from per-node aggregation over all incident edges
_NODE_AGG_FEATURES = ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS"]
# Fallback when dataset uses alternate column names
_NODE_AGG_FALLBACKS = {
    "IN_BYTES": ["tot_fwd_byts", "Tot Fwd Pkts"],
    "OUT_BYTES": ["tot_bwd_byts"],
    "IN_PKTS": ["tot_fwd_pkts", "TotLen Fwd Pkts"],
    "OUT_PKTS": ["tot_bwd_pkts"],
    "FLOW_DURATION_MILLISECONDS": ["Flow Duration", "flow_duration"],
}


def _resolve_col(df: pd.DataFrame, primary: str) -> str | None:
    """Return the actual column name for a primary key (with fallback lookup)."""
    if primary in df.columns:
        return primary
    for alt in _NODE_AGG_FALLBACKS.get(primary, []):
        if alt in df.columns:
            return alt
    return None


def _build_node_index(df_window: pd.DataFrame) -> dict[tuple, int]:
    """Map (src_ip, src_port) and (dst_ip, dst_port) tuples to integer indices."""
    src_ip_col = next((c for c in ["IPV4_SRC_ADDR", "src_ip", "Src IP"] if c in df_window.columns), None)
    dst_ip_col = next((c for c in ["IPV4_DST_ADDR", "dst_ip", "Dst IP"] if c in df_window.columns), None)
    src_port_col = next((c for c in ["L4_SRC_PORT", "src_port", "Src Port"] if c in df_window.columns), None)
    dst_port_col = next((c for c in ["L4_DST_PORT", "dst_port", "Dst Port"] if c in df_window.columns), None)

    node2idx: dict[tuple, int] = {}
    counter = 0

    for _, row in df_window.iterrows():
        src_key = (
            row[src_ip_col] if src_ip_col else "unknown_src",
            row[src_port_col] if src_port_col else 0,
        )
        dst_key = (
            row[dst_ip_col] if dst_ip_col else "unknown_dst",
            row[dst_port_col] if dst_port_col else 0,
        )
        if src_key not in node2idx:
            node2idx[src_key] = counter
            counter += 1
        if dst_key not in node2idx:
            node2idx[dst_key] = counter
            counter += 1

    return node2idx


def _compute_node_features(df_window: pd.DataFrame, node2idx: dict[tuple, int]) -> torch.Tensor:
    """Aggregate edge-level statistics per node to produce node feature matrix.

    Returns a tensor of shape ``[num_nodes, 5]``.
    """
    n_nodes = len(node2idx)
    node_feat = np.zeros((n_nodes, 5), dtype=np.float32)

    src_ip_col = next((c for c in ["IPV4_SRC_ADDR", "src_ip", "Src IP"] if c in df_window.columns), None)
    src_port_col = next((c for c in ["L4_SRC_PORT", "src_port", "Src Port"] if c in df_window.columns), None)

    resolved = [_resolve_col(df_window, p) for p in _NODE_AGG_FEATURES]

    for _, row in df_window.iterrows():
        src_key = (
            row[src_ip_col] if src_ip_col else "unknown_src",
            row[src_port_col] if src_port_col else 0,
        )
        if src_key in node2idx:
            idx = node2idx[src_key]
            for fi, col in enumerate(resolved):
                if col is not None and col in df_window.columns:
                    val = row[col]
                    if pd.notna(val):
                        node_feat[idx, fi] += float(val)

    return torch.from_numpy(node_feat)


def _build_pyg_graph(
    df_window: pd.DataFrame,
    feature_cols: list[str],
    node2idx: dict[tuple, int],
    label2idx: dict[str, int],
    scaler: StandardScaler | None = None,
) -> Data:
    """Build a PyG Data object from a single time-window DataFrame."""
    src_ip_col = next((c for c in ["IPV4_SRC_ADDR", "src_ip", "Src IP"] if c in df_window.columns), None)
    dst_ip_col = next((c for c in ["IPV4_DST_ADDR", "dst_ip", "Dst IP"] if c in df_window.columns), None)
    src_port_col = next((c for c in ["L4_SRC_PORT", "src_port", "Src Port"] if c in df_window.columns), None)
    dst_port_col = next((c for c in ["L4_DST_PORT", "dst_port", "Dst Port"] if c in df_window.columns), None)

    # Edge index
    src_indices, dst_indices = [], []
    for _, row in df_window.iterrows():
        src_key = (
            row[src_ip_col] if src_ip_col else "unknown_src",
            row[src_port_col] if src_port_col else 0,
        )
        dst_key = (
            row[dst_ip_col] if dst_ip_col else "unknown_dst",
            row[dst_port_col] if dst_port_col else 0,
        )
        src_indices.append(node2idx[src_key])
        dst_indices.append(node2idx[dst_key])

    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)

    # Edge attributes (normalised feature columns)
    edge_attr_raw = df_window[feature_cols].fillna(0.0).values.astype(np.float32)
    if scaler is not None:
        edge_attr_raw = scaler.transform(edge_attr_raw).astype(np.float32)
    edge_attr = torch.from_numpy(edge_attr_raw)

    # Labels
    y_multi = torch.tensor(
        df_window["_label"].map(label2idx).values, dtype=torch.long
    )
    y = (y_multi > 0).long()

    # Node features
    x = _compute_node_features(df_window, node2idx)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        y_multi=y_multi,
        num_nodes=len(node2idx),
    )


def build_static_graphs(
    csv_path: str | Path,
    output_dir: str | Path,
    window_size_s: float = 60.0,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    clip_sigma: float = 3.0,
    timestamp_col: str | None = None,
    label_col: str | None = None,
) -> dict[str, Any]:
    """Build and save static snapshot graphs from a NetFlow CSV.

    Parameters
    ----------
    csv_path:
        Path to raw CSV file.
    output_dir:
        Root directory for output (split subdirs created automatically).
    window_size_s:
        Time window duration in seconds.
    ratios:
        (train, val, test) fractions.
    clip_sigma:
        Clip features at ±clip_sigma standard deviations during normalisation.
    timestamp_col, label_col:
        Override auto-detection (passed to :func:`load_csv`).

    Returns
    -------
    dict
        Metadata written to ``meta.json``.
    """
    output_dir = Path(output_dir)
    for split in ("train", "val", "test"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    # ── Load ────────────────────────────────────────────────────────────────
    df = load_csv(csv_path, timestamp_col=timestamp_col, label_col=label_col)
    encoded, label2idx = encode_labels(df)
    df["_encoded"] = encoded

    # ── Chronological split ─────────────────────────────────────────────────
    train_df, val_df, test_df = chronological_split(df, ratios=ratios)
    split_dfs = {"train": train_df, "val": val_df, "test": test_df}

    # ── Fit scaler on train only ────────────────────────────────────────────
    feature_cols = get_feature_columns(df)
    train_feat = train_df[feature_cols].fillna(0.0).values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(train_feat)

    # Apply ±clip_sigma clipping
    clip_lo = scaler.mean_ - clip_sigma * scaler.scale_
    clip_hi = scaler.mean_ + clip_sigma * scaler.scale_
    # Store clipping bounds as attributes for downstream use
    scaler.clip_lo_ = clip_lo  # type: ignore[attr-defined]
    scaler.clip_hi_ = clip_hi  # type: ignore[attr-defined]

    # ── Save scaler and label map ───────────────────────────────────────────
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(output_dir / "label2idx.json", "w") as f:
        json.dump(label2idx, f, indent=2)

    # ── Build graphs per split ──────────────────────────────────────────────
    split_counts: dict[str, int] = {}

    for split_name, split_df in split_dfs.items():
        if split_df.empty:
            split_counts[split_name] = 0
            continue

        ts_min = split_df["_ts"].min()
        ts_max = split_df["_ts"].max()
        # Ensure at least one window when all flows fall in the same second
        if ts_max <= ts_min + window_size_s:
            window_starts = np.array([ts_min])
        else:
            window_starts = np.arange(ts_min, ts_max, window_size_s)

        graph_idx = 0
        for i, w_start in enumerate(window_starts):
            is_last_window = i == len(window_starts) - 1
            w_end = w_start + window_size_s
            if is_last_window:
                # Include all remaining rows (handles ts_max == w_start case)
                mask = split_df["_ts"] >= w_start
            else:
                mask = (split_df["_ts"] >= w_start) & (split_df["_ts"] < w_end)
            df_window = split_df[mask]

            if df_window.empty:
                continue

            node2idx = _build_node_index(df_window)
            if len(node2idx) == 0:
                continue

            graph = _build_pyg_graph(
                df_window,
                feature_cols,
                node2idx,
                label2idx,
                scaler=scaler if split_name == "train" else scaler,
            )
            torch.save(graph, output_dir / split_name / f"{graph_idx:05d}.pt")
            graph_idx += 1

        split_counts[split_name] = graph_idx

    # ── Save meta.json ──────────────────────────────────────────────────────
    meta: dict[str, Any] = {
        "csv_path": str(csv_path),
        "window_size_s": window_size_s,
        "ratios": list(ratios),
        "clip_sigma": clip_sigma,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "label2idx": label2idx,
        "split_counts": split_counts,
        "n_classes": len(label2idx),
    }

    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


@hydra.main(config_path="../../configs/data", config_name="static_default", version_base=None)
def main(cfg: DictConfig) -> None:
    from omegaconf import OmegaConf

    print(OmegaConf.to_yaml(cfg))

    meta = build_static_graphs(
        csv_path=cfg.raw_path,
        output_dir=cfg.processed_dir,
        window_size_s=cfg.window_size_s,
        ratios=tuple(cfg.splits),
        clip_sigma=cfg.normalization.clip_sigma,
    )

    print(f"Built {sum(meta['split_counts'].values())} total windows.")
    for split, count in meta["split_counts"].items():
        print(f"  {split}: {count} windows")


if __name__ == "__main__":
    main()
