"""Build temporal graph data (TemporalData) from NetFlow CSV.

Produces one TemporalData object per chronological split, stored as::

    data/processed/temporal/
    ├── meta.json        # num_nodes, label2idx, feat_dim, …
    ├── scaler.pkl       # StandardScaler fitted on train only
    ├── train.pt         # TemporalData (chronological order)
    ├── val.pt
    └── test.pt

Unlike the static builder, no windowing is performed — every NetFlow record
becomes one event in the continuous-time graph.  Node identity is IP-level
(each unique IP address = one node), which is natural for TGN's per-node
memory.

Usage::

    uv run python src/data/temporal_builder.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import TemporalData

from src.data.loader import (
    encode_labels,
    get_feature_columns,
    load_csv,
)

# IP address columns to use for node identity (not included as edge features)
_IP_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "src_ip", "dst_ip", "Src IP", "Dst IP"]


def _get_ip_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def build_temporal_data(cfg: DictConfig) -> None:
    """Build and save TemporalData objects for train / val / test splits."""
    out_dir = Path(cfg.data.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load CSV ───────────────────────────────────────────────────────────
    print(f"Loading {cfg.data.raw_path} …")
    df = load_csv(cfg.data.raw_path, label_col=cfg.data.label_col)
    df = df.sort_values("_ts").reset_index(drop=True)
    print(f"  {len(df):,} events loaded")

    # ── 2. Label encoding ─────────────────────────────────────────────────────
    labels, label2idx = encode_labels(df)

    # ── 3. Node identity: IP address → integer ID ─────────────────────────────
    src_ip_col = _get_ip_col(df, ["IPV4_SRC_ADDR", "src_ip", "Src IP"])
    dst_ip_col = _get_ip_col(df, ["IPV4_DST_ADDR", "dst_ip", "Dst IP"])

    if src_ip_col is None or dst_ip_col is None:
        raise ValueError(
            "IP address columns not found. "
            "Expected IPV4_SRC_ADDR / IPV4_DST_ADDR in the CSV."
        )

    all_ips = pd.unique(
        np.concatenate([df[src_ip_col].values, df[dst_ip_col].values]).astype(str)
    )
    ip2id: dict[str, int] = {ip: i for i, ip in enumerate(all_ips)}
    num_nodes = len(ip2id)
    print(f"  {num_nodes} unique IP nodes")

    src_nodes = np.array(
        [ip2id[str(ip)] for ip in df[src_ip_col].values], dtype=np.int64
    )
    dst_nodes = np.array(
        [ip2id[str(ip)] for ip in df[dst_ip_col].values], dtype=np.int64
    )

    # ── 4. Timestamps (float seconds, relative to first event) ────────────────
    timestamps = df["_ts"].values.astype(np.float32)
    t0 = timestamps.min()
    timestamps = timestamps - t0

    # ── 5. Feature matrix (all numeric cols except _ts, _label, IP columns) ───
    exclude = {c for c in _IP_COLS if c in df.columns}
    feat_cols = get_feature_columns(df, exclude=list(exclude))
    feats = df[feat_cols].fillna(0.0).values.astype(np.float32)
    feat_dim = feats.shape[1]
    print(f"  {feat_dim} edge features")

    # ── 6. Chronological split ────────────────────────────────────────────────
    ratios = (
        cfg.data.splits.get("train", 0.6),
        cfg.data.splits.get("val", 0.2),
        cfg.data.splits.get("test", 0.2),
    )
    n = len(df)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    idx: dict[str, slice] = {
        "train": slice(0, n_train),
        "val":   slice(n_train, n_train + n_val),
        "test":  slice(n_train + n_val, n),
    }

    # ── 7. Fit scaler on train split only ─────────────────────────────────────
    scaler = StandardScaler()
    feats[idx["train"]] = scaler.fit_transform(feats[idx["train"]])
    feats[idx["val"]]   = scaler.transform(feats[idx["val"]])
    feats[idx["test"]]  = scaler.transform(feats[idx["test"]])

    sigma = cfg.data.normalization.clip_sigma
    feats = np.clip(feats, -sigma, sigma)

    labels_arr = labels.values.astype(np.int64)

    # ── 8. Build and save TemporalData per split ───────────────────────────────
    counts: dict[str, int] = {}
    for split, sl in idx.items():
        td = TemporalData(
            src=torch.from_numpy(src_nodes[sl]),
            dst=torch.from_numpy(dst_nodes[sl]),
            t=torch.from_numpy(timestamps[sl]),
            msg=torch.from_numpy(feats[sl]),
            y=torch.from_numpy(labels_arr[sl]),
        )
        torch.save(td, out_dir / f"{split}.pt")
        counts[split] = int(sl.stop - sl.start) if sl.stop else n - sl.start
        print(f"  {split}: {counts[split]:,} events → {out_dir / split}.pt")

    # ── 9. Save scaler and metadata ───────────────────────────────────────────
    scaler_path = out_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "num_nodes":    num_nodes,
        "n_features":   feat_dim,
        "n_classes":    len(label2idx),
        "label2idx":    label2idx,
        "ip2id":        ip2id,
        "feat_cols":    feat_cols,
        "t0_seconds":   float(t0),
        "split_counts": counts,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved meta.json and scaler.pkl to {out_dir}")


@hydra.main(version_base=None, config_path="../../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    import logging
    logging.basicConfig(level=logging.INFO)
    build_temporal_data(cfg)


if __name__ == "__main__":
    main()
