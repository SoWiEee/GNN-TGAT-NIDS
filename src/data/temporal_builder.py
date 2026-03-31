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

    uv run python src/data/temporal_builder.py +data=temporal_default
"""

from __future__ import annotations

import glob
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

from src.data.loader import get_feature_columns

# Standard 49-column schema for the raw UNSW-NB15 CSV files (no header row).
# Source: UNSW-NB15 dataset documentation / NUSW-NB15_features.csv
_UNSW_NB15_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state",
    "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
    "service", "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin",
    "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt",
    "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
    "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label",
]

# Columns excluded from edge features (identity / label / timestamp)
_EXCLUDE_COLS = {"srcip", "dstip", "Stime", "Ltime", "attack_cat", "label"}


def _load_unsw_nb15_raw(raw_dir: str | Path) -> pd.DataFrame:
    """Load all UNSW-NB15_*.csv files from *raw_dir*, assign column names.

    Benign rows have NaN in attack_cat; these are filled with "Benign".
    Returns a DataFrame sorted by Stime with a ``_ts`` and ``_label`` column.
    """
    raw_dir = Path(raw_dir)
    # Match only UNSW-NB15_{number}.csv — exclude LIST_EVENTS and similar files
    import re
    csv_files = sorted(
        p for p in glob.glob(str(raw_dir / "UNSW-NB15_*.csv"))
        if re.search(r"UNSW-NB15_\d+\.csv$", p)
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No UNSW-NB15_*.csv files found in {raw_dir}. "
            "Expected UNSW-NB15_1.csv … UNSW-NB15_4.csv."
        )

    parts: list[pd.DataFrame] = []
    for path in csv_files:
        df = pd.read_csv(path, header=None, names=_UNSW_NB15_COLUMNS, low_memory=False)
        parts.append(df)
        print(f"  Loaded {path} — {len(df):,} rows")

    df = pd.concat(parts, ignore_index=True)
    print(f"  Total: {len(df):,} rows across {len(csv_files)} files")

    # Fill NaN attack_cat (benign traffic), strip whitespace, normalise labels
    df["attack_cat"] = (
        df["attack_cat"]
        .fillna("Benign")
        .str.strip()
        .replace({"Backdoors": "Backdoor"})  # fix inconsistent plural form
    )

    # Unified timestamp and label columns expected by downstream code
    df["_ts"] = df["Stime"].astype(np.float64)
    df["_label"] = df["attack_cat"]

    return df.sort_values("_ts").reset_index(drop=True)


def build_temporal_data(cfg: DictConfig) -> None:
    """Build and save TemporalData objects for train / val / test splits."""
    out_dir = Path(cfg.data.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load CSV ───────────────────────────────────────────────────────────
    raw_format = cfg.data.get("raw_format", "single_csv")

    if raw_format == "unsw_nb15_raw":
        print(f"Loading raw UNSW-NB15 files from {cfg.data.raw_dir} …")
        df = _load_unsw_nb15_raw(cfg.data.raw_dir)
    else:
        from src.data.loader import load_csv
        print(f"Loading {cfg.data.raw_path} …")
        df = load_csv(cfg.data.raw_path, label_col=cfg.data.label_col)
        df = df.sort_values("_ts").reset_index(drop=True)

    print(f"  {len(df):,} events loaded")

    # ── 2. Label encoding ─────────────────────────────────────────────────────
    unique_labels = df["_label"].unique().tolist()
    benign_raw = next(
        (lbl for lbl in unique_labels if lbl.lower() in ("benign", "normal")), None
    )
    attack_classes = sorted(lbl for lbl in unique_labels if lbl != benign_raw)
    label2idx: dict[str, int] = {}
    if benign_raw is not None:
        label2idx[benign_raw] = 0
    for i, cls in enumerate(attack_classes, start=1):
        label2idx[cls] = i

    labels_arr = df["_label"].map(label2idx).values.astype(np.int64)
    print(f"  {len(label2idx)} classes: {list(label2idx.keys())}")

    # ── 3. Node identity: IP address → integer ID ─────────────────────────────
    src_col = next((c for c in ["srcip", "IPV4_SRC_ADDR", "src_ip"] if c in df.columns), None)
    dst_col = next((c for c in ["dstip", "IPV4_DST_ADDR", "dst_ip"] if c in df.columns), None)

    if src_col is None or dst_col is None:
        raise ValueError(
            "IP address columns not found. "
            "Expected srcip/dstip (UNSW-NB15 raw) or IPV4_SRC_ADDR/IPV4_DST_ADDR."
        )

    all_ips = pd.unique(
        np.concatenate([df[src_col].values, df[dst_col].values]).astype(str)
    )
    ip2id: dict[str, int] = {ip: i for i, ip in enumerate(all_ips)}
    num_nodes = len(ip2id)
    print(f"  {num_nodes} unique IP nodes")

    src_nodes = np.array([ip2id[str(ip)] for ip in df[src_col].values], dtype=np.int64)
    dst_nodes = np.array([ip2id[str(ip)] for ip in df[dst_col].values], dtype=np.int64)

    # ── 4. Timestamps (float seconds, relative to first event) ────────────────
    timestamps = df["_ts"].values.astype(np.float32)
    t0 = timestamps.min()
    timestamps = timestamps - t0

    # ── 5. Feature matrix ─────────────────────────────────────────────────────
    exclude = set(_EXCLUDE_COLS) | {"_ts", "_label"}
    exclude.update(c for c in [src_col, dst_col] if c not in _EXCLUDE_COLS)
    feat_cols = get_feature_columns(df, exclude=list(exclude))
    feats = df[feat_cols].fillna(0.0).values.astype(np.float32)
    feat_dim = feats.shape[1]
    print(f"  {feat_dim} edge features")

    # ── 6. Chronological split ────────────────────────────────────────────────
    splits_cfg = cfg.data.splits
    r_train = splits_cfg.get("train", 0.6)
    r_val = splits_cfg.get("val", 0.2)
    n = len(df)
    n_train = int(n * r_train)
    n_val = int(n * r_val)

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

    # ── 8. Build and save TemporalData per split ───────────────────────────────
    counts: dict[str, int] = {}
    for split, sl in idx.items():
        n_split = (sl.stop or n) - sl.start
        td = TemporalData(
            src=torch.from_numpy(src_nodes[sl]),
            dst=torch.from_numpy(dst_nodes[sl]),
            t=torch.from_numpy(timestamps[sl]),
            msg=torch.from_numpy(feats[sl]),
            y=torch.from_numpy(labels_arr[sl]),
        )
        torch.save(td, out_dir / f"{split}.pt")
        counts[split] = n_split
        pct_attack = float((labels_arr[sl] > 0).mean() * 100)
        print(f"  {split}: {n_split:,} events → {out_dir / split}.pt  "
              f"({pct_attack:.1f}% attack)")

    # ── 9. Save scaler and metadata ───────────────────────────────────────────
    with open(out_dir / "scaler.pkl", "wb") as f:
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
