"""Prepare UNSW-NB15 data for training and create a demo CSV for the web UI.

Reads the original training and testing set CSVs, normalises label names,
merges them into a single file (data/raw/NF-UNSW-NB15-v2.csv), and creates
a small stratified sample for the web UI demo (data/demo/demo_flows.csv).

Usage:
    uv run python scripts/create_demo_dataset.py
    uv run python scripts/create_demo_dataset.py --demo-size 500 --seed 42
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Source files (relative to project root)
_TRAIN_CSV = Path(
    "src/data/raw/NF-UNSW-NB15-v2/CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv"
)
_TEST_CSV = Path(
    "src/data/raw/NF-UNSW-NB15-v2/CSV Files/Training and Testing Sets/UNSW_NB15_testing-set.csv"
)

# Output paths
_MERGED_CSV = Path("data/raw/NF-UNSW-NB15-v2.csv")
_DEMO_CSV = Path("data/demo/demo_flows.csv")

# Columns not useful as features or labels — drop from output
_DROP_COLS = ["id", "label"]


def _load_and_clean(path: Path) -> pd.DataFrame:
    """Load a UNSW-NB15 CSV, rename 'Normal' → 'Benign', drop unused cols."""
    logger.info("Loading %s …", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("  Loaded %d rows × %d columns", *df.shape)

    # Rename benign class so encode_labels() recognises it (case-insensitive check)
    df["attack_cat"] = df["attack_cat"].str.strip().replace("Normal", "Benign")

    # Drop columns that would leak the label or are pure identifiers
    cols_to_drop = [c for c in _DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df


def _stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Return n rows sampled proportionally from each attack_cat class."""
    rng = np.random.default_rng(seed)
    counts = df["attack_cat"].value_counts()
    fractions = (counts / len(df))

    parts: list[pd.DataFrame] = []
    remaining = n
    classes = list(fractions.index)

    for i, cls in enumerate(classes):
        is_last = i == len(classes) - 1
        quota = remaining if is_last else int(np.round(fractions[cls] * n))
        quota = min(quota, len(df[df["attack_cat"] == cls]))
        quota = max(quota, 1)  # ensure at least one row per class

        rows = df[df["attack_cat"] == cls]
        idx = rng.choice(len(rows), size=min(quota, len(rows)), replace=False)
        parts.append(rows.iloc[idx])
        remaining -= len(parts[-1])

    sample = pd.concat(parts).sort_index().reset_index(drop=True)
    logger.info("Stratified sample: %d rows across %d classes", len(sample), len(parts))
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare UNSW-NB15 datasets")
    parser.add_argument(
        "--demo-size", type=int, default=1000,
        help="Number of flows in the demo CSV (default: 1000)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-merge", action="store_true",
        help="Skip writing the large merged CSV (only write demo CSV)",
    )
    args = parser.parse_args()

    if not _TRAIN_CSV.exists():
        logger.error("Training CSV not found: %s", _TRAIN_CSV)
        raise SystemExit(1)
    if not _TEST_CSV.exists():
        logger.error("Testing CSV not found: %s", _TEST_CSV)
        raise SystemExit(1)

    train_df = _load_and_clean(_TRAIN_CSV)
    test_df = _load_and_clean(_TEST_CSV)

    merged = pd.concat([train_df, test_df], ignore_index=True)
    logger.info("Merged: %d rows × %d columns", *merged.shape)

    # Distribution summary
    logger.info("Class distribution:")
    for cls, cnt in merged["attack_cat"].value_counts().items():
        logger.info("  %-20s  %6d  (%.1f%%)", cls, cnt, 100 * cnt / len(merged))

    if not args.skip_merge:
        _MERGED_CSV.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(_MERGED_CSV, index=False)
        logger.info("Merged CSV saved → %s  (%d MB)", _MERGED_CSV,
                    _MERGED_CSV.stat().st_size // 1_000_000)

    # Demo dataset
    _DEMO_CSV.parent.mkdir(parents=True, exist_ok=True)
    demo = _stratified_sample(merged, args.demo_size, args.seed)
    demo.to_csv(_DEMO_CSV, index=False)
    logger.info("Demo CSV saved → %s", _DEMO_CSV)

    print("\n── Dataset Summary ──────────────────────────────────────")
    if not args.skip_merge:
        print(f"  Full dataset : {_MERGED_CSV}  ({len(merged):,} flows)")
    print(f"  Demo dataset : {_DEMO_CSV}  ({len(demo):,} flows)")
    n_feat = len([c for c in merged.columns if c != "attack_cat"])
    print(f"  Feature cols : {n_feat} numeric + 3 categorical")
    print(f"  Classes      : {sorted(merged['attack_cat'].unique())}")
    print()
    print("Next step:")
    print("  uv run python src/data/static_builder.py --config-name static_default")
    print()


if __name__ == "__main__":
    main()
