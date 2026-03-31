"""CSV loading, label encoding, and chronological splitting for NF-UNSW-NB15-v2."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# Column name candidates for timestamps (tried in order)
_TIMESTAMP_CANDIDATES = ["FIRST_SEEN", "Timestamp", "timestamp", "first_seen", "ts"]

# Label column name candidates
_LABEL_CANDIDATES = ["Label", "label", "Attack", "attack", "class", "Class"]


def load_csv(
    path: str | Path, timestamp_col: str | None = None, label_col: str | None = None
) -> pd.DataFrame:
    """Load a NetFlow CSV and ensure timestamp and label columns are present.

    Auto-detects timestamp and label columns if not specified.  If no timestamp
    column is found the row index is used as a synthetic monotone timestamp.

    Parameters
    ----------
    path:
        Path to the CSV file.
    timestamp_col:
        Override the timestamp column name.  If None, auto-detected.
    label_col:
        Override the label column name.  If None, auto-detected.

    Returns
    -------
    pd.DataFrame
        DataFrame with a guaranteed ``_ts`` column (float seconds since epoch
        or integer row index) and a ``_label`` column (original string labels).
    """
    df = pd.read_csv(path, low_memory=False)

    # --- timestamp ---
    if timestamp_col is not None:
        ts_col = timestamp_col
    else:
        ts_col = next((c for c in _TIMESTAMP_CANDIDATES if c in df.columns), None)

    if ts_col is not None:
        ts_parsed = pd.to_datetime(df[ts_col], errors="coerce")
        # Convert to whole seconds (avoids nanosecond/microsecond ambiguity
        # across pandas versions — pandas 2.x may use μs internally).
        df["_ts"] = (
            ts_parsed.astype("datetime64[s]")
            .astype(np.int64)
            .astype(np.float64)
        )
        # Fall back to row position for rows that couldn't be parsed (NaT).
        # Using np.where gives actual integer positions (0-based), which are
        # guaranteed to be monotone regardless of the DataFrame's index type.
        failed_mask = ts_parsed.isna()
        if failed_mask.any():
            positions = np.where(failed_mask.values)[0].astype(np.float64)
            df.loc[failed_mask, "_ts"] = positions
    else:
        # No time column — use row index as a proxy
        df["_ts"] = np.arange(len(df), dtype=np.float64)

    # --- label ---
    if label_col is not None:
        lbl_col = label_col
    else:
        lbl_col = next((c for c in _LABEL_CANDIDATES if c in df.columns), None)

    if lbl_col is None:
        raise ValueError(
            f"Cannot find label column in {list(df.columns)}. "
            "Pass label_col= explicitly."
        )

    df["_label"] = df[lbl_col].astype(str).str.strip()

    return df


def encode_labels(df: pd.DataFrame) -> tuple[pd.Series, dict[str, int]]:
    """Encode string labels to integers.

    Encoding rule:
    - ``Benign`` / ``BENIGN`` / ``benign`` → 0
    - All other classes sorted alphabetically → 1, 2, 3, …

    Parameters
    ----------
    df:
        DataFrame returned by :func:`load_csv` (must have ``_label`` column).

    Returns
    -------
    encoded : pd.Series
        Integer labels aligned with ``df.index``.
    label2idx : dict[str, int]
        Mapping from original string label to integer index.
    """
    if "_label" not in df.columns:
        raise ValueError("DataFrame must have a '_label' column (call load_csv first).")

    unique_labels = df["_label"].unique().tolist()

    # Identify benign class — accept "Benign"/"BENIGN" or "Normal"/"NORMAL"
    benign_raw = next(
        (lbl for lbl in unique_labels if lbl.lower() in ("benign", "normal")),
        None,
    )

    attack_classes = sorted(lbl for lbl in unique_labels if lbl != benign_raw)
    label2idx: dict[str, int] = {}

    if benign_raw is not None:
        label2idx[benign_raw] = 0

    for i, cls in enumerate(attack_classes, start=1):
        label2idx[cls] = i

    encoded = df["_label"].map(label2idx)
    if encoded.isna().any():
        unknown = df.loc[encoded.isna(), "_label"].unique().tolist()
        raise ValueError(f"Unmapped labels found: {unknown}")

    return encoded.astype(np.int64), label2idx


def chronological_split(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    ts_col: str = "_ts",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train / val / test by time order (no shuffling).

    Parameters
    ----------
    df:
        DataFrame with a ``_ts`` column (or custom ``ts_col``).
    ratios:
        (train, val, test) fractions — must sum to 1.0.
    ts_col:
        Name of the timestamp column used for sorting.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
        Non-overlapping chronological splits.

    Raises
    ------
    AssertionError
        If temporal leakage is detected (train max_ts ≥ val min_ts, etc.).
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios):.4f}")

    # Sort by timestamp
    df_sorted = df.sort_values(ts_col).reset_index(drop=True)
    n = len(df_sorted)

    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    n_test = n - n_train - n_val  # absorb rounding remainder

    train_df = df_sorted.iloc[:n_train].copy()
    val_df = df_sorted.iloc[n_train : n_train + n_val].copy()
    test_df = df_sorted.iloc[n_train + n_val :].copy()

    assert len(test_df) == n_test, "Split size mismatch."

    # Temporal leakage checks
    # Strictly non-overlapping: each split's last timestamp must be strictly
    # earlier than the next split's first timestamp.  Equal boundary timestamps
    # would place the same real-world moment in two splits, which constitutes
    # temporal leakage for time-series classification.
    if len(train_df) > 0 and len(val_df) > 0:
        assert train_df[ts_col].max() < val_df[ts_col].min(), (
            "Temporal leakage: train max_ts >= val min_ts. "
            f"(train_max={train_df[ts_col].max()}, val_min={val_df[ts_col].min()})"
        )

    if len(val_df) > 0 and len(test_df) > 0:
        assert val_df[ts_col].max() < test_df[ts_col].min(), (
            "Temporal leakage: val max_ts >= test min_ts. "
            f"(val_max={val_df[ts_col].max()}, test_min={test_df[ts_col].min()})"
        )

    return train_df, val_df, test_df


def get_feature_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """Return numeric feature column names, excluding metadata columns.

    Parameters
    ----------
    df:
        Input DataFrame.
    exclude:
        Additional column names to exclude.

    Returns
    -------
    list[str]
        Sorted list of numeric feature column names.
    """
    always_exclude = {"_ts", "_label", "_encoded"}
    if exclude:
        always_exclude.update(exclude)

    return sorted(
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in always_exclude
    )


SplitName = Literal["train", "val", "test"]
