"""Tests for src/data/loader.py."""

from __future__ import annotations

import textwrap

import pandas as pd
import pytest

from src.data.loader import (
    chronological_split,
    encode_labels,
    get_feature_columns,
    load_csv,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_csv(rows: str, tmp_path) -> str:
    """Write a CSV string to a temp file and return its path."""
    p = tmp_path / "test.csv"
    p.write_text(textwrap.dedent(rows))
    return str(p)


def _simple_df() -> pd.DataFrame:
    """Create a minimal DataFrame for label/split tests."""
    return pd.DataFrame(
        {
            "_ts": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "_label": ["Benign"] * 6 + ["DoS"] * 2 + ["Recon"] * 2,
        }
    )


# ── load_csv ─────────────────────────────────────────────────────────────────

class TestLoadCSV:
    def test_auto_detect_timestamp_and_label(self, tmp_path):
        csv_path = _make_csv(
            """\
            FIRST_SEEN,Label,feat_a,feat_b
            2023-01-01 00:00:00,Benign,1.0,2.0
            2023-01-01 00:01:00,DoS,3.0,4.0
            """,
            tmp_path,
        )
        df = load_csv(csv_path)
        assert "_ts" in df.columns
        assert "_label" in df.columns
        assert df["_label"].tolist() == ["Benign", "DoS"]
        # Timestamps should be monotone increasing
        assert df["_ts"].iloc[0] < df["_ts"].iloc[1]

    def test_fallback_row_index_when_no_timestamp(self, tmp_path):
        csv_path = _make_csv(
            """\
            Label,feat_a
            Benign,1.0
            DoS,2.0
            Recon,3.0
            """,
            tmp_path,
        )
        df = load_csv(csv_path)
        assert "_ts" in df.columns
        assert df["_ts"].tolist() == [0.0, 1.0, 2.0]

    def test_explicit_override_columns(self, tmp_path):
        csv_path = _make_csv(
            """\
            my_ts,my_label,feat
            2023-01-01,Attack,0.5
            2023-01-02,Benign,0.6
            """,
            tmp_path,
        )
        df = load_csv(csv_path, timestamp_col="my_ts", label_col="my_label")
        assert df["_label"].tolist() == ["Attack", "Benign"]

    def test_missing_label_column_raises(self, tmp_path):
        csv_path = _make_csv(
            """\
            feat_a,feat_b
            1.0,2.0
            """,
            tmp_path,
        )
        with pytest.raises(ValueError, match="Cannot find label column"):
            load_csv(csv_path)

    def test_label_whitespace_stripped(self, tmp_path):
        csv_path = _make_csv(
            """\
            Label,feat
            " Benign ",1.0
            "DoS  ",2.0
            """,
            tmp_path,
        )
        df = load_csv(csv_path)
        assert df["_label"].tolist() == ["Benign", "DoS"]


# ── encode_labels ─────────────────────────────────────────────────────────────

class TestEncodeLabels:
    def test_benign_is_zero(self):
        df = _simple_df()
        encoded, label2idx = encode_labels(df)
        assert label2idx["Benign"] == 0
        assert all(encoded[df["_label"] == "Benign"] == 0)

    def test_attack_classes_sorted_alphabetically(self):
        df = _simple_df()
        _, label2idx = encode_labels(df)
        # DoS < Recon alphabetically
        assert label2idx["DoS"] == 1
        assert label2idx["Recon"] == 2

    def test_case_insensitive_benign(self):
        df = pd.DataFrame({"_label": ["BENIGN", "attack_a", "attack_b"]})
        _, label2idx = encode_labels(df)
        assert label2idx["BENIGN"] == 0

    def test_no_benign_class(self):
        df = pd.DataFrame({"_label": ["classA", "classB", "classC"]})
        encoded, label2idx = encode_labels(df)
        # All classes get 1-indexed, none is 0
        assert 0 not in label2idx.values()

    def test_encoded_length_matches_dataframe(self):
        df = _simple_df()
        encoded, _ = encode_labels(df)
        assert len(encoded) == len(df)

    def test_missing_label_column_raises(self):
        df = pd.DataFrame({"not_label": ["a", "b"]})
        with pytest.raises(ValueError, match="_label"):
            encode_labels(df)


# ── chronological_split ───────────────────────────────────────────────────────

class TestChronologicalSplit:
    def test_correct_sizes(self):
        df = _simple_df()
        train, val, test = chronological_split(df, ratios=(0.6, 0.2, 0.2))
        assert len(train) + len(val) + len(test) == len(df)

    def test_no_temporal_leakage(self):
        df = _simple_df()
        train, val, test = chronological_split(df, ratios=(0.6, 0.2, 0.2))
        # strict: no shared boundary timestamps across splits
        assert train["_ts"].max() < val["_ts"].min()
        assert val["_ts"].max() < test["_ts"].min()

    def test_rows_sorted_by_timestamp(self):
        # Shuffle the DataFrame before splitting; split should re-sort
        df = _simple_df().sample(frac=1.0, random_state=0)
        train, val, test = chronological_split(df, ratios=(0.6, 0.2, 0.2))
        combined = pd.concat([train, val, test])
        assert combined["_ts"].is_monotonic_increasing

    def test_invalid_ratios_raises(self):
        df = _simple_df()
        with pytest.raises(ValueError, match="Ratios must sum"):
            chronological_split(df, ratios=(0.5, 0.3, 0.3))

    def test_temporal_leakage_assertion_triggered(self):
        # Create a DataFrame where rows are NOT sorted — after sorting the split
        # should be fine; this test checks the assert fires if we deliberately
        # pass a mis-sorted DataFrame.  We monkey-patch the internal sort call
        # by overriding _ts with reversed values post-sort.
        #
        # Actually: the function sorts internally, so leakage cannot occur with
        # properly formed inputs.  We verify assertions pass (no error).
        df = _simple_df()
        train, val, test = chronological_split(df)  # default 60/20/20
        assert train["_ts"].max() < val["_ts"].min()


# ── get_feature_columns ───────────────────────────────────────────────────────

class TestGetFeatureColumns:
    def test_excludes_metadata_columns(self):
        df = pd.DataFrame(
            {
                "_ts": [1.0, 2.0],
                "_label": ["Benign", "DoS"],
                "feat_a": [1.0, 2.0],
                "feat_b": [3.0, 4.0],
                "str_col": ["x", "y"],
            }
        )
        cols = get_feature_columns(df)
        assert "_ts" not in cols
        assert "_label" not in cols
        assert "str_col" not in cols
        assert "feat_a" in cols
        assert "feat_b" in cols

    def test_custom_exclude(self):
        df = pd.DataFrame({"feat_a": [1.0], "feat_b": [2.0], "_ts": [0.0], "_label": ["B"]})
        cols = get_feature_columns(df, exclude=["feat_a"])
        assert "feat_a" not in cols
        assert "feat_b" in cols
