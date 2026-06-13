"""Tests for temporal_builder helper functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestLoadUnswNb15Raw:
    def test_raises_when_no_files(self, tmp_path):
        from src.data.temporal_builder import _load_unsw_nb15_raw

        with pytest.raises(FileNotFoundError, match="No UNSW-NB15_"):
            _load_unsw_nb15_raw(tmp_path)

    def test_loads_single_csv(self, tmp_path):
        from src.data.temporal_builder import _UNSW_NB15_COLUMNS, _load_unsw_nb15_raw

        n_rows = 50
        rng = np.random.default_rng(42)
        data = {}
        for col in _UNSW_NB15_COLUMNS:
            if col in ("srcip", "dstip"):
                data[col] = [f"192.168.1.{rng.integers(1, 254)}" for _ in range(n_rows)]
            elif col == "attack_cat":
                cats = [np.nan, "Fuzzers", "DoS", np.nan, "Exploits"]
                data[col] = [cats[i % len(cats)] for i in range(n_rows)]
            elif col == "label":
                data[col] = rng.integers(0, 2, n_rows).tolist()
            elif col in ("proto", "state", "service"):
                data[col] = ["tcp"] * n_rows
            elif col in ("Stime", "Ltime"):
                data[col] = (rng.random(n_rows) * 1e6 + 1e9).tolist()
            else:
                data[col] = rng.random(n_rows).tolist()

        df = pd.DataFrame(data)
        df.to_csv(tmp_path / "UNSW-NB15_1.csv", index=False, header=False)

        result = _load_unsw_nb15_raw(tmp_path)
        assert len(result) == n_rows
        assert "_ts" in result.columns
        assert "_label" in result.columns
        assert result["_label"].isin(["Benign", "Fuzzers", "DoS", "Exploits"]).all()

    def test_ignores_non_numbered_csvs(self, tmp_path):
        from src.data.temporal_builder import _load_unsw_nb15_raw

        (tmp_path / "UNSW-NB15_LIST_EVENTS.csv").write_text("a,b,c\n")
        with pytest.raises(FileNotFoundError):
            _load_unsw_nb15_raw(tmp_path)

    def test_backdoors_normalized(self, tmp_path):
        from src.data.temporal_builder import _UNSW_NB15_COLUMNS, _load_unsw_nb15_raw

        n_rows = 5
        data = {}
        for col in _UNSW_NB15_COLUMNS:
            if col in ("srcip", "dstip"):
                data[col] = [f"10.0.0.{i}" for i in range(n_rows)]
            elif col == "attack_cat":
                data[col] = ["Backdoors"] * n_rows
            elif col in ("proto", "state", "service"):
                data[col] = ["tcp"] * n_rows
            elif col in ("Stime", "Ltime"):
                data[col] = list(range(n_rows))
            else:
                data[col] = [0.0] * n_rows

        df = pd.DataFrame(data)
        df.to_csv(tmp_path / "UNSW-NB15_1.csv", index=False, header=False)

        result = _load_unsw_nb15_raw(tmp_path)
        assert "Backdoors" not in result["_label"].values
        assert "Backdoor" in result["_label"].values

    def test_sorted_by_time(self, tmp_path):
        from src.data.temporal_builder import _UNSW_NB15_COLUMNS, _load_unsw_nb15_raw

        n_rows = 10
        data = {}
        for col in _UNSW_NB15_COLUMNS:
            if col in ("srcip", "dstip"):
                data[col] = [f"10.0.0.{i}" for i in range(n_rows)]
            elif col == "attack_cat":
                data[col] = [np.nan] * n_rows
            elif col in ("proto", "state", "service"):
                data[col] = ["tcp"] * n_rows
            elif col == "Stime":
                data[col] = list(reversed(range(n_rows)))
            elif col == "Ltime":
                data[col] = list(range(n_rows))
            else:
                data[col] = [0.0] * n_rows

        df = pd.DataFrame(data)
        df.to_csv(tmp_path / "UNSW-NB15_1.csv", index=False, header=False)

        result = _load_unsw_nb15_raw(tmp_path)
        ts = result["_ts"].values
        assert (ts[1:] >= ts[:-1]).all()


class TestExcludeCols:
    def test_identity_columns_excluded(self):
        from src.data.temporal_builder import _EXCLUDE_COLS
        assert "srcip" in _EXCLUDE_COLS
        assert "dstip" in _EXCLUDE_COLS
        assert "attack_cat" in _EXCLUDE_COLS
        assert "label" in _EXCLUDE_COLS

    def test_timestamps_excluded(self):
        from src.data.temporal_builder import _EXCLUDE_COLS
        assert "Stime" in _EXCLUDE_COLS
        assert "Ltime" in _EXCLUDE_COLS
