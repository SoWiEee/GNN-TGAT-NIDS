"""Tests for multi-seed evaluation script."""
from unittest.mock import MagicMock, patch

import numpy as np

from scripts.multi_seed_eval import (
    TEST_METRIC_RE,
    _aggregate,
    _run_training,
)


class TestMetricRegex:
    def test_matches_standard_output(self):
        line = "TEST | f1=0.9712 | precision=0.9792 | recall=0.9660 | roc_auc=0.9992"
        m = TEST_METRIC_RE.search(line)
        assert m is not None
        assert float(m.group("f1")) == 0.9712
        assert float(m.group("precision")) == 0.9792
        assert float(m.group("recall")) == 0.9660
        assert float(m.group("roc_auc")) == 0.9992

    def test_matches_embedded_in_log(self):
        log = (
            "INFO  epoch 30/30 | train_loss=0.12\n"
            "INFO  Loading best checkpoint...\n"
            "INFO  TEST | f1=0.9500 | precision=0.9600 | recall=0.9400 | roc_auc=0.9900\n"
        )
        m = TEST_METRIC_RE.search(log)
        assert m is not None
        assert float(m.group("f1")) == 0.9500

    def test_no_match_on_garbage(self):
        assert TEST_METRIC_RE.search("hello world") is None


class TestAggregate:
    def test_empty_input(self):
        assert _aggregate([]) == {}

    def test_single_run(self):
        runs = [{"seed": 42, "f1": 0.95, "precision": 0.96,
                 "recall": 0.94, "roc_auc": 0.99}]
        agg = _aggregate(runs)
        assert agg["n_seeds"] == 1
        assert agg["f1"]["mean"] == 0.95
        assert agg["f1"]["std"] == 0.0

    def test_multiple_runs(self):
        runs = [
            {"seed": 42, "f1": 0.90, "precision": 0.92,
             "recall": 0.88, "roc_auc": 0.98},
            {"seed": 123, "f1": 0.94, "precision": 0.95,
             "recall": 0.93, "roc_auc": 0.99},
            {"seed": 456, "f1": 0.92, "precision": 0.93,
             "recall": 0.91, "roc_auc": 0.985},
        ]
        agg = _aggregate(runs)
        assert agg["n_seeds"] == 3
        assert agg["f1"]["min"] == 0.9
        assert agg["f1"]["max"] == 0.94
        assert abs(agg["f1"]["mean"] - np.mean([0.90, 0.94, 0.92])) < 1e-4
        assert agg["f1"]["std"] > 0

    def test_seeds_preserved(self):
        runs = [
            {"seed": 42, "f1": 0.9, "precision": 0.9,
             "recall": 0.9, "roc_auc": 0.9},
            {"seed": 123, "f1": 0.9, "precision": 0.9,
             "recall": 0.9, "roc_auc": 0.9},
        ]
        agg = _aggregate(runs)
        assert agg["seeds"] == [42, 123]


class TestRunTraining:
    @patch("scripts.multi_seed_eval.subprocess.run")
    def test_parses_successful_output(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="TEST | f1=0.9500 | precision=0.9600 | recall=0.9400 | roc_auc=0.9900\n",
            stderr="",
            returncode=0,
        )
        result = _run_training("graphsage", 42, 30, False)
        assert result is not None
        assert result["seed"] == 42
        assert result["f1"] == 0.95

    @patch("scripts.multi_seed_eval.subprocess.run")
    def test_returns_none_on_failure(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="", stderr="RuntimeError: CUDA OOM", returncode=1,
        )
        result = _run_training("graphsage", 42, 30, False)
        assert result is None

    @patch("scripts.multi_seed_eval.subprocess.run")
    def test_temporal_flag_adds_data_arg(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="TEST | f1=0.9400 | precision=0.9500 | recall=0.9300 | roc_auc=0.9800\n",
            stderr="",
            returncode=0,
        )
        _run_training("tgat", 42, 30, True)
        cmd = mock_run.call_args[0][0]
        assert "data=temporal_default" in cmd

    @patch("scripts.multi_seed_eval.subprocess.run")
    def test_seed_in_checkpoint_dir(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="TEST | f1=0.9400 | precision=0.9500 | recall=0.9300 | roc_auc=0.9800\n",
            stderr="",
            returncode=0,
        )
        _run_training("gat", 789, 30, False)
        cmd = mock_run.call_args[0][0]
        ckpt_args = [a for a in cmd if "checkpoint_dir" in a]
        assert len(ckpt_args) == 1
        assert "gat_s789" in ckpt_args[0]
