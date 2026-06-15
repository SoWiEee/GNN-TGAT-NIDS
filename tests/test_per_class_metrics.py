"""Tests for per-class metrics and confusion matrix."""
import numpy as np
import torch

from src.eval.metrics import compute_per_class_metrics


class TestComputePerClassMetrics:
    def test_basic_output_shape(self):
        y_true = torch.tensor([0, 0, 1, 1, 2])
        y_pred = torch.tensor([0, 1, 1, 1, 2])
        result = compute_per_class_metrics(y_true, y_pred)

        assert "per_class" in result
        assert "confusion_matrix" in result
        assert "class_names" in result
        assert len(result["per_class"]) == 3
        assert len(result["confusion_matrix"]) == 3

    def test_perfect_predictions(self):
        y_true = torch.tensor([0, 0, 1, 1, 2, 2])
        y_pred = torch.tensor([0, 0, 1, 1, 2, 2])
        result = compute_per_class_metrics(y_true, y_pred)

        for cls in result["per_class"]:
            assert cls["precision"] == 1.0
            assert cls["recall"] == 1.0
            assert cls["f1"] == 1.0

    def test_custom_label_names(self):
        y_true = torch.tensor([0, 1, 2])
        y_pred = torch.tensor([0, 1, 2])
        result = compute_per_class_metrics(
            y_true, y_pred, label_names=["Benign", "DoS", "Exploit"]
        )

        names = [cls["name"] for cls in result["per_class"]]
        assert names == ["Benign", "DoS", "Exploit"]

    def test_confusion_matrix_diagonal(self):
        y_true = torch.tensor([0, 0, 1, 1])
        y_pred = torch.tensor([0, 0, 1, 1])
        result = compute_per_class_metrics(y_true, y_pred)
        cm = result["confusion_matrix"]
        assert cm[0][0] == 2
        assert cm[1][1] == 2
        assert cm[0][1] == 0
        assert cm[1][0] == 0

    def test_support_counts(self):
        y_true = torch.tensor([0, 0, 0, 1, 1, 2])
        y_pred = torch.tensor([0, 0, 0, 1, 1, 2])
        result = compute_per_class_metrics(y_true, y_pred)

        supports = {cls["class_id"]: cls["support"] for cls in result["per_class"]}
        assert supports[0] == 3
        assert supports[1] == 2
        assert supports[2] == 1

    def test_numpy_input(self):
        y_true = np.array([0, 1, 2, 0])
        y_pred = np.array([0, 1, 0, 0])
        result = compute_per_class_metrics(y_true, y_pred)
        assert len(result["per_class"]) == 3

    def test_zero_recall_class(self):
        y_true = torch.tensor([0, 0, 1, 1, 2, 2])
        y_pred = torch.tensor([0, 0, 0, 0, 0, 0])
        result = compute_per_class_metrics(y_true, y_pred)

        cls_map = {c["class_id"]: c for c in result["per_class"]}
        assert cls_map[1]["recall"] == 0.0
        assert cls_map[2]["recall"] == 0.0
