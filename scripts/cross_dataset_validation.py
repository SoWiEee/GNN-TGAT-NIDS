"""Run cross-dataset validation for trained static models and ensemble.

The validation CSV must use the same column schema as the training data
(NF-UNSW-NB15-v2 format).  If columns don't match, the script reports
which columns are missing instead of silently producing dimension errors.

Example:
    python scripts/cross_dataset_validation.py --csv data/raw/NF-UNSW-NB15-v2.csv --name nf-unsw
    python scripts/cross_dataset_validation.py --csv data/demo/demo_flows.csv --name demo
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from scripts.compute_reliability_metrics import evaluate_clean
from src.data.static_builder import build_static_graphs
from src.data.static_dataset import StaticNIDSDataset
from src.models.ensemble import EnsembleModel

logger = logging.getLogger(__name__)

STATIC_MODELS = ["graphsage", "gat", "egraphsage"]
PROCESSED_DIR = Path("data/processed/static")


def _empty_result(reason: str) -> dict:
    return {
        "clean_f1": None,
        "clean_precision": None,
        "clean_recall": None,
        "clean_roc_auc": None,
        "clean_macro_f1": None,
        "status": "skipped",
        "reason": reason,
    }


def _format_metrics(metrics: dict[str, float]) -> dict:
    return {
        "clean_f1": round(float(metrics["f1"]), 4),
        "clean_precision": round(float(metrics["precision"]), 4),
        "clean_recall": round(float(metrics["recall"]), 4),
        "clean_roc_auc": round(float(metrics.get("roc_auc", 0)), 4),
        "clean_macro_f1": round(float(metrics.get("macro_f1", 0)), 4),
        "status": "ok",
    }


def _load_model(name: str, checkpoints_dir: Path):
    path = checkpoints_dir / f"{name}_best.pt"
    if not path.exists():
        return None
    model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    return model


def _safe_evaluate(model, loader) -> dict:
    try:
        return _format_metrics(evaluate_clean(model, loader))
    except RuntimeError as exc:
        return _empty_result(f"evaluation failed: {exc}")


def _check_schema_compatibility(csv_path: Path, processed_dir: Path) -> tuple[bool, str, list[str]]:
    """Check if the CSV has the same feature columns as the training pipeline."""
    meta_path = processed_dir / "meta.json"
    if not meta_path.exists():
        return False, "training meta.json not found", []

    meta = json.loads(meta_path.read_text())
    train_features = meta.get("feature_cols", [])
    if not train_features:
        return False, "no feature_cols in training meta.json", []

    df_head = pd.read_csv(csv_path, nrows=0)
    csv_cols = set(df_head.columns)

    missing = sorted(set(train_features) - csv_cols)
    if missing:
        return (
            False,
            f"CSV missing {len(missing)}/{len(train_features)} training features: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}",
            train_features,
        )
    return True, "ok", train_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset validation")
    parser.add_argument("--csv", required=True, help="Target CSV dataset to validate on")
    parser.add_argument("--name", default="external", help="Dataset label for output metadata")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--output", default="data/metrics/cross_dataset_validation.json")
    parser.add_argument("--window-size-s", type=float, default=120.0)
    parser.add_argument("--label-col", default="attack_cat")
    parser.add_argument("--timestamp-col", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    csv_path = Path(args.csv)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path(args.checkpoints_dir)
    processed_dir = Path(args.processed_dir)

    if not csv_path.exists():
        payload = {
            "dataset": args.name,
            "csv": str(csv_path),
            "status": "skipped",
            "reason": "CSV not found",
            "models": {},
        }
        out_path.write_text(json.dumps(payload, indent=2))
        logger.warning("CSV not found: %s", csv_path)
        return

    compatible, reason, train_features = _check_schema_compatibility(csv_path, processed_dir)
    if not compatible:
        logger.warning("Schema mismatch: %s", reason)
        payload = {
            "dataset": args.name,
            "csv": str(csv_path),
            "status": "schema_mismatch",
            "reason": reason,
            "n_windows": 0,
            "meta": {"n_flows": None, "n_features": None, "window_size_s": args.window_size_s},
            "models": {
                name: _empty_result(f"schema mismatch: {reason}")
                for name in STATIC_MODELS + ["ensemble"]
            },
        }
        out_path.write_text(json.dumps(payload, indent=2))
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        meta = build_static_graphs(
            csv_path=csv_path,
            output_dir=tmpdir,
            window_size_s=args.window_size_s,
            ratios=(1.0, 0.0, 0.0),
            label_col=args.label_col,
            timestamp_col=args.timestamp_col,
        )
        ds = StaticNIDSDataset(tmpdir, split="train")
        loader = DataLoader(ds, batch_size=1, shuffle=False)

        n_features_csv = meta.get("n_features", 0)
        n_features_model = len(train_features)
        if n_features_csv != n_features_model:
            logger.warning(
                "Feature count mismatch after build: CSV=%d, model=%d",
                n_features_csv, n_features_model,
            )

        results: dict[str, dict] = {}
        loaded = {}
        for name in STATIC_MODELS:
            model = _load_model(name, checkpoints_dir)
            if model is None:
                results[name] = _empty_result("checkpoint not found")
                continue
            loaded[name] = model
            logger.info("[%s] evaluating %d windows", name, len(ds))
            results[name] = _safe_evaluate(model, loader)

        if len(loaded) >= 2:
            ensemble = EnsembleModel(loaded, strategy="soft_vote")
            results["ensemble"] = _safe_evaluate(ensemble, loader)
        else:
            results["ensemble"] = _empty_result("need at least two static checkpoints")

        payload = {
            "dataset": args.name,
            "csv": str(csv_path),
            "status": "ok",
            "n_windows": len(ds),
            "meta": {
                "n_flows": meta.get("n_flows"),
                "n_features": meta.get("n_features"),
                "window_size_s": meta.get("window_size_s"),
            },
            "models": results,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        logger.info("Saved cross-dataset validation -> %s", out_path)


if __name__ == "__main__":
    main()
