"""Evaluate static models at different time-window sizes to quantify the
training (120s) vs web-inference (60s) mismatch.

Usage:
    uv run python scripts/window_size_eval.py
    uv run python scripts/window_size_eval.py --windows 30,60,120,300
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_geometric.loader import DataLoader

from scripts.compute_reliability_metrics import evaluate_clean, load_model
from src.data.static_builder import build_static_graphs
from src.data.static_dataset import StaticNIDSDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

MODELS = ["graphsage", "gat"]
RAW_CSV = Path("data/raw/NF-UNSW-NB15-v2.csv")
OUTPUT_PATH = Path("data/metrics/window_size_eval.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Window-size sensitivity evaluation")
    parser.add_argument(
        "--windows", default="60,120,300",
        help="Comma-separated window sizes in seconds",
    )
    parser.add_argument("--csv", default=str(RAW_CSV), help="Raw CSV path")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--models", default=",".join(MODELS))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        return

    window_sizes = [int(w) for w in args.windows.split(",")]
    model_names = [m.strip() for m in args.models.split(",")]

    models = {}
    for name in model_names:
        m = load_model(name)
        if m is not None:
            models[name] = m

    if not models:
        logger.error("No model checkpoints found")
        return

    results: dict[str, list[dict]] = {name: [] for name in models}

    for ws in window_sizes:
        logger.info("Building graphs with window_size=%ds ...", ws)
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = build_static_graphs(
                csv_path=csv_path,
                output_dir=tmpdir,
                window_size_s=float(ws),
                ratios=(0.0, 0.0, 1.0),
            )
            ds = StaticNIDSDataset(tmpdir, split="test")
            if len(ds) == 0:
                logger.warning("No windows at %ds — skipping", ws)
                continue
            loader = DataLoader(ds, batch_size=16, shuffle=False)
            logger.info("  %d windows, %d features", len(ds), meta.get("n_features", "?"))

            for name, model in models.items():
                try:
                    metrics = evaluate_clean(model, loader)
                    results[name].append({
                        "window_size_s": ws,
                        "n_windows": len(ds),
                        "f1": round(float(metrics["f1"]), 4),
                        "precision": round(float(metrics["precision"]), 4),
                        "recall": round(float(metrics["recall"]), 4),
                        "macro_f1": round(float(metrics.get("macro_f1", 0)), 4),
                    })
                    logger.info("  [%s] window=%ds  f1=%.4f", name, ws, metrics["f1"])
                except RuntimeError as exc:
                    logger.warning("  [%s] window=%ds  failed: %s", name, ws, exc)
                    results[name].append({
                        "window_size_s": ws,
                        "n_windows": len(ds),
                        "error": str(exc),
                    })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved -> %s", output_path)

    print("\n── Window Size Sensitivity ──────────────────")
    for name, rows in results.items():
        print(f"  {name}:")
        for r in rows:
            if "error" in r:
                print(f"    {r['window_size_s']:4d}s  ERROR: {r['error']}")
            else:
                ws = r['window_size_s']
                print(
                    f"    {ws:4d}s  f1={r['f1']:.4f}"
                    f"  macro_f1={r['macro_f1']:.4f}"
                    f"  ({r['n_windows']} windows)"
                )
    print()


if __name__ == "__main__":
    main()
