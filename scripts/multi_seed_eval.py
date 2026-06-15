"""Multi-seed training & evaluation: measure variance across random seeds.

Trains each model architecture with N different seeds, collects test metrics,
and reports mean +/- std to separate architecture effects from seed noise.

Usage:
    uv run python scripts/multi_seed_eval.py
    uv run python scripts/multi_seed_eval.py --seeds 42,123,456,789,1024
    uv run python scripts/multi_seed_eval.py --models graphsage,gat --epochs 30
    uv run python scripts/multi_seed_eval.py --temporal --models tgat,tgn
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
STATIC_MODELS = ["graphsage", "gat", "egraphsage"]
TEMPORAL_MODELS = ["tgat", "tgn"]
OUTPUT_PATH = Path("data/metrics/multi_seed_eval.json")

TEST_METRIC_RE = re.compile(
    r"TEST \| f1=(?P<f1>[\d.]+) \| precision=(?P<precision>[\d.]+)"
    r" \| recall=(?P<recall>[\d.]+) \| roc_auc=(?P<roc_auc>[\d.]+)"
)


def _run_training(
    model: str,
    seed: int,
    epochs: int,
    is_temporal: bool,
) -> dict | None:
    """Run a single training run via subprocess, return parsed test metrics."""
    cmd = [
        sys.executable, "train.py",
        f"model={model}",
        f"seed={seed}",
        f"train.epochs={epochs}",
        f"train.checkpoint_dir=checkpoints/seed_eval/{model}_s{seed}",
    ]
    if is_temporal:
        cmd.append("data=temporal_default")

    logger.info("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            cwd=str(Path(__file__).parent.parent),
        )
    except subprocess.TimeoutExpired:
        logger.warning("[%s seed=%d] timed out", model, seed)
        return None

    output = result.stdout + "\n" + result.stderr
    match = TEST_METRIC_RE.search(output)
    if not match:
        logger.warning(
            "[%s seed=%d] could not parse test metrics (exit=%d)",
            model, seed, result.returncode,
        )
        if result.returncode != 0:
            last_lines = output.strip().split("\n")[-5:]
            for line in last_lines:
                logger.warning("  %s", line)
        return None

    return {
        "seed": seed,
        "f1": float(match.group("f1")),
        "precision": float(match.group("precision")),
        "recall": float(match.group("recall")),
        "roc_auc": float(match.group("roc_auc")),
    }


def _aggregate(runs: list[dict]) -> dict:
    """Compute mean, std, min, max for each metric across seed runs."""
    import numpy as np

    if not runs:
        return {}

    metrics = ["f1", "precision", "recall", "roc_auc"]
    agg = {}
    for m in metrics:
        values = [r[m] for r in runs]
        arr = np.array(values)
        agg[m] = {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "values": [round(v, 4) for v in values],
        }
    agg["n_seeds"] = len(runs)
    agg["seeds"] = [r["seed"] for r in runs]
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed training evaluation",
    )
    parser.add_argument(
        "--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seed values",
    )
    parser.add_argument(
        "--models", default=None,
        help="Comma-separated model names (default: all static)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--temporal", action="store_true",
        help="Evaluate temporal models instead of static",
    )
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        model_names = TEMPORAL_MODELS if args.temporal else STATIC_MODELS

    results: dict[str, dict] = {}

    for model_name in model_names:
        logger.info(
            "=== %s: %d seeds, %d epochs ===",
            model_name, len(seeds), args.epochs,
        )
        runs: list[dict] = []
        for seed in seeds:
            run = _run_training(
                model_name, seed, args.epochs, args.temporal,
            )
            if run is not None:
                runs.append(run)
                logger.info(
                    "  [%s seed=%d] f1=%.4f",
                    model_name, seed, run["f1"],
                )

        if not runs:
            logger.warning("[%s] no successful runs", model_name)
            results[model_name] = {"error": "no successful runs"}
            continue

        agg = _aggregate(runs)
        results[model_name] = {
            "summary": agg,
            "runs": runs,
        }
        logger.info(
            "[%s] f1=%.4f ± %.4f  (n=%d)",
            model_name, agg["f1"]["mean"], agg["f1"]["std"], agg["n_seeds"],
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved -> %s", output_path)

    _print_summary(results)


def _print_summary(results: dict) -> None:
    """Print a formatted comparison table."""
    print("\n── Multi-Seed Evaluation Summary ────────────────")
    print(f"  {'Model':<15} {'F1':>14} {'Precision':>14} {'Recall':>14} {'n':>4}")
    print("  " + "─" * 65)

    for name, data in results.items():
        if "error" in data:
            print(f"  {name:<15} {'ERROR':>14}")
            continue

        s = data["summary"]
        f1 = s["f1"]
        p = s["precision"]
        r = s["recall"]
        n = s["n_seeds"]
        print(
            f"  {name:<15}"
            f" {f1['mean']:.4f}±{f1['std']:.4f}"
            f" {p['mean']:.4f}±{p['std']:.4f}"
            f" {r['mean']:.4f}±{r['std']:.4f}"
            f" {n:>4}"
        )

    print()


if __name__ == "__main__":
    main()
