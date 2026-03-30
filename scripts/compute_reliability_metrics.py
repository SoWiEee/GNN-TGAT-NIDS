"""Offline script: compute model reliability metrics and write data/metrics/reliability.json.

Run once after training is complete:

    uv run python scripts/compute_reliability_metrics.py

Metrics written per model:
    clean_f1                    — weighted F1 on NF-UNSW-NB15-v2 test split
    dr_under_cpgd_eps01         — detection rate after C-PGD attack (ε=0.1, 40 steps)
    delta_f1_after_adv_training — improvement after adversarial training (optional)

The output is served statically by GET /api/metrics in the web app (ReliabilityPanel view).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'src' is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch_geometric.loader import DataLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed/static")
CHECKPOINTS_DIR = Path("checkpoints")
OUTPUT_PATH = Path("data/metrics/reliability.json")

MODEL_NAMES = ["graphsage", "gat"]
CPGD_EPSILON = 0.1
CPGD_STEPS = 40


def load_model(name: str) -> torch.nn.Module | None:
    path = CHECKPOINTS_DIR / f"{name}_best.pt"
    if not path.exists():
        logger.warning("Checkpoint not found: %s — skipping %s", path, name)
        return None
    model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    logger.info("Loaded %s from %s", name, path)
    return model


def evaluate_clean(model, loader) -> float:
    """Return weighted F1 on the given loader."""
    from src.eval.metrics import compute_metrics

    all_true, all_pred, all_proba = [], [], []
    with torch.inference_mode():
        for data in loader:
            logits = model(data)
            pred = logits.argmax(dim=-1)
            proba = torch.softmax(logits, dim=-1)
            all_true.append(data.y_multi)
            all_pred.append(pred)
            all_proba.append(proba)

    metrics = compute_metrics(
        torch.cat(all_true),
        torch.cat(all_pred),
        torch.cat(all_proba),
    )
    return round(float(metrics["f1"]), 4)


def evaluate_under_cpgd(model, loader, epsilon: float, steps: int) -> float:
    """Return detection rate (fraction of attack edges still correctly detected) under C-PGD."""
    from src.attack.cpgd import CPGDAttack

    scaler_path = PROCESSED_DIR / "scaler.json"
    attacker = CPGDAttack(epsilon=epsilon, steps=steps, scaler_path=scaler_path)

    total_attack = 0
    still_detected = 0

    for data in loader:
        with torch.inference_mode():
            orig_preds = model(data).argmax(dim=-1)

        adv_data = attacker.generate(model, data)

        with torch.inference_mode():
            adv_preds = model(adv_data).argmax(dim=-1)

        attack_mask = orig_preds > 0
        n_attack = int(attack_mask.sum())
        if n_attack == 0:
            continue

        # Still detected = was attack AND adversarial prediction is still non-benign
        detected = int(((orig_preds > 0) & (adv_preds > 0)).sum())
        total_attack += n_attack
        still_detected += detected

    if total_attack == 0:
        return 0.0
    return round(still_detected / total_attack, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute reliability metrics for trained GNN models")
    parser.add_argument(
        "--processed-dir", default=str(PROCESSED_DIR),
        help="Root directory of processed static graphs",
    )
    parser.add_argument(
        "--checkpoints-dir", default=str(CHECKPOINTS_DIR),
        help="Directory containing {name}_best.pt files",
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_PATH),
        help="Output JSON path",
    )
    parser.add_argument(
        "--epsilon", type=float, default=CPGD_EPSILON,
        help="C-PGD perturbation budget",
    )
    parser.add_argument(
        "--steps", type=int, default=CPGD_STEPS,
        help="C-PGD steps",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not (processed_dir / "meta.json").exists():
        logger.error("Processed data not found at %s — run static_builder.py first.", processed_dir)
        raise SystemExit(1)

    from src.data.static_dataset import StaticNIDSDataset

    test_ds = StaticNIDSDataset(processed_dir, split="test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    logger.info("Test split: %d windows", len(test_ds))

    results: dict[str, dict] = {}

    for name in MODEL_NAMES:
        model = load_model(name)
        if model is None:
            results[name] = {
                "clean_f1": None,
                "dr_under_cpgd_eps01": None,
                "delta_f1_after_adv_training": None,
            }
            continue

        logger.info("[%s] Computing clean F1 …", name)
        clean_f1 = evaluate_clean(model, test_loader)
        logger.info("[%s] clean_f1 = %.4f", name, clean_f1)

        logger.info("[%s] Computing DR under C-PGD (ε=%.2f, steps=%d) …", name, args.epsilon, args.steps)
        dr = evaluate_under_cpgd(model, test_loader, args.epsilon, args.steps)
        logger.info("[%s] dr_under_cpgd_eps01 = %.4f", name, dr)

        results[name] = {
            "clean_f1": clean_f1,
            "dr_under_cpgd_eps01": dr,
            # delta_f1_after_adv_training requires a separately adversarially-trained
            # checkpoint (checkpoints/{name}_adv_best.pt). Set to None until Phase 2.
            "delta_f1_after_adv_training": None,
        }

        # If adversarially-trained checkpoint exists, compute delta
        adv_path = Path(args.checkpoints_dir) / f"{name}_adv_best.pt"
        if adv_path.exists():
            logger.info("[%s] Adversarially-trained checkpoint found — computing ΔF1 …", name)
            adv_model = torch.load(adv_path, map_location="cpu", weights_only=False)
            adv_model.eval()
            adv_f1 = evaluate_clean(adv_model, test_loader)
            delta = round(adv_f1 - clean_f1, 4)
            results[name]["delta_f1_after_adv_training"] = delta
            logger.info("[%s] ΔF1 = %+.4f (%.4f → %.4f)", name, delta, clean_f1, adv_f1)

    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Reliability metrics saved → %s", output_path)

    # Print summary
    print("\n── Model Reliability Summary ────────────────")
    for name, m in results.items():
        f1 = f"{m['clean_f1']:.4f}" if m["clean_f1"] is not None else "N/A"
        dr = f"{m['dr_under_cpgd_eps01']:.4f}" if m["dr_under_cpgd_eps01"] is not None else "N/A"
        delta = f"+{m['delta_f1_after_adv_training']:.4f}" if m["delta_f1_after_adv_training"] is not None else "N/A"
        print(f"  {name:12s}  clean_f1={f1}  dr_cpgd={dr}  Δf1_adv={delta}")
    print()


if __name__ == "__main__":
    main()
