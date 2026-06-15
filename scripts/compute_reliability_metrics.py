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

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch_geometric.loader import DataLoader, TemporalDataLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed/static")
TEMPORAL_DIR = Path("data/processed/temporal")
CHECKPOINTS_DIR = Path("checkpoints")
OUTPUT_PATH = Path("data/metrics/reliability.json")

STATIC_MODELS = ["graphsage", "gat"]
TEMPORAL_MODELS = ["tgat", "tgn"]
MODEL_NAMES = STATIC_MODELS + TEMPORAL_MODELS
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
    """Return weighted F1 on a static DataLoader."""
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


def evaluate_clean_temporal(model, train_loader, test_loader) -> float:
    """Return weighted F1 for temporal models. Replays train to warm memory."""
    from src.eval.metrics import compute_metrics

    model.reset_memory()
    with torch.no_grad():
        for batch in train_loader:
            model(batch)
            model.update_state(batch.src, batch.dst, batch.t, batch.msg)

    all_true, all_pred, all_proba = [], [], []
    with torch.inference_mode():
        for batch in test_loader:
            logits = model(batch)
            pred = logits.argmax(dim=-1)
            proba = torch.softmax(logits, dim=-1)
            all_true.append(batch.y)
            all_pred.append(pred)
            all_proba.append(proba)

    metrics = compute_metrics(
        torch.cat(all_true),
        torch.cat(all_pred),
        torch.cat(all_proba),
    )
    return round(float(metrics["f1"]), 4)


def evaluate_under_cpgd(model, loader, epsilon: float, steps: int) -> float:
    """Return detection rate under C-PGD for static models."""
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

        detected = int(((orig_preds > 0) & (adv_preds > 0)).sum())
        total_attack += n_attack
        still_detected += detected

    if total_attack == 0:
        return 0.0
    return round(still_detected / total_attack, 4)


def evaluate_under_cpgd_temporal(
    model, train_loader, test_loader, epsilon: float, steps: int,
) -> float:
    """Return detection rate under C-PGD for temporal models.

    Perturbs edge features (msg) on test batches after warming memory on train.
    """
    model.reset_memory()
    with torch.no_grad():
        for batch in train_loader:
            model(batch)
            model.update_state(batch.src, batch.dst, batch.t, batch.msg)

    total_attack = 0
    still_detected = 0

    for batch in test_loader:
        with torch.inference_mode():
            orig_preds = model(batch).argmax(dim=-1)

        msg_orig = batch.msg.clone()
        batch.msg.requires_grad_(True)

        for _ in range(steps):
            logits = model(batch)
            loss = -torch.nn.functional.cross_entropy(logits, orig_preds)
            loss.backward()
            with torch.no_grad():
                batch.msg.data = batch.msg.data - epsilon / steps * batch.msg.grad.sign()
                delta = batch.msg.data - msg_orig
                delta.clamp_(-epsilon, epsilon)
                batch.msg.data = msg_orig + delta
            batch.msg.grad.zero_()

        batch.msg.requires_grad_(False)

        with torch.inference_mode():
            adv_preds = model(batch).argmax(dim=-1)

        attack_mask = orig_preds > 0
        n_attack = int(attack_mask.sum())
        if n_attack == 0:
            continue

        detected = int(((orig_preds > 0) & (adv_preds > 0)).sum())
        total_attack += n_attack
        still_detected += detected

    if total_attack == 0:
        return 0.0
    return round(still_detected / total_attack, 4)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute reliability metrics for trained GNN models"
    )
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

    # ── Static data ──────────────────────────────────────────────────────────
    static_available = (processed_dir / "meta.json").exists()
    test_loader = None
    if static_available:
        from src.data.static_dataset import StaticNIDSDataset
        test_ds = StaticNIDSDataset(processed_dir, split="test")
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        logger.info("Static test split: %d windows", len(test_ds))
    else:
        logger.warning("Static processed data not found at %s", processed_dir)

    # ── Temporal data ────────────────────────────────────────────────────────
    temporal_available = (TEMPORAL_DIR / "meta.json").exists()
    temporal_train_loader = None
    temporal_test_loader = None
    if temporal_available:
        train_td = torch.load(TEMPORAL_DIR / "train.pt", weights_only=False)
        test_td = torch.load(TEMPORAL_DIR / "test.pt", weights_only=False)
        temporal_train_loader = TemporalDataLoader(train_td, batch_size=200)
        temporal_test_loader = TemporalDataLoader(test_td, batch_size=200)
        logger.info("Temporal test split: %d events", len(test_td.src))
    else:
        logger.warning("Temporal processed data not found at %s", TEMPORAL_DIR)

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

        is_temporal = name in TEMPORAL_MODELS

        # ── Clean F1 ────────────────────────────────────────────────────────
        logger.info("[%s] Computing clean F1 …", name)
        if is_temporal and temporal_available:
            clean_f1 = evaluate_clean_temporal(model, temporal_train_loader, temporal_test_loader)
        elif not is_temporal and static_available:
            clean_f1 = evaluate_clean(model, test_loader)
        else:
            logger.warning("[%s] No matching data — skipping", name)
            results[name] = {
                "clean_f1": None, "dr_under_cpgd_eps01": None,
                "delta_f1_after_adv_training": None,
            }
            continue
        logger.info("[%s] clean_f1 = %.4f", name, clean_f1)

        # ── DR under C-PGD ──────────────────────────────────────────────────
        logger.info(
            "[%s] Computing DR under C-PGD (ε=%.2f, steps=%d) …",
            name, args.epsilon, args.steps,
        )
        if is_temporal and temporal_available:
            dr = evaluate_under_cpgd_temporal(
                model, temporal_train_loader, temporal_test_loader,
                args.epsilon, args.steps,
            )
        else:
            dr = evaluate_under_cpgd(model, test_loader, args.epsilon, args.steps)
        logger.info("[%s] dr_under_cpgd_eps01 = %.4f", name, dr)

        results[name] = {
            "clean_f1": clean_f1,
            "dr_under_cpgd_eps01": dr,
            "delta_f1_after_adv_training": None,
        }

        # ── Adversarially-trained checkpoint (optional) ──────────────────────
        adv_path = Path(args.checkpoints_dir) / f"{name}_adv_best.pt"
        if adv_path.exists():
            logger.info("[%s] Adversarially-trained checkpoint found — computing ΔF1 …", name)
            adv_model = torch.load(adv_path, map_location="cpu", weights_only=False)
            adv_model.eval()
            if is_temporal and temporal_available:
                adv_f1 = evaluate_clean_temporal(
                    adv_model, temporal_train_loader, temporal_test_loader,
                )
            else:
                adv_f1 = evaluate_clean(adv_model, test_loader)
            delta = round(adv_f1 - clean_f1, 4)
            results[name]["delta_f1_after_adv_training"] = delta
            logger.info("[%s] ΔF1 = %+.4f (%.4f → %.4f)", name, delta, clean_f1, adv_f1)

    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Reliability metrics saved → %s", output_path)

    print("\n── Model Reliability Summary ────────────────")
    for name, m in results.items():
        f1 = f"{m['clean_f1']:.4f}" if m["clean_f1"] is not None else "N/A"
        dr = f"{m['dr_under_cpgd_eps01']:.4f}" if m["dr_under_cpgd_eps01"] is not None else "N/A"
        adv = m["delta_f1_after_adv_training"]
        delta = f"+{adv:.4f}" if adv is not None else "N/A"
        print(f"  {name:12s}  clean_f1={f1}  dr_cpgd={dr}  Δf1_adv={delta}")
    print()


if __name__ == "__main__":
    main()
