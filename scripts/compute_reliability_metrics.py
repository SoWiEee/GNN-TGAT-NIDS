"""Offline script: compute model reliability metrics and write data/metrics/reliability.json.

Run once after training is complete:

    uv run python scripts/compute_reliability_metrics.py

Metrics written per model:
    clean_f1                    — weighted F1 on NF-UNSW-NB15-v2 test split
    clean_precision             — weighted precision on the test split
    clean_recall                — weighted recall on the test split
    clean_roc_auc               — weighted multiclass ROC-AUC on the test split
    clean_macro_f1              — macro F1 on the test split
    dr_under_cpgd_eps01         — detection rate after C-PGD attack (ε=0.1, 40 steps)
    delta_f1_after_adv_training — improvement after adversarial training (optional)

The output is served statically by GET /api/metrics in the web app (ReliabilityPanel view).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable
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


def _empty_metrics() -> dict:
    return {
        "clean_f1": None,
        "clean_precision": None,
        "clean_recall": None,
        "clean_roc_auc": None,
        "clean_macro_f1": None,
        "dr_under_cpgd_eps01": None,
        "dr_under_cpgd_eps01_sampled": None,
        "cpgd_epsilon": None,
        "cpgd_steps": None,
        "cpgd_sample_windows": None,
        "cpgd_attack_edges": None,
        "delta_f1_after_adv_training": None,
    }


def _round_metric(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _format_metrics(metrics: dict[str, float]) -> dict:
    return {
        "clean_f1": _round_metric(metrics.get("f1")),
        "clean_precision": _round_metric(metrics.get("precision")),
        "clean_recall": _round_metric(metrics.get("recall")),
        "clean_roc_auc": _round_metric(metrics.get("roc_auc")),
        "clean_macro_f1": _round_metric(metrics.get("macro_f1")),
    }


def load_model(name: str) -> torch.nn.Module | None:
    path = CHECKPOINTS_DIR / f"{name}_best.pt"
    if not path.exists():
        logger.warning("Checkpoint not found: %s — skipping %s", path, name)
        return None
    model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    logger.info("Loaded %s from %s", name, path)
    return model


def _collect_metrics(all_true, all_pred, all_proba) -> dict[str, float]:
    from src.eval.metrics import compute_metrics

    y_true = torch.cat(all_true)
    y_pred = torch.cat(all_pred)
    y_proba = torch.cat(all_proba)
    weighted = compute_metrics(y_true, y_pred, y_proba, average="weighted")
    macro = compute_metrics(y_true, y_pred, y_proba, average="macro")
    weighted["macro_f1"] = macro["f1"]
    return weighted


def evaluate_clean(model, loader) -> dict[str, float]:
    """Return clean weighted and macro metrics on a static DataLoader."""
    all_true, all_pred, all_proba = [], [], []
    with torch.inference_mode():
        for data in loader:
            logits = model(data)
            pred = logits.argmax(dim=-1)
            proba = torch.softmax(logits, dim=-1)
            all_true.append(data.y_multi)
            all_pred.append(pred)
            all_proba.append(proba)

    return _collect_metrics(all_true, all_pred, all_proba)


def evaluate_clean_temporal(model, train_loader, test_loader) -> dict[str, float]:
    """Return clean weighted and macro metrics for temporal models.

    Replays train to warm memory before evaluating the test split.
    """
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

    return _collect_metrics(all_true, all_pred, all_proba)


def _iter_with_limit(loader: Iterable, limit: int | None) -> Iterable:
    for idx, data in enumerate(loader):
        if limit is not None and idx >= limit:
            break
        yield data


def evaluate_under_cpgd(
    model,
    loader,
    epsilon: float,
    steps: int,
    max_windows: int | None = None,
) -> dict[str, float | int | bool]:
    """Return detection rate under C-PGD for static models."""
    from src.attack.cpgd import CPGDAttack

    scaler_path = PROCESSED_DIR / "scaler.json"
    attacker = CPGDAttack(epsilon=epsilon, steps=steps, scaler_path=scaler_path)

    total_attack = 0
    still_detected = 0
    n_windows = 0

    for data in _iter_with_limit(loader, max_windows):
        n_windows += 1
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
        dr = 0.0
    else:
        dr = still_detected / total_attack
    return {
        "dr": round(dr, 4),
        "sampled": max_windows is not None,
        "sample_windows": n_windows,
        "attack_edges": total_attack,
    }


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
    parser.add_argument(
        "--cpgd-max-windows", type=int, default=None,
        help=(
            "Optional cap for static C-PGD evaluation windows. "
            "Useful because full-test C-PGD is expensive."
        ),
    )
    parser.add_argument(
        "--skip-temporal-cpgd", action="store_true",
        help="Leave temporal dr_under_cpgd_eps01 as null; compute clean temporal metrics only.",
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
            results[name] = _empty_metrics()
            continue

        is_temporal = name in TEMPORAL_MODELS

        # ── Clean F1 ────────────────────────────────────────────────────────
        logger.info("[%s] Computing clean F1 …", name)
        if is_temporal and temporal_available:
            clean_metrics = evaluate_clean_temporal(
                model, temporal_train_loader, temporal_test_loader
            )
        elif not is_temporal and static_available:
            clean_metrics = evaluate_clean(model, test_loader)
        else:
            logger.warning("[%s] No matching data — skipping", name)
            results[name] = _empty_metrics()
            continue

        result = _empty_metrics()
        result.update(_format_metrics(clean_metrics))
        logger.info(
            "[%s] clean: f1=%.4f precision=%.4f recall=%.4f roc_auc=%.4f macro_f1=%.4f",
            name,
            result["clean_f1"] or 0.0,
            result["clean_precision"] or 0.0,
            result["clean_recall"] or 0.0,
            result["clean_roc_auc"] or 0.0,
            result["clean_macro_f1"] or 0.0,
        )

        # ── DR under C-PGD ──────────────────────────────────────────────────
        logger.info(
            "[%s] Computing DR under C-PGD (ε=%.2f, steps=%d) …",
            name, args.epsilon, args.steps,
        )
        if is_temporal:
            if args.skip_temporal_cpgd:
                logger.info("[%s] Temporal C-PGD skipped by request", name)
                dr_meta = None
            elif temporal_available:
                dr = evaluate_under_cpgd_temporal(
                    model, temporal_train_loader, temporal_test_loader,
                    args.epsilon, args.steps,
                )
                dr_meta = {
                    "dr": dr,
                    "sampled": False,
                    "sample_windows": None,
                    "attack_edges": None,
                }
            else:
                dr_meta = None
        else:
            dr_meta = evaluate_under_cpgd(
                model, test_loader, args.epsilon, args.steps, args.cpgd_max_windows
            )
        if dr_meta is not None:
            result["dr_under_cpgd_eps01"] = dr_meta["dr"]
            result["dr_under_cpgd_eps01_sampled"] = dr_meta["sampled"]
            result["cpgd_epsilon"] = args.epsilon
            result["cpgd_steps"] = args.steps
            result["cpgd_sample_windows"] = dr_meta["sample_windows"]
            result["cpgd_attack_edges"] = dr_meta["attack_edges"]
            logger.info("[%s] dr_under_cpgd_eps01 = %.4f", name, dr_meta["dr"])
        results[name] = result

        # ── Adversarially-trained checkpoint (optional) ──────────────────────
        adv_path = Path(args.checkpoints_dir) / f"{name}_adv_best.pt"
        if adv_path.exists() and adv_path.stat().st_size > 0:
            logger.info("[%s] Adversarially-trained checkpoint found — computing ΔF1 …", name)
            adv_model = torch.load(adv_path, map_location="cpu", weights_only=False)
            adv_model.eval()
            if is_temporal and temporal_available:
                adv_metrics = evaluate_clean_temporal(
                    adv_model, temporal_train_loader, temporal_test_loader,
                )
            else:
                adv_metrics = evaluate_clean(adv_model, test_loader)
            adv_f1 = float(adv_metrics["f1"])
            clean_f1 = float(clean_metrics["f1"])
            delta = round(adv_f1 - clean_f1, 4)
            results[name]["delta_f1_after_adv_training"] = delta
            logger.info("[%s] ΔF1 = %+.4f (%.4f → %.4f)", name, delta, clean_f1, adv_f1)
        elif adv_path.exists():
            logger.warning("[%s] Ignoring empty adversarial checkpoint: %s", name, adv_path)

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
