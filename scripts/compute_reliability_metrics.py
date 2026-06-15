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
import torch.nn.functional as F
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
TEMPORAL_CLIP_SIGMA = 3.0


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
        "cpgd_scope": None,
        "cpgd_constraint": None,
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


def _as_probabilities(output: torch.Tensor) -> torch.Tensor:
    row_sum = output.sum(dim=-1)
    if bool((output >= 0).all()) and torch.allclose(
        row_sum, torch.ones_like(row_sum), atol=1e-3, rtol=1e-3
    ):
        return output
    return torch.softmax(output, dim=-1)


def evaluate_clean(model, loader) -> dict[str, float]:
    """Return clean weighted and macro metrics on a static DataLoader."""
    all_true, all_pred, all_proba = [], [], []
    with torch.inference_mode():
        for data in loader:
            logits = model(data)
            proba = _as_probabilities(logits)
            pred = proba.argmax(dim=-1)
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
            proba = _as_probabilities(logits)
            pred = proba.argmax(dim=-1)
            all_true.append(batch.y)
            all_pred.append(pred)
            all_proba.append(proba)

    return _collect_metrics(all_true, all_pred, all_proba)


def _iter_with_limit(loader: Iterable, limit: int | None) -> Iterable:
    seen = 0
    for data in loader:
        if limit is not None and seen >= limit:
            break
        yield data
        seen += int(getattr(data, "num_graphs", 1))


def evaluate_under_cpgd(
    model,
    loader,
    epsilon: float,
    steps: int,
    max_windows: int | None = None,
) -> dict[str, float | int | bool]:
    """Return detection rate under full/sampled C-PGD for static models.

    This evaluator perturbs all attack-predicted edges in a window together,
    then projects the batch through the same raw-space ConstraintSet used by
    ``CPGDAttack``. It is intended for whole-split experiments where the
    per-edge attack class would be prohibitively slow.
    """
    from src.attack.cpgd import CPGDAttack

    scaler_path = PROCESSED_DIR / "scaler.json"
    attacker = CPGDAttack(epsilon=epsilon, steps=steps, scaler_path=scaler_path)

    total_attack = 0
    still_detected = 0
    n_windows = 0

    for batch_idx, data in enumerate(_iter_with_limit(loader, max_windows), start=1):
        n_windows += int(getattr(data, "num_graphs", 1))
        with torch.inference_mode():
            orig_preds = model(data).argmax(dim=-1)

        adv_data = _generate_static_cpgd_batch(model, data, attacker, epsilon, steps)

        with torch.inference_mode():
            adv_preds = model(adv_data).argmax(dim=-1)

        attack_mask = orig_preds > 0
        n_attack = int(attack_mask.sum())
        if n_attack == 0:
            continue

        detected = int(((orig_preds > 0) & (adv_preds > 0)).sum())
        total_attack += n_attack
        still_detected += detected
        if batch_idx == 1 or batch_idx % 10 == 0:
            logger.info(
                "  C-PGD progress: %d windows, %d attack edges",
                n_windows,
                total_attack,
            )

    if total_attack == 0:
        dr = 0.0
    else:
        dr = still_detected / total_attack
    return {
        "dr": round(dr, 4),
        "sampled": max_windows is not None,
        "sample_windows": n_windows,
        "attack_edges": total_attack,
        "constraint": "raw_constraint_set",
    }


def _generate_static_cpgd_batch(model, data, attacker, epsilon: float, steps: int):
    """Vectorised equivalent of ``CPGDAttack.generate`` for one static graph."""
    with torch.no_grad():
        orig_preds = model(data).argmax(dim=-1)
    attack_idx = (orig_preds > 0).nonzero(as_tuple=True)[0]
    if len(attack_idx) == 0:
        return data

    alpha = epsilon / max(steps, 1) * 2.5
    x_all = data.edge_attr.detach()
    x_orig = x_all[attack_idx].clone()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
    x_adv = x_adv.clamp(x_orig - epsilon, x_orig + epsilon)
    target = torch.zeros(len(attack_idx), dtype=torch.long, device=x_all.device)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        edge_attr_mod = x_all.clone()
        edge_attr_mod[attack_idx] = x_adv
        mod_data = data.clone()
        mod_data.edge_attr = edge_attr_mod

        logits = model(mod_data)[attack_idx]
        loss = F.cross_entropy(logits, target)
        loss.backward()
        grad = x_adv.grad
        if grad is None:
            break

        grad_norm = grad.flatten(1).norm(p=2, dim=1).clamp_min(1e-8).unsqueeze(1)
        x_next = (x_adv + alpha * grad / grad_norm).detach()
        x_raw = attacker._inverse_transform(x_next.cpu().numpy())
        x_raw = attacker.cs.project(x_raw)
        x_next = torch.from_numpy(attacker._transform(x_raw)).float().to(x_all.device)
        x_adv = x_next.clamp(x_orig - epsilon, x_orig + epsilon)

    adv_data = data.clone()
    edge_attr = x_all.clone()
    edge_attr[attack_idx] = x_adv.detach()
    adv_data.edge_attr = edge_attr
    return adv_data


def evaluate_under_cpgd_temporal(
    model,
    train_loader,
    test_loader,
    epsilon: float,
    steps: int,
    max_batches: int | None = None,
    warmup_max_batches: int | None = None,
) -> dict[str, float | int | bool | None | str]:
    """Return detection rate under constrained temporal C-PGD."""
    from src.attack.temporal_cpgd import ConstrainedTemporalCPGDAttack

    attacker = ConstrainedTemporalCPGDAttack(
        epsilon=epsilon,
        steps=steps,
        clip_min=-TEMPORAL_CLIP_SIGMA,
        clip_max=TEMPORAL_CLIP_SIGMA,
    )
    model.reset_memory()
    warmup_batches = 0
    with torch.no_grad():
        for batch in _iter_with_limit(train_loader, warmup_max_batches):
            warmup_batches += 1
            model(batch)
            model.update_state(batch.src, batch.dst, batch.t, batch.msg)

    total_attack = 0
    still_detected = 0
    n_batches = 0

    for batch in _iter_with_limit(test_loader, max_batches):
        n_batches += 1
        with torch.inference_mode():
            orig_preds = model(batch).argmax(dim=-1)

        adv_batch = attacker.generate(model, batch)

        with torch.inference_mode():
            adv_preds = model(adv_batch).argmax(dim=-1)

        attack_mask = orig_preds > 0
        n_attack = int(attack_mask.sum())
        if n_attack == 0:
            continue

        detected = int(((orig_preds > 0) & (adv_preds > 0)).sum())
        total_attack += n_attack
        still_detected += detected
        if n_batches == 1 or n_batches % 10 == 0:
            logger.info(
                "  Temporal C-PGD progress: %d batches, %d attack edges",
                n_batches,
                total_attack,
            )

    if total_attack == 0:
        dr = 0.0
    else:
        dr = still_detected / total_attack
    return {
        "dr": round(dr, 4),
        "sampled": max_batches is not None,
        "sample_windows": n_batches if max_batches is not None else None,
        "attack_edges": total_attack,
        "constraint": (
            "normalized_clip_and_linf"
            if warmup_max_batches is None
            else f"normalized_clip_and_linf,warmup_batches={warmup_batches}"
        ),
    }


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
        "--static-batch-size", type=int, default=16,
        help="Static graph DataLoader batch size for clean and C-PGD evaluation.",
    )
    parser.add_argument(
        "--temporal-cpgd-max-batches", type=int, default=None,
        help="Optional cap for temporal C-PGD batches.",
    )
    parser.add_argument(
        "--temporal-warmup-max-batches", type=int, default=None,
        help="Optional cap for train batches used to warm temporal memory before C-PGD.",
    )
    parser.add_argument(
        "--skip-temporal-cpgd", action="store_true",
        help="Leave temporal dr_under_cpgd_eps01 as null; compute clean temporal metrics only.",
    )
    parser.add_argument(
        "--include-ensemble", action="store_true",
        help="Add a soft-vote static ensemble clean-metric experiment.",
    )
    parser.add_argument(
        "--models",
        default=",".join(MODEL_NAMES),
        help="Comma-separated model list to update, e.g. graphsage,gat or tgat,tgn.",
    )
    parser.add_argument(
        "--attack-only",
        action="store_true",
        help="Reuse existing clean metrics and update only C-PGD fields for selected models.",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = sorted(set(selected_models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}")

    # ── Static data ──────────────────────────────────────────────────────────
    static_available = (processed_dir / "meta.json").exists()
    test_loader = None
    if static_available:
        from src.data.static_dataset import StaticNIDSDataset
        test_ds = StaticNIDSDataset(processed_dir, split="test")
        test_loader = DataLoader(test_ds, batch_size=args.static_batch_size, shuffle=False)
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

    if output_path.exists():
        results: dict[str, dict] = json.loads(output_path.read_text())
    else:
        results = {}

    for name in selected_models:
        model = load_model(name)
        if model is None:
            results[name] = _empty_metrics()
            continue

        is_temporal = name in TEMPORAL_MODELS

        clean_metrics = None
        if args.attack_only and name in results:
            result = {**_empty_metrics(), **results[name]}
            logger.info("[%s] Reusing existing clean metrics", name)
        else:
            # ── Clean F1 ────────────────────────────────────────────────────
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
                dr_meta = evaluate_under_cpgd_temporal(
                    model, temporal_train_loader, temporal_test_loader,
                    args.epsilon, args.steps, args.temporal_cpgd_max_batches,
                    args.temporal_warmup_max_batches,
                )
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
            result["cpgd_scope"] = "sampled" if dr_meta["sampled"] else "full_test"
            result["cpgd_constraint"] = dr_meta.get("constraint")
            logger.info("[%s] dr_under_cpgd_eps01 = %.4f", name, dr_meta["dr"])
        results[name] = result

        # ── Adversarially-trained checkpoint (optional) ──────────────────────
        adv_path = Path(args.checkpoints_dir) / f"{name}_adv_best.pt"
        if not args.attack_only and adv_path.exists() and adv_path.stat().st_size > 0:
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

    if args.include_ensemble and static_available:
        from src.models.ensemble import EnsembleModel

        ensemble_models = {
            name: load_model(name)
            for name in STATIC_MODELS
            if (Path(args.checkpoints_dir) / f"{name}_best.pt").exists()
        }
        ensemble_models = {k: v for k, v in ensemble_models.items() if v is not None}
        if len(ensemble_models) >= 2:
            logger.info("[ensemble] Computing static soft-vote clean metrics …")
            ensemble = EnsembleModel(ensemble_models, strategy="soft_vote")
            clean_metrics = evaluate_clean(ensemble, test_loader)
            result = _empty_metrics()
            result.update(_format_metrics(clean_metrics))
            result["cpgd_scope"] = "not_applicable"
            result["cpgd_constraint"] = "ensemble_clean_only"
            results["ensemble"] = result
        else:
            logger.warning("[ensemble] Need at least two static checkpoints; skipping")

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
