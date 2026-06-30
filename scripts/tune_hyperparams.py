"""Bayesian hyperparameter search using Optuna.

Searches the joint space of architecture + optimiser knobs and writes the best
configuration to results/best_hparams_{model}.json.

Usage:
    uv run python scripts/tune_hyperparams.py --model graphsage --trials 50
    uv run python scripts/tune_hyperparams.py --model gat --trials 50
    uv run python scripts/tune_hyperparams.py --model tgat --trials 30
    uv run python scripts/tune_hyperparams.py --model tgn --trials 30

    # Live dashboard (open http://localhost:8080 while running):
    uv run optuna-dashboard sqlite:///results/optuna.db

Notes:
    - Each trial trains for --epochs epochs (default 30) — enough to rank
      hyperparameters without running full 200-epoch training.
    - Pruning (MedianPruner) stops unpromising trials early after 10 epochs.
    - Results are stored in results/optuna.db (SQLite) and survive crashes.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
import torch
from torch_geometric.loader import DataLoader, TemporalDataLoader

from src.eval.metrics import compute_class_weights, compute_metrics
from src.utils.seed import set_global_seed

logging.basicConfig(level=logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)

STATIC_DIR = Path("data/processed/static")
TEMPORAL_DIR = Path("data/processed/temporal")
RESULTS_DIR = Path("results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATIC_MODELS = {"graphsage", "gat"}
TEMPORAL_MODELS = {"tgat", "tgn"}


# ── Search space ──────────────────────────────────────────────────────────────

def _suggest(trial: optuna.Trial, model_name: str) -> dict:
    params: dict = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "focal_gamma": trial.suggest_float("focal_gamma", 0.5, 3.0, step=0.5),
        "oversample_factor": trial.suggest_int("oversample_factor", 1, 20),
        "weight_strategy": trial.suggest_categorical(
            "weight_strategy", ["inverse", "sqrt_inverse", "effective"],
        ),
    }

    if model_name in STATIC_MODELS:
        params["hidden_dim"] = trial.suggest_categorical("hidden_dim", [128, 256, 512])
        params["num_layers"] = trial.suggest_int("num_layers", 2, 4)
        params["dropout"] = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        params["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64])

        if model_name == "gat":
            params["num_heads"] = trial.suggest_categorical("num_heads", [2, 4, 8])
            if params["hidden_dim"] % params["num_heads"] != 0:
                raise optuna.TrialPruned()
        if model_name == "graphsage":
            params["aggregation"] = trial.suggest_categorical("aggregation", ["mean", "max"])
    else:
        params["hidden_dim"] = trial.suggest_categorical("hidden_dim", [64, 128, 172, 256])
        params["heads"] = trial.suggest_categorical("heads", [1, 2, 4])
        params["n_neighbors"] = trial.suggest_categorical("n_neighbors", [10, 20, 30])
        params["batch_size"] = trial.suggest_categorical("batch_size", [100, 200, 400])

        if model_name == "tgn":
            params["memory_dim"] = trial.suggest_categorical("memory_dim", [64, 100, 128])

    return params


# ── Build model from suggested params ────────────────────────────────────────

def _build_model(model_name: str, params: dict, **kwargs):
    if model_name == "graphsage":
        from src.models.graphsage import GraphSAGEModel
        return GraphSAGEModel(
            in_node_channels=kwargs["n_node"],
            in_edge_channels=kwargs["n_edge"],
            hidden_dim=params["hidden_dim"],
            num_classes=kwargs["n_classes"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            aggregation=params.get("aggregation", "mean"),
        )
    if model_name == "gat":
        from src.models.gat import GATModel
        return GATModel(
            in_node_channels=kwargs["n_node"],
            in_edge_channels=kwargs["n_edge"],
            hidden_dim=params["hidden_dim"],
            num_classes=kwargs["n_classes"],
            num_layers=params["num_layers"],
            num_heads=params.get("num_heads", 4),
            dropout=params["dropout"],
        )
    if model_name == "tgat":
        from src.models.tgat import TGATModel
        return TGATModel(
            num_nodes=kwargs["num_nodes"],
            raw_msg_dim=kwargs["n_edge"],
            num_classes=kwargs["n_classes"],
            hidden_dim=params["hidden_dim"],
            heads=params["heads"],
            n_neighbors=params["n_neighbors"],
        )
    from src.models.tgn import TGNModel
    return TGNModel(
        num_nodes=kwargs["num_nodes"],
        raw_msg_dim=kwargs["n_edge"],
        num_classes=kwargs["n_classes"],
        hidden_dim=params["hidden_dim"],
        heads=params["heads"],
        num_neighbors=params["n_neighbors"],
        memory_dim=params.get("memory_dim", 100),
    )


# ── Objective (static) ──────────────────────────────────────────────────────

def _oversample_edges(logits, labels, criterion, factor):
    """Replicate rare-class edges in the loss computation."""
    loss = criterion(logits, labels)
    if loss.dim() == 0:
        return loss
    if factor <= 1:
        return loss.mean()
    counts = torch.bincount(labels, minlength=logits.shape[1]).float()
    median_count = counts[counts > 0].median()
    weights_per_sample = torch.ones(len(labels), device=labels.device)
    for cls_id in range(logits.shape[1]):
        mask = labels == cls_id
        n = mask.sum().item()
        if 0 < n < median_count:
            repeats = min(int(median_count / n), factor)
            weights_per_sample[mask] = float(repeats)
    return (loss * weights_per_sample).mean()


def _objective_static(trial: optuna.Trial, model_name: str, n_epochs: int) -> float:
    from src.data.static_dataset import StaticNIDSDataset

    params = _suggest(trial, model_name)

    train_ds = StaticNIDSDataset(STATIC_DIR, split="train")
    val_ds = StaticNIDSDataset(STATIC_DIR, split="val")

    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"],
                              shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=params["batch_size"],
                            shuffle=False, pin_memory=pin)

    sample = next(iter(train_loader))
    n_node = sample.x.shape[1]

    model = _build_model(
        model_name, params,
        n_node=n_node, n_edge=train_ds.n_edge_features, n_classes=train_ds.n_classes,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs,
    )

    all_labels = torch.cat([d.y_multi for d in train_loader])
    weights = compute_class_weights(
        all_labels, train_ds.n_classes, DEVICE,
        strategy=params.get("weight_strategy", "inverse"),
    )
    from src.eval.losses import FocalLoss
    criterion = FocalLoss(
        weight=weights, gamma=params["focal_gamma"], reduction="none",
    )

    use_amp = DEVICE.type == "cuda"
    amp_scaler = torch.amp.GradScaler() if use_amp else None
    osf = params.get("oversample_factor", 1)

    best_macro_f1 = 0.0

    for epoch in range(n_epochs):
        model.train()
        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                logits = model(data)
                loss = _oversample_edges(logits, data.y_multi, criterion, osf)
            if amp_scaler:
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss.backward()
                optimizer.step()
        scheduler.step()

        if (epoch + 1) % 5 != 0 and epoch + 1 < n_epochs:
            continue

        model.eval()
        all_true, all_pred, all_proba = [], [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(DEVICE)
                logits = model(data)
                all_true.append(data.y_multi.cpu())
                all_pred.append(logits.argmax(-1).cpu())
                all_proba.append(torch.softmax(logits, -1).cpu())

        metrics = compute_metrics(
            torch.cat(all_true), torch.cat(all_pred), torch.cat(all_proba),
        )
        macro_f1 = metrics.get("macro_f1", metrics["f1"])

        best_macro_f1 = max(best_macro_f1, macro_f1)
        trial.report(macro_f1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_macro_f1


# ── Objective (temporal) ─────────────────────────────────────────────────────

def _objective_temporal(trial: optuna.Trial, model_name: str, n_epochs: int) -> float:
    params = _suggest(trial, model_name)

    with open(TEMPORAL_DIR / "meta.json") as f:
        meta = json.load(f)

    from app.services.torch_load import load_torch_artifact
    train_td = load_torch_artifact(TEMPORAL_DIR / "train.pt")
    val_td = load_torch_artifact(TEMPORAL_DIR / "val.pt")

    bs = params["batch_size"]
    train_loader = TemporalDataLoader(train_td, batch_size=bs)
    val_loader = TemporalDataLoader(val_td, batch_size=bs)

    model = _build_model(
        model_name, params,
        num_nodes=meta["num_nodes"], n_edge=meta["n_features"], n_classes=meta["n_classes"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    all_labels = torch.cat([batch.y for batch in TemporalDataLoader(train_td, batch_size=10000)])
    weights = compute_class_weights(all_labels, meta["n_classes"], DEVICE)
    from src.eval.losses import FocalLoss
    criterion = FocalLoss(weight=weights, gamma=params["focal_gamma"])

    use_amp = DEVICE.type == "cuda"
    amp_scaler = torch.amp.GradScaler() if use_amp else None

    best_macro_f1 = 0.0

    for epoch in range(n_epochs):
        model.train()
        model.reset_memory()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp,
                                    dtype=torch.bfloat16):
                logits = model(batch)
                loss = criterion(logits, batch.y)
            if amp_scaler:
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss.backward()
                optimizer.step()
            model.update_state(batch.src, batch.dst, batch.t, batch.msg)

        if (epoch + 1) % 5 != 0 and epoch + 1 < n_epochs:
            continue

        model.eval()
        all_true, all_pred, all_proba = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                logits = model(batch)
                all_true.append(batch.y.cpu())
                all_pred.append(logits.argmax(-1).cpu())
                all_proba.append(torch.softmax(logits, -1).cpu())

        macro = compute_metrics(
            torch.cat(all_true), torch.cat(all_pred), y_proba=None, average="macro",
        )
        macro_f1 = macro["f1"]

        best_macro_f1 = max(best_macro_f1, macro_f1)
        trial.report(macro_f1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_macro_f1


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="graphsage",
                        choices=["graphsage", "gat", "tgat", "tgn"])
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30,
                        help="Epochs per trial (shorter = faster search)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    db_path = RESULTS_DIR / "optuna.db"
    study = optuna.create_study(
        study_name=f"{args.model}_search",
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        load_if_exists=True,
    )

    print(f"Searching {args.model} hyperparams — {args.trials} trials × {args.epochs} epochs")
    print(f"  device : {DEVICE}")
    print(f"  storage: {db_path}  (resume-safe)")
    print(f"  dashboard: uv run optuna-dashboard sqlite:///{db_path}")
    print()

    def objective(trial: optuna.Trial) -> float:
        if args.model in TEMPORAL_MODELS:
            return _objective_temporal(trial, args.model, args.epochs)
        return _objective_static(trial, args.model, args.epochs)

    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\nBest val macro_f1 : {best.value:.4f}")
    print(f"Best params       : {best.params}")

    out = RESULTS_DIR / f"best_hparams_{args.model}.json"
    out.write_text(json.dumps({"val_macro_f1": best.value, **best.params}, indent=2))
    print(f"Saved → {out}")

    print("\nTop 5 trials:")
    for t in sorted(study.trials, key=lambda t: t.value or 0, reverse=True)[:5]:
        print(f"  #{t.number:3d}  macro_f1={t.value:.4f}  {t.params}")


if __name__ == "__main__":
    main()
