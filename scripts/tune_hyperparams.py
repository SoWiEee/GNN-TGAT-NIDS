"""Bayesian hyperparameter search using Optuna.

Searches the joint space of architecture + optimiser knobs for GraphSAGE and GAT
and writes the best configuration to results/best_hparams_{model}.json.

Usage:
    uv run python scripts/tune_hyperparams.py --model graphsage --trials 50
    uv run python scripts/tune_hyperparams.py --model gat --trials 50
    uv run python scripts/tune_hyperparams.py --model graphsage --trials 100 --epochs 40

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

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
import torch
from torch_geometric.loader import DataLoader

from src.data.static_dataset import StaticNIDSDataset
from src.eval.metrics import compute_class_weights, compute_metrics
from src.utils.seed import set_global_seed

logging.basicConfig(level=logging.WARNING)  # suppress per-epoch noise during search
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED_DIR = Path("data/processed/static")
RESULTS_DIR = Path("results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Search space ──────────────────────────────────────────────────────────────

def _suggest(trial: optuna.Trial, model_name: str) -> dict:
    params = {
        "lr":         trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
        "num_layers": trial.suggest_int("num_layers", 2, 4),
        "dropout":    trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }
    if model_name == "gat":
        params["num_heads"] = trial.suggest_categorical("num_heads", [2, 4, 8])
        # hidden_dim must be divisible by num_heads
        if params["hidden_dim"] % params["num_heads"] != 0:
            raise optuna.TrialPruned()
    if model_name == "graphsage":
        params["aggregation"] = trial.suggest_categorical(
            "aggregation", ["mean", "max"]
        )
    return params


# ── Build model from suggested params ────────────────────────────────────────

def _build_model(model_name: str, params: dict, n_node: int, n_edge: int, n_classes: int):
    if model_name == "graphsage":
        from src.models.graphsage import GraphSAGEModel
        return GraphSAGEModel(
            in_node_channels=n_node,
            in_edge_channels=n_edge,
            hidden_dim=params["hidden_dim"],
            num_classes=n_classes,
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            aggregation=params.get("aggregation", "mean"),
        )
    from src.models.gat import GATModel
    return GATModel(
        in_node_channels=n_node,
        in_edge_channels=n_edge,
        hidden_dim=params["hidden_dim"],
        num_classes=n_classes,
        num_layers=params["num_layers"],
        num_heads=params.get("num_heads", 4),
        dropout=params["dropout"],
    )


# ── Objective ────────────────────────────────────────────────────────────────

def _objective(trial: optuna.Trial, model_name: str, n_epochs: int) -> float:
    params = _suggest(trial, model_name)

    train_ds = StaticNIDSDataset(PROCESSED_DIR, split="train")
    val_ds   = StaticNIDSDataset(PROCESSED_DIR, split="val")

    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"],
                              shuffle=True, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=params["batch_size"],
                              shuffle=False, pin_memory=pin)

    sample = next(iter(train_loader))
    n_node = sample.x.shape[1]

    model = _build_model(
        model_name, params,
        n_node, train_ds.n_edge_features, train_ds.n_classes,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    all_labels = torch.cat([d.y_multi for d in train_loader])
    weights = compute_class_weights(all_labels, train_ds.n_classes, DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    use_amp = DEVICE.type == "cuda"
    amp_scaler = torch.amp.GradScaler() if use_amp else None

    best_val_f1 = 0.0

    for epoch in range(n_epochs):
        # ── train ──
        model.train()
        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                loss = criterion(model(data), data.y_multi)
            if amp_scaler:
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss.backward()
                optimizer.step()

        # ── val every 5 epochs to keep trials fast ──
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

        val_f1 = compute_metrics(
            torch.cat(all_true), torch.cat(all_pred), torch.cat(all_proba)
        )["f1"]

        best_val_f1 = max(best_val_f1, val_f1)

        # Report intermediate value for pruning
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_f1


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="graphsage", choices=["graphsage", "gat"])
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30,
                        help="Epochs per trial (shorter = faster search, noisier estimate)")
    parser.add_argument("--seed",   type=int, default=42)
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

    study.optimize(
        lambda trial: _objective(trial, args.model, args.epochs),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\nBest val F1 : {best.value:.4f}")
    print(f"Best params : {best.params}")

    out = RESULTS_DIR / f"best_hparams_{args.model}.json"
    out.write_text(json.dumps({"val_f1": best.value, **best.params}, indent=2))
    print(f"Saved → {out}")

    # Print top-5 for manual comparison
    print("\nTop 5 trials:")
    for t in sorted(study.trials, key=lambda t: t.value or 0, reverse=True)[:5]:
        print(f"  #{t.number:3d}  f1={t.value:.4f}  {t.params}")


if __name__ == "__main__":
    main()
