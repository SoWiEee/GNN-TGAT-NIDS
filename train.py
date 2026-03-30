"""Entry point for model training.

Usage:
    uv run python train.py model=graphsage data=static_default
    uv run python train.py model=gat data=static_default train.epochs=100
    uv run python train.py model=tgat data=temporal_default   # Phase 3
    uv run python train.py model=tgn  data=temporal_default   # Phase 3
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

log = logging.getLogger(__name__)


def _build_static_loaders(cfg: DictConfig) -> tuple:
    """Return (train_loader, val_loader, test_loader) for static models."""
    from src.data.static_dataset import StaticNIDSDataset

    processed_dir = Path(cfg.paths.data_processed) / "static"
    train_ds = StaticNIDSDataset(processed_dir, split="train")
    val_ds = StaticNIDSDataset(processed_dir, split="val")
    test_ds = StaticNIDSDataset(processed_dir, split="test")

    batch = cfg.train.get("batch_size", 1)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False)

    return train_loader, val_loader, test_loader, train_ds.n_classes, train_ds.n_edge_features


def _compute_class_weights(loader: DataLoader, n_classes: int, device: torch.device) -> torch.Tensor:
    """Aggregate all y_multi labels from the loader to compute class weights."""
    from src.eval.metrics import compute_class_weights

    all_labels = []
    for data in loader:
        all_labels.append(data.y_multi)
    all_labels_t = torch.cat(all_labels, dim=0)
    return compute_class_weights(all_labels_t, n_classes, device=device)


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch; return mean loss."""
    model.train()
    total_loss = 0.0
    total_edges = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, data.y_multi)
        loss.backward()
        optimizer.step()

        n_edges = data.y_multi.numel()
        total_loss += loss.item() * n_edges
        total_edges += n_edges

    return total_loss / max(total_edges, 1)


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on a loader; return dict of metrics."""
    from src.eval.metrics import compute_metrics

    model.eval()
    total_loss = 0.0
    total_edges = 0
    all_true, all_pred, all_proba = [], [], []

    for data in loader:
        data = data.to(device)
        logits = model(data)
        loss = criterion(logits, data.y_multi)

        proba = torch.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)

        n_edges = data.y_multi.numel()
        total_loss += loss.item() * n_edges
        total_edges += n_edges

        all_true.append(data.y_multi.cpu())
        all_pred.append(pred.cpu())
        all_proba.append(proba.cpu())

    y_true = torch.cat(all_true)
    y_pred = torch.cat(all_pred)
    y_proba = torch.cat(all_proba)

    metrics = compute_metrics(y_true, y_pred, y_proba)
    metrics["loss"] = total_loss / max(total_edges, 1)
    return metrics


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    from hydra.utils import instantiate

    from src.utils.checkpoint import load_checkpoint, save_checkpoint
    from src.utils.seed import set_global_seed

    set_global_seed(cfg.seed)
    log.info("seed=%d | model=%s", cfg.seed, cfg.model._target_)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    # ── Data ─────────────────────────────────────────────────────────────────
    dataset_type = cfg.data.get("graph_type", "static")
    if dataset_type == "static":
        train_loader, val_loader, test_loader, n_classes, n_edge_feat = (
            _build_static_loaders(cfg)
        )
        # Infer node feature dim from first batch
        sample = next(iter(train_loader))
        n_node_feat = sample.x.shape[1]
    else:
        raise NotImplementedError(
            "Temporal data pipeline (TGAT/TGN) will be implemented in Phase 3."
        )

    log.info("n_classes=%d  n_edge_feat=%d  n_node_feat=%d", n_classes, n_edge_feat, n_node_feat)

    # ── Model ────────────────────────────────────────────────────────────────
    model: torch.nn.Module = instantiate(
        cfg.model,
        in_node_channels=n_node_feat,
        in_edge_channels=n_edge_feat,
        num_classes=n_classes,
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    # Weighted cross-entropy (fitted on train split)
    class_weights = _compute_class_weights(train_loader, n_classes, device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ── Checkpoint resume ────────────────────────────────────────────────────
    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0

    resume_path = ckpt_dir / "latest.pt"
    if resume_path.exists():
        start_epoch = load_checkpoint(model, optimizer, str(resume_path), map_location=device)
        log.info("Resumed from epoch %d", start_epoch)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_f1 = 0.0
    epochs = cfg.train.epochs
    save_every = cfg.train.save_every

    for epoch in range(start_epoch, epochs):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = _evaluate(model, val_loader, criterion, device)

        log.info(
            "epoch %d/%d | train_loss=%.4f | val_f1=%.4f | val_loss=%.4f",
            epoch + 1,
            epochs,
            train_loss,
            val_metrics["f1"],
            val_metrics["loss"],
        )

        # Save latest checkpoint
        save_checkpoint(model, optimizer, epoch + 1, str(resume_path))

        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, optimizer, epoch + 1,
                str(ckpt_dir / f"epoch{epoch + 1:04d}.pt"),
            )

        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            save_checkpoint(
                model, optimizer, epoch + 1,
                str(ckpt_dir / "best.pt"),
                extra={"val_metrics": val_metrics},
            )
            # Also save a complete model object for the web inference service.
            # This file is loaded by app/services/inference.py at startup.
            inference_path = ckpt_dir.parent / f"{Path(ckpt_dir).name}_best.pt"
            torch.save(model.cpu(), inference_path)
            model.to(device)
            log.info("New best val_f1=%.4f saved → inference: %s", best_val_f1, inference_path)

    # ── Final test evaluation ────────────────────────────────────────────────
    log.info("Loading best checkpoint for final test evaluation …")
    load_checkpoint(model, None, str(ckpt_dir / "best.pt"), map_location=device)
    test_metrics = _evaluate(model, test_loader, criterion, device)

    log.info(
        "TEST | f1=%.4f | precision=%.4f | recall=%.4f | roc_auc=%.4f",
        test_metrics.get("f1", 0.0),
        test_metrics.get("precision", 0.0),
        test_metrics.get("recall", 0.0),
        test_metrics.get("roc_auc", 0.0),
    )


if __name__ == "__main__":
    main()
