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
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader, TemporalDataLoader

log = logging.getLogger(__name__)


def _build_static_loaders(cfg: DictConfig) -> tuple:
    """Return (train_loader, val_loader, test_loader) for static models."""
    from src.data.static_dataset import StaticNIDSDataset

    processed_dir = Path(cfg.paths.data_processed) / "static"
    train_ds = StaticNIDSDataset(processed_dir, split="train")
    val_ds = StaticNIDSDataset(processed_dir, split="val")
    test_ds = StaticNIDSDataset(processed_dir, split="test")

    batch = cfg.train.get("batch_size", 32)
    num_workers = cfg.train.get("num_workers", 0)
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return train_loader, val_loader, test_loader, train_ds.n_classes, train_ds.n_edge_features


def _compute_class_weights(
    loader: DataLoader, n_classes: int, device: torch.device, strategy: str = "inverse",
) -> torch.Tensor:
    """Aggregate all y_multi labels from the loader to compute class weights."""
    from src.eval.metrics import compute_class_weights

    all_labels = []
    for data in loader:
        all_labels.append(data.y_multi)
    all_labels_t = torch.cat(all_labels, dim=0)
    return compute_class_weights(all_labels_t, n_classes, device=device, strategy=strategy)


def _oversample_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: torch.nn.Module,
    oversample_factor: int,
) -> torch.Tensor:
    """Compute loss with edge-level oversampling for rare classes.

    Duplicates minority-class edges in the loss so the model sees more
    gradient signal from rare attack types each batch.
    """
    if oversample_factor <= 1:
        return criterion(logits, labels)

    counts = torch.bincount(labels, minlength=logits.shape[1]).float()
    median_count = counts[counts > 0].median()

    extra_logits, extra_labels = [], []
    for cls in range(logits.shape[1]):
        mask = labels == cls
        n = mask.sum().item()
        if 0 < n < median_count:
            repeats = min(int(median_count / n), oversample_factor)
            if repeats > 1:
                extra_logits.append(logits[mask].repeat(repeats - 1, 1))
                extra_labels.append(labels[mask].repeat(repeats - 1))

    if extra_logits:
        all_logits = torch.cat([logits] + extra_logits)
        all_labels = torch.cat([labels] + extra_labels)
        return criterion(all_logits, all_labels)
    return criterion(logits, labels)


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    oversample_factor: int = 1,
) -> float:
    """Run one training epoch; return mean loss."""
    model.train()
    total_loss = 0.0
    total_edges = 0
    use_amp = scaler is not None

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=use_amp,
                                dtype=torch.bfloat16):
            logits = model(data)
            loss = _oversample_loss(
                logits, data.y_multi, criterion, oversample_factor,
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
    macro = compute_metrics(y_true, y_pred, y_proba=None, average="macro")
    metrics["macro_f1"] = macro["f1"]
    metrics["loss"] = total_loss / max(total_edges, 1)
    return metrics


def _build_temporal_loaders(cfg: DictConfig) -> tuple:
    """Return (train_loader, val_loader, test_loader, num_nodes, n_edge_feat, n_classes)."""
    import json

    from torch_geometric.data import TemporalData

    processed_dir = Path(cfg.paths.data_processed) / "temporal"
    if not (processed_dir / "train.pt").exists():
        raise FileNotFoundError(
            f"Temporal data not found at {processed_dir}. "
            "Run: uv run python src/data/temporal_builder.py"
        )

    with open(processed_dir / "meta.json") as f:
        meta = json.load(f)

    batch = cfg.train.get("batch_size", 200)

    def _load(split: str) -> TemporalData:
        from app.services.torch_load import load_torch_artifact
        return load_torch_artifact(processed_dir / f"{split}.pt")

    train_loader = TemporalDataLoader(_load("train"), batch_size=batch)
    val_loader   = TemporalDataLoader(_load("val"),   batch_size=batch)
    test_loader  = TemporalDataLoader(_load("test"),  batch_size=batch)

    return (
        train_loader, val_loader, test_loader,
        meta["num_nodes"], meta["n_features"], meta["n_classes"],
    )


def _compute_class_weights_temporal(
    loader: TemporalDataLoader, n_classes: int, device: torch.device,
    strategy: str = "inverse",
) -> torch.Tensor:
    from src.eval.metrics import compute_class_weights
    all_labels = [batch.y for batch in loader]
    return compute_class_weights(torch.cat(all_labels), n_classes, device=device, strategy=strategy)


def _train_epoch_temporal(
    model,
    loader: TemporalDataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    amp_scaler: torch.amp.GradScaler | None,
) -> float:
    """One training epoch for TGN — updates memory after each batch."""
    model.train()
    model.reset_memory()
    total_loss = 0.0
    total_edges = 0
    use_amp = amp_scaler is not None

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=use_amp,
                                dtype=torch.bfloat16):
            logits = model(batch)
            loss = criterion(logits, batch.y)

        if use_amp:
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Update memory AFTER backward (also detaches to prevent BPTT through history)
        model.update_state(batch.src, batch.dst, batch.t, batch.msg)

        n = batch.y.numel()
        total_loss += loss.item() * n
        total_edges += n

    return total_loss / max(total_edges, 1)


@torch.no_grad()
def _evaluate_temporal(
    model,
    loader: TemporalDataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate TGN — memory is read-only (not updated)."""
    from src.eval.metrics import compute_metrics

    model.eval()
    total_loss = 0.0
    total_edges = 0
    all_true, all_pred, all_proba = [], [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = criterion(logits, batch.y)

        proba = torch.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)

        n = batch.y.numel()
        total_loss += loss.item() * n
        total_edges += n
        all_true.append(batch.y.cpu())
        all_pred.append(pred.cpu())
        all_proba.append(proba.cpu())

    y_true_cat = torch.cat(all_true)
    y_pred_cat = torch.cat(all_pred)
    y_proba_cat = torch.cat(all_proba)
    metrics = compute_metrics(y_true_cat, y_pred_cat, y_proba_cat)
    macro = compute_metrics(y_true_cat, y_pred_cat, y_proba=None, average="macro")
    metrics["macro_f1"] = macro["f1"]
    metrics["loss"] = total_loss / max(total_edges, 1)
    return metrics


def _log_per_class_f1(
    loader,
    model: torch.nn.Module,
    device: torch.device,
    is_temporal: bool,
    n_classes: int,
) -> None:
    """Log per-class precision/recall/F1 breakdown after test evaluation."""
    import json

    from src.eval.metrics import compute_per_class_metrics

    label2idx_path = Path("data/processed/static/label2idx.json")
    label_names = None
    if label2idx_path.exists():
        label2idx = json.loads(label2idx_path.read_text())
        idx2label = {v: k for k, v in label2idx.items()}
        label_names = [idx2label.get(i, str(i)) for i in range(n_classes)]

    all_true, all_pred = [], []
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data)
            labels = data.y if is_temporal else data.y_multi
            all_true.append(labels.cpu())
            all_pred.append(logits.argmax(-1).cpu())

    report = compute_per_class_metrics(
        torch.cat(all_true), torch.cat(all_pred), label_names,
    )

    log.info("Per-class F1 breakdown:")
    for entry in report["per_class"]:
        log.info(
            "  %-16s  P=%.4f  R=%.4f  F1=%.4f  support=%d",
            entry["name"], entry["precision"], entry["recall"],
            entry["f1"], entry["support"],
        )


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    from hydra.utils import instantiate

    from src.utils.checkpoint import load_checkpoint, save_checkpoint
    from src.utils.seed import set_global_seed

    set_global_seed(cfg.seed)
    log.info("seed=%d | model=%s", cfg.seed, cfg.model._target_)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    # AMP GradScaler — only active on CUDA when use_amp=true
    use_amp = cfg.train.get("use_amp", True) and device.type == "cuda"
    scaler: torch.amp.GradScaler | None = torch.amp.GradScaler() if use_amp else None
    log.info("AMP=%s", use_amp)

    # ── Data ─────────────────────────────────────────────────────────────────
    dataset_type = cfg.data.get("graph_type", "static")
    is_temporal = dataset_type == "temporal"

    if is_temporal:
        train_loader, val_loader, test_loader, num_nodes, n_edge_feat, n_classes = (
            _build_temporal_loaders(cfg)
        )
        log.info("n_classes=%d  n_edge_feat=%d  num_nodes=%d", n_classes, n_edge_feat, num_nodes)
    else:
        train_loader, val_loader, test_loader, n_classes, n_edge_feat = (
            _build_static_loaders(cfg)
        )
        sample = next(iter(train_loader))
        n_node_feat = sample.x.shape[1]
        log.info(
            "n_classes=%d  n_edge_feat=%d  n_node_feat=%d", n_classes, n_edge_feat, n_node_feat
        )

    # ── Model ────────────────────────────────────────────────────────────────
    # Strip non-constructor keys that live in the model YAML but aren't
    # __init__ parameters (loss config, memory_reset_policy, nested train).
    _NON_MODEL_KEYS = {"loss", "memory_reset_policy", "train"}
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    for k in _NON_MODEL_KEYS:
        model_cfg.pop(k, None)
    model_cfg = OmegaConf.create(model_cfg)

    if is_temporal:
        model: torch.nn.Module = instantiate(
            model_cfg,
            num_nodes=num_nodes,
            raw_msg_dim=n_edge_feat,
            num_classes=n_classes,
        )
    else:
        model: torch.nn.Module = instantiate(
            model_cfg,
            in_node_channels=n_node_feat,
            in_edge_channels=n_edge_feat,
            num_classes=n_classes,
        )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    # LR scheduler
    scheduler_type = cfg.train.get("scheduler", "none")
    scheduler = None
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.lr * 0.01,
        )
        log.info("scheduler=CosineAnnealingLR(T_max=%d)", cfg.train.epochs)
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5,
        )
        log.info("scheduler=ReduceLROnPlateau(factor=0.5, patience=5)")

    # Loss function — class weights always applied; focal loss reduces recall gap
    cw_strategy = cfg.train.get("class_weight_strategy", "inverse")
    class_weights = (
        _compute_class_weights_temporal(train_loader, n_classes, device, strategy=cw_strategy)
        if is_temporal
        else _compute_class_weights(train_loader, n_classes, device, strategy=cw_strategy)
    )
    log.info("class_weight_strategy=%s  weights=%s", cw_strategy, class_weights.cpu().tolist())
    loss_type = cfg.train.get("loss", "focal")
    if loss_type == "focal":
        from src.eval.losses import FocalLoss
        gamma = float(cfg.train.get("focal_gamma", 2.0))
        criterion = FocalLoss(weight=class_weights, gamma=gamma)
        log.info("loss=FocalLoss(gamma=%.1f)", gamma)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        log.info("loss=CrossEntropyLoss")

    # ── Training config ─────────────────────────────────────────────────────
    epochs = cfg.train.epochs
    save_every = cfg.train.save_every
    val_every = cfg.train.get("val_every", 1)
    patience = int(cfg.train.get("patience", 0))
    val_metric_key = cfg.train.get("val_metric", "f1")
    oversample_factor = int(cfg.train.get("oversample_factor", 1))
    log.info("val_metric=%s (checkpoint selection)", val_metric_key)
    if oversample_factor > 1:
        log.info("edge-level oversampling factor=%d", oversample_factor)

    # ── Checkpoint resume ────────────────────────────────────────────────────
    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0

    resume_path = ckpt_dir / "latest.pt"
    best_val_score = 0.0
    if resume_path.exists():
        start_epoch = load_checkpoint(model, optimizer, str(resume_path), map_location=device)
        best_ckpt_path = ckpt_dir / "best.pt"
        if best_ckpt_path.exists():
            best_payload = torch.load(
                best_ckpt_path, map_location="cpu", weights_only=True,
            )
            stored = best_payload.get("extra", {}).get("val_metrics", {})
            best_val_score = stored.get(val_metric_key, stored.get("f1", 0.0))
            if best_val_score > 0:
                log.info(
                    "Restored best val_%s=%.4f from best checkpoint",
                    val_metric_key, best_val_score,
                )
        log.info("Resumed from epoch %d", start_epoch)

    # ── Training loop ────────────────────────────────────────────────────────
    epochs_no_improve = 0

    # ── Adversarial training setup ──────────────────────────────────────────
    use_adv = cfg.train.get("adversarial_training", False)
    adv_cfg = None
    if use_adv:
        from src.defense.adversarial_training import AdvTrainingConfig, adversarial_train_epoch

        if is_temporal:
            from src.defense.adversarial_training import adversarial_train_epoch_temporal

        adv_cfg = AdvTrainingConfig(
            epsilon=float(cfg.train.get("adv_epsilon", 0.1)),
            steps=int(cfg.train.get("adv_steps", 10)),
            ratio=float(cfg.train.get("adv_ratio", 0.3)),
            scaler_path=str(Path(cfg.paths.data_processed) / "static" / "scaler.json")
            if (Path(cfg.paths.data_processed) / "static" / "scaler.json").exists()
            else None,
        )
        log.info(
            "Adversarial training ENABLED (%s): ε=%.3f, steps=%d, ratio=%.2f",
            "temporal" if is_temporal else "static",
            adv_cfg.epsilon, adv_cfg.steps, adv_cfg.ratio,
        )

    for epoch in range(start_epoch, epochs):
        if is_temporal and use_adv:
            train_loss = adversarial_train_epoch_temporal(
                model, train_loader, optimizer, criterion, device, adv_cfg, scaler
            )
        elif is_temporal:
            train_loss = _train_epoch_temporal(
                model, train_loader, optimizer, criterion, device, scaler
            )
        elif use_adv:
            train_loss = adversarial_train_epoch(
                model, train_loader, optimizer, criterion, device, adv_cfg, scaler
            )
        else:
            train_loss = _train_epoch(
                model, train_loader, optimizer, criterion, device, scaler,
                oversample_factor=oversample_factor,
            )

        # Step LR scheduler
        if scheduler is not None and not isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step()

        # Skip val evaluation on non-val epochs (log train loss only)
        if (epoch + 1) % val_every != 0 and epoch + 1 < epochs:
            log.info("epoch %d/%d | train_loss=%.4f", epoch + 1, epochs, train_loss)
            save_checkpoint(model, optimizer, epoch + 1, str(resume_path))
            continue

        val_metrics = (
            _evaluate_temporal(model, val_loader, criterion, device)
            if is_temporal
            else _evaluate(model, val_loader, criterion, device)
        )
        current_lr = optimizer.param_groups[0]["lr"]
        log.info(
            "epoch %d/%d | train_loss=%.4f | val_f1=%.4f | val_macro_f1=%.4f"
            " | val_loss=%.4f | lr=%.6f",
            epoch + 1, epochs, train_loss,
            val_metrics["f1"], val_metrics.get("macro_f1", 0.0),
            val_metrics["loss"], current_lr,
        )

        # Step ReduceLROnPlateau after validation
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics.get(val_metric_key, val_metrics["f1"]))

        save_checkpoint(model, optimizer, epoch + 1, str(resume_path))

        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, optimizer, epoch + 1,
                str(ckpt_dir / f"epoch{epoch + 1:04d}.pt"),
            )

        current_score = val_metrics.get(val_metric_key, val_metrics["f1"])
        if current_score > best_val_score:
            best_val_score = current_score
            epochs_no_improve = 0
            save_checkpoint(
                model, optimizer, epoch + 1,
                str(ckpt_dir / "best.pt"),
                extra={"val_metrics": val_metrics},
            )
            model_key = cfg.model._target_.rsplit(".", 1)[-1].replace("Model", "").lower()
            suffix = "_adv_best.pt" if use_adv else "_best.pt"
            inference_path = ckpt_dir.parent / f"{model_key}{suffix}"
            torch.save(model.cpu(), inference_path)
            model.to(device)
            log.info(
                "New best val_%s=%.4f (weighted_f1=%.4f, macro_f1=%.4f) → %s",
                val_metric_key, best_val_score,
                val_metrics["f1"], val_metrics.get("macro_f1", 0.0),
                inference_path,
            )
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                log.info("Early stopping: no improvement for %d epochs", patience)
                break

    # ── Final test evaluation ────────────────────────────────────────────────
    log.info("Loading best checkpoint for final test evaluation …")
    load_checkpoint(model, None, str(ckpt_dir / "best.pt"), map_location=device)
    if is_temporal:
        model.reset_memory()
        # Replay train split to warm up memory before test evaluation
        log.info("Replaying train split to warm up TGN memory …")
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(device)
                model(batch)
                model.update_state(batch.src, batch.dst, batch.t, batch.msg)
        test_metrics = _evaluate_temporal(model, test_loader, criterion, device)
    else:
        test_metrics = _evaluate(model, test_loader, criterion, device)

    log.info(
        "TEST | f1=%.4f | macro_f1=%.4f | precision=%.4f | recall=%.4f | roc_auc=%.4f",
        test_metrics.get("f1", 0.0),
        test_metrics.get("macro_f1", 0.0),
        test_metrics.get("precision", 0.0),
        test_metrics.get("recall", 0.0),
        test_metrics.get("roc_auc", 0.0),
    )

    _log_per_class_f1(test_loader, model, device, is_temporal, n_classes)


if __name__ == "__main__":
    main()
