"""Train a two-stage hierarchical classifier and evaluate combined 10-class performance.

Stage 1: Binary (Benign vs Attack)
Stage 2: 9-class attack subtype (loss masked on benign edges, full graph for message passing)

Usage:
    uv run python scripts/train_hierarchical.py
    uv run python scripts/train_hierarchical.py --base-model egraphsage
    uv run python scripts/train_hierarchical.py --epochs 200 --patience 20
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed" / "static"
CHECKPOINT_DIR = ROOT / "checkpoints"


def _build_loaders(batch_size: int = 32) -> tuple:
    from src.data.static_dataset import StaticNIDSDataset

    train_ds = StaticNIDSDataset(PROCESSED_DIR, split="train")
    val_ds = StaticNIDSDataset(PROCESSED_DIR, split="val")
    test_ds = StaticNIDSDataset(PROCESSED_DIR, split="test")

    pin = torch.cuda.is_available()
    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=pin)

    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    return train_loader, val_loader, test_loader, train_ds.n_classes, train_ds.n_edge_features


def _create_model(
    base_model: str,
    in_node_channels: int,
    in_edge_channels: int,
    num_classes: int,
) -> torch.nn.Module:
    if base_model == "egraphsage":
        from src.models.egraphsage import EGraphSAGEModel
        return EGraphSAGEModel(
            in_node_channels=in_node_channels,
            in_edge_channels=in_edge_channels,
            hidden_dim=256,
            num_classes=num_classes,
            num_layers=3,
            dropout=0.3,
        )
    else:
        from src.models.graphsage import GraphSAGEModel
        return GraphSAGEModel(
            in_node_channels=in_node_channels,
            in_edge_channels=in_edge_channels,
            hidden_dim=256,
            num_classes=num_classes,
            num_layers=3,
            dropout=0.3,
        )


def _compute_weights(loader: DataLoader, n_classes: int, device: torch.device,
                     label_fn=None) -> torch.Tensor:
    from src.eval.metrics import compute_class_weights
    all_labels = []
    for data in loader:
        labels = data.y_multi if label_fn is None else label_fn(data.y_multi)
        all_labels.append(labels)
    return compute_class_weights(
        torch.cat(all_labels), n_classes, device=device, strategy="sqrt_inverse",
    )


def _train_stage1_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
) -> float:
    model.train()
    total_loss, total_edges = 0.0, 0
    use_amp = scaler is not None

    for data in loader:
        data = data.to(device)
        binary_labels = (data.y_multi > 0).long()
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            logits = model(data)
            loss = criterion(logits, binary_labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        n = binary_labels.numel()
        total_loss += loss.item() * n
        total_edges += n

    return total_loss / max(total_edges, 1)


def _train_stage2_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
) -> float:
    """Train stage 2 on attack edges only, but keep full graph for message passing."""
    model.train()
    total_loss, total_edges = 0.0, 0
    use_amp = scaler is not None

    for data in loader:
        data = data.to(device)
        attack_mask = data.y_multi > 0
        if not attack_mask.any():
            continue

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            logits = model(data)
            attack_logits = logits[attack_mask]
            attack_labels = data.y_multi[attack_mask] - 1
            loss = criterion(attack_logits, attack_labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        n = attack_labels.numel()
        total_loss += loss.item() * n
        total_edges += n

    return total_loss / max(total_edges, 1)


@torch.no_grad()
def _eval_stage(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    stage: int,
) -> dict:
    """Evaluate a single stage. Returns metrics dict."""
    from src.eval.metrics import compute_metrics

    model.eval()
    all_true, all_pred, all_proba = [], [], []

    for data in loader:
        data = data.to(device)
        logits = model(data)
        proba = torch.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)

        if stage == 1:
            labels = (data.y_multi > 0).long()
            all_true.append(labels.cpu())
            all_pred.append(pred.cpu())
            all_proba.append(proba.cpu())
        else:
            mask = data.y_multi > 0
            if mask.any():
                all_true.append((data.y_multi[mask] - 1).cpu())
                all_pred.append(pred[mask].cpu())
                all_proba.append(proba[mask].cpu())

    y_true = torch.cat(all_true)
    y_pred = torch.cat(all_pred)
    y_proba = torch.cat(all_proba)
    metrics = compute_metrics(y_true, y_pred, y_proba)
    macro = compute_metrics(y_true, y_pred, y_proba=None, average="macro")
    metrics["macro_f1"] = macro["f1"]
    return metrics


@torch.no_grad()
def _eval_combined(
    stage1: torch.nn.Module,
    stage2: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate the combined hierarchical model on original 10-class labels."""
    from src.eval.metrics import compute_metrics, compute_per_class_metrics

    stage1.eval()
    stage2.eval()
    all_true, all_pred, all_proba = [], [], []

    for data in loader:
        data = data.to(device)

        binary_logits = stage1(data)
        binary_proba = torch.softmax(binary_logits, dim=-1)

        attack_logits = stage2(data)
        attack_proba = torch.softmax(attack_logits, dim=-1)

        p_benign = binary_proba[:, 0:1]
        p_attack = binary_proba[:, 1:2]
        combined_proba = torch.cat([p_benign, p_attack * attack_proba], dim=-1)

        pred = combined_proba.argmax(dim=-1)
        all_true.append(data.y_multi.cpu())
        all_pred.append(pred.cpu())
        all_proba.append(combined_proba.cpu())

    y_true = torch.cat(all_true)
    y_pred = torch.cat(all_pred)
    y_proba = torch.cat(all_proba)

    metrics = compute_metrics(y_true, y_pred, y_proba)
    macro = compute_metrics(y_true, y_pred, y_proba=None, average="macro")
    metrics["macro_f1"] = macro["f1"]

    label_names = [
        "Benign", "Analysis", "Backdoor", "DoS", "Exploits",
        "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
    ]
    per_class = compute_per_class_metrics(y_true, y_pred, label_names)
    return metrics, per_class


def _train_stage(
    stage_name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    train_fn,
    stage_num: int,
    epochs: int = 200,
    patience: int = 20,
    lr: float = 0.001,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None

    best_score = 0.0
    epochs_no_improve = 0
    best_state = None

    for epoch in range(epochs):
        train_loss = train_fn(model, train_loader, optimizer, criterion, device, scaler)
        scheduler.step()

        val_metrics = _eval_stage(model, val_loader, device, stage_num)
        score = val_metrics["macro_f1"]
        current_lr = optimizer.param_groups[0]["lr"]

        log.info(
            "%s epoch %d/%d | loss=%.4f | val_f1=%.4f | val_macro_f1=%.4f | lr=%.6f",
            stage_name, epoch + 1, epochs, train_loss,
            val_metrics["f1"], score, current_lr,
        )

        if score > best_score:
            best_score = score
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            log.info("  → new best val_macro_f1=%.4f", best_score)
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                log.info("%s early stopping after %d epochs", stage_name, patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hierarchical NIDS classifier")
    parser.add_argument("--base-model", default="graphsage", choices=["graphsage", "egraphsage"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--focal-gamma", type=float, default=3.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s  base_model=%s", device, args.base_model)

    train_loader, val_loader, test_loader, n_classes, n_edge_feat = _build_loaders(args.batch_size)
    sample = next(iter(train_loader))
    n_node_feat = sample.x.shape[1]
    log.info(
        "n_node_feat=%d  n_edge_feat=%d  original_classes=%d",
        n_node_feat, n_edge_feat, n_classes,
    )

    # ── Stage 1: Binary (Benign vs Attack) ──────────────────────────────────
    log.info("=" * 60)
    log.info("STAGE 1: Binary classifier (Benign vs Attack)")
    log.info("=" * 60)

    stage1 = _create_model(args.base_model, n_node_feat, n_edge_feat, num_classes=2).to(device)

    def _binary_label(y: torch.Tensor) -> torch.Tensor:
        return (y > 0).long()

    s1_weights = _compute_weights(train_loader, 2, device, label_fn=_binary_label)
    log.info("Stage 1 class weights: %s", s1_weights.cpu().tolist())

    from src.eval.losses import FocalLoss
    s1_criterion = FocalLoss(weight=s1_weights, gamma=args.focal_gamma)

    _train_stage(
        "Stage1", stage1, train_loader, val_loader, device, s1_criterion,
        _train_stage1_epoch, stage_num=1,
        epochs=args.epochs, patience=args.patience, lr=args.lr,
    )

    s1_test = _eval_stage(stage1, test_loader, device, stage=1)
    log.info("Stage 1 TEST | f1=%.4f | macro_f1=%.4f | precision=%.4f | recall=%.4f",
             s1_test["f1"], s1_test["macro_f1"], s1_test["precision"], s1_test["recall"])

    # ── Stage 2: Attack subtype (9 classes) ─────────────────────────────────
    log.info("=" * 60)
    log.info("STAGE 2: Attack subtype classifier (9 classes, masked loss)")
    log.info("=" * 60)

    stage2 = _create_model(args.base_model, n_node_feat, n_edge_feat, num_classes=9).to(device)

    def _attack_label(y: torch.Tensor) -> torch.Tensor:
        return y[y > 0] - 1

    s2_weights = _compute_weights(train_loader, 9, device, label_fn=_attack_label)
    log.info("Stage 2 class weights: %s", s2_weights.cpu().tolist())

    s2_criterion = FocalLoss(weight=s2_weights, gamma=args.focal_gamma)

    _train_stage(
        "Stage2", stage2, train_loader, val_loader, device, s2_criterion,
        _train_stage2_epoch, stage_num=2,
        epochs=args.epochs, patience=args.patience, lr=args.lr,
    )

    s2_test = _eval_stage(stage2, test_loader, device, stage=2)
    log.info("Stage 2 TEST | f1=%.4f | macro_f1=%.4f | precision=%.4f | recall=%.4f",
             s2_test["f1"], s2_test["macro_f1"], s2_test["precision"], s2_test["recall"])

    # ── Combined evaluation ─────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("COMBINED: Hierarchical 10-class evaluation")
    log.info("=" * 60)

    combined_metrics, per_class = _eval_combined(stage1, stage2, test_loader, device)
    log.info(
        "COMBINED TEST | weighted_f1=%.4f | macro_f1=%.4f | precision=%.4f | recall=%.4f",
        combined_metrics["f1"], combined_metrics["macro_f1"],
        combined_metrics["precision"], combined_metrics["recall"],
    )

    log.info("Per-class F1:")
    for cls_info in per_class["per_class"]:
        log.info(
            "  %-15s support=%6d  f1=%.4f  precision=%.4f  recall=%.4f",
            cls_info["name"], cls_info["support"],
            cls_info["f1"], cls_info["precision"], cls_info["recall"],
        )

    # ── Save combined model ─────────────────────────────────────────────────
    from src.models.hierarchical import HierarchicalNIDSModel

    combined_model = HierarchicalNIDSModel(stage1.cpu(), stage2.cpu())
    save_path = CHECKPOINT_DIR / f"{args.base_model}_hierarchical.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(combined_model, save_path)
    log.info("Saved combined model to %s", save_path)


if __name__ == "__main__":
    main()
