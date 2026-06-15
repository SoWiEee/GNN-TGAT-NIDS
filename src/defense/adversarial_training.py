"""Adversarial training defense for NIDS GNN models.

Augments each training batch with C-PGD adversarial examples to improve
model robustness. The adversarially-trained model is saved as a separate
checkpoint ({name}_adv_best.pt) for comparison with the clean-trained model.

Supports both static models (edge_attr perturbation) and temporal models
(message-space perturbation).

Usage:
    uv run python train.py model=graphsage train.adversarial_training=true
    uv run python train.py model=tgat data=temporal_default train.adversarial_training=true

Or programmatically:
    from src.defense.adversarial_training import adversarial_train_epoch
    loss = adversarial_train_epoch(model, loader, optimizer, criterion, device, adv_cfg)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class AdvTrainingConfig:
    """Configuration for adversarial training.

    Attributes
    ----------
    epsilon:
        C-PGD perturbation budget (normalised space).
    steps:
        Number of PGD steps for generating adversarial examples.
    alpha:
        PGD step size. None uses the CPGDAttack default.
    ratio:
        Fraction of each batch to replace with adversarial examples.
        0.0 = pure clean training, 1.0 = fully adversarial.
    scaler_path:
        Path to scaler for constraint projection during PGD.
    """

    epsilon: float = 0.1
    steps: int = 10
    alpha: float | None = None
    ratio: float = 0.3
    scaler_path: str | None = None


def _fast_pgd_batch(
    model: torch.nn.Module,
    data: Data,
    epsilon: float,
    steps: int,
    alpha: float | None = None,
) -> Data:
    """Fast batch-level PGD: perturb ALL edge features simultaneously.

    Unlike per-edge C-PGD (which runs steps × n_attack_edges forward passes),
    this uses only ``steps`` forward passes total, making it practical for
    adversarial training where speed matters more than per-edge precision.
    """
    if alpha is None:
        alpha = epsilon / max(steps, 1) * 2.5

    model.eval()
    x_orig = data.edge_attr.detach().clone()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
    x_adv = x_adv.clamp(x_orig - epsilon, x_orig + epsilon)

    target = torch.zeros(data.y_multi.shape[0], dtype=torch.long, device=x_orig.device)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        mod_data = data.clone()
        mod_data.edge_attr = x_adv

        logits = model(mod_data)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()

        grad = x_adv.grad
        if grad is None:
            break

        norm = grad.norm(2, dim=-1, keepdim=True).clamp(min=1e-8)
        x_adv = (x_adv + alpha * grad / norm).detach()
        x_adv = x_adv.clamp(x_orig - epsilon, x_orig + epsilon)

    adv_data = data.clone()
    adv_data.edge_attr = x_adv.detach()
    model.train()
    return adv_data


def adversarial_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    adv_cfg: AdvTrainingConfig,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Run one adversarial training epoch.

    For each batch, generates adversarial examples via fast batch-level PGD
    and mixes them with clean examples according to ``adv_cfg.ratio``.

    Parameters
    ----------
    model:
        GNN model to train.
    loader:
        Training data loader.
    optimizer:
        Optimizer instance.
    criterion:
        Loss function (FocalLoss or CrossEntropyLoss).
    device:
        Target device (cpu or cuda).
    adv_cfg:
        Adversarial training configuration.
    scaler:
        Optional AMP GradScaler for mixed precision.

    Returns
    -------
    float
        Mean loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    total_edges = 0
    use_amp = scaler is not None

    for data in loader:
        data = data.to(device)
        n_edges = data.y_multi.numel()

        if adv_cfg.ratio > 0:
            adv_data = _fast_pgd_batch(
                model, data, adv_cfg.epsilon, adv_cfg.steps, adv_cfg.alpha,
            )
            n_adv = int(n_edges * adv_cfg.ratio)
            if n_adv > 0:
                perm = torch.randperm(n_edges, device=device)[:n_adv]
                mixed_attr = data.edge_attr.clone()
                mixed_attr[perm] = adv_data.edge_attr[perm]
                data = data.clone()
                data.edge_attr = mixed_attr

        optimizer.zero_grad()

        with torch.amp.autocast(
            device_type=device.type, enabled=use_amp, dtype=torch.bfloat16
        ):
            logits = model(data)
            loss = criterion(logits, data.y_multi)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * n_edges
        total_edges += n_edges

    return total_loss / max(total_edges, 1)


def _fast_pgd_temporal_batch(
    model: torch.nn.Module,
    batch,
    epsilon: float,
    steps: int,
    alpha: float | None = None,
    clip_min: float = -3.0,
    clip_max: float = 3.0,
):
    """Fast batch-level PGD for temporal models: perturb message features."""
    if alpha is None:
        alpha = epsilon / max(steps, 1) * 2.5

    model.eval()
    msg_orig = batch.msg.detach().clone()
    msg_adv = msg_orig + torch.empty_like(msg_orig).uniform_(-epsilon, epsilon)
    msg_adv = msg_adv.clamp(msg_orig - epsilon, msg_orig + epsilon).clamp(clip_min, clip_max)

    target = torch.zeros(batch.y.shape[0], dtype=torch.long, device=msg_orig.device)

    for _ in range(steps):
        msg_adv = msg_adv.detach().requires_grad_(True)
        adv_batch = batch.clone()
        adv_batch.msg = msg_adv

        logits = model(adv_batch)
        loss = F.cross_entropy(logits, target)
        grad = torch.autograd.grad(loss, msg_adv, retain_graph=True, allow_unused=True)[0]
        if grad is None:
            break

        norm = grad.flatten(1).norm(2, dim=1).clamp_min(1e-8).unsqueeze(1)
        msg_adv = (msg_adv + alpha * grad / norm).detach()
        msg_adv = msg_adv.clamp(msg_orig - epsilon, msg_orig + epsilon).clamp(clip_min, clip_max)

    adv_batch = batch.clone()
    adv_batch.msg = msg_adv.detach()
    model.train()
    return adv_batch


def adversarial_train_epoch_temporal(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    adv_cfg: AdvTrainingConfig,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Run one adversarial training epoch for temporal models."""
    model.train()
    model.reset_memory()
    total_loss = 0.0
    total_edges = 0
    use_amp = scaler is not None

    for batch in loader:
        batch = batch.to(device)
        n_edges = batch.y.numel()

        if adv_cfg.ratio > 0:
            adv_batch = _fast_pgd_temporal_batch(
                model, batch, adv_cfg.epsilon, adv_cfg.steps,
            )
            n_adv = int(n_edges * adv_cfg.ratio)
            if n_adv > 0:
                perm = torch.randperm(n_edges, device=device)[:n_adv]
                mixed_msg = batch.msg.clone()
                mixed_msg[perm] = adv_batch.msg[perm]
                batch = batch.clone()
                batch.msg = mixed_msg

        optimizer.zero_grad()
        with torch.amp.autocast(
            device_type=device.type, enabled=use_amp, dtype=torch.bfloat16
        ):
            logits = model(batch)
            loss = criterion(logits, batch.y)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        model.update_state(batch.src, batch.dst, batch.t, batch.msg)

        total_loss += loss.item() * n_edges
        total_edges += n_edges

    return total_loss / max(total_edges, 1)
