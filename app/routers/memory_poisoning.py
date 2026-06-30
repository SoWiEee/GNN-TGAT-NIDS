"""Memory poisoning experiment router for temporal models."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader

from app.schemas import MemoryPoisoningRequest

logger = logging.getLogger(__name__)
router = APIRouter(tags=["memory-poisoning"])

ENABLE_ATTACK_ENDPOINTS = os.getenv("ENABLE_ATTACK_ENDPOINTS", "true").lower() == "true"

TEMPORAL_DIR = Path("data/processed/temporal")
CHECKPOINTS_DIR = Path("checkpoints")


def _load_model(name: str):
    from app.services.torch_load import load_torch_artifact

    path = CHECKPOINTS_DIR / f"{name}_best.pt"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = load_torch_artifact(path)
    model.eval()
    return model


def _sync_run_memory_poisoning(req: MemoryPoisoningRequest) -> dict:
    from src.attack.memory_poisoning import MemoryPoisoningAttack

    train_path = TEMPORAL_DIR / "train.pt"
    test_path = TEMPORAL_DIR / "test.pt"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Temporal processed data not found")

    from app.services.torch_load import load_torch_artifact
    train_td = load_torch_artifact(train_path)
    test_td = load_torch_artifact(test_path)
    train_loader = TemporalDataLoader(train_td, batch_size=req.batch_size)
    test_loader = TemporalDataLoader(test_td, batch_size=req.batch_size)

    clean_model = _load_model(req.model)
    poison_model = _load_model(req.model)

    for model in (clean_model, poison_model):
        if hasattr(model, "reset_memory"):
            model.reset_memory()
        with torch.no_grad():
            for batch in train_loader:
                model(batch)
                if hasattr(model, "update_state"):
                    model.update_state(batch.src, batch.dst, batch.t, batch.msg)

    attacker = MemoryPoisoningAttack(
        n_poison=req.n_poison,
        poison_strategy=req.poison_strategy,
        memory_reset_policy="none",
    )

    rows = []
    total_attack = 0
    total_evaded = 0
    total_poison = 0

    for idx, batch in enumerate(test_loader):
        if idx >= req.max_batches:
            break

        with torch.no_grad():
            orig_preds = clean_model(batch).argmax(dim=-1)

        adv_data = attacker.generate(poison_model, batch)
        n_orig = len(batch.src)
        n_poison = len(adv_data.src) - n_orig
        if n_poison > 0 and hasattr(poison_model, "update_state"):
            poison_model.update_state(
                adv_data.src[:n_poison],
                adv_data.dst[:n_poison],
                adv_data.t[:n_poison],
                adv_data.msg[:n_poison],
            )

        orig_batch = TemporalData(
            src=adv_data.src[n_poison:],
            dst=adv_data.dst[n_poison:],
            t=adv_data.t[n_poison:],
            msg=adv_data.msg[n_poison:],
        )
        with torch.no_grad():
            adv_preds = poison_model(orig_batch).argmax(dim=-1)

        attack_mask = orig_preds > 0
        n_attack = int(attack_mask.sum())
        evaded = int(((orig_preds > 0) & (adv_preds == 0)).sum())
        asr = evaded / n_attack if n_attack else 0.0
        rows.append(
            {
                "batch": idx,
                "events": int(len(batch.src)),
                "attack_edges": n_attack,
                "evaded": evaded,
                "poison_events": int(n_poison),
                "asr": round(asr, 4),
            }
        )
        total_attack += n_attack
        total_evaded += evaded
        total_poison += int(n_poison)

        if hasattr(clean_model, "update_state"):
            clean_model.update_state(batch.src, batch.dst, batch.t, batch.msg)
        if hasattr(poison_model, "update_state"):
            poison_model.update_state(batch.src, batch.dst, batch.t, batch.msg)

    return {
        "model": req.model,
        "n_poison": req.n_poison,
        "poison_strategy": req.poison_strategy,
        "batches": len(rows),
        "total_attack_edges": total_attack,
        "total_evaded": total_evaded,
        "total_poison_events": total_poison,
        "asr": round(total_evaded / total_attack, 4) if total_attack else 0.0,
        "rows": rows,
    }


@router.post("/memory-poisoning")
async def run_memory_poisoning(req: MemoryPoisoningRequest):
    if not ENABLE_ATTACK_ENDPOINTS:
        raise HTTPException(403, detail="Attack endpoints are disabled in this environment")
    try:
        return await run_in_threadpool(_sync_run_memory_poisoning, req)
    except FileNotFoundError as exc:
        raise HTTPException(404, detail=str(exc))
    except Exception:
        logger.exception("Memory poisoning experiment failed")
        raise HTTPException(500, detail="Memory poisoning experiment failed")
