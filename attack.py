"""Entry point for adversarial example generation.

Usage:
    uv run python attack.py attack=cpgd model=graphsage
    uv run python attack.py attack=cpgd model=graphsage attack.epsilon=0.05 attack.steps=20
    uv run python attack.py attack=edge_injection model=graphsage attack.n_inject=100
    uv run python attack.py attack=gan model=graphsage

The output path is automatically parameterised by hyperparameters to prevent
overwriting (see configs/run_attack.yaml: output_dir).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="run_attack")
def main(cfg: DictConfig) -> None:
    from hydra.utils import instantiate

    from src.utils.seed import set_global_seed

    set_global_seed(cfg.seed)
    log.info(
        "seed=%d | attack=%s | model=%s | split=%s",
        cfg.seed,
        cfg.attack._target_,
        cfg.model._target_,
        cfg.attack.target_split,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    # ── Load model checkpoint ────────────────────────────────────────────
    model_name = Path(cfg.model._target_).stem.replace("Model", "").lower()
    ckpt_path = Path(cfg.paths.checkpoints) / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        alt_name = cfg.model._target_.rsplit(".", 1)[-1].replace("Model", "").lower()
        ckpt_path = Path(cfg.paths.checkpoints) / f"{alt_name}_best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {ckpt_path}. "
            f"Train the model first: uv run python train.py model={model_name}"
        )

    from app.services.torch_load import load_torch_artifact
    model = load_torch_artifact(ckpt_path, map_location=str(device))
    model.eval()
    log.info("Loaded model from %s", ckpt_path)

    # ── Load data split ──────────────────────────────────────────────────
    dataset_type = cfg.data.get("graph_type", "static")
    target_split = cfg.attack.get("target_split", "test")

    if dataset_type == "temporal":
        from torch_geometric.loader import TemporalDataLoader

        processed_dir = Path(cfg.paths.data_processed) / "temporal"
        split_path = processed_dir / f"{target_split}.pt"
        if not split_path.exists():
            raise FileNotFoundError(
                f"Temporal data not found at {split_path}. "
                "Run: uv run python src/data/temporal_builder.py"
            )
        data = load_torch_artifact(split_path)
        loader = TemporalDataLoader(data, batch_size=200)
        log.info("Loaded temporal %s split (%d events)", target_split, len(data.src))
    else:
        from torch_geometric.loader import DataLoader

        from src.data.static_dataset import StaticNIDSDataset

        processed_dir = Path(cfg.paths.data_processed) / "static"
        dataset = StaticNIDSDataset(processed_dir, split=target_split)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        log.info("Loaded static %s split (%d windows)", target_split, len(dataset))

    # ── Instantiate attack ───────────────────────────────────────────────
    attack_kwargs = {}
    scaler_path = cfg.attack.get("constraints", {}).get("scaler_path", None)
    if scaler_path is not None:
        attack_kwargs["scaler_path"] = scaler_path

    attack = instantiate(cfg.attack, **attack_kwargs)
    log.info("Attack: %s", type(attack).__name__)

    # ── Run attack ───────────────────────────────────────────────────────
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    total_attack_edges = 0
    total_evaded = 0

    for batch_idx, data_batch in enumerate(loader):
        data_batch = data_batch.to(device)

        with torch.no_grad():
            orig_preds = model(data_batch).argmax(dim=-1)

        adv_data = attack.generate(model, data_batch)

        with torch.no_grad():
            adv_preds = model(adv_data).argmax(dim=-1)

        n_orig = orig_preds.shape[0]
        adv_preds_orig = adv_preds[:n_orig]

        attack_mask = orig_preds > 0
        n_attack = int(attack_mask.sum())
        evaded = int(((orig_preds > 0) & (adv_preds_orig == 0)).sum())

        total_attack_edges += n_attack
        total_evaded += evaded

        # Save perturbed data
        save_path = output_dir / f"batch_{batch_idx:04d}.pt"
        torch.save(adv_data.cpu(), save_path)

        batch_asr = evaded / max(n_attack, 1)
        all_results.append({
            "batch": batch_idx,
            "n_attack_edges": n_attack,
            "n_evaded": evaded,
            "asr": round(batch_asr, 4),
        })

        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            log.info(
                "batch %d | attack_edges=%d  evaded=%d  asr=%.4f",
                batch_idx, n_attack, evaded, batch_asr,
            )

    # ── Summary ──────────────────────────────────────────────────────────
    overall_asr = total_evaded / max(total_attack_edges, 1)

    summary = {
        "attack": cfg.attack._target_,
        "model": cfg.model._target_,
        "split": target_split,
        "seed": cfg.seed,
        "total_attack_edges": total_attack_edges,
        "total_evaded": total_evaded,
        "overall_asr": round(overall_asr, 4),
        "batches": all_results,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    log.info(
        "DONE | attack_edges=%d  evaded=%d  ASR=%.4f  → %s",
        total_attack_edges, total_evaded, overall_asr, output_dir,
    )


if __name__ == "__main__":
    main()
