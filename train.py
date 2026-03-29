"""
train.py
Entry point for model training.

Usage:
    uv run python train.py model=graphsage data=static_default
    uv run python train.py model=tgat data=temporal_default seed=0
    uv run python train.py model=gat train.lr=0.0005 train.epochs=100
"""
from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    from src.utils.seed import set_global_seed

    set_global_seed(cfg.seed)
    log.info("seed=%d | model=%s | data=%s", cfg.seed, cfg.model._target_, cfg.data.dataset)

    # ── TODO (Phase 1): Build data pipeline ───────────────────────────────
    # from src.data.static_dataset import StaticNIDSDataset
    # from src.data.temporal_builder import build_temporal_data
    # train_loader, val_loader, test_loader = ...

    # ── TODO (Phase 2): Instantiate model ─────────────────────────────────
    # import hydra.utils
    # model = hydra.utils.instantiate(cfg.model)
    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    # ── TODO (Phase 2): Training loop with checkpointing ──────────────────
    # from src.utils.checkpoint import save_checkpoint, load_checkpoint
    # for epoch in range(cfg.train.epochs):
    #     train_one_epoch(model, optimizer, train_loader)
    #     if epoch % cfg.train.save_every == 0:
    #         save_checkpoint(model, optimizer, epoch,
    #                         path=f"{cfg.train.checkpoint_dir}/epoch{epoch}.ckpt")

    raise NotImplementedError(
        "Model training not yet implemented. "
        "Complete Phase 1 (data pipeline) and Phase 2 (models) first."
    )


if __name__ == "__main__":
    main()
