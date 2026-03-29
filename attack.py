"""
attack.py
Entry point for adversarial example generation.

Usage:
    uv run python attack.py attack=cpgd model=graphsage
    uv run python attack.py attack=cpgd model=graphsage attack.epsilon=0.05 attack.steps=20
    uv run python attack.py attack=edge_injection model=tgat attack.n_inject=100
    uv run python attack.py attack=gan model=graphsage

The output path is automatically parameterised by hyperparameters to prevent
overwriting (see configs/run_attack.yaml: output_dir).
"""
from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="run_attack")
def main(cfg: DictConfig) -> None:
    from src.utils.seed import set_global_seed

    set_global_seed(cfg.seed)
    log.info(
        "seed=%d | attack=%s | model=%s | split=%s",
        cfg.seed,
        cfg.attack._target_,
        cfg.model._target_,
        cfg.attack.target_split,
    )

    # ── TODO (Phase 3): Load model checkpoint ─────────────────────────────
    # import hydra.utils, torch
    # model = hydra.utils.instantiate(cfg.model)
    # from src.utils.checkpoint import load_checkpoint
    # load_checkpoint(model, optimizer=None,
    #                 path=f"{cfg.paths.checkpoints}/{cfg.model._target_}/best.ckpt")
    # model.eval()

    # ── TODO (Phase 3): Load data split ───────────────────────────────────
    # data = load_split(cfg.data, split=cfg.attack.target_split)

    # ── TODO (Phase 3): Instantiate attack and run ─────────────────────────
    # attack = hydra.utils.instantiate(cfg.attack)
    # adv_data = attack.generate(model, data)
    # torch.save(adv_data, cfg.output_dir / f"{model_name}_test.pt")
    # log.info("CSR=%.4f  ASR=%.4f", attack.last_csr, attack.last_asr)

    raise NotImplementedError(
        "Adversarial generation not yet implemented. "
        "Complete Phase 2 (models) and Phase 3 (CAAG) first."
    )


if __name__ == "__main__":
    main()
