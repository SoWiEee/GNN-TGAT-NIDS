"""
src/utils/seed.py
Global random seed management for full experiment reproducibility.

Call ``set_global_seed(cfg.seed)`` at the very start of every entry-point
script (train.py, attack.py, eval/comparison.py) before any model or data
initialisation.
"""
from __future__ import annotations

import logging
import random

import numpy as np
import torch

log = logging.getLogger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """Fix all random sources to make experiments reproducible.

    Covers: Python ``random``, NumPy, PyTorch (CPU + all CUDA devices),
    and cuDNN determinism flags.

    Args:
        seed: Integer seed value. Default matches ``configs/base.yaml``.

    Note:
        Setting ``cudnn.deterministic = True`` may reduce GPU throughput for
        some operations. For pure-inference benchmarks (latency measurements),
        consider disabling this flag.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    log.debug("Global seed set to %d", seed)
