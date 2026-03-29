"""
tests/test_seed.py
Tests for src/utils/seed.py.

Verifies that set_global_seed() produces reproducible outputs across all
random sources (Python random, NumPy, PyTorch CPU).
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.utils.seed import set_global_seed


class TestSetGlobalSeed:
    def test_python_random_reproducible(self):
        set_global_seed(0)
        a = [random.random() for _ in range(10)]
        set_global_seed(0)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_numpy_reproducible(self):
        set_global_seed(42)
        a = np.random.rand(20)
        set_global_seed(42)
        b = np.random.rand(20)
        np.testing.assert_array_equal(a, b)

    def test_torch_reproducible(self):
        set_global_seed(7)
        a = torch.rand(15)
        set_global_seed(7)
        b = torch.rand(15)
        assert torch.allclose(a, b)

    def test_different_seeds_differ(self):
        set_global_seed(1)
        a = torch.rand(10)
        set_global_seed(2)
        b = torch.rand(10)
        assert not torch.allclose(a, b)

    def test_default_seed_value(self):
        """Default seed (42) should not raise and should be reproducible."""
        set_global_seed()
        a = np.random.rand(5)
        set_global_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_reproducible(self):
        set_global_seed(99)
        a = torch.rand(10, device="cuda")
        set_global_seed(99)
        b = torch.rand(10, device="cuda")
        assert torch.allclose(a, b)
