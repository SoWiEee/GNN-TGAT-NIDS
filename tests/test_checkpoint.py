"""
tests/test_checkpoint.py
Tests for src/utils/checkpoint.py.

Verifies save/load round-trip fidelity for model and optimizer states,
handles missing files gracefully, and checks that extra metadata persists.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam

from src.utils.checkpoint import load_checkpoint, save_checkpoint


# ── Minimal test model ─────────────────────────────────────────────────────────

class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def model_and_optimizer():
    model = _TinyNet()
    opt = Adam(model.parameters(), lr=0.01)
    # Take one gradient step so optimizer state is non-trivial
    loss = model(torch.randn(3, 4)).sum()
    loss.backward()
    opt.step()
    return model, opt


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestSaveLoadRoundTrip:
    def test_model_weights_restored(self, tmp_path, model_and_optimizer):
        model, opt = model_and_optimizer
        ckpt = tmp_path / "test.ckpt"

        original_weights = {k: v.clone() for k, v in model.state_dict().items()}
        save_checkpoint(model, opt, epoch=5, path=ckpt)

        # Corrupt the model weights
        for param in model.parameters():
            param.data.fill_(0.0)

        loaded_epoch = load_checkpoint(model, opt, path=ckpt)

        assert loaded_epoch == 5
        for key, val in model.state_dict().items():
            torch.testing.assert_close(val, original_weights[key])

    def test_optimizer_state_restored(self, tmp_path, model_and_optimizer):
        model, opt = model_and_optimizer
        ckpt = tmp_path / "opt.ckpt"

        # Snapshot the full state_dict (keyed by param index → {k: v})
        original_state = {
            param_idx: {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in param_state.items()
            }
            for param_idx, param_state in opt.state_dict()["state"].items()
        }
        save_checkpoint(model, opt, epoch=10, path=ckpt)

        # Reset optimizer state and reload
        opt.state.clear()
        load_checkpoint(model, opt, path=ckpt)

        for param_idx, param_state in opt.state_dict()["state"].items():
            for k, v in param_state.items():
                if isinstance(v, torch.Tensor):
                    torch.testing.assert_close(v, original_state[param_idx][k])

    def test_epoch_value_preserved(self, tmp_path, model_and_optimizer):
        model, opt = model_and_optimizer
        for epoch in (0, 1, 99, 200):
            ckpt = tmp_path / f"e{epoch}.ckpt"
            save_checkpoint(model, opt, epoch=epoch, path=ckpt)
            assert load_checkpoint(model, opt, path=ckpt) == epoch

    def test_extra_metadata_persisted(self, tmp_path, model_and_optimizer):
        model, opt = model_and_optimizer
        ckpt = tmp_path / "meta.ckpt"
        extra = {"val_f1": 0.923, "config": "graphsage"}
        save_checkpoint(model, opt, epoch=3, path=ckpt, extra=extra)

        payload = torch.load(ckpt, weights_only=False)
        assert payload["extra"]["val_f1"] == pytest.approx(0.923)
        assert payload["extra"]["config"] == "graphsage"

    def test_parent_dirs_created(self, tmp_path, model_and_optimizer):
        model, opt = model_and_optimizer
        deep_path = tmp_path / "a" / "b" / "c" / "model.ckpt"
        save_checkpoint(model, opt, epoch=1, path=deep_path)
        assert deep_path.exists()


class TestLoadCheckpointErrors:
    def test_missing_file_raises(self, tmp_path, model_and_optimizer):
        model, opt = model_and_optimizer
        with pytest.raises(FileNotFoundError):
            load_checkpoint(model, opt, tmp_path / "nonexistent.ckpt")

    def test_inference_mode_no_optimizer(self, tmp_path, model_and_optimizer):
        model, opt = model_and_optimizer
        ckpt = tmp_path / "infer.ckpt"
        save_checkpoint(model, opt, epoch=7, path=ckpt)

        # Passing optimizer=None should not raise
        epoch = load_checkpoint(model, optimizer=None, path=ckpt)
        assert epoch == 7
