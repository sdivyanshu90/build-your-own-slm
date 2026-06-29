"""Tests for checkpoint save/load."""

from __future__ import annotations

import pytest
import torch

from slm.config import ModelConfig
from slm.model.gpt import GPT
from slm.training.checkpoint import (
    Checkpoint,
    build_model,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture()
def model() -> GPT:
    return GPT(
        ModelConfig(vocab_size=48, block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    )


def test_save_and_load_roundtrip(model: GPT, tmp_path):
    opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    path = save_checkpoint(
        tmp_path / "model.pt", model=model, step=10, best_val_loss=1.23, optimizer=opt
    )
    assert path.exists()
    ckpt = load_checkpoint(path)
    assert ckpt.step == 10
    assert ckpt.best_val_loss == 1.23
    assert ckpt.optimizer_state is not None
    assert ckpt.model_config.n_layer == 2


def test_build_model_matches_weights(model: GPT, tmp_path):
    path = save_checkpoint(tmp_path / "model.pt", model=model, step=1, best_val_loss=9.9)
    rebuilt = build_model(load_checkpoint(path), device=torch.device("cpu"))
    x = torch.randint(0, model.config.vocab_size, (1, 8))
    with torch.no_grad():
        a, _ = model(x)
        b, _ = rebuilt(x)
    assert torch.allclose(a, b, atol=1e-6)


def test_strip_compiled_prefix(model: GPT, tmp_path):
    # Simulate a torch.compile state_dict by prefixing keys.
    state = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
    payload = {
        "version": 1,
        "model_config": model.config.model_dump(),
        "model_state": state,
        "step": 0,
        "best_val_loss": float("inf"),
    }
    path = tmp_path / "compiled.pt"
    torch.save(payload, path)
    ckpt = load_checkpoint(path)
    assert all(not k.startswith("_orig_mod.") for k in ckpt.model_state)
    build_model(ckpt, device=torch.device("cpu"))  # loads without error


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "absent.pt")


def test_load_bad_version_raises(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"version": 99, "model_config": {}, "model_state": {}}, path)
    with pytest.raises(ValueError, match="version"):
        load_checkpoint(path)


def test_checkpoint_dataclass_defaults():
    ckpt = Checkpoint(model_config=ModelConfig(), model_state={}, step=0, best_val_loss=1.0)
    assert ckpt.optimizer_state is None
    assert ckpt.experiment_config is None
