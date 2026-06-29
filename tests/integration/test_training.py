"""Integration tests for the training loop."""

from __future__ import annotations

import math

import pytest

from slm.data import prepare_dataset
from slm.training.trainer import Trainer

pytestmark = pytest.mark.integration


def test_training_reduces_loss(experiment_config):
    prepare_dataset(experiment_config)
    trainer = Trainer(experiment_config)
    initial = trainer.estimate_loss()["val"]
    state = trainer.train()
    assert math.isfinite(state.best_val_loss)
    # The model must learn *something* on a repetitive corpus.
    assert state.best_val_loss < initial


def test_training_writes_artifacts(experiment_config):
    prepare_dataset(experiment_config)
    Trainer(experiment_config).train()
    assert (experiment_config.train.out_dir / "model.pt").exists()
    assert (experiment_config.train.out_dir / "tokenizer.json").exists()


def test_training_resume_continues(experiment_config):
    prepare_dataset(experiment_config)
    Trainer(experiment_config).train()  # trains to max_steps (40)

    extended = experiment_config.model_copy(
        update={"train": experiment_config.train.model_copy(update={"max_steps": 60})}
    )
    resumed = Trainer(extended, resume=True)
    assert resumed.start_step == 40
    state = resumed.train()
    assert state.step == 60


def test_estimate_loss_reports_both_splits(experiment_config):
    prepare_dataset(experiment_config)
    losses = Trainer(experiment_config).estimate_loss()
    assert set(losses) == {"train", "val"}
    assert all(math.isfinite(v) for v in losses.values())


def test_constant_lr_when_decay_disabled(experiment_config):
    prepare_dataset(experiment_config)
    cfg = experiment_config.model_copy(
        update={"optim": experiment_config.optim.model_copy(update={"decay_lr": False})}
    )
    trainer = Trainer(cfg)
    assert trainer._lr_at(0) == cfg.optim.lr
    assert trainer._lr_at(1000) == cfg.optim.lr


def test_training_invokes_progress_callback(experiment_config):
    prepare_dataset(experiment_config)
    seen: list[int] = []
    Trainer(experiment_config).train(progress_callback=lambda state: seen.append(state.step))
    assert seen  # callback fired at least once
    assert all(s > 0 for s in seen)


def test_tokenizer_colocated_when_path_differs(experiment_config):
    # Point the tokenizer outside the checkpoint dir so training copies it in.
    relocated = experiment_config.model_copy(
        update={
            "tokenizer": experiment_config.tokenizer.model_copy(
                update={"path": experiment_config.data.data_dir / "tokenizer.json"}
            )
        }
    )
    prepare_dataset(relocated)
    Trainer(relocated).train()
    assert (relocated.train.out_dir / "tokenizer.json").exists()
