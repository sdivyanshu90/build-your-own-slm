"""Shared pytest fixtures.

A single tiny model is trained once per test session on a synthetic corpus and
reused across the inference/API tests, keeping the whole suite fast (seconds)
while still exercising the real training → checkpoint → serving path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slm.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimConfig,
    Settings,
    TokenizerConfig,
    TrainConfig,
)

# A small but structured corpus: enough repetition for BPE to learn merges and
# for the model to drive its loss down within a handful of steps.
SYNTHETIC_CORPUS = (
    "the quick brown fox jumps over the lazy dog . "
    "a wizard's job is to vex chumps quickly in fog . "
    "the cat sat on the mat while the dog ran in the park . "
) * 300


def build_experiment_config(root: Path) -> ExperimentConfig:
    """Construct a self-contained tiny experiment rooted at ``root``."""
    data_dir = root / "data"
    out_dir = root / "ckpt"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "input.txt").write_text(SYNTHETIC_CORPUS, encoding="utf-8")
    return ExperimentConfig(
        name="test",
        model=ModelConfig(
            vocab_size=384, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0
        ),
        tokenizer=TokenizerConfig(
            vocab_size=384, path=out_dir / "tokenizer.json", special_tokens=["<|endoftext|>"]
        ),
        data=DataConfig(data_dir=data_dir, source_url=None, raw_file="input.txt", val_fraction=0.1),
        optim=OptimConfig(lr=1e-3, warmup_steps=5, lr_decay_steps=40, min_lr=1e-4),
        train=TrainConfig(
            out_dir=out_dir,
            max_steps=40,
            batch_size=16,
            eval_interval=20,
            eval_steps=5,
            log_interval=10,
            device="cpu",
            dtype="float32",
            seed=1234,
        ),
    )


@pytest.fixture()
def experiment_config(tmp_path: Path) -> ExperimentConfig:
    """A fresh, isolated experiment config in a per-test temp dir."""
    return build_experiment_config(tmp_path)


@pytest.fixture()
def tokenizer():
    """A trained BPE tokenizer over the synthetic corpus."""
    from slm.tokenizer import BPETokenizer

    return BPETokenizer.train([SYNTHETIC_CORPUS], vocab_size=384, special_tokens=["<|endoftext|>"])


@pytest.fixture()
def tiny_model():
    """An untrained tiny GPT for fast unit tests."""
    from slm.model.gpt import GPT

    return GPT(
        ModelConfig(vocab_size=128, block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    )


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a tiny model once and return its checkpoint directory."""
    from slm.data import prepare_dataset
    from slm.training.trainer import Trainer

    root = tmp_path_factory.mktemp("trained")
    config = build_experiment_config(root)
    prepare_dataset(config)
    Trainer(config).train()
    return config.train.out_dir


@pytest.fixture()
def inference_engine(trained_model_dir: Path):
    """An :class:`InferenceEngine` over the session-trained model."""
    from slm.inference.engine import InferenceEngine

    return InferenceEngine.from_pretrained(trained_model_dir, device="cpu")


@pytest.fixture()
def auth_settings() -> Settings:
    return Settings(
        api_keys="test-key",
        rate_limit_per_minute=6000,
        rate_limit_burst=1000,
        log_level="WARNING",
        env="development",
    )


@pytest.fixture()
def client(auth_settings: Settings, inference_engine):
    """A TestClient with auth enabled and a generous rate limit."""
    from fastapi.testclient import TestClient

    from slm.api.app import create_app

    app = create_app(auth_settings, engine=inference_engine, load_model=False)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def auth_header() -> dict[str, str]:
    return {"X-API-Key": "test-key"}
