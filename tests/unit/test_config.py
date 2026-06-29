"""Tests for configuration models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from slm.config import (
    ExperimentConfig,
    ModelConfig,
    Settings,
    load_experiment_config,
)


def test_settings_defaults():
    settings = Settings(_env_file=None)
    assert settings.env == "development"
    assert settings.api_port == 8000
    assert settings.auth_enabled is False
    assert settings.is_production is False


def test_settings_cors_and_keys_parse_csv():
    settings = Settings(
        _env_file=None,
        cors_origins="https://a.com, https://b.com",
        api_keys="k1, k2 ,k3",
    )
    assert settings.cors_origins == ["https://a.com", "https://b.com"]
    assert settings.api_keys == frozenset({"k1", "k2", "k3"})
    assert settings.auth_enabled is True


def test_settings_log_level_validation():
    assert Settings(_env_file=None, log_level="debug").log_level == "DEBUG"
    with pytest.raises(ValidationError):
        Settings(_env_file=None, log_level="VERBOSE")


def test_settings_production_flag():
    assert Settings(_env_file=None, env="production").is_production is True


def test_model_config_head_divisibility():
    with pytest.raises(ValidationError):
        ModelConfig(n_embd=30, n_head=4)
    cfg = ModelConfig(n_embd=32, n_head=4)
    assert cfg.head_dim == 8


def test_experiment_config_syncs_vocab():
    cfg = ExperimentConfig.model_validate(
        {"tokenizer": {"vocab_size": 512}, "model": {"vocab_size": 8192, "n_embd": 32, "n_head": 4}}
    )
    assert cfg.model.vocab_size == 512


def test_load_experiment_config(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(
        "name: demo\nmodel:\n  n_layer: 3\n  n_embd: 64\n  n_head: 4\n", encoding="utf-8"
    )
    cfg = load_experiment_config(path)
    assert cfg.name == "demo"
    assert cfg.model.n_layer == 3


def test_load_experiment_config_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_experiment_config(tmp_path / "nope.yaml")


def test_load_experiment_config_non_mapping(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_experiment_config(path)


def test_config_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ModelConfig(unknown_field=1)
