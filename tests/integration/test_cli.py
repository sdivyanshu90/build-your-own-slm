"""Integration tests for the Typer CLI."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from slm.cli import app

pytestmark = pytest.mark.integration

runner = CliRunner()

CONFIG_TEMPLATE = """
name: cli-test
model:
  vocab_size: 384
  block_size: 32
  n_layer: 2
  n_head: 2
  n_embd: 32
  dropout: 0.0
tokenizer:
  vocab_size: 384
  path: {out_dir}/tokenizer.json
  special_tokens: ["<|endoftext|>"]
data:
  data_dir: {data_dir}
  raw_file: input.txt
  val_fraction: 0.1
optim:
  lr: 1.0e-3
  warmup_steps: 5
  lr_decay_steps: 40
  min_lr: 1.0e-4
train:
  out_dir: {out_dir}
  max_steps: 30
  batch_size: 16
  eval_interval: 15
  eval_steps: 5
  log_interval: 10
  device: cpu
  dtype: float32
"""

CORPUS = "the quick brown fox jumps over the lazy dog . " * 300


@pytest.fixture()
def config_file(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "ckpt"
    data_dir.mkdir(parents=True)
    (data_dir / "input.txt").write_text(CORPUS, encoding="utf-8")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        CONFIG_TEMPLATE.format(data_dir=data_dir.as_posix(), out_dir=out_dir.as_posix()),
        encoding="utf-8",
    )
    return cfg_path, out_dir


def test_info_command():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "BYO-SLM" in result.stdout


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    # Typer's no_args_is_help prints usage and exits with Click's code 2.
    assert result.exit_code in (0, 2)
    assert "Build, train, and serve" in result.output


def test_generate_streaming_path(config_file):
    cfg_path, out_dir = config_file
    assert runner.invoke(app, ["prepare-data", "--config", str(cfg_path)]).exit_code == 0
    assert runner.invoke(app, ["train", "--config", str(cfg_path)]).exit_code == 0
    gen = runner.invoke(
        app,
        ["generate", "--model-dir", str(out_dir), "--prompt", "the", "--max-tokens", "5"],
    )
    assert gen.exit_code == 0, gen.output


def test_serve_invokes_uvicorn(monkeypatch):
    calls = {}

    def fake_run(target, **kwargs):
        calls["target"] = target
        calls.update(kwargs)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)
    result = runner.invoke(app, ["serve", "--port", "9001", "--host", "127.0.0.1"])
    assert result.exit_code == 0
    assert calls["target"] == "slm.api.app:create_app"
    assert calls["port"] == 9001


def test_full_cli_pipeline(config_file):
    cfg_path, out_dir = config_file

    prep = runner.invoke(app, ["prepare-data", "--config", str(cfg_path)])
    assert prep.exit_code == 0, prep.output
    assert (out_dir / "tokenizer.json").exists()

    train = runner.invoke(app, ["train", "--config", str(cfg_path)])
    assert train.exit_code == 0, train.output
    assert (out_dir / "model.pt").exists()

    gen = runner.invoke(
        app,
        [
            "generate",
            "--model-dir",
            str(out_dir),
            "--prompt",
            "the",
            "--max-tokens",
            "5",
            "--no-stream",
        ],
    )
    assert gen.exit_code == 0, gen.output


def test_generate_missing_model_dir_fails(tmp_path):
    result = runner.invoke(
        app, ["generate", "--model-dir", str(tmp_path / "absent"), "--prompt", "hi"]
    )
    assert result.exit_code != 0
