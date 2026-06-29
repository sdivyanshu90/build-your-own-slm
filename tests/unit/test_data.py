"""Tests for dataset preparation and batching."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from slm.data import DatasetStats, get_batch, load_token_bin, prepare_dataset


def test_prepare_dataset_writes_binaries(experiment_config):
    stats = prepare_dataset(experiment_config)
    assert isinstance(stats, DatasetStats)
    assert stats.train_tokens > 0
    assert stats.val_tokens > 0
    assert stats.train_bin.exists()
    assert stats.val_bin.exists()
    assert stats.tokenizer_path.exists()


def test_prepare_dataset_reuse_tokenizer(experiment_config):
    prepare_dataset(experiment_config)
    # Second call reusing the tokenizer should not raise and produces same vocab.
    stats = prepare_dataset(experiment_config, retrain_tokenizer=False)
    assert stats.vocab_size > 256


def test_prepare_missing_raw_no_url_raises(experiment_config, tmp_path):
    cfg = experiment_config.model_copy(
        update={"data": experiment_config.data.model_copy(update={"raw_file": "missing.txt"})}
    )
    with pytest.raises(FileNotFoundError):
        prepare_dataset(cfg)


def test_get_batch_shapes_and_shift(experiment_config):
    prepare_dataset(experiment_config)
    data = load_token_bin(experiment_config.data.data_dir / experiment_config.data.train_bin)
    gen = torch.Generator().manual_seed(0)
    x, y = get_batch(data, block_size=16, batch_size=4, device=torch.device("cpu"), generator=gen)
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    # y is x shifted left by one position.
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_get_batch_is_deterministic_with_generator(experiment_config):
    prepare_dataset(experiment_config)
    data = load_token_bin(experiment_config.data.data_dir / experiment_config.data.train_bin)
    x1, _ = get_batch(
        data,
        block_size=16,
        batch_size=4,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(1),
    )
    x2, _ = get_batch(
        data,
        block_size=16,
        batch_size=4,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(1),
    )
    assert torch.equal(x1, x2)


def test_load_token_bin_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_token_bin(tmp_path / "nope.bin")


def test_prepare_downloads_when_url_set(experiment_config, monkeypatch):
    import contextlib
    import io

    corpus = ("the cat sat on the mat . " * 200).encode("utf-8")

    @contextlib.contextmanager
    def fake_urlopen(url, timeout=0):
        yield io.BytesIO(corpus)

    # Remove the local raw file and point at a URL so the download path runs.
    (experiment_config.data.data_dir / experiment_config.data.raw_file).unlink()
    cfg = experiment_config.model_copy(
        update={
            "data": experiment_config.data.model_copy(
                update={"source_url": "https://example.test/input.txt"}
            )
        }
    )
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    stats = prepare_dataset(cfg)
    assert stats.train_tokens > 0
    assert (cfg.data.data_dir / cfg.data.raw_file).exists()


def test_get_batch_too_small_raises(tmp_path):
    data = np.array([1, 2, 3], dtype=np.uint16)
    path = tmp_path / "tiny.bin"
    data.tofile(path)
    mm = load_token_bin(path)
    with pytest.raises(ValueError, match="too small"):
        get_batch(mm, block_size=16, batch_size=2, device=torch.device("cpu"))
