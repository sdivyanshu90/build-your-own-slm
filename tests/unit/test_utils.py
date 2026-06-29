"""Tests for utility helpers."""

from __future__ import annotations

import numpy as np
import torch

from slm.config import ModelConfig
from slm.model.gpt import GPT
from slm.utils import (
    autocast_dtype,
    count_parameters,
    human_readable_count,
    numpy_seed,
    resolve_device,
    set_seed,
)


def test_set_seed_is_reproducible():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_set_seed_deterministic_flag():
    set_seed(1, deterministic=True)
    # Reset so deterministic-algorithm mode does not leak into other tests.
    torch.use_deterministic_algorithms(False)


def test_resolve_device_cpu():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


def test_autocast_dtype_cpu():
    cpu = torch.device("cpu")
    assert autocast_dtype("float32", cpu) is torch.float32
    assert autocast_dtype("bfloat16", cpu) is torch.bfloat16
    # float16 autocast on CPU is unsupported -> degrade to float32.
    assert autocast_dtype("float16", cpu) is torch.float32


def test_count_parameters():
    model = GPT(
        ModelConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    )
    assert count_parameters(model) > 0
    assert count_parameters(model, trainable_only=False) >= count_parameters(model)


def test_human_readable_count():
    assert human_readable_count(500) == "500"
    assert human_readable_count(1_500) == "1.50K"
    assert human_readable_count(2_500_000) == "2.50M"
    assert human_readable_count(3_000_000_000) == "3.00B"


def test_numpy_seed_restores_state():
    np.random.seed(7)
    before = np.random.get_state()[1][0]
    with numpy_seed(99):
        inside = np.random.randint(0, 1000)
    after = np.random.get_state()[1][0]
    assert before == after  # global state restored
    with numpy_seed(99):
        assert np.random.randint(0, 1000) == inside  # deterministic within block
