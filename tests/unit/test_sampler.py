"""Tests for sampling strategies and token generation."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from slm.config import ModelConfig
from slm.generation.sampler import (
    GenerationConfig,
    generate_tokens,
    sample_next_token,
)
from slm.model.gpt import GPT


def _empty_generated(batch: int = 1) -> torch.Tensor:
    return torch.zeros((batch, 0), dtype=torch.long)


def test_generation_config_bounds():
    with pytest.raises(ValidationError):
        GenerationConfig(temperature=-1)
    with pytest.raises(ValidationError):
        GenerationConfig(top_p=2.0)
    with pytest.raises(ValidationError):
        GenerationConfig(max_new_tokens=0)


def test_greedy_is_argmax():
    logits = torch.tensor([[0.1, 5.0, 0.2, -1.0]])
    cfg = GenerationConfig(temperature=0.0)
    out = sample_next_token(logits, cfg, _empty_generated())
    assert out.item() == 1


def test_top_k_restricts_support():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    cfg = GenerationConfig(temperature=1.0, top_k=2, top_p=None)
    samples = {sample_next_token(logits, cfg, _empty_generated()).item() for _ in range(50)}
    assert samples.issubset({3, 4})  # only the top-2 logits survive


def test_top_p_keeps_at_least_one():
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    cfg = GenerationConfig(temperature=1.0, top_k=None, top_p=0.1)
    out = sample_next_token(logits, cfg, _empty_generated())
    assert out.item() == 0


def test_repetition_penalty_reduces_probability():
    # With equal logits, penalising the already-seen token 0 makes it strictly
    # less likely than the others, so greedy decoding must avoid it.
    logits = torch.tensor([[2.0, 2.0, 2.0]])
    generated = torch.tensor([[0]])
    cfg = GenerationConfig(temperature=0.0, repetition_penalty=2.0)
    out = sample_next_token(logits.clone(), cfg, generated)
    assert out.item() != 0


@pytest.fixture()
def model() -> GPT:
    cfg = ModelConfig(vocab_size=48, block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    return GPT(cfg).eval()


def test_generate_tokens_length(model: GPT):
    prompt = torch.zeros((1, 3), dtype=torch.long)
    cfg = GenerationConfig(max_new_tokens=7, seed=0)
    out = list(generate_tokens(model, prompt, cfg, generator=torch.Generator().manual_seed(0)))
    assert len(out) == 7


def test_generate_tokens_stop_id(model: GPT):
    prompt = torch.zeros((1, 3), dtype=torch.long)
    cfg = GenerationConfig(max_new_tokens=20, temperature=0.0)
    # Force a stop on whatever greedy first produces.
    first = next(iter(generate_tokens(model, prompt, cfg)))
    out = list(generate_tokens(model, prompt, cfg, stop_ids=frozenset({first})))
    assert out == []  # stops immediately on the first (stop) token


def test_generate_tokens_seed_reproducible(model: GPT):
    prompt = torch.zeros((1, 2), dtype=torch.long)
    cfg = GenerationConfig(max_new_tokens=8, temperature=0.9, seed=42)
    a = list(generate_tokens(model, prompt, cfg, generator=torch.Generator().manual_seed(42)))
    b = list(generate_tokens(model, prompt, cfg, generator=torch.Generator().manual_seed(42)))
    assert a == b


def test_generate_tokens_rejects_bad_shape(model: GPT):
    with pytest.raises(ValueError, match=r"\(1, T\)"):
        list(generate_tokens(model, torch.zeros((2, 3), dtype=torch.long), GenerationConfig()))


def test_generate_restores_training_mode(model: GPT):
    model.train()
    list(
        generate_tokens(
            model, torch.zeros((1, 2), dtype=torch.long), GenerationConfig(max_new_tokens=2)
        )
    )
    assert model.training is True
