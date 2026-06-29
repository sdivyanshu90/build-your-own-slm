"""Tests for the GPT model and its layers."""

from __future__ import annotations

import pytest
import torch

from slm.config import ModelConfig
from slm.model.gpt import GPT
from slm.model.layers import CausalSelfAttention, LayerNorm


@pytest.fixture()
def config() -> ModelConfig:
    return ModelConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0)


def test_forward_with_targets_returns_loss(config: ModelConfig):
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (3, 8))
    logits, loss = model(x, x)
    assert logits.shape == (3, 8, config.vocab_size)
    assert loss is not None and loss.item() > 0


def test_forward_without_targets_is_last_position_only(config: ModelConfig):
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (3, 8))
    logits, loss = model(x)
    assert logits.shape == (3, 1, config.vocab_size)
    assert loss is None


def test_weight_tying(config: ModelConfig):
    model = GPT(config)
    assert model.token_embedding.weight is model.lm_head.weight


def test_block_size_overflow_raises(config: ModelConfig):
    model = GPT(config)
    too_long = torch.randint(0, config.vocab_size, (1, config.block_size + 1))
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(too_long)


def test_num_params_excludes_position_embedding(config: ModelConfig):
    model = GPT(config)
    full = model.get_num_params(non_embedding=False)
    non_emb = model.get_num_params(non_embedding=True)
    assert full - non_emb == model.position_embedding.weight.numel()


def test_parameter_groups_numel(config: ModelConfig):
    model = GPT(config)
    breakdown = model.parameter_groups_numel()
    assert set(breakdown) == {"embedding", "position", "total", "non_embedding"}
    assert breakdown["total"] > breakdown["non_embedding"]


def test_configure_optimizers_groups(config: ModelConfig):
    model = GPT(config)
    opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == 0.1
    assert opt.param_groups[1]["weight_decay"] == 0.0


def test_gradient_checkpointing_runs():
    cfg = ModelConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    model = GPT(cfg, gradient_checkpointing=True)
    model.train()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss = model(x, x)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_model_with_bias_enabled():
    cfg = ModelConfig(vocab_size=48, block_size=16, n_layer=2, n_head=2, n_embd=16, bias=True)
    model = GPT(cfg)
    # At least one Linear now carries a (zero-initialised) bias.
    linear_biases = [m.bias for m in model.modules() if isinstance(m, torch.nn.Linear)]
    assert any(b is not None for b in linear_biases)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    logits, _ = model(x)
    assert logits.shape == (1, 1, cfg.vocab_size)


def test_layernorm_without_bias():
    ln = LayerNorm(8, bias=False)
    assert ln.bias is None
    out = ln(torch.randn(2, 8))
    assert out.shape == (2, 8)


def test_attention_is_causal(config: ModelConfig):
    # A change to a future token must not affect the current token's output.
    attn = CausalSelfAttention(config).eval()
    x = torch.randn(1, 6, config.n_embd)
    with torch.no_grad():
        base = attn(x)
        perturbed = x.clone()
        perturbed[0, -1] += 10.0  # change only the last (future) position
        after = attn(perturbed)
    assert torch.allclose(base[0, 0], after[0, 0], atol=1e-5)
