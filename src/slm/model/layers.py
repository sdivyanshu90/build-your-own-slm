"""Transformer building blocks: LayerNorm, causal self-attention, MLP, Block.

The implementation favours clarity and correctness while still using PyTorch's
fused scaled-dot-product-attention kernel (FlashAttention / mem-efficient) when
available, with a numerically identical manual fallback for older runtimes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.config import ModelConfig

__all__ = ["MLP", "Block", "CausalSelfAttention", "LayerNorm"]


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias.

    ``torch.nn.LayerNorm`` does not support disabling the bias, which GPT-2 style
    models often do; this thin wrapper exposes that knob.
    """

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Queries, keys and values are produced by a single fused projection. The
    causal mask is applied by ``scaled_dot_product_attention(is_causal=True)``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self._flash = hasattr(F, "scaled_dot_product_attention")
        self.causal_mask: torch.Tensor
        if not self._flash:  # pragma: no cover - fallback for torch < 2.0
            # Pre-compute a causal mask buffer for the fallback path.
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer(
                "causal_mask", mask.view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()  # batch, time, channels
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = c // self.n_head
        # (B, nh, T, hd)
        q = q.view(b, t, self.n_head, head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, head_dim).transpose(1, 2)

        if self._flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:  # pragma: no cover - fallback for torch < 2.0
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
            att = att.masked_fill(self.causal_mask[:, :, :t, :t] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network with a 4x inner expansion and GELU."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """A pre-norm transformer block: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
