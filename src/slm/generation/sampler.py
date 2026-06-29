"""Token sampling strategies for autoregressive decoding.

Supported controls (composable):

* **temperature** — flattens (>1) or sharpens (<1) the distribution; ``0`` means
  greedy/argmax decoding.
* **top_k** — keep only the ``k`` most probable tokens.
* **top_p** (nucleus) — keep the smallest set of tokens whose cumulative
  probability exceeds ``p``.
* **repetition_penalty** — divide the logits of already-generated tokens to
  discourage loops (Keskar et al., 2019).
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

from slm.model.gpt import GPT

__all__ = ["GenerationConfig", "generate_tokens", "sample_next_token"]


class GenerationConfig(BaseModel):
    """Validated decoding parameters."""

    model_config = {"frozen": True, "extra": "forbid"}

    max_new_tokens: int = Field(default=128, ge=1, le=8192)
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    top_k: int | None = Field(default=40, ge=1)
    top_p: float | None = Field(default=0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0)
    seed: int | None = Field(default=None)


def _apply_repetition_penalty(
    logits: torch.Tensor, generated: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Penalise logits of previously generated tokens in-place-safe fashion."""
    if penalty == 1.0 or generated.numel() == 0:
        return logits
    unique = torch.unique(generated)
    selected = logits[..., unique]
    # Positive logits are divided, negative logits are multiplied, so that the
    # penalty always *reduces* the probability of seen tokens.
    selected = torch.where(selected > 0, selected / penalty, selected * penalty)
    logits[..., unique] = selected
    return logits


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, logits.size(-1))
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens once cumulative probability has exceeded p, but always keep
    # at least the single most probable token.
    sorted_remove = cum_probs > p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False
    remove = sorted_remove.scatter(-1, sorted_idx, sorted_remove)
    return logits.masked_fill(remove, float("-inf"))


def sample_next_token(
    logits: torch.Tensor,
    config: GenerationConfig,
    generated: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample (or argmax) the next token id from final-position ``logits``.

    Args:
        logits: ``(B, vocab)`` raw logits for the next position.
        config: Decoding parameters.
        generated: ``(B, t)`` ids generated so far (for repetition penalty).
        generator: Optional torch RNG for reproducible sampling.

    Returns:
        ``(B, 1)`` int64 tensor of sampled token ids.
    """
    logits = logits.float()
    logits = _apply_repetition_penalty(logits, generated, config.repetition_penalty)

    if config.temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / config.temperature
    if config.top_k is not None:
        logits = _top_k_filter(logits, config.top_k)
    if config.top_p is not None:
        logits = _top_p_filter(logits, config.top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def generate_tokens(
    model: GPT,
    prompt_ids: torch.Tensor,
    config: GenerationConfig,
    *,
    stop_ids: frozenset[int] | None = None,
    generator: torch.Generator | None = None,
) -> Iterator[int]:
    """Yield generated token ids one at a time (single sequence, ``B == 1``).

    The context is cropped to ``block_size`` on every step, so generation works
    for prompts of any length and runs indefinitely up to ``max_new_tokens``.

    Args:
        model: A GPT model in eval mode.
        prompt_ids: ``(1, T)`` int64 tensor of prompt token ids.
        config: Decoding parameters.
        stop_ids: Token ids that terminate generation when produced.
        generator: Optional torch RNG for reproducibility.

    Yields:
        Generated token ids (excluding the prompt).
    """
    if prompt_ids.dim() != 2 or prompt_ids.size(0) != 1:
        raise ValueError("generate_tokens expects a (1, T) prompt tensor.")

    was_training = model.training
    model.eval()
    block_size = model.config.block_size
    idx = prompt_ids
    generated = prompt_ids.new_zeros((1, 0))
    stop_ids = stop_ids or frozenset()

    try:
        for _ in range(config.max_new_tokens):
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits, _ = model(idx_cond)
            next_id = sample_next_token(logits[:, -1, :], config, generated, generator=generator)
            token = int(next_id.item())
            if token in stop_ids:
                break
            yield token
            idx = torch.cat((idx, next_id), dim=1)
            generated = torch.cat((generated, next_id), dim=1)
    finally:
        if was_training:
            model.train()
