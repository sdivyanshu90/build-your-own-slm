"""Autoregressive text generation: sampling strategies and configuration."""

from __future__ import annotations

from slm.generation.sampler import GenerationConfig, generate_tokens, sample_next_token

__all__ = ["GenerationConfig", "generate_tokens", "sample_next_token"]
