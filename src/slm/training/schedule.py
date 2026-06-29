"""Learning-rate schedules."""

from __future__ import annotations

import math

__all__ = ["cosine_lr"]


def cosine_lr(
    step: int,
    *,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    decay_steps: int,
) -> float:
    """Linear warmup followed by cosine decay to ``min_lr``.

    * For ``step < warmup_steps``: linearly ramp from 0 to ``base_lr``.
    * For ``warmup_steps <= step <= decay_steps``: cosine-decay to ``min_lr``.
    * For ``step > decay_steps``: hold at ``min_lr``.

    Args:
        step: Current optimisation step (0-indexed).
        base_lr: Peak learning rate reached at the end of warmup.
        min_lr: Floor learning rate.
        warmup_steps: Number of warmup steps.
        decay_steps: Step at which decay reaches ``min_lr``.

    Returns:
        The learning rate to use at ``step``.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / (warmup_steps + 1)
    if step > decay_steps:
        return min_lr
    span = max(decay_steps - warmup_steps, 1)
    progress = (step - warmup_steps) / span
    progress = min(max(progress, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (base_lr - min_lr)
