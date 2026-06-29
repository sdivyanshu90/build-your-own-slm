"""Tests for the learning-rate schedule."""

from __future__ import annotations

import math

from slm.training.schedule import cosine_lr


def _lr(step: int) -> float:
    return cosine_lr(step, base_lr=1.0, min_lr=0.1, warmup_steps=10, decay_steps=110)


def test_warmup_is_linear_and_increasing():
    assert _lr(0) < _lr(5) < _lr(9)
    assert _lr(9) <= 1.0


def test_peak_after_warmup():
    assert math.isclose(_lr(10), 1.0, rel_tol=0.05)


def test_midpoint_is_between_floor_and_peak():
    mid = _lr(60)  # halfway through the 100-step decay window
    assert 0.1 < mid < 1.0
    assert math.isclose(mid, 0.55, abs_tol=0.05)


def test_floor_after_decay():
    assert math.isclose(_lr(110), 0.1, abs_tol=1e-9)
    assert math.isclose(_lr(500), 0.1, abs_tol=1e-9)


def test_no_warmup():
    assert cosine_lr(0, base_lr=1.0, min_lr=0.1, warmup_steps=0, decay_steps=100) == 1.0
