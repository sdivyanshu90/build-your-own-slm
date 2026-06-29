"""Small, dependency-light utilities shared across the codebase."""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

__all__ = [
    "autocast_dtype",
    "count_parameters",
    "human_readable_count",
    "resolve_device",
    "set_seed",
]


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs.

    Args:
        seed: Non-negative seed value.
        deterministic: When ``True``, request deterministic CUDA kernels at the
            cost of throughput. Use for tests and debugging, not training.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CUDA not present in CI
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def resolve_device(choice: str = "auto") -> torch.device:
    """Resolve a device preference string to a concrete :class:`torch.device`.

    ``"auto"`` prefers CUDA, then Apple MPS, then CPU.
    """
    import torch

    if choice == "auto":
        if torch.cuda.is_available():  # pragma: no cover - CUDA not present in CI
            return torch.device("cuda")
        if (  # pragma: no cover - MPS not present in CI
            getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


def autocast_dtype(name: str, device: torch.device) -> torch.dtype:
    """Map a dtype name to a torch dtype, downgrading bf16->fp16->fp32 safely.

    CPU autocast only reliably supports bfloat16; CUDA bf16 needs Ampere+. We
    degrade gracefully so the same config runs everywhere without crashing.
    """
    import torch

    requested = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(name, torch.float32)

    if requested is torch.float32:
        return torch.float32
    if device.type == "cuda":  # pragma: no cover - CUDA not present in CI
        if requested is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            return torch.float16
        return requested
    if device.type == "cpu":
        # float16 autocast on CPU is poorly supported; prefer bfloat16.
        return torch.bfloat16 if requested is torch.bfloat16 else torch.float32
    return torch.float32  # pragma: no cover - MPS/other: stay in fp32


def count_parameters(module: torch.nn.Module, *, trainable_only: bool = True) -> int:
    """Count (optionally only trainable) parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def human_readable_count(n: int) -> str:
    """Render an integer count compactly, e.g. ``1_500_000 -> '1.50M'``."""
    for unit, threshold in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= threshold:
            return f"{n / threshold:.2f}{unit}"
    return str(n)


@contextmanager
def numpy_seed(seed: int) -> Iterator[None]:
    """Temporarily set the NumPy global seed, restoring prior state on exit."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
