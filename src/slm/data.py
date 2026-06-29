"""Dataset preparation and efficient batch loading.

The pipeline follows the well-trodden nanoGPT layout: raw UTF-8 text is encoded
once into a flat array of ``uint16`` token ids and written to disk as a binary
file. Training then memory-maps that file and samples random fixed-length
windows, so the dataset never has to fit in RAM and there is zero per-batch
tokenisation cost.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from slm.config import ExperimentConfig
from slm.logging_config import get_logger
from slm.tokenizer import BPETokenizer

__all__ = ["DatasetStats", "get_batch", "load_token_bin", "prepare_dataset"]

log = get_logger(__name__)

# uint16 stores ids 0..65535; guard against vocabularies that would overflow it.
_MAX_UINT16_VOCAB = 1 << 16


@dataclass(frozen=True)
class DatasetStats:
    """Summary of a prepared dataset."""

    train_tokens: int
    val_tokens: int
    vocab_size: int
    train_bin: Path
    val_bin: Path
    tokenizer_path: Path


def _resolve_raw_text(config: ExperimentConfig) -> str:
    """Return the raw training text, downloading it if a source URL is given."""
    data = config.data
    raw_path = data.data_dir / data.raw_file
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        if data.source_url is None:
            raise FileNotFoundError(
                f"Raw text {raw_path} not found and no data.source_url configured."
            )
        log.info("dataset.download", url=data.source_url, dest=str(raw_path))
        with urllib.request.urlopen(data.source_url, timeout=60) as response:
            raw_path.write_bytes(response.read())

    return raw_path.read_text(encoding="utf-8")


def prepare_dataset(config: ExperimentConfig, *, retrain_tokenizer: bool = True) -> DatasetStats:
    """Train (or reuse) the tokenizer and write train/val token binaries.

    Args:
        config: The experiment recipe.
        retrain_tokenizer: When ``False`` and a tokenizer already exists at the
            configured path, it is reused instead of retrained.

    Returns:
        A :class:`DatasetStats` describing the prepared artifacts.
    """
    text = _resolve_raw_text(config)
    tok_cfg = config.tokenizer

    if tok_cfg.path.exists() and not retrain_tokenizer:
        log.info("tokenizer.reuse", path=str(tok_cfg.path))
        tokenizer = BPETokenizer.load(tok_cfg.path)
    else:
        log.info("tokenizer.train", vocab_size=tok_cfg.vocab_size)
        tokenizer = BPETokenizer.train(
            texts=[text],
            vocab_size=tok_cfg.vocab_size,
            special_tokens=tok_cfg.special_tokens,
        )
        tokenizer.save(tok_cfg.path)
        log.info("tokenizer.saved", path=str(tok_cfg.path), vocab_size=tokenizer.vocab_size)

    if tokenizer.vocab_size >= _MAX_UINT16_VOCAB:  # pragma: no cover - guard
        raise ValueError(
            f"vocab_size {tokenizer.vocab_size} does not fit in uint16; "
            "reduce tokenizer.vocab_size or change the storage dtype."
        )

    ids = np.array(tokenizer.encode(text), dtype=np.uint16)
    split = int(len(ids) * (1.0 - config.data.val_fraction))
    train_ids, val_ids = ids[:split], ids[split:]

    data_dir = config.data.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    train_bin = data_dir / config.data.train_bin
    val_bin = data_dir / config.data.val_bin
    train_ids.tofile(train_bin)
    val_ids.tofile(val_bin)

    log.info(
        "dataset.prepared",
        train_tokens=int(train_ids.size),
        val_tokens=int(val_ids.size),
        vocab_size=tokenizer.vocab_size,
    )
    return DatasetStats(
        train_tokens=int(train_ids.size),
        val_tokens=int(val_ids.size),
        vocab_size=tokenizer.vocab_size,
        train_bin=train_bin,
        val_bin=val_bin,
        tokenizer_path=tok_cfg.path,
    )


def load_token_bin(path: str | Path) -> np.memmap:
    """Memory-map a ``uint16`` token binary written by :func:`prepare_dataset`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Token binary not found: {path}. Run `slm prepare-data` first.")
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    data: np.memmap,
    *,
    block_size: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random training batch of ``(x, y)`` next-token pairs.

    ``y`` is ``x`` shifted left by one position (the language-modelling target).
    Pinned-memory + non-blocking transfer is used on CUDA to overlap H2D copies.
    """
    if len(data) <= block_size:
        raise ValueError(
            f"Dataset has {len(data)} tokens, which is too small for block_size {block_size}."
        )
    max_start = len(data) - block_size - 1
    ix = torch.randint(max_start + 1, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
    )
    if device.type == "cuda":  # pragma: no cover - CUDA not present in CI
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    return x.to(device), y.to(device)
