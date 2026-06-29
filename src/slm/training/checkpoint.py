"""Checkpoint serialisation and restoration.

A checkpoint is a single ``model.pt`` file (PyTorch pickle) capturing everything
needed to resume training *or* to run inference: the model weights, the exact
architecture config, the optimiser state, and training progress. The tokenizer
lives alongside it as ``tokenizer.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from slm.config import ExperimentConfig, ModelConfig
from slm.logging_config import get_logger
from slm.model.gpt import GPT

__all__ = ["Checkpoint", "build_model", "load_checkpoint", "save_checkpoint"]

log = get_logger(__name__)

CHECKPOINT_VERSION = 1
_COMPILED_PREFIX = "_orig_mod."


@dataclass
class Checkpoint:
    """In-memory representation of a loaded checkpoint."""

    model_config: ModelConfig
    model_state: dict[str, Any]
    step: int
    best_val_loss: float
    optimizer_state: dict[str, Any] | None = None
    experiment_config: dict[str, Any] | None = None


def _strip_compiled_prefix(state: dict[str, Any]) -> dict[str, Any]:
    """Remove the ``_orig_mod.`` prefix added by ``torch.compile`` wrappers."""
    if any(k.startswith(_COMPILED_PREFIX) for k in state):
        return {k.removeprefix(_COMPILED_PREFIX): v for k, v in state.items()}
    return state


def save_checkpoint(
    path: str | Path,
    *,
    model: GPT,
    step: int,
    best_val_loss: float,
    optimizer: torch.optim.Optimizer | None = None,
    experiment_config: ExperimentConfig | None = None,
) -> Path:
    """Atomically persist a checkpoint to ``path``.

    The write goes to a temporary file which is then renamed, so a crash mid-save
    can never corrupt an existing good checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    raw_model = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    payload: dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "model_config": raw_model.config.model_dump(),
        "model_state": _strip_compiled_prefix(raw_model.state_dict()),
        "step": step,
        "best_val_loss": best_val_loss,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if experiment_config is not None:
        payload["experiment_config"] = experiment_config.model_dump(mode="json")

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    log.info("checkpoint.saved", path=str(path), step=step, best_val_loss=best_val_loss)
    return path


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> Checkpoint:
    """Load a checkpoint file into a :class:`Checkpoint`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    version = payload.get("version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {version} (expected {CHECKPOINT_VERSION})."
        )
    return Checkpoint(
        model_config=ModelConfig.model_validate(payload["model_config"]),
        model_state=_strip_compiled_prefix(payload["model_state"]),
        step=int(payload.get("step", 0)),
        best_val_loss=float(payload.get("best_val_loss", float("inf"))),
        optimizer_state=payload.get("optimizer_state"),
        experiment_config=payload.get("experiment_config"),
    )


def build_model(checkpoint: Checkpoint, *, device: torch.device) -> GPT:
    """Instantiate a :class:`GPT` from a checkpoint and load its weights."""
    model = GPT(checkpoint.model_config)
    model.load_state_dict(checkpoint.model_state)
    model.to(device)
    model.eval()
    return model
