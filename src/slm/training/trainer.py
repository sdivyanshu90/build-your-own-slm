"""The training loop.

Implements the standard ingredients of a robust LM training run:

* Mixed-precision autocast (bf16/fp16) with a gradient scaler for fp16.
* Gradient accumulation to reach large effective batch sizes on small GPUs.
* Cosine LR schedule with linear warmup.
* Gradient clipping.
* Periodic evaluation on a held-out split and best-checkpoint saving.
* Resumable state (optimizer + step) via :mod:`slm.training.checkpoint`.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from slm.config import ExperimentConfig
from slm.data import get_batch, load_token_bin
from slm.logging_config import get_logger
from slm.model.gpt import GPT
from slm.training.checkpoint import build_model, load_checkpoint, save_checkpoint
from slm.training.schedule import cosine_lr
from slm.utils import (
    autocast_dtype,
    count_parameters,
    human_readable_count,
    resolve_device,
    set_seed,
)

__all__ = ["TrainState", "Trainer"]

log = get_logger(__name__)

ProgressCallback = Callable[["TrainState"], None]


@dataclass
class TrainState:
    """A snapshot of training progress emitted to callbacks and logs."""

    step: int
    max_steps: int
    lr: float
    train_loss: float
    val_loss: float | None = None
    best_val_loss: float = float("inf")
    tokens_seen: int = 0
    dt_ms: float = 0.0


class Trainer:
    """Owns the model, optimiser, data, and the optimisation loop."""

    def __init__(self, config: ExperimentConfig, *, resume: bool = False) -> None:
        self.config = config
        set_seed(config.train.seed)
        self.device = resolve_device(config.train.device)
        self.device_type = self.device.type
        self._generator = torch.Generator().manual_seed(config.train.seed)

        # Data (memory-mapped; never fully loaded into RAM).
        data_dir = config.data.data_dir
        self.train_data = load_token_bin(data_dir / config.data.train_bin)
        self.val_data = load_token_bin(data_dir / config.data.val_bin)

        # Mixed precision setup.
        self.amp_dtype = autocast_dtype(config.train.dtype, self.device)
        self._ctx = (
            torch.autocast(device_type=self.device_type, dtype=self.amp_dtype)
            if self.amp_dtype is not torch.float32 and self.device_type in ("cuda", "cpu")
            else nullcontext()
        )
        self.scaler = torch.amp.GradScaler(
            device=self.device_type,
            enabled=(self.amp_dtype is torch.float16 and self.device_type == "cuda"),
        )

        ckpt_path = config.train.out_dir / "model.pt"
        self.start_step = 0
        self.best_val_loss = float("inf")
        if resume and ckpt_path.exists():
            self._resume(ckpt_path)
        else:
            self.model = GPT(
                config.model, gradient_checkpointing=config.train.gradient_checkpointing
            )
            self.model.to(self.device)
            self.optimizer = self.model.configure_optimizers(
                weight_decay=config.optim.weight_decay,
                learning_rate=config.optim.lr,
                betas=(config.optim.beta1, config.optim.beta2),
                device_type=self.device_type,
            )

        self._raw_model = self.model
        if config.train.compile and hasattr(torch, "compile"):  # pragma: no cover - slow JIT
            log.info("trainer.compile")
            self.model = torch.compile(self.model)  # type: ignore[assignment]

        log.info(
            "trainer.init",
            device=str(self.device),
            dtype=str(self.amp_dtype),
            params=human_readable_count(count_parameters(self._raw_model)),
            train_tokens=int(self.train_data.size),
            val_tokens=int(self.val_data.size),
        )

    def _resume(self, ckpt_path: Path) -> None:
        log.info("trainer.resume", path=str(ckpt_path))
        ckpt = load_checkpoint(ckpt_path, map_location=self.device)
        self.model = build_model(ckpt, device=self.device)
        self.model.train()
        self.optimizer = self.model.configure_optimizers(
            weight_decay=self.config.optim.weight_decay,
            learning_rate=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            device_type=self.device_type,
        )
        if ckpt.optimizer_state is not None:
            self.optimizer.load_state_dict(ckpt.optimizer_state)
        self.start_step = ckpt.step
        self.best_val_loss = ckpt.best_val_loss

    def _batch(self, data: np.memmap) -> tuple[torch.Tensor, torch.Tensor]:
        return get_batch(
            data,
            block_size=self.config.model.block_size,
            batch_size=self.config.train.batch_size,
            device=self.device,
            generator=self._generator,
        )

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        """Average loss over ``eval_steps`` batches for each split."""
        self.model.eval()
        out: dict[str, float] = {}
        for split, data in (("train", self.train_data), ("val", self.val_data)):
            losses = torch.zeros(self.config.train.eval_steps)
            for i in range(self.config.train.eval_steps):
                x, y = self._batch(data)
                with self._ctx:
                    _, loss = self.model(x, y)
                losses[i] = loss.item()
            out[split] = float(losses.mean())
        self.model.train()
        return out

    def _lr_at(self, step: int) -> float:
        opt = self.config.optim
        if not opt.decay_lr:
            return opt.lr
        return cosine_lr(
            step,
            base_lr=opt.lr,
            min_lr=opt.min_lr,
            warmup_steps=opt.warmup_steps,
            decay_steps=opt.lr_decay_steps,
        )

    def _save(self, step: int) -> None:
        save_checkpoint(
            self.config.train.out_dir / "model.pt",
            model=self._raw_model,
            step=step,
            best_val_loss=self.best_val_loss,
            optimizer=self.optimizer,
            experiment_config=self.config,
        )
        self._ensure_tokenizer_colocated()

    def _ensure_tokenizer_colocated(self) -> None:
        """Guarantee the tokenizer sits next to the checkpoint for serving."""
        target = self.config.train.out_dir / "tokenizer.json"
        source = self.config.tokenizer.path
        if source.exists() and source.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    def train(self, progress_callback: ProgressCallback | None = None) -> TrainState:
        """Run the optimisation loop and return the final :class:`TrainState`."""
        cfg = self.config
        self.model.train()
        grad_accum = cfg.train.grad_accum_steps
        tokens_per_step = cfg.train.batch_size * grad_accum * cfg.model.block_size
        state = TrainState(
            step=self.start_step,
            max_steps=cfg.train.max_steps,
            lr=self._lr_at(self.start_step),
            train_loss=float("nan"),
            best_val_loss=self.best_val_loss,
        )

        for step in range(self.start_step + 1, cfg.train.max_steps + 1):
            t0 = time.perf_counter()
            lr = self._lr_at(step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            # Evaluation + checkpointing on the cadence (also at the final step).
            if step % cfg.train.eval_interval == 0 or step == cfg.train.max_steps:
                losses = self.estimate_loss()
                state.val_loss = losses["val"]
                improved = losses["val"] < self.best_val_loss
                if improved or cfg.train.always_save_checkpoint:
                    if improved:
                        self.best_val_loss = losses["val"]
                        state.best_val_loss = self.best_val_loss
                    self._save(step)
                log.info(
                    "trainer.eval",
                    step=step,
                    train_loss=round(losses["train"], 4),
                    val_loss=round(losses["val"], 4),
                    best=round(self.best_val_loss, 4),
                )

            # Forward/backward with gradient accumulation.
            running = 0.0
            for _ in range(grad_accum):
                x, y = self._batch(self.train_data)
                with self._ctx:
                    _, loss = self.model(x, y)
                    loss = loss / grad_accum
                self.scaler.scale(loss).backward()
                running += loss.item()

            if cfg.optim.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.optim.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            dt_ms = (time.perf_counter() - t0) * 1000
            state.step = step
            state.lr = lr
            state.train_loss = running
            state.tokens_seen += tokens_per_step
            state.dt_ms = dt_ms

            if step % cfg.train.log_interval == 0:
                log.info(
                    "trainer.step",
                    step=step,
                    loss=round(running, 4),
                    lr=round(lr, 6),
                    dt_ms=round(dt_ms, 1),
                )
                if progress_callback is not None:
                    progress_callback(state)

        log.info("trainer.done", steps=cfg.train.max_steps, best_val_loss=self.best_val_loss)
        return state
