"""Configuration models for BYO-SLM.

Two distinct configuration surfaces live here:

* :class:`Settings` — *runtime* configuration sourced from environment variables
  (Twelve-Factor App, factor III). Consumed by the inference engine and API.
* :class:`ExperimentConfig` and friends — *experiment* configuration sourced from
  YAML files under ``configs/``. Consumed by data preparation and training.

Keeping the two apart means an operator can change serving behaviour (device,
rate limits, API keys) without touching the immutable recipe that produced a
checkpoint, and a researcher can version experiment recipes in git without
leaking secrets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "ModelConfig",
    "OptimConfig",
    "Settings",
    "TokenizerConfig",
    "TrainConfig",
    "get_settings",
    "load_experiment_config",
]

Environment = Literal["development", "staging", "production"]
DeviceChoice = Literal["auto", "cpu", "cuda", "mps"]


# ===========================================================================
# Runtime settings (environment driven)
# ===========================================================================
class Settings(BaseSettings):
    """Process-wide runtime settings, populated from the environment.

    Every field maps to an ``SLM_``-prefixed environment variable (see
    ``.env.example``). Instances are immutable once constructed.
    """

    model_config = SettingsConfigDict(
        env_prefix="SLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # Runtime
    env: Environment = "development"
    log_level: str = "INFO"
    log_json: bool = False

    # Model / inference
    model_dir: Path = Path("./checkpoints/tiny")
    device: DeviceChoice = "auto"
    max_new_tokens: int = Field(default=256, ge=1, le=8192)
    max_prompt_tokens: int = Field(default=1024, ge=1, le=131072)

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_workers: int = Field(default=1, ge=1, le=64)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # Security
    api_keys: frozenset[str] = Field(default_factory=frozenset)
    rate_limit_per_minute: int = Field(default=60, ge=1)
    rate_limit_burst: int = Field(default=10, ge=1)

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        level = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return level

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("api_keys", mode="before")
    @classmethod
    def _split_api_keys(cls, value: object) -> object:
        if isinstance(value, str):
            return frozenset(key.strip() for key in value.split(",") if key.strip())
        if isinstance(value, (list, set, tuple)):
            return frozenset(str(v).strip() for v in value if str(v).strip())
        return value

    @property
    def auth_enabled(self) -> bool:
        """Authentication is enforced only when at least one API key is set."""
        return len(self.api_keys) > 0

    @property
    def is_production(self) -> bool:
        return self.env == "production"


def get_settings() -> Settings:
    """Return a fresh :class:`Settings` instance read from the environment.

    Intentionally *not* cached: tests construct settings with patched env vars,
    and the API wires a single instance into application state at startup.
    """
    return Settings()


# ===========================================================================
# Experiment configuration (YAML driven)
# ===========================================================================
class ModelConfig(BaseModel):
    """Architecture hyper-parameters defining a concrete GPT instance."""

    model_config = {"frozen": True, "extra": "forbid"}

    vocab_size: int = Field(default=8192, ge=1, description="Token vocabulary size.")
    block_size: int = Field(default=256, ge=1, description="Maximum context length (tokens).")
    n_layer: int = Field(default=6, ge=1, description="Number of transformer blocks.")
    n_head: int = Field(default=6, ge=1, description="Attention heads per block.")
    n_embd: int = Field(default=384, ge=1, description="Embedding / residual stream width.")
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    bias: bool = Field(default=False, description="Use bias terms in Linear/LayerNorm.")

    @model_validator(mode="after")
    def _check_divisible(self) -> ModelConfig:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head}).")
        return self

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


class OptimConfig(BaseModel):
    """Optimiser and learning-rate schedule hyper-parameters."""

    model_config = {"frozen": True, "extra": "forbid"}

    lr: float = Field(default=6e-4, gt=0.0)
    weight_decay: float = Field(default=0.1, ge=0.0)
    beta1: float = Field(default=0.9, ge=0.0, lt=1.0)
    beta2: float = Field(default=0.95, ge=0.0, lt=1.0)
    grad_clip: float = Field(default=1.0, ge=0.0, description="0 disables clipping.")
    decay_lr: bool = True
    warmup_steps: int = Field(default=100, ge=0)
    lr_decay_steps: int = Field(default=5000, ge=1)
    min_lr: float = Field(default=6e-5, ge=0.0)


class DataConfig(BaseModel):
    """Where raw text lives and where tokenised binaries are written."""

    model_config = {"frozen": True, "extra": "forbid"}

    data_dir: Path = Path("./data")
    # Either a URL to download or a local path; resolved by prepare-data.
    source_url: str | None = None
    raw_file: str = "input.txt"
    train_bin: str = "train.bin"
    val_bin: str = "val.bin"
    val_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)


class TokenizerConfig(BaseModel):
    """Byte-level BPE tokenizer training parameters."""

    model_config = {"frozen": True, "extra": "forbid"}

    vocab_size: int = Field(default=8192, ge=256)
    path: Path = Path("./checkpoints/tiny/tokenizer.json")
    special_tokens: list[str] = Field(default_factory=lambda: ["<|endoftext|>"])


class TrainConfig(BaseModel):
    """Training loop control: budget, batching, evaluation, checkpoints."""

    model_config = {"frozen": True, "extra": "forbid"}

    out_dir: Path = Path("./checkpoints/tiny")
    max_steps: int = Field(default=2000, ge=1)
    batch_size: int = Field(default=32, ge=1)
    grad_accum_steps: int = Field(default=1, ge=1)
    eval_interval: int = Field(default=250, ge=1)
    eval_steps: int = Field(default=100, ge=1)
    log_interval: int = Field(default=10, ge=1)
    seed: int = 1337
    device: DeviceChoice = "auto"
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    compile: bool = False
    gradient_checkpointing: bool = False
    always_save_checkpoint: bool = False


class ExperimentConfig(BaseModel):
    """Top-level recipe binding all sub-configs together."""

    model_config = {"frozen": True, "extra": "forbid"}

    name: str = "tiny"
    model: ModelConfig = Field(default_factory=ModelConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)

    @model_validator(mode="after")
    def _sync_vocab(self) -> ExperimentConfig:
        # The tokenizer's vocab size is authoritative; keep the model in sync so
        # the recipe is internally consistent before training begins.
        if self.model.vocab_size != self.tokenizer.vocab_size:
            object.__setattr__(
                self,
                "model",
                self.model.model_copy(update={"vocab_size": self.tokenizer.vocab_size}),
            )
        return self


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an :class:`ExperimentConfig` from a YAML file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}: {path}")
    return ExperimentConfig.model_validate(raw)
