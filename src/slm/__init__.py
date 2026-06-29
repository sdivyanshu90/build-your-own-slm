"""BYO-SLM: Build Your Own Small Language Model.

A compact, dependency-light, production-grade implementation of a GPT-style
decoder-only transformer — tokenizer, model, training loop, text generation,
and a serving API — built from first principles in PyTorch.

Public API re-exports keep import paths short and stable::

    from slm import GPT, ModelConfig, BPETokenizer, InferenceEngine
"""

from __future__ import annotations

from slm.config import (
    ExperimentConfig,
    ModelConfig,
    Settings,
    get_settings,
    load_experiment_config,
)

__version__ = "0.1.0"

# Names marked (lazy) are resolved on first access via __getattr__ to avoid
# importing torch at package import time.
__all__ = [
    "GPT",  # lazy
    "BPETokenizer",  # lazy
    "ExperimentConfig",
    "GenerationConfig",  # lazy
    "InferenceEngine",  # lazy
    "ModelConfig",
    "Settings",
    "__version__",
    "get_settings",
    "load_experiment_config",
]


def __getattr__(name: str) -> object:
    """Lazily resolve heavy (torch-dependent) symbols on first access.

    Importing ``slm`` should be cheap (it is imported by the CLI for ``--help``
    and by config-only tooling). Torch is only pulled in when a model/tokenizer
    symbol is actually requested.
    """
    if name == "GPT":
        from slm.model.gpt import GPT

        return GPT
    if name == "BPETokenizer":
        from slm.tokenizer import BPETokenizer

        return BPETokenizer
    if name == "InferenceEngine":
        from slm.inference.engine import InferenceEngine

        return InferenceEngine
    if name == "GenerationConfig":
        from slm.generation.sampler import GenerationConfig

        return GenerationConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
