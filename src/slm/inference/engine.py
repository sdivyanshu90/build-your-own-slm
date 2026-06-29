"""High-level inference engine.

Wraps a trained model + tokenizer behind a small, safe API used by both the CLI
and the HTTP service. Responsibilities:

* Load a model directory (``model.pt`` + ``tokenizer.json``).
* Enforce input limits (prompt length, max new tokens).
* Provide both buffered (:meth:`generate`) and streaming (:meth:`stream`)
  decoding, with correct incremental UTF-8 handling for multi-byte characters.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from slm.generation.sampler import GenerationConfig, generate_tokens
from slm.logging_config import get_logger
from slm.model.gpt import GPT
from slm.tokenizer import BPETokenizer
from slm.training.checkpoint import build_model, load_checkpoint
from slm.utils import count_parameters, human_readable_count, resolve_device

__all__ = ["GenerationResult", "InferenceEngine", "ModelMetadata"]

log = get_logger(__name__)


@dataclass(frozen=True)
class ModelMetadata:
    """Static information describing the loaded model."""

    model_dir: str
    device: str
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    num_parameters: int
    num_parameters_human: str


@dataclass(frozen=True)
class GenerationResult:
    """The outcome of a buffered generation call."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str  # "stop" | "length"


def _incremental_decode(buffer: bytearray) -> tuple[str, int]:
    """Decode the maximal valid UTF-8 prefix of ``buffer``.

    Returns the decoded text and the number of bytes consumed, leaving any
    trailing incomplete multi-byte sequence in the buffer for the next token.
    Genuinely invalid leading bytes are emitted as the replacement character so
    streaming never deadlocks.
    """
    if not buffer:
        return "", 0
    try:
        return bytes(buffer).decode("utf-8"), len(buffer)
    except UnicodeDecodeError as exc:
        if exc.start > 0:
            return bytes(buffer[: exc.start]).decode("utf-8"), exc.start
        if "unexpected end of data" in str(exc.reason):
            return "", 0  # incomplete tail; wait for more bytes
        return "�", 1  # invalid start byte; consume one and continue


class InferenceEngine:
    """Thread-safe-by-construction text generation over a single model."""

    def __init__(
        self,
        model: GPT,
        tokenizer: BPETokenizer,
        *,
        device: torch.device,
        model_dir: str = "",
        max_new_tokens_limit: int = 1024,
        max_prompt_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens_limit = max_new_tokens_limit
        self.max_prompt_tokens = max_prompt_tokens
        self.model.eval()

        self.metadata = ModelMetadata(
            model_dir=model_dir,
            device=str(device),
            vocab_size=model.config.vocab_size,
            block_size=model.config.block_size,
            n_layer=model.config.n_layer,
            n_head=model.config.n_head,
            n_embd=model.config.n_embd,
            num_parameters=count_parameters(model, trainable_only=False),
            num_parameters_human=human_readable_count(
                count_parameters(model, trainable_only=False)
            ),
        )

    # -- construction -------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        device: str = "auto",
        max_new_tokens_limit: int = 1024,
        max_prompt_tokens: int = 1024,
    ) -> InferenceEngine:
        """Load an engine from a directory containing ``model.pt`` + tokenizer."""
        model_dir = Path(model_dir)
        resolved = resolve_device(device)
        ckpt = load_checkpoint(model_dir / "model.pt", map_location=resolved)
        model = build_model(ckpt, device=resolved)
        tokenizer = BPETokenizer.load(model_dir / "tokenizer.json")
        log.info(
            "engine.loaded",
            model_dir=str(model_dir),
            device=str(resolved),
            params=human_readable_count(count_parameters(model, trainable_only=False)),
        )
        return cls(
            model,
            tokenizer,
            device=resolved,
            model_dir=str(model_dir),
            max_new_tokens_limit=max_new_tokens_limit,
            max_prompt_tokens=max_prompt_tokens,
        )

    # -- helpers ------------------------------------------------------------
    def _prepare(
        self, prompt: str, config: GenerationConfig
    ) -> tuple[torch.Tensor, GenerationConfig]:
        ids = self.tokenizer.encode(prompt)
        if len(ids) > self.max_prompt_tokens:
            raise ValueError(
                f"Prompt has {len(ids)} tokens, exceeding the limit of {self.max_prompt_tokens}."
            )
        if len(ids) >= self.model.config.block_size:
            raise ValueError(
                f"Prompt has {len(ids)} tokens, which leaves no room within the "
                f"model context window of {self.model.config.block_size}."
            )
        if not ids:
            # Seed with the end-of-text token (or token 0) for unconditional gen.
            ids = [self.tokenizer.eot_id if self.tokenizer.eot_id is not None else 0]
        if config.max_new_tokens > self.max_new_tokens_limit:
            config = config.model_copy(update={"max_new_tokens": self.max_new_tokens_limit})
        prompt_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        return prompt_tensor, config

    def _make_generator(self, config: GenerationConfig) -> torch.Generator | None:
        if config.seed is None:
            return None
        return torch.Generator(device="cpu").manual_seed(config.seed)

    def _stop_ids(self) -> frozenset[int]:
        return (
            frozenset({self.tokenizer.eot_id}) if self.tokenizer.eot_id is not None else frozenset()
        )

    @staticmethod
    def _truncate_at_stop(text: str, stop: Sequence[str] | None) -> tuple[str, bool]:
        if not stop:
            return text, False
        cut = len(text)
        hit = False
        for s in stop:
            if not s:
                continue
            idx = text.find(s)
            if idx != -1 and idx < cut:
                cut, hit = idx, True
        return text[:cut], hit

    # -- generation ---------------------------------------------------------
    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        *,
        stop: Sequence[str] | None = None,
    ) -> GenerationResult:
        """Generate a completion and return it as a single buffered result."""
        config = config or GenerationConfig()
        prompt_tensor, config = self._prepare(prompt, config)
        generator = self._make_generator(config)

        produced = 0
        buffer = bytearray()
        text_parts: list[str] = []
        for token in generate_tokens(
            self.model, prompt_tensor, config, stop_ids=self._stop_ids(), generator=generator
        ):
            produced += 1
            buffer += self.tokenizer.id_to_bytes(token)
            chunk, consumed = _incremental_decode(buffer)
            if consumed:
                del buffer[:consumed]
                text_parts.append(chunk)
            current = "".join(text_parts)
            trimmed, hit = self._truncate_at_stop(current, stop)
            if hit:
                return GenerationResult(
                    text=trimmed,
                    prompt_tokens=int(prompt_tensor.size(1)),
                    completion_tokens=produced,
                    finish_reason="stop",
                )

        # Flush any remaining bytes (replacing invalid trailing data).
        if buffer:
            text_parts.append(bytes(buffer).decode("utf-8", errors="replace"))
        text = "".join(text_parts)
        text, _ = self._truncate_at_stop(text, stop)
        finish = "length" if produced >= config.max_new_tokens else "stop"
        return GenerationResult(
            text=text,
            prompt_tokens=int(prompt_tensor.size(1)),
            completion_tokens=produced,
            finish_reason=finish,
        )

    def stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        *,
        stop: Sequence[str] | None = None,
    ) -> Iterator[str]:
        """Yield decoded text chunks incrementally as tokens are produced."""
        config = config or GenerationConfig()
        prompt_tensor, config = self._prepare(prompt, config)
        generator = self._make_generator(config)

        buffer = bytearray()
        emitted = ""
        for token in generate_tokens(
            self.model, prompt_tensor, config, stop_ids=self._stop_ids(), generator=generator
        ):
            buffer += self.tokenizer.id_to_bytes(token)
            chunk, consumed = _incremental_decode(buffer)
            if not consumed:
                continue
            del buffer[:consumed]
            emitted += chunk
            trimmed, hit = self._truncate_at_stop(emitted, stop)
            if hit:
                remaining = trimmed[len(emitted) - len(chunk) :]
                if remaining:
                    yield remaining
                return
            yield chunk
        if buffer:
            tail = bytes(buffer).decode("utf-8", errors="replace")
            trimmed, _ = self._truncate_at_stop(emitted + tail, stop)
            extra = trimmed[len(emitted) :]
            if extra:
                yield extra
