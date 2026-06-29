"""Pydantic request/response models for the HTTP API.

The completion shape is intentionally close to the OpenAI ``/v1/completions``
contract so existing client tooling and mental models transfer directly.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from slm.generation.sampler import GenerationConfig

__all__ = [
    "CompletionChoice",
    "CompletionRequest",
    "CompletionResponse",
    "ErrorBody",
    "ErrorResponse",
    "HealthResponse",
    "ModelInfo",
    "ModelList",
    "Usage",
]


class CompletionRequest(BaseModel):
    """A text-completion request."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=0, max_length=100_000, description="Prompt text.")
    max_tokens: int = Field(default=128, ge=1, le=8192, description="Max tokens to generate.")
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    top_k: int | None = Field(default=40, ge=1)
    top_p: float | None = Field(default=0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0)
    seed: int | None = Field(default=None, description="Set for reproducible sampling.")
    stop: list[str] | None = Field(
        default=None, max_length=8, description="Up to 8 strings that halt generation."
    )
    stream: bool = Field(default=False, description="Stream tokens via SSE when true.")

    def to_generation_config(self) -> GenerationConfig:
        """Project the request onto the internal :class:`GenerationConfig`."""
        return GenerationConfig(
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            seed=self.seed,
        )


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: Literal["stop", "length"] | None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    model: str
    choices: list[CompletionChoice]
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    parameters: int
    parameters_human: str
    context_length: int
    n_layer: int
    n_head: int
    n_embd: int
    device: str


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    status: Literal["ok", "starting", "unavailable"]
    version: str
    model_loaded: bool


class ErrorBody(BaseModel):
    type: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
