"""HTTP route handlers."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from slm.api.schemas import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    HealthResponse,
    ModelInfo,
    ModelList,
    Usage,
)
from slm.api.security import rate_limit_dependency
from slm.inference.engine import InferenceEngine

__all__ = ["health_router", "v1_router"]

health_router = APIRouter(tags=["health"])
v1_router = APIRouter(prefix="/v1", tags=["inference"])

_MODEL_ID = "byo-slm"


def _engine(request: Request) -> InferenceEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:  # pragma: no cover - defensive; readiness gate prevents this
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded."
        )
    return engine


# ---------------------------------------------------------------------------
# Health & observability (unauthenticated)
# ---------------------------------------------------------------------------
@health_router.get("/healthz", response_model=HealthResponse, summary="Liveness probe")
async def healthz(request: Request) -> HealthResponse:
    """Liveness: the process is up and serving. Always ``ok`` if reachable."""
    from slm import __version__

    return HealthResponse(
        status="ok",
        version=__version__,
        model_loaded=getattr(request.app.state, "engine", None) is not None,
    )


@health_router.get("/readyz", response_model=HealthResponse, summary="Readiness probe")
async def readyz(request: Request, response: Response) -> HealthResponse:
    """Readiness: returns ``503`` until the model is loaded and able to serve."""
    from fastapi import status

    from slm import __version__

    loaded = getattr(request.app.state, "engine", None) is not None
    if not loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if loaded else "starting",
        version=__version__,
        model_loaded=loaded,
    )


@health_router.get("/metrics", summary="Prometheus metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    data = generate_latest(request.app.state.metrics.registry)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Inference (authenticated + rate limited)
# ---------------------------------------------------------------------------
@v1_router.get("/models", response_model=ModelList, summary="List available models")
async def list_models(
    request: Request, _identity: str = Depends(rate_limit_dependency)
) -> ModelList:
    meta = _engine(request).metadata
    return ModelList(
        data=[
            ModelInfo(
                id=_MODEL_ID,
                parameters=meta.num_parameters,
                parameters_human=meta.num_parameters_human,
                context_length=meta.block_size,
                n_layer=meta.n_layer,
                n_head=meta.n_head,
                n_embd=meta.n_embd,
                device=meta.device,
            )
        ]
    )


@v1_router.post(
    "/completions",
    response_model=CompletionResponse,
    summary="Create a text completion",
    responses={
        200: {"description": "Completion (JSON) or an SSE token stream."},
        401: {"description": "Missing or invalid API key."},
        422: {"description": "Validation error."},
        429: {"description": "Rate limit exceeded."},
    },
)
async def create_completion(
    request: Request,
    body: CompletionRequest,
    _identity: str = Depends(rate_limit_dependency),
) -> Response:
    """Generate a completion, optionally streaming tokens over SSE."""
    engine = _engine(request)
    gen_config = body.to_generation_config()
    completion_id = f"cmpl-{uuid.uuid4().hex}"

    if body.stream:
        return EventSourceResponse(
            _stream_completion(request, engine, body, gen_config, completion_id)
        )

    result = await run_in_threadpool(engine.generate, body.prompt, gen_config, stop=body.stop)
    request.app.state.metrics.tokens_generated.inc(result.completion_tokens)
    return _json_completion(completion_id, result)


def _json_completion(completion_id: str, result: object) -> Response:
    from slm.inference.engine import GenerationResult

    assert isinstance(result, GenerationResult)
    payload = CompletionResponse(
        id=completion_id,
        model=_MODEL_ID,
        choices=[CompletionChoice(index=0, text=result.text, finish_reason=result.finish_reason)],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )
    return Response(content=payload.model_dump_json(), media_type="application/json")


async def _stream_completion(
    request: Request,
    engine: InferenceEngine,
    body: CompletionRequest,
    gen_config: object,
    completion_id: str,
) -> AsyncIterator[dict[str, str]]:
    """Yield Server-Sent Events, one per decoded text chunk, then ``[DONE]``."""
    from slm.generation.sampler import GenerationConfig

    assert isinstance(gen_config, GenerationConfig)
    sync_iter = engine.stream(body.prompt, gen_config, stop=body.stop)
    chunks = 0
    async for chunk in iterate_in_threadpool(sync_iter):
        if await request.is_disconnected():
            break
        chunks += 1
        event = {
            "id": completion_id,
            "object": "text_completion.chunk",
            "model": _MODEL_ID,
            "choices": [{"index": 0, "text": chunk, "finish_reason": None}],
        }
        yield {"data": json.dumps(event)}
    request.app.state.metrics.tokens_generated.inc(chunks)
    final = {
        "id": completion_id,
        "object": "text_completion.chunk",
        "model": _MODEL_ID,
        "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
    }
    yield {"data": json.dumps(final)}
    yield {"data": "[DONE]"}
