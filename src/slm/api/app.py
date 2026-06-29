"""FastAPI application factory.

``create_app`` wires together settings, the inference engine, middleware,
routers and exception handlers. It is deliberately a *factory* (not a module
global) so tests can construct isolated apps with injected dependencies.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from slm import __version__
from slm.api.metrics import Metrics
from slm.api.middleware import RequestContextMiddleware, SecurityHeadersMiddleware
from slm.api.routes import health_router, v1_router
from slm.api.schemas import ErrorBody, ErrorResponse
from slm.api.security import RateLimiter
from slm.config import Settings, get_settings
from slm.inference.engine import InferenceEngine
from slm.logging_config import configure_logging, get_logger

__all__ = ["create_app"]

log = get_logger(__name__)


def _error_response(status_code: int, type_: str, message: str, request: Request) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    body = ErrorResponse(error=ErrorBody(type=type_, message=message, request_id=request_id))
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _load_engine(settings: Settings) -> InferenceEngine | None:
    model_path = settings.model_dir / "model.pt"
    if not model_path.exists():
        log.warning("engine.not_found", model_dir=str(settings.model_dir))
        return None
    try:
        return InferenceEngine.from_pretrained(
            settings.model_dir,
            device=settings.device,
            max_new_tokens_limit=settings.max_new_tokens,
            max_prompt_tokens=settings.max_prompt_tokens,
        )
    except Exception:  # pragma: no cover - surfaced via readiness probe
        log.exception("engine.load_failed", model_dir=str(settings.model_dir))
        return None


def create_app(
    settings: Settings | None = None,
    *,
    engine: InferenceEngine | None = None,
    load_model: bool = True,
) -> FastAPI:
    """Create and configure a FastAPI application.

    Args:
        settings: Runtime settings (defaults to environment-derived settings).
        engine: A pre-built inference engine (skips disk loading; used in tests).
        load_model: When ``True`` and no ``engine`` is supplied, the model is
            loaded from ``settings.model_dir`` at startup.
    """
    settings = settings or get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.engine is None and load_model:
            app.state.engine = _load_engine(settings)
        log.info(
            "api.startup",
            env=settings.env,
            model_loaded=app.state.engine is not None,
            auth_enabled=settings.auth_enabled,
        )
        yield
        log.info("api.shutdown")

    app = FastAPI(
        title="BYO-SLM Inference API",
        version=__version__,
        description="Serve a from-scratch GPT-style Small Language Model.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.state.settings = settings
    app.state.metrics = Metrics.create()
    app.state.rate_limiter = RateLimiter(
        rate_per_minute=settings.rate_limit_per_minute, burst=settings.rate_limit_burst
    )
    app.state.engine = engine

    # Middleware is applied bottom-up: security headers (outermost) wraps the
    # request-context/metrics middleware (innermost).
    app.add_middleware(RequestContextMiddleware, metrics=app.state.metrics)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
    )

    app.include_router(health_router)
    app.include_router(v1_router)

    # ---- Exception handlers: consistent error envelope --------------------
    @app.exception_handler(StarletteHTTPException)
    async def _http_exc(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        response = _error_response(exc.status_code, "http_error", str(exc.detail), request)
        if exc.headers:
            response.headers.update(exc.headers)
        return response

    @app.exception_handler(RequestValidationError)
    async def _validation_exc(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(422, "validation_error", str(exc.errors()), request)

    @app.exception_handler(ValueError)
    async def _value_exc(request: Request, exc: ValueError) -> JSONResponse:
        return _error_response(400, "invalid_request", str(exc), request)

    @app.exception_handler(Exception)
    async def _unhandled_exc(request: Request, exc: Exception) -> JSONResponse:
        log.exception("api.unhandled_exception")
        message = "Internal server error." if settings.is_production else str(exc)
        return _error_response(500, "internal_error", message, request)

    return app
