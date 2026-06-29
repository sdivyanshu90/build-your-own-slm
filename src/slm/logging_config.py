"""Centralised structured logging configuration built on ``structlog``.

The same call site (``log = get_logger(__name__)``) produces human-friendly,
coloured console output in development and machine-parseable JSON in production,
selected purely by configuration. Request-scoped context (request id, api key
id) is bound via ``structlog.contextvars`` so it is automatically attached to
every log line emitted while handling a request.
"""

from __future__ import annotations

import logging
import sys

import structlog

__all__ = ["configure_logging", "get_logger"]

_CONFIGURED = False


def configure_logging(*, level: str = "INFO", json_logs: bool = False) -> None:
    """Configure stdlib logging and structlog once per process.

    Args:
        level: Minimum level name (e.g. ``"INFO"``).
        json_logs: Emit newline-delimited JSON when ``True``; otherwise a
            coloured, human-readable console renderer is used.
    """
    global _CONFIGURED

    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        timestamper,
    ]

    if json_logs:
        renderer: structlog.typing.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging (uvicorn, etc.) through the same renderer.
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=numeric_level)
    for noisy in ("uvicorn.access",):
        logging.getLogger(noisy).handlers.clear()

    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger, configuring defaults lazily if needed."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)
