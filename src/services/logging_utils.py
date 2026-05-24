"""Logging helpers for request-scoped diagnostics."""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Iterator
from uuid import uuid4


request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def new_request_id() -> str:
    """Create a short request id for correlating logs."""
    return uuid4().hex[:10]


def get_request_id() -> str:
    """Return the current request id."""
    return request_id_var.get()


def set_request_id(request_id: str):
    """Set the current request id and return the context token."""
    return request_id_var.set(request_id)


def reset_request_id(token) -> None:
    """Reset request id context from a token."""
    request_id_var.reset(token)


class RequestIdFilter(logging.Filter):
    """Inject request_id into stdlib log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


def truncate(value: object, limit: int = 240) -> str:
    """Return a compact single-line representation for logs."""
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


@contextmanager
def log_duration(logger: logging.Logger, label: str, **fields: object) -> Iterator[None]:
    """Log start/end/failure with elapsed milliseconds."""
    start = time.perf_counter()
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("%s start %s", label, field_text)
    try:
        yield
    except Exception:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.exception("%s failed elapsed_ms=%s %s", label, elapsed_ms, field_text)
        raise
    else:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info("%s done elapsed_ms=%s %s", label, elapsed_ms, field_text)
