"""FastAPI entrypoint exposing the DeepResearchAgent via HTTP."""

from __future__ import annotations

import sys
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Iterator

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from agent import DeepResearchAgent, _create_llm
from config import Configuration, SearchAPI
from services.logging_utils import (
    RequestIdFilter,
    get_request_id,
    new_request_id,
    reset_request_id,
    set_request_id,
    truncate,
)


class _LoguruHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        request_id = getattr(record, "request_id", "-")
        logger.bind(request_id=request_id).opt(depth=6, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )


def _configure_logging() -> None:
    handler = _LoguruHandler()
    handler.addFilter(RequestIdFilter())
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))


LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | rid={extra[request_id]} | "
    "{name}:{function}:{line} | {message}"
)


logger.remove()
logger.add(
    sys.stderr,
    level=os.getenv("LOG_LEVEL", "INFO"),
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <7}</level> | rid={extra[request_id]} | "
        "<cyan>{name}:{function}:{line}</cyan> | <level>{message}</level>"
    ),
    colorize=True,
    filter=lambda record: record["extra"].update({"request_id": record["extra"].get("request_id", get_request_id())}) or True,
)

log_file = os.getenv("LOG_FILE", "logs/app.log").strip()
if log_file:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path,
        level=os.getenv("LOG_LEVEL", "INFO"),
        format=LOG_FORMAT,
        rotation=os.getenv("LOG_ROTATION", "20 MB"),
        retention=os.getenv("LOG_RETENTION", "7 days"),
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=False,
        filter=lambda record: record["extra"].update({"request_id": record["extra"].get("request_id", get_request_id())}) or True,
    )
_configure_logging()


class ResearchRequest(BaseModel):
    """Payload for triggering a research run."""

    topic: str = Field(..., min_length=1, max_length=500, description="Research topic (1-500 chars)")
    search_api: SearchAPI | None = Field(
        default=None,
        description="Override the default search backend configured via env",
    )


class ResearchResponse(BaseModel):
    """HTTP response containing the generated report and structured tasks."""

    report_markdown: str = Field(
        ..., description="Markdown-formatted research report including sections"
    )
    todo_items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured TODO items with summaries and sources",
    )


def _mask_secret(value: str | None, visible: int = 4) -> str:
    """Mask sensitive tokens while keeping leading and trailing characters."""
    if not value:
        return "unset"

    if len(value) <= visible * 2:
        return "*" * len(value)

    return f"{value[:visible]}...{value[-visible:]}"


def _build_config(payload: ResearchRequest) -> Configuration:
    overrides: Dict[str, Any] = {}

    if payload.search_api is not None:
        overrides["search_api"] = payload.search_api

    return Configuration.from_env(overrides=overrides)


def _log_startup_configuration() -> None:
    config = Configuration.from_env()

    if config.llm_provider == "ollama":
        base_url = config.sanitized_ollama_url()
    elif config.llm_provider == "lmstudio":
        base_url = config.lmstudio_base_url
    else:
        base_url = config.llm_base_url or "unset"

    logger.info(
        "DeepResearch configuration loaded: provider={} model={} base_url={} search_api={} "
        "fetch_full_page={} strip_thinking={} enable_notes={} notes_workspace={} api_key={}",
        config.llm_provider,
        config.resolved_model() or "unset",
        base_url,
        (config.search_api.value if isinstance(config.search_api, SearchAPI) else config.search_api),
        config.fetch_full_page,
        config.strip_thinking_tokens,
        config.enable_notes,
        config.notes_workspace,
        _mask_secret(config.llm_api_key),
    )


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    _log_startup_configuration()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Deep Researcher", lifespan=lifespan)

    cors_origins = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    def health_check() -> Dict[str, Any]:
        request_id = new_request_id()
        token = set_request_id(request_id)
        llm_ok = False
        try:
            logger.info("healthz start")
            cfg = Configuration.from_env()
            llm = _create_llm(cfg)
            llm.invoke([HumanMessage(content="ping")])
            llm_ok = True
        except Exception as exc:
            logger.warning("healthz llm probe failed: {}", exc)
        finally:
            logger.info("healthz done llm={}", llm_ok)
            reset_request_id(token)
        return {"status": "ok", "llm": llm_ok, "request_id": request_id}

    @app.post("/research", response_model=ResearchResponse)
    def run_research(payload: ResearchRequest) -> ResearchResponse:
        request_id = new_request_id()
        token = set_request_id(request_id)
        try:
            logger.info(
                "research request start topic={} search_api={}",
                truncate(payload.topic),
                payload.search_api.value if payload.search_api else "default",
            )
            config = _build_config(payload)
            agent = DeepResearchAgent(config=config)
            result = agent.run(payload.topic)
        except ValueError as exc:  # Likely due to unsupported configuration
            logger.warning("research request rejected: {}", exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.exception("research request failed")
            raise HTTPException(status_code=500, detail="Research failed") from exc
        finally:
            reset_request_id(token)

        todo_payload = [
            {
                "id": item.id,
                "title": item.title,
                "intent": item.intent,
                "query": item.query,
                "status": item.status,
                "summary": item.summary,
                "sources_summary": item.sources_summary,
                "note_id": item.note_id,
                "note_path": item.note_path,
            }
            for item in result.todo_items
        ]

        return ResearchResponse(
            report_markdown=(result.report_markdown or result.running_summary or ""),
            todo_items=todo_payload,
        )

    @app.post("/research/stream")
    def stream_research(payload: ResearchRequest) -> StreamingResponse:
        request_id = new_request_id()
        token = set_request_id(request_id)
        try:
            logger.info(
                "stream request accepted topic={} search_api={}",
                truncate(payload.topic),
                payload.search_api.value if payload.search_api else "default",
            )
            config = _build_config(payload)
            agent = DeepResearchAgent(config=config)
        except ValueError as exc:
            reset_request_id(token)
            logger.warning("stream request rejected: {}", exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def event_iterator() -> Iterator[str]:
            set_request_id(request_id)
            event_count = 0
            try:
                for event in agent.run_stream(payload.topic):
                    event_count += 1
                    logger.debug(
                        "sse event #{} type={} task_id={}",
                        event_count,
                        event.get("type"),
                        event.get("task_id"),
                    )
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:  # pragma: no cover - defensive guardrail
                logger.exception("stream request failed events_sent={}", event_count)
                error_payload = {"type": "error", "detail": str(exc), "request_id": request_id}
                yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
            finally:
                logger.info("stream request finished events_sent={}", event_count)

        reset_request_id(token)
        return StreamingResponse(
            event_iterator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id,
            },
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
