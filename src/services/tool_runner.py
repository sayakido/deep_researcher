"""Tool runner handling LLM tool call loops."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)


class ToolRunner:
    """Wraps an LLM with bound tools, handling tool call loops automatically."""

    def __init__(self, llm: Any, tools: list) -> None:
        self._llm = llm.bind_tools(tools)
        self._tool_map: dict[str, Any] = {t.name: t for t in tools}

    def run(self, messages: list) -> str:
        """Run LLM with tool support. Returns final text content."""
        max_turns = 6
        for turn in range(1, max_turns + 1):
            logger.debug("tool_runner run turn=%s messages=%s", turn, len(messages))
            response = self._llm.invoke(messages)
            tool_calls = getattr(response, "tool_calls", None)
            if not tool_calls:
                content = response.content if hasattr(response, "content") else str(response)
                logger.info(
                    "tool_runner run completed turn=%s content_chars=%s",
                    turn,
                    len(str(content or "")),
                )
                return content

            logger.info(
                "tool_runner run tool_calls turn=%s count=%s names=%s",
                turn,
                len(tool_calls),
                [tc.get("name") for tc in tool_calls],
            )
            messages.append(response)
            for tc in tool_calls:
                tool_fn = self._tool_map.get(tc["name"])
                if tool_fn:
                    logger.debug("tool_runner invoking tool=%s", tc["name"])
                    result = tool_fn.invoke(tc["args"])
                else:
                    logger.warning("tool_runner unknown tool=%s", tc["name"])
                    result = f"Unknown tool: {tc['name']}"
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        logger.warning("tool_runner run reached max_turns=%s", max_turns)
        response = self._llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    def stream(self, messages: list) -> Iterator[str]:
        """Run LLM with tools, streaming final text after tool calls are resolved."""
        max_turns = 6
        for turn in range(1, max_turns + 1):
            logger.debug("tool_runner stream turn=%s messages=%s", turn, len(messages))
            response = self._llm.invoke(messages)
            tool_calls = getattr(response, "tool_calls", None)
            if not tool_calls:
                chunk_count = 0
                char_count = 0
                for chunk in self._llm.stream(messages):
                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if content:
                        chunk_count += 1
                        char_count += len(content)
                        yield content
                logger.info(
                    "tool_runner stream completed turn=%s chunks=%s chars=%s",
                    turn,
                    chunk_count,
                    char_count,
                )
                return

            logger.info(
                "tool_runner stream tool_calls turn=%s count=%s names=%s",
                turn,
                len(tool_calls),
                [tc.get("name") for tc in tool_calls],
            )
            messages.append(response)
            for tc in tool_calls:
                tool_fn = self._tool_map.get(tc["name"])
                if tool_fn:
                    logger.debug("tool_runner invoking tool=%s", tc["name"])
                    result = tool_fn.invoke(tc["args"])
                else:
                    logger.warning("tool_runner unknown tool=%s", tc["name"])
                    result = f"Unknown tool: {tc['name']}"
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        logger.warning("tool_runner stream reached max_turns=%s", max_turns)
        chunk_count = 0
        char_count = 0
        for chunk in self._llm.stream(messages):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                chunk_count += 1
                char_count += len(content)
                yield content
        logger.info("tool_runner stream fallback completed chunks=%s chars=%s", chunk_count, char_count)
