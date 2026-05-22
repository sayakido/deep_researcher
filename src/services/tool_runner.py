"""Tool runner handling LLM tool call loops."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain_core.messages import ToolMessage


class ToolRunner:
    """Wraps an LLM with bound tools, handling tool call loops automatically."""

    def __init__(self, llm: Any, tools: list) -> None:
        self._llm = llm.bind_tools(tools)
        self._tool_map: dict[str, Any] = {t.name: t for t in tools}

    def run(self, messages: list) -> str:
        """Run LLM with tool support. Returns final text content."""
        max_turns = 6
        for _ in range(max_turns):
            response = self._llm.invoke(messages)
            tool_calls = getattr(response, "tool_calls", None)
            if not tool_calls:
                return response.content if hasattr(response, "content") else str(response)

            for tc in tool_calls:
                tool_fn = self._tool_map.get(tc["name"])
                if not tool_fn:
                    continue
                result = tool_fn.invoke(tc["args"])
                messages.append(response)
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        response = self._llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)

    def stream(self, messages: list) -> Iterator[str]:
        """Run LLM with tools, streaming final text after tool calls are resolved."""
        max_turns = 6
        for _ in range(max_turns):
            response = self._llm.invoke(messages)
            tool_calls = getattr(response, "tool_calls", None)
            if not tool_calls:
                for chunk in self._llm.stream(messages):
                    content = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if content:
                        yield content
                return

            for tc in tool_calls:
                tool_fn = self._tool_map.get(tc["name"])
                if not tool_fn:
                    continue
                result = tool_fn.invoke(tc["args"])
                messages.append(response)
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        for chunk in self._llm.stream(messages):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                yield content
