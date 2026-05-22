"""Tests for ToolRunner."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from services.tool_runner import ToolRunner


def test_run_returns_text(mock_llm):
    mock_llm.responses = ["hello world"]
    runner = ToolRunner(mock_llm, [])
    result = runner.run([HumanMessage(content="hi")])
    assert result == "hello world"


def test_stream_returns_chunks(mock_llm):
    mock_llm.responses = ["abc"]
    runner = ToolRunner(mock_llm, [])
    chunks = list(runner.stream([HumanMessage(content="hi")]))
    assert len(chunks) > 0
    assert "".join(chunks) == "abc"


def test_tool_call_executed():
    call_log = []

    @tool
    def my_tool(name: str) -> str:
        """A test tool that greets a name."""
        call_log.append(name)
        return f"hello {name}"

    llm = MagicMock()
    llm.bind_tools.return_value = llm

    # First call returns a tool call
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "my_tool", "args": {"name": "world"}, "id": "call_1"}],
    )
    # Second call returns final text
    final_msg = AIMessage(content="done with tools")

    llm.invoke.side_effect = [tool_call_msg, final_msg]

    runner = ToolRunner(llm, [my_tool])
    result = runner.run([HumanMessage(content="do it")])

    assert result == "done with tools"
    assert call_log == ["world"]


def test_max_turns_limit():
    @tool
    def loop_tool() -> str:
        """A test tool that always triggers again."""
        return "again"

    llm = MagicMock()
    llm.bind_tools.return_value = llm

    # Always returns a tool call (infinite loop scenario)
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "loop_tool", "args": {}, "id": "call_1"}],
    )
    llm.invoke.return_value = tool_call_msg

    runner = ToolRunner(llm, [loop_tool])
    result = runner.run([HumanMessage(content="loop")])

    # max_turns=6 iterations inside loop + 1 fallback call after loop
    assert result == ""
    assert llm.invoke.call_count == 7
