"""Shared fixtures for all tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from config import Configuration
from services.notes import NoteStore
from services.tool_runner import ToolRunner


# ------------------------------------------------------------------
# LLM mocks
# ------------------------------------------------------------------
class MockChatModel:
    """Mock LLM that returns predefined responses."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or ["mock response"]
        self._call_count = 0
        self.tool_responses: dict[str, list[dict]] = {}

    def bind_tools(self, tools: list) -> MockChatModel:
        return self

    def invoke(self, messages: list) -> AIMessage:
        if self._call_count < len(self.responses):
            resp = self.responses[self._call_count]
            self._call_count += 1
            return AIMessage(content=resp)
        return AIMessage(content=self.responses[-1])

    def stream(self, messages: list):
        resp = self.responses[min(self._call_count, len(self.responses) - 1)]
        self._call_count += 1
        for char in resp:
            yield AIMessageChunk(content=char)


@pytest.fixture
def mock_llm() -> MockChatModel:
    return MockChatModel()


@pytest.fixture
def note_store() -> NoteStore:
    tmp = tempfile.mkdtemp()
    return NoteStore(workspace=tmp)


@pytest.fixture
def empty_tool_runner(mock_llm: MockChatModel) -> ToolRunner:
    return ToolRunner(mock_llm, [])


@pytest.fixture
def config() -> Configuration:
    return Configuration()


@pytest.fixture
def sample_todo_json() -> str:
    return json.dumps({
        "tasks": [
            {
                "title": "背景调研",
                "intent": "了解主题的核心概念和发展历程",
                "query": "主题 背景 发展历程",
            },
            {
                "title": "现状分析",
                "intent": "分析当前的最新进展和关键技术",
                "query": "主题 最新进展 2026",
            },
        ]
    }, ensure_ascii=False)
