"""Tests for the LangGraph agent graph and Send API routing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent import DeepResearchAgent
from models import LangGraphState, TodoItem
from services.tool_runner import ToolRunner


@pytest.fixture
def agent() -> DeepResearchAgent:
    with patch("agent._create_llm") as mock_create:
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content='{"tasks": []}')
        mock_create.return_value = mock_llm
        agent = DeepResearchAgent()
        return agent


class TestRouteTasks:
    def test_no_pending_routes_to_report(self, agent: DeepResearchAgent):
        state: LangGraphState = {
            "research_topic": "test",
            "todo_items": [
                TodoItem(id=1, title="t1", intent="i1", query="q1", status="completed"),
            ],
            "web_research_results": [],
            "sources_gathered": [],
            "research_loop_count": 1,
            "running_summary": "",
            "structured_report": None,
            "report_note_id": None,
            "report_note_path": None,
        }
        result = agent._route_tasks(state)
        assert result == "generate_report"

    def test_pending_returns_send_list(self, agent: DeepResearchAgent):
        state: LangGraphState = {
            "research_topic": "test",
            "todo_items": [
                TodoItem(id=1, title="t1", intent="i1", query="q1", status="pending"),
                TodoItem(id=2, title="t2", intent="i2", query="q2", status="pending"),
            ],
            "web_research_results": [],
            "sources_gathered": [],
            "research_loop_count": 0,
            "running_summary": "",
            "structured_report": None,
            "report_note_id": None,
            "report_note_path": None,
        }
        result = agent._route_tasks(state)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(s.node == "execute_one" for s in result)

    def test_mixed_status(self, agent: DeepResearchAgent):
        state: LangGraphState = {
            "research_topic": "test",
            "todo_items": [
                TodoItem(id=1, title="t1", intent="i1", query="q1", status="completed"),
                TodoItem(id=2, title="t2", intent="i2", query="q2", status="pending"),
                TodoItem(id=3, title="t3", intent="i3", query="q3", status="skipped"),
            ],
            "web_research_results": [],
            "sources_gathered": [],
            "research_loop_count": 1,
            "running_summary": "",
            "structured_report": None,
            "report_note_id": None,
            "report_note_path": None,
        }
        result = agent._route_tasks(state)
        assert isinstance(result, list)
        assert len(result) == 1  # only task 2 is pending
        assert result[0].node == "execute_one"


class TestExecuteOneTask:
    def test_empty_task_returns_empty(self, agent: DeepResearchAgent):
        state: LangGraphState = {
            "research_topic": "test",
            "todo_items": [],
            "web_research_results": [],
            "sources_gathered": [],
            "research_loop_count": 0,
            "running_summary": "",
            "structured_report": None,
            "report_note_id": None,
            "report_note_path": None,
        }
        result = agent._execute_one_task(state)
        assert result == {}

    def test_completed_task_returns_empty(self, agent: DeepResearchAgent):
        state: LangGraphState = {
            "research_topic": "test",
            "todo_items": [TodoItem(id=1, title="t", intent="i", query="q", status="completed")],
            "web_research_results": [],
            "sources_gathered": [],
            "research_loop_count": 0,
            "running_summary": "",
            "structured_report": None,
            "report_note_id": None,
            "report_note_path": None,
        }
        result = agent._execute_one_task(state)
        assert result == {}


class TestPlanTasksNode:
    def test_creates_todo_items(self, agent: DeepResearchAgent):
        state: LangGraphState = {
            "research_topic": "test",
            "todo_items": [],
            "web_research_results": [],
            "sources_gathered": [],
            "research_loop_count": 0,
            "running_summary": "",
            "structured_report": None,
            "report_note_id": None,
            "report_note_path": None,
        }
        # Mock planner to return items
        agent.planner.plan_todo_list = MagicMock(return_value=[
            TodoItem(id=1, title="背景调研", intent="了解背景", query="背景"),
        ])
        result = agent._plan_tasks_node(state)
        assert "todo_items" in result
        assert len(result["todo_items"]) == 1
        assert result["todo_items"][0].title == "背景调研"

    def test_fallback_on_empty(self, agent: DeepResearchAgent):
        state: LangGraphState = {
            "research_topic": "test",
            "todo_items": [],
            "web_research_results": [],
            "sources_gathered": [],
            "research_loop_count": 0,
            "running_summary": "",
            "structured_report": None,
            "report_note_id": None,
            "report_note_path": None,
        }
        agent.planner.plan_todo_list = MagicMock(return_value=[])
        result = agent._plan_tasks_node(state)
        assert len(result["todo_items"]) == 1  # fallback task
        assert result["todo_items"][0].id == 1
        assert result["todo_items"][0].title == "需求与架构初设"
        assert "hardware architecture" in result["todo_items"][0].query


class TestRunStream:
    def test_yields_direct_custom_events(self, agent: DeepResearchAgent):
        agent.graph.stream = MagicMock(return_value=iter([
            {"type": "todo_list", "tasks": []},
            {"type": "done"},
        ]))

        events = list(agent.run_stream("test"))

        assert events[0]["type"] == "status"
        assert events[1] == {"type": "todo_list", "tasks": []}
        assert events[2] == {"type": "done"}

    def test_yields_wrapped_custom_events(self, agent: DeepResearchAgent):
        agent.graph.stream = MagicMock(return_value=iter([
            {"type": "custom", "data": {"type": "done"}},
        ]))

        events = list(agent.run_stream("test"))

        assert events[0]["type"] == "status"
        assert events[1] == {"type": "done"}
