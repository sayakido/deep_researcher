"""Tests for planner response parsing."""

from __future__ import annotations

from config import Configuration
from services.planner import PlanningService


def test_extracts_tasks_from_markdown_table(empty_tool_runner):
    service = PlanningService(empty_tool_runner, Configuration())
    raw = """
## Task overview

| Task | Name | Engineering question |
|------|------|----------------------|
| 1 | **Power budget** | Estimate battery life and duty cycle |
| 2 | **MCU selection** | Compare low-power MCU candidates |
"""

    tasks = service._extract_tasks(raw)

    assert len(tasks) == 2
    assert tasks[0]["title"] == "Power budget"
    assert tasks[0]["intent"] == "Estimate battery life and duty cycle"
    assert "datasheet reference design" in tasks[0]["query"]


def test_extracts_tasks_from_nested_payload(empty_tool_runner):
    service = PlanningService(empty_tool_runner, Configuration())
    raw = '{"result": {"todo_items": [{"title": "MCU", "intent": "Pick MCU", "query": "MCU datasheet"}]}}'

    tasks = service._extract_tasks(raw)

    assert tasks == [{"title": "MCU", "intent": "Pick MCU", "query": "MCU datasheet"}]
