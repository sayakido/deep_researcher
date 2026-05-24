"""Service responsible for converting the research topic into actionable tasks."""

from __future__ import annotations

import json
import logging
from typing import Any, List

from langchain_core.messages import HumanMessage, SystemMessage

from config import Configuration
from models import TodoItem
from prompts import (
    get_current_date,
    todo_planner_instructions,
    todo_planner_system_prompt,
)
from services.tool_runner import ToolRunner
from utils import strip_thinking_tokens

logger = logging.getLogger(__name__)


class PlanningService:
    """Wraps the planner LLM to produce structured TODO items."""

    def __init__(self, tool_runner: ToolRunner, config: Configuration) -> None:
        self._tool_runner = tool_runner
        self._config = config

    def plan_todo_list(self, topic: str) -> List[TodoItem]:
        """Ask the planner to break the topic into actionable tasks."""
        prompt = todo_planner_instructions.format(
            current_date=get_current_date(),
            research_topic=topic,
        )

        messages = [
            SystemMessage(content=todo_planner_system_prompt.strip()),
            HumanMessage(content=prompt),
        ]

        content = self._tool_runner.run(messages)

        logger.info("Planner raw output (truncated): %s", content[:500])

        tasks_payload = self._extract_tasks(content)
        todo_items: List[TodoItem] = []

        for idx, item in enumerate(tasks_payload, start=1):
            title = str(item.get("title") or f"任务{idx}").strip()
            intent = str(item.get("intent") or "聚焦主题的关键问题").strip()
            query = str(item.get("query") or topic).strip()

            if not query:
                query = topic

            task = TodoItem(id=idx, title=title, intent=intent, query=query)
            todo_items.append(task)

        titles = [task.title for task in todo_items]
        logger.info("Planner produced %d tasks: %s", len(todo_items), titles)
        return todo_items

    @staticmethod
    def create_fallback_task(topic: str) -> TodoItem:
        """Create a minimal fallback task when planning failed."""
        return TodoItem(
            id=1,
            title="需求与架构初设",
            intent="梳理产品需求、关键约束和初步系统架构，为后续核心器件选型建立边界条件。",
            query=(
                f"{topic} hardware architecture component selection reference design"
                if topic
                else "hardware architecture component selection reference design"
            ),
        )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _extract_tasks(self, raw_response: str) -> List[dict[str, Any]]:
        """Parse planner output into a list of task dictionaries."""
        text = raw_response.strip()
        if self._config.strip_thinking_tokens:
            text = strip_thinking_tokens(text)

        json_payload = self._extract_json_payload(text)
        tasks: List[dict[str, Any]] = []

        if isinstance(json_payload, dict):
            candidate = json_payload.get("tasks")
            if isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, dict):
                        tasks.append(item)
        elif isinstance(json_payload, list):
            for item in json_payload:
                if isinstance(item, dict):
                    tasks.append(item)

        return tasks

    def _extract_json_payload(self, text: str) -> dict[str, Any] | list | None:
        """Try to locate and parse a JSON object or array from the text."""
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None

        return None
