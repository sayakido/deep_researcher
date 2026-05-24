"""Service responsible for converting the research topic into actionable tasks."""

from __future__ import annotations

import json
import logging
import re
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
from services.logging_utils import log_duration, truncate
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

        logger.info("Planner request topic=%s", truncate(topic))
        with log_duration(logger, "planner_llm"):
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
        if not todo_items:
            logger.warning("Planner produced no valid tasks")
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
            candidate = self._find_task_list(json_payload)
            if isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, dict):
                        tasks.append(item)
        elif isinstance(json_payload, list):
            for item in json_payload:
                if isinstance(item, dict):
                    tasks.append(item)

        if tasks:
            return tasks

        markdown_tasks = self._extract_markdown_table_tasks(text)
        if markdown_tasks:
            logger.info("Planner parsed %d tasks from markdown table fallback", len(markdown_tasks))
            return markdown_tasks

        return tasks

    def _find_task_list(self, payload: dict[str, Any]) -> list | None:
        """Find a task list in common LLM response shapes."""
        for key in ("tasks", "todo_items", "todoItems", "items", "task_list", "taskList"):
            value = payload.get(key)
            if isinstance(value, list):
                return value

        for value in payload.values():
            if isinstance(value, dict):
                found = self._find_task_list(value)
                if found is not None:
                    return found
            elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return value

        return None

    @staticmethod
    def _extract_markdown_table_tasks(text: str) -> List[dict[str, Any]]:
        """Parse planner fallback output such as a Markdown task overview table."""
        tasks: List[dict[str, Any]] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("|") or "---" in stripped:
                continue

            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if len(cells) < 3:
                continue
            if not re.fullmatch(r"\d+", cells[0]):
                continue

            title = re.sub(r"[*_`]", "", cells[1]).strip()
            intent = re.sub(r"[*_`]", "", cells[2]).strip()
            if not title:
                continue

            query_parts = [title, intent, "datasheet reference design price lifecycle"]
            tasks.append(
                {
                    "title": title,
                    "intent": intent or title,
                    "query": " ".join(part for part in query_parts if part),
                }
            )

        return tasks

    def _extract_json_payload(self, text: str) -> dict[str, Any] | list | None:
        """Try to locate and parse a JSON object or array from the text."""
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                logger.debug("Planner JSON object parse failed: %s", exc)

        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                logger.debug("Planner JSON array parse failed: %s", exc)
                return None

        logger.debug("Planner response did not contain JSON payload")
        return None
