"""Task summarization using langchain ChatOpenAI with tool support."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from config import Configuration
from models import SummaryState, TodoItem
from prompts import task_summarizer_instructions
from services.tool_runner import ToolRunner
from services.logging_utils import log_duration
from utils import strip_thinking_tokens

logger = logging.getLogger(__name__)


class SummarizationService:
    """Handles synchronous and streaming task summarization with note tools."""

    def __init__(self, tool_runner: ToolRunner, config: Configuration) -> None:
        self._tool_runner = tool_runner
        self._config = config

    def summarize_task(self, state: SummaryState, task: TodoItem, context: str) -> str:
        """Generate a task-specific summary using the LLM."""
        prompt = self._build_prompt(state, task, context)

        messages = [
            SystemMessage(content=task_summarizer_instructions),
            HumanMessage(content=prompt),
        ]

        with log_duration(logger, "summarize_task_llm", task_id=task.id):
            content = self._tool_runner.run(messages)

        summary_text = content.strip()
        if self._config.strip_thinking_tokens:
            summary_text = strip_thinking_tokens(summary_text)

        return summary_text or "暂无可用信息"

    def stream_task_summary(
        self, state: SummaryState, task: TodoItem, context: str
    ) -> Tuple[Iterator[str], Callable[[], str]]:
        """Stream the summary text for a task while collecting full output."""
        prompt = self._build_prompt(state, task, context)
        remove_thinking = self._config.strip_thinking_tokens
        raw_buffer = ""
        visible_output = ""
        emit_index = 0

        messages = [
            SystemMessage(content=task_summarizer_instructions),
            HumanMessage(content=prompt),
        ]
        logger.info(
            "summarizer stream prepared task_id=%s context_chars=%s remove_thinking=%s",
            task.id,
            len(context),
            remove_thinking,
        )

        def flush_visible() -> Iterator[str]:
            nonlocal emit_index, raw_buffer
            while True:
                start = raw_buffer.find("<think>", emit_index)
                if start == -1:
                    if emit_index < len(raw_buffer):
                        segment = raw_buffer[emit_index:]
                        emit_index = len(raw_buffer)
                        if segment:
                            yield segment
                    break

                if start > emit_index:
                    segment = raw_buffer[emit_index:start]
                    emit_index = start
                    if segment:
                        yield segment

                end = raw_buffer.find("</think>", start)
                if end == -1:
                    break
                emit_index = end + len("</think>")

        def generator() -> Iterator[str]:
            nonlocal raw_buffer, visible_output, emit_index
            try:
                for chunk in self._tool_runner.stream(messages):
                    raw_buffer += chunk
                    if remove_thinking:
                        for segment in flush_visible():
                            visible_output += segment
                            if segment:
                                yield segment
                    else:
                        visible_output += chunk
                        if chunk:
                            yield chunk
            finally:
                if remove_thinking:
                    for segment in flush_visible():
                        visible_output += segment
                        if segment:
                            yield segment
                logger.info(
                    "summarizer stream finished task_id=%s raw_chars=%s visible_chars=%s",
                    task.id,
                    len(raw_buffer),
                    len(visible_output),
                )

        def get_summary() -> str:
            if remove_thinking:
                cleaned = strip_thinking_tokens(visible_output)
            else:
                cleaned = visible_output
            return cleaned.strip()

        return generator(), get_summary

    def _build_prompt(self, state: SummaryState, task: TodoItem, context: str) -> str:
        """Construct the summarization prompt shared by both modes."""
        note_hint = ""
        if task.note_id:
            note_hint = (
                f"\n已有笔记 ID：{task.note_id}。"
                "可使用 read_note 工具获取已有内容，"
                "使用 update_note 工具追加新的总结。"
            )
        else:
            note_hint = (
                "\n可使用 create_note 工具创建任务笔记来记录硬件方案分析结果，"
                "标签需包含 hardware_design 和 task_{task.id}。"
            )

        return (
            f"产品需求：{state.research_topic}\n"
            f"任务名称：{task.title}\n"
            f"任务目标：{task.intent}\n"
            f"检索查询：{task.query}\n"
            f"任务上下文：\n{context}\n"
            f"{note_hint}\n"
            "请基于以上上下文，生成一份面向硬件方案评审的 Markdown 分析总结。"
        )
