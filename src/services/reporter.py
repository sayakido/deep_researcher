"""Service that consolidates task results into the final report."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from config import Configuration
from models import SummaryState
from prompts import report_writer_instructions
from services.tool_runner import ToolRunner
from utils import strip_thinking_tokens


class ReportingService:
    """Generates the final structured report."""

    def __init__(self, tool_runner: ToolRunner, config: Configuration) -> None:
        self._tool_runner = tool_runner
        self._config = config

    def generate_report(self, state: SummaryState) -> str:
        """Generate a structured report based on completed tasks."""

        tasks_block = []
        for task in state.todo_items:
            summary_block = task.summary or "暂无可用信息"
            sources_block = task.sources_summary or "暂无来源"
            tasks_block.append(
                f"### 任务 {task.id}: {task.title}\n"
                f"- 任务目标：{task.intent}\n"
                f"- 检索查询：{task.query}\n"
                f"- 执行状态：{task.status}\n"
                f"- 任务总结：\n{summary_block}\n"
                f"- 来源概览：\n{sources_block}\n"
            )

        note_references = []
        for task in state.todo_items:
            if task.note_id:
                note_references.append(
                    f"- 任务 {task.id}《{task.title}》：note_id={task.note_id}"
                )

        notes_section = "\n".join(note_references) if note_references else "- 暂无可用任务笔记"

        prompt = (
            f"研究主题：{state.research_topic}\n"
            f"任务概览：\n{''.join(tasks_block)}\n"
            f"可用任务笔记：\n{notes_section}\n"
            "请整合所有信息后撰写一份结构化研究报告。"
            "完成后可使用 create_note 工具（type=conclusion, tags=deep_research,report）保存报告要点。"
        )

        messages = [
            SystemMessage(content=report_writer_instructions.strip()),
            HumanMessage(content=prompt),
        ]

        content = self._tool_runner.run(messages)

        report_text = content.strip()
        if self._config.strip_thinking_tokens:
            report_text = strip_thinking_tokens(report_text)

        return report_text or "报告生成失败，请检查输入。"
