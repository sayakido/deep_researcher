"""Orchestrator coordinating the deep research workflow using LangGraph."""

from __future__ import annotations

import logging
from typing import Any, Iterator

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from config import Configuration
from models import LangGraphState, SummaryState, SummaryStateOutput, TodoItem
from services.notes import NoteStore, create_note_tools
from services.planner import PlanningService
from services.reporter import ReportingService
from services.search import dispatch_search, prepare_research_context
from services.summarizer import SummarizationService
from services.tool_runner import ToolRunner

logger = logging.getLogger(__name__)


def _create_llm(config: Configuration) -> ChatOpenAI:
    """Create a langchain ChatOpenAI instance from configuration."""
    kwargs: dict[str, Any] = {"temperature": 0.0, "timeout": 120}

    model_id = config.llm_model_id or config.local_llm
    if model_id:
        kwargs["model"] = model_id

    provider = (config.llm_provider or "").strip()

    if provider == "ollama":
        kwargs["base_url"] = config.sanitized_ollama_url()
        kwargs["api_key"] = config.llm_api_key or "ollama"
    elif provider == "lmstudio":
        kwargs["base_url"] = config.lmstudio_base_url
        if config.llm_api_key:
            kwargs["api_key"] = config.llm_api_key
    else:
        if config.llm_base_url:
            kwargs["base_url"] = config.llm_base_url
        if config.llm_api_key:
            kwargs["api_key"] = config.llm_api_key

    return ChatOpenAI(**kwargs)


def _graph_state_to_summary(state: LangGraphState) -> SummaryState:
    return SummaryState(
        research_topic=state["research_topic"],
        todo_items=list(state.get("todo_items", [])),
        web_research_results=list(state.get("web_research_results", [])),
        sources_gathered=list(state.get("sources_gathered", [])),
        research_loop_count=state.get("research_loop_count", 0),
        running_summary=state.get("running_summary", ""),
        structured_report=state.get("structured_report"),
        report_note_id=state.get("report_note_id"),
        report_note_path=state.get("report_note_path"),
    )


class DeepResearchAgent:
    """Coordinator orchestrating the research workflow using LangGraph."""

    def __init__(self, config: Configuration | None = None) -> None:
        self.config = config or Configuration.from_env()
        self.llm = _create_llm(self.config)

        self.note_store = (
            NoteStore(workspace=self.config.notes_workspace)
            if self.config.enable_notes
            else None
        )

        self.tools = create_note_tools(self.note_store) if self.note_store else []
        self.tool_runner = ToolRunner(self.llm, self.tools) if self.tools else ToolRunner(self.llm, [])

        self.planner = PlanningService(self.tool_runner, self.config)
        self.summarizer = SummarizationService(self.tool_runner, self.config)
        self.reporting = ReportingService(self.tool_runner, self.config)

        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph definition
    # ------------------------------------------------------------------
    def _build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(LangGraphState)

        builder.add_node("plan_tasks", self._plan_tasks_node)
        builder.add_node("execute_one", self._execute_one_task)
        builder.add_node("generate_report", self._generate_report_node)

        builder.add_edge(START, "plan_tasks")
        builder.add_conditional_edges(
            "plan_tasks",
            self._route_tasks,
            {"generate_report": "generate_report"},
        )
        builder.add_edge("execute_one", "generate_report")
        builder.add_edge("generate_report", END)

        return builder.compile()

    def _plan_tasks_node(self, state: LangGraphState) -> dict:
        todo_items = self.planner.plan_todo_list(state["research_topic"])
        if not todo_items:
            todo_items = [self.planner.create_fallback_task(state["research_topic"])]
        return {"todo_items": todo_items}

    def _route_tasks(self, state: LangGraphState) -> list[Send] | str:
        """Route to parallel execute_one tasks, or go to report if all done."""
        pending = [t for t in state["todo_items"] if t.status == "pending"]
        if not pending:
            return "generate_report"

        return [
            Send("execute_one", {
                **state,
                "todo_items": [task],
                "web_research_results": [],
                "sources_gathered": [],
                "research_loop_count": 0,
            })
            for task in pending
        ]

    def _execute_one_task(self, state: LangGraphState) -> dict:
        """Execute research for a single task (used by parallel Send API)."""
        task = state["todo_items"][0] if state["todo_items"] else None
        if not task or task.status in ("completed", "skipped"):
            return {}

        summary_state = _graph_state_to_summary(state)
        self._execute_single_task(summary_state, task)

        return {
            "todo_items": [task],
            "web_research_results": summary_state.web_research_results,
            "sources_gathered": summary_state.sources_gathered,
            "research_loop_count": 1,
        }

    def _generate_report_node(self, state: LangGraphState) -> dict:
        summary_state = _graph_state_to_summary(state)
        summary_state.todo_items = state["todo_items"]

        report = self.reporting.generate_report(summary_state)
        self._persist_final_report(summary_state, report)

        return {
            "structured_report": report,
            "running_summary": report,
            "report_note_id": summary_state.report_note_id,
            "report_note_path": summary_state.report_note_path,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, topic: str) -> SummaryStateOutput:
        """Execute the research workflow and return the final report."""
        initial_state: LangGraphState = {
            "research_topic": topic,
            "todo_items": [],
            "web_research_results": [],
            "sources_gathered": [],
            "research_loop_count": 0,
            "running_summary": "",
            "structured_report": None,
            "report_note_id": None,
            "report_note_path": None,
        }

        result = self.graph.invoke(initial_state)

        report = result.get("structured_report") or result.get("running_summary") or ""
        todo_items = result.get("todo_items", [])

        return SummaryStateOutput(
            running_summary=report,
            report_markdown=report,
            todo_items=todo_items,
        )

    def run_stream(self, topic: str) -> Iterator[dict[str, Any]]:
        """Execute the workflow yielding incremental progress events."""
        logger.debug("Starting streaming research: topic=%s", topic)
        yield {"type": "status", "message": "初始化研究流程"}

        # ---- 1. Plan tasks ----
        todo_items = self.planner.plan_todo_list(topic)
        if not todo_items:
            todo_items = [self.planner.create_fallback_task(topic)]

        yield {
            "type": "todo_list",
            "tasks": [self._serialize_task(t) for t in todo_items],
            "step": 0,
        }

        # ---- 2. Execute tasks ----
        summary_state = SummaryState(
            research_topic=topic,
            todo_items=todo_items,
        )

        for task in summary_state.todo_items:
            if task.status in ("completed", "skipped"):
                continue

            yield {
                "type": "task_status",
                "task_id": task.id,
                "status": "in_progress",
                "title": task.title,
                "intent": task.intent,
                "note_id": task.note_id,
                "note_path": task.note_path,
            }

            search_result, notices, answer_text, backend = dispatch_search(
                task.query,
                self.config,
            )

            if notices:
                for notice in notices:
                    if notice:
                        yield {
                            "type": "status",
                            "message": notice,
                            "task_id": task.id,
                        }

            if not search_result or not search_result.get("results"):
                task.status = "skipped"
                yield {
                    "type": "task_status",
                    "task_id": task.id,
                    "status": "skipped",
                    "title": task.title,
                    "intent": task.intent,
                    "note_id": task.note_id,
                    "note_path": task.note_path,
                }
                continue

            sources_summary, context = prepare_research_context(
                search_result, answer_text, self.config,
            )
            task.sources_summary = sources_summary
            summary_state.web_research_results.append(context)
            summary_state.sources_gathered.append(sources_summary)
            summary_state.research_loop_count += 1

            yield {
                "type": "sources",
                "task_id": task.id,
                "latest_sources": sources_summary,
                "raw_context": context,
                "backend": backend,
                "note_id": task.note_id,
                "note_path": task.note_path,
            }

            summary_text: str | None = None
            summary_stream, summary_getter = self.summarizer.stream_task_summary(
                summary_state, task, context,
            )

            try:
                for chunk in summary_stream:
                    if chunk:
                        yield {
                            "type": "task_summary_chunk",
                            "task_id": task.id,
                            "content": chunk,
                            "note_id": task.note_id,
                        }
            finally:
                summary_text = summary_getter()

            task.summary = summary_text.strip() if summary_text else "暂无可用信息"
            task.status = "completed"

            yield {
                "type": "task_status",
                "task_id": task.id,
                "status": "completed",
                "summary": task.summary,
                "sources_summary": task.sources_summary,
                "note_id": task.note_id,
                "note_path": task.note_path,
            }

        # ---- 3. Generate report ----
        report = self.reporting.generate_report(summary_state)
        self._persist_final_report(summary_state, report)

        yield {
            "type": "final_report",
            "report": report,
            "note_id": summary_state.report_note_id,
            "note_path": summary_state.report_note_path,
        }
        yield {"type": "done"}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _execute_single_task(self, summary_state: SummaryState, task: TodoItem) -> None:
        """Run search + summarization for a single task."""
        task.status = "in_progress"

        search_result, notices, answer_text, _ = dispatch_search(
            task.query,
            self.config,
        )

        if not search_result or not search_result.get("results"):
            task.status = "skipped"
            task.notices = notices
            return

        sources_summary, context = prepare_research_context(
            search_result, answer_text, self.config,
        )
        task.sources_summary = sources_summary
        summary_state.web_research_results.append(context)
        summary_state.sources_gathered.append(sources_summary)
        summary_state.research_loop_count += 1

        summary_text = self.summarizer.summarize_task(summary_state, task, context)
        task.summary = summary_text.strip() if summary_text else "暂无可用信息"
        task.status = "completed"

    def _serialize_task(self, task: TodoItem) -> dict[str, Any]:
        return {
            "id": task.id,
            "title": task.title,
            "intent": task.intent,
            "query": task.query,
            "status": task.status,
            "summary": task.summary,
            "sources_summary": task.sources_summary,
            "note_id": task.note_id,
            "note_path": task.note_path,
            "stream_token": task.stream_token,
        }

    def _persist_final_report(self, summary_state: SummaryState, report: str) -> None:
        if not self.note_store or not report or not report.strip():
            return

        note_title = f"研究报告：{summary_state.research_topic}".strip() or "研究报告"
        content = report.strip()

        note_id = summary_state.report_note_id
        if note_id and self.note_store.read(note_id):
            self.note_store.update(
                note_id=note_id,
                title=note_title,
                note_type="conclusion",
                tags=["deep_research", "report"],
                content=content,
            )
        else:
            note_id = self.note_store.create(
                title=note_title,
                note_type="conclusion",
                tags=["deep_research", "report"],
                content=content,
            )
            summary_state.report_note_id = note_id
            summary_state.report_note_path = self.note_store.get_path(note_id)


def run_deep_research(topic: str, config: Configuration | None = None) -> SummaryStateOutput:
    """Convenience function mirroring the class-based API."""
    agent = DeepResearchAgent(config=config)
    return agent.run(topic)
