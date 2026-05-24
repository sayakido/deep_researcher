"""Orchestrator coordinating the deep research workflow using LangGraph."""

from __future__ import annotations

import logging
from typing import Any, Iterator

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send, StreamWriter

from config import Configuration
from models import LangGraphState, SummaryState, SummaryStateOutput, TodoItem
from services.notes import NoteStore, create_note_tools
from services.logging_utils import truncate
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

    def _plan_tasks_node(self, state: LangGraphState, writer: StreamWriter = lambda _: None) -> dict:
        logger.info("plan_tasks start topic=%s", truncate(state["research_topic"]))
        todo_items = self.planner.plan_todo_list(state["research_topic"])
        if not todo_items:
            logger.warning("planner returned no tasks; using fallback task")
            todo_items = [self.planner.create_fallback_task(state["research_topic"])]

        logger.info(
            "plan_tasks done task_count=%s titles=%s",
            len(todo_items),
            [task.title for task in todo_items],
        )
        writer({
            "type": "todo_list",
            "tasks": [self._serialize_task(t) for t in todo_items],
            "step": 0,
        })

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

    def _execute_one_task(self, state: LangGraphState, writer: StreamWriter = lambda _: None) -> dict:
        """Execute research for a single task, emitting SSE events via StreamWriter."""
        task = state["todo_items"][0] if state["todo_items"] else None
        if not task or task.status in ("completed", "skipped"):
            logger.debug("execute_one skipped empty_or_done=%s", bool(task))
            return {}

        summary_state = _graph_state_to_summary(state)
        task.status = "in_progress"
        logger.info(
            "execute_one start task_id=%s title=%s query=%s",
            task.id,
            task.title,
            truncate(task.query),
        )

        writer({
            "type": "task_status",
            "task_id": task.id,
            "status": "in_progress",
            "title": task.title,
            "intent": task.intent,
            "note_id": task.note_id,
            "note_path": task.note_path,
        })

        search_result, notices, answer_text, backend = dispatch_search(
            task.query,
            self.config,
        )

        if notices:
            for notice in notices:
                if notice:
                    writer({"type": "status", "message": notice, "task_id": task.id})

        if not search_result or not search_result.get("results"):
            task.status = "skipped"
            task.notices = notices
            logger.warning(
                "execute_one no search results task_id=%s backend=%s notices=%s",
                task.id,
                backend,
                notices,
            )
            writer({
                "type": "task_status",
                "task_id": task.id,
                "status": "skipped",
                "title": task.title,
                "intent": task.intent,
                "note_id": task.note_id,
                "note_path": task.note_path,
            })
            return {
                "todo_items": [task],
                "web_research_results": [],
                "sources_gathered": [],
                "research_loop_count": 0,
            }

        sources_summary, context = prepare_research_context(
            search_result, answer_text, self.config,
        )
        task.sources_summary = sources_summary
        summary_state.web_research_results.append(context)
        summary_state.sources_gathered.append(sources_summary)
        summary_state.research_loop_count += 1
        logger.info(
            "execute_one search done task_id=%s backend=%s results=%s context_chars=%s",
            task.id,
            backend,
            len((search_result or {}).get("results", [])),
            len(context),
        )

        writer({
            "type": "sources",
            "task_id": task.id,
            "latest_sources": sources_summary,
            "raw_context": context,
            "backend": backend,
            "note_id": task.note_id,
            "note_path": task.note_path,
        })

        summary_text: str | None = None
        summary_stream, summary_getter = self.summarizer.stream_task_summary(
            summary_state, task, context,
        )

        try:
            for chunk in summary_stream:
                if chunk:
                    writer({
                        "type": "task_summary_chunk",
                        "task_id": task.id,
                        "content": chunk,
                        "note_id": task.note_id,
                    })
        finally:
            summary_text = summary_getter()

        task.summary = summary_text.strip() if summary_text else "暂无可用信息"
        task.status = "completed"
        logger.info(
            "execute_one completed task_id=%s summary_chars=%s",
            task.id,
            len(task.summary or ""),
        )

        writer({
            "type": "task_status",
            "task_id": task.id,
            "status": "completed",
            "summary": task.summary,
            "sources_summary": task.sources_summary,
            "note_id": task.note_id,
            "note_path": task.note_path,
        })

        return {
            "todo_items": [task],
            "web_research_results": summary_state.web_research_results,
            "sources_gathered": summary_state.sources_gathered,
            "research_loop_count": 1,
        }

    def _generate_report_node(self, state: LangGraphState, writer: StreamWriter = lambda _: None) -> dict:
        summary_state = _graph_state_to_summary(state)
        summary_state.todo_items = state["todo_items"]
        completed = [task for task in summary_state.todo_items if task.status == "completed"]
        skipped = [task for task in summary_state.todo_items if task.status == "skipped"]
        logger.info(
            "generate_report start tasks=%s completed=%s skipped=%s",
            len(summary_state.todo_items),
            len(completed),
            len(skipped),
        )

        report = self.reporting.generate_report(summary_state)
        self._persist_final_report(summary_state, report)
        logger.info(
            "generate_report done report_chars=%s note_id=%s",
            len(report or ""),
            summary_state.report_note_id,
        )

        writer({
            "type": "final_report",
            "report": report,
            "note_id": summary_state.report_note_id,
            "note_path": summary_state.report_note_path,
        })
        writer({"type": "done"})

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
        """Execute the workflow yielding incremental progress events.

        Uses LangGraph's stream_mode='custom' so all SSE events are
        emitted by the graph nodes via StreamWriter.  Tasks run in
        parallel through the Send API with zero duplicated logic.
        """
        logger.info("run_stream start topic=%s", truncate(topic))
        yield {"type": "status", "message": "初始化硬件方案设计流程"}

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

        for event in self.graph.stream(initial_state, stream_mode="custom"):
            if not isinstance(event, dict):
                continue

            if event.get("type") and event.get("type") != "custom":
                logger.debug("run_stream yield type=%s task_id=%s", event.get("type"), event.get("task_id"))
                yield event
                continue

            if event.get("type") == "custom":
                data = event.get("data")
                if isinstance(data, dict) and data.get("type"):
                    logger.debug("run_stream yield type=%s task_id=%s", data.get("type"), data.get("task_id"))
                    yield data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
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

        note_title = f"硬件方案设计报告：{summary_state.research_topic}".strip() or "硬件方案设计报告"
        content = report.strip()

        note_id = summary_state.report_note_id
        if note_id and self.note_store.read(note_id):
            self.note_store.update(
                note_id=note_id,
                title=note_title,
                note_type="conclusion",
                tags=["hardware_design", "report"],
                content=content,
            )
        else:
            note_id = self.note_store.create(
                title=note_title,
                note_type="conclusion",
                tags=["hardware_design", "report"],
                content=content,
            )
            summary_state.report_note_id = note_id
            summary_state.report_note_path = self.note_store.get_path(note_id)


def run_deep_research(topic: str, config: Configuration | None = None) -> SummaryStateOutput:
    """Convenience function mirroring the class-based API."""
    agent = DeepResearchAgent(config=config)
    return agent.run(topic)
