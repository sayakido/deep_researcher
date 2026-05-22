"""State models used by the deep research workflow."""

import operator
from dataclasses import dataclass, field
from typing import Annotated, List, TypedDict


@dataclass(kw_only=True)
class TodoItem:
    """单个待办任务项。"""

    id: int
    title: str
    intent: str
    query: str
    status: str = field(default="pending")
    summary: str | None = field(default=None)
    sources_summary: str | None = field(default=None)
    notices: list[str] = field(default_factory=list)
    note_id: str | None = field(default=None)
    note_path: str | None = field(default=None)
    stream_token: str | None = field(default=None)


@dataclass(kw_only=True)
class SummaryState:
    research_topic: str = field(default=None)
    search_query: str = field(default=None)
    web_research_results: Annotated[list, operator.add] = field(default_factory=list)
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    research_loop_count: int = field(default=0)
    running_summary: str = field(default=None)
    todo_items: Annotated[list, operator.add] = field(default_factory=list)
    structured_report: str | None = field(default=None)
    report_note_id: str | None = field(default=None)
    report_note_path: str | None = field(default=None)


@dataclass(kw_only=True)
class SummaryStateOutput:
    running_summary: str = field(default=None)
    report_markdown: str | None = field(default=None)
    todo_items: List[TodoItem] = field(default_factory=list)


class LangGraphState(TypedDict):
    """State type used by the LangGraph workflow."""

    research_topic: str
    todo_items: Annotated[List[TodoItem], operator.add]
    web_research_results: Annotated[List[str], operator.add]
    sources_gathered: Annotated[List[str], operator.add]
    research_loop_count: Annotated[int, operator.add]
    running_summary: str
    structured_report: str | None
    report_note_id: str | None
    report_note_path: str | None
