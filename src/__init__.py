"""Deep Research - A LangGraph-powered deep research assistant."""

from __future__ import annotations

import sys
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

__version__ = "0.0.1"

from agent import DeepResearchAgent
from config import Configuration, SearchAPI
from models import SummaryState, SummaryStateOutput, TodoItem

__all__ = [
    "DeepResearchAgent",
    "Configuration",
    "SearchAPI",
    "SummaryState",
    "SummaryStateOutput",
    "TodoItem",
]
