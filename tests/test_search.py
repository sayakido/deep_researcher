"""Tests for search dispatch (mocked backends)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from config import Configuration, SearchAPI
from services.search import dispatch_search


def test_dispatch_search_duckduckgo():
    """DuckDuckGo path: no API key required, just needs search_api config."""
    config = Configuration(search_api=SearchAPI.DUCKDUCKGO, fetch_full_page=False)
    with patch("services.search._duckduckgo_search", return_value=[
        {"title": "R1", "url": "https://r1.com", "content": "content1"},
    ]):
        result, notices, answer, backend = dispatch_search("test query", config)
        assert result is not None
        assert len(result.get("results", [])) == 1
        assert backend == "duckduckgo"
