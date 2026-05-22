"""Tests for utility functions."""

from __future__ import annotations

from utils import deduplicate_and_format_sources, format_sources, strip_thinking_tokens


class TestStripThinkingTokens:
    def test_no_think_tags(self):
        assert strip_thinking_tokens("Hello world") == "Hello world"

    def test_simple_think_block(self):
        assert strip_thinking_tokens("A<think>inner</think>B") == "AB"

    def test_multiple_think_blocks(self):
        result = strip_thinking_tokens("a<think>1</think>b<think>2</think>c")
        assert result == "abc"

    def test_empty_string(self):
        assert strip_thinking_tokens("") == ""

    def test_unclosed_tag(self):
        assert strip_thinking_tokens("a<think>unclosed") == "a<think>unclosed"


class TestFormatSources:
    def test_empty_input(self):
        assert format_sources(None) == ""
        assert format_sources({}) == ""

    def test_single_result(self):
        data = {"results": [{"title": "Foo", "url": "https://foo.com"}]}
        result = format_sources(data)
        assert "Foo" in result
        assert "https://foo.com" in result

    def test_skips_missing_url(self):
        data = {"results": [{"title": "No URL"}, {"title": "Bar", "url": "https://bar.com"}]}
        result = format_sources(data)
        assert "No URL" not in result
        assert "Bar" in result


class TestDeduplicateAndFormatSources:
    def test_deduplicates_by_url(self):
        data = {
            "results": [
                {"url": "https://a.com", "title": "A", "content": "content a"},
                {"url": "https://a.com", "title": "A dup", "content": "content a dup"},
                {"url": "https://b.com", "title": "B", "content": "content b"},
            ]
        }
        result = deduplicate_and_format_sources(data, max_tokens_per_source=100)
        assert result.count("信息来源:") == 2
        assert "A" in result

    def test_empty_results(self):
        result = deduplicate_and_format_sources({"results": []}, max_tokens_per_source=100)
        assert result == ""

    def test_no_url_filter(self):
        data = {
            "results": [
                {"url": "", "title": "No URL", "content": "skip"},
                {"url": "https://c.com", "title": "C", "content": "keep"},
            ]
        }
        result = deduplicate_and_format_sources(data, max_tokens_per_source=100)
        assert "No URL" not in result
        assert "C" in result
