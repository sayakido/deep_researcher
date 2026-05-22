"""Search dispatch helpers — direct implementations replacing SearchTool."""

from __future__ import annotations

import logging
from typing import Any

from config import Configuration, SearchAPI
from utils import (
    deduplicate_and_format_sources,
    format_sources,
    get_config_value,
)

logger = logging.getLogger(__name__)

MAX_TOKENS_PER_SOURCE = 2000


def _duckduckgo_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    from duckduckgo_search import DDGS
    results: list[dict[str, Any]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "content": r.get("body", ""),
            })
    return results


def _tavily_search(query: str, api_key: str, max_results: int = 5) -> list[dict[str, Any]]:
    from tavily import TavilyClient
    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, max_results=max_results)
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in response.get("results", [])
    ]


def _fetch_page_text(url: str, timeout: int = 10) -> str | None:
    try:
        import requests
        from bs4 import BeautifulSoup
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (compatible; DeepResearch/1.0)"
        })
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text[:8000]
    except Exception as exc:
        logger.debug("Failed to fetch page %s: %s", url, exc)
        return None


def _fetch_full_page_contents(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched = []
    for r in results:
        url = r.get("url", "")
        raw = _fetch_page_text(url) if url else None
        r["raw_content"] = raw or r.get("content", "")
        enriched.append(r)
    return enriched


def dispatch_search(
    query: str,
    config: Configuration,
) -> tuple[dict[str, Any] | None, list[str], str | None, str]:
    """Execute configured search backend directly."""

    search_api_str = get_config_value(config.search_api)
    search_api = SearchAPI(search_api_str)
    notices: list[str] = []
    answer_text: str | None = None

    try:
        if search_api == SearchAPI.DUCKDUCKGO:
            results = _duckduckgo_search(query)
            backend_label = "duckduckgo"
        elif search_api == SearchAPI.TAVILY:
            api_key = config.tavily_api_key or ""
            if not api_key:
                raise ValueError("TAVILY_API_KEY is not configured")
            results = _tavily_search(query, api_key)
            backend_label = "tavily"
        elif search_api == SearchAPI.PERPLEXITY:
            from openai import OpenAI
            api_key = config.perplexity_api_key or ""
            if not api_key:
                raise ValueError("PERPLEXITY_API_KEY is not configured")
            client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
            chat_resp = client.chat.completions.create(
                model=config.perplexity_model or "sonar-pro",
                messages=[{"role": "user", "content": query}],
            )
            answer_text = chat_resp.choices[0].message.content or ""
            results = []
            citations = getattr(chat_resp, "citations", None) or []
            for i, c in enumerate(citations):
                results.append({
                    "title": f"Source {i + 1}",
                    "url": c if isinstance(c, str) else "",
                    "content": "",
                })
            backend_label = "perplexity"
        elif search_api == SearchAPI.SEARXNG:
            searxng_url = config.searxng_base_url or "http://localhost:4000"
            import requests
            http_resp = requests.get(
                f"{searxng_url}/search",
                params={"q": query, "format": "json"},
                timeout=15,
            )
            http_resp.raise_for_status()
            data = http_resp.json()
            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                }
                for r in data.get("results", [])
            ]
            backend_label = "searxng"
        else:
            results = _duckduckgo_search(query)
            backend_label = "duckduckgo"

        if config.fetch_full_page and results:
            results = _fetch_full_page_contents(results)

    except Exception as exc:
        logger.exception("Search backend %s failed: %s", search_api_str, exc)
        raise

    payload: dict[str, Any] = {
        "results": results,
        "backend": backend_label,
        "answer": answer_text,
        "notices": notices,
    }

    logger.info(
        "Search backend=%s answer=%s results=%s",
        backend_label,
        bool(answer_text),
        len(results),
    )

    return payload, notices, answer_text, backend_label


def prepare_research_context(
    search_result: dict[str, Any] | None,
    answer_text: str | None,
    config: Configuration,
) -> tuple[str, str]:
    """Build structured context and source summary for downstream agents."""
    sources_summary = format_sources(search_result)
    context = deduplicate_and_format_sources(
        search_result or {"results": []},
        max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
        fetch_full_page=config.fetch_full_page,
    )

    if answer_text:
        context = f"AI直接答案：\n{answer_text}\n\n{context}"

    return sources_summary, context
