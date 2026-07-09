"""The continuity critic's one tool: a bounded web search.

v1 has exactly one tool. There is no headless browser, no page fetcher, no
search API key — ``ddgs`` scrapes public result pages and needs no credential,
which is why it is the whole of the tool surface.

**This function never raises.** It sits inside an agentic loop that a model
drives, and a search that throws would abort a draft over a rate limit or a
transient DNS failure. Every fault — a blank query, a timeout, a scraper the
upstream engine broke, an empty result set — collapses to ``[]`` plus one INFO
line in ``fsm.log``. The model then sees "no results" and writes around it,
which is the correct behaviour for a critic checking a fact it cannot confirm.
"""

from __future__ import annotations

from typing import Any

from ddgs import DDGS

from museai.core.logging_setup import get_fsm_logger
from museai.fsm.nodes.deps import get_node_config

# Snippets are grounding material for a critic, not documents. Long ones only
# spend context.
_SNIPPET_PREVIEW_CHARS = 400

# ddgs names its result fields ``href``/``body``; some of its engines emit
# ``url``/``snippet``. Accept both rather than pin the tool to one spelling.
_URL_KEYS = ("href", "url", "link")
_SNIPPET_KEYS = ("body", "snippet", "description")


WEB_SEARCH_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the public web for factual grounding. Returns a list of "
            "results, each with a title, url, and snippet. An empty list means "
            "the fact could not be confirmed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5,
                    "minimum": 1,
                },
            },
            "required": ["query"],
        },
    },
}


def _first_value(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Return the first non-empty string among ``keys``, or an empty string."""
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalise(rows: Any, max_results: int) -> list[dict[str, str]]:
    """Coerce raw ddgs rows into ``{"title", "url", "snippet"}`` dicts.

    A row carrying no URL is dropped: a citation the critic cannot follow is not
    evidence.
    """
    results: list[dict[str, str]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        url = _first_value(row, _URL_KEYS)
        if not url:
            continue
        snippet = _first_value(row, _SNIPPET_KEYS)
        results.append(
            {
                "title": _first_value(row, ("title",)),
                "url": url,
                "snippet": snippet[:_SNIPPET_PREVIEW_CHARS],
            }
        )
        if len(results) >= max_results:
            break
    return results


def _log_empty(reason: str) -> None:
    """Write the one line a failed or empty search is allowed to emit."""
    get_fsm_logger().info("web_search empty/failed: %s", reason)


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web and return up to ``max_results`` normalized results.

    Returns a list of ``{"title", "url", "snippet"}`` dicts, or ``[]`` on any
    failure or empty response. Never raises.
    """
    text = (query or "").strip()
    if not text:
        _log_empty("blank query")
        return []

    try:
        max_results = max(1, int(max_results))
    except (TypeError, ValueError):
        max_results = 5

    try:
        # The config read sits inside the try deliberately: an unreadable
        # config.yaml must degrade this tool to "no results", not tear down the
        # draft loop from inside a tool call.
        timeout = get_node_config().web_search_timeout
        rows = DDGS(timeout=timeout).text(text, max_results=max_results)
    except Exception as exc:  # noqa: BLE001 - the contract: nothing escapes
        _log_empty(f"{type(exc).__name__}: {exc}")
        return []

    results = _normalise(rows, max_results)
    if not results:
        _log_empty(f"no results for {text!r}")
    return results


# Tool name -> callable, as the agentic loop resolves them.
TOOL_IMPLS: dict[str, Any] = {"web_search": web_search}
