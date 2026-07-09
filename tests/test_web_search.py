"""Tests for museai.fsm.tools.web_search.

``DDGS`` is replaced in the module's namespace, so no test reaches the network.
The contract under test is the one the agentic loop depends on: whatever goes
wrong, ``web_search`` returns a list and logs a line — it never raises.
"""

from __future__ import annotations

import logging

import pytest

from museai.core.logging_setup import get_fsm_logger
from museai.fsm.nodes.deps import set_node_config
from museai.fsm.tools import web_search as web_search_module
from museai.fsm.tools.web_search import TOOL_IMPLS, WEB_SEARCH_TOOL_SPEC, web_search

ROWS = [
    {
        "title": "Perseids peak",
        "href": "https://example.org/perseids",
        "body": "The shower peaks around August 12.",
    },
    {
        "title": "Meteor guide",
        "href": "https://example.org/guide",
        "body": "Best viewed after midnight.",
    },
]


class _FakeDDGS:
    """Stands in for ddgs.DDGS, recording how it was constructed and called."""

    instances: list["_FakeDDGS"] = []

    def __init__(self, timeout=None, **kwargs):
        self.timeout = timeout
        self.calls: list[tuple[str, int]] = []
        _FakeDDGS.instances.append(self)

    def text(self, query, max_results=None, **kwargs):
        self.calls.append((query, max_results))
        return ROWS


@pytest.fixture(autouse=True)
def _config(config_factory):
    set_node_config(config_factory(web_search_timeout=7))
    _FakeDDGS.instances = []


@pytest.fixture
def fsm_logs():
    """Capture fsm.log records directly.

    The ``museai`` logger does not propagate to root, so ``caplog`` sees nothing;
    the handler has to be attached to the logger itself.
    """
    logger = get_fsm_logger()
    captured: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = Capture()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield captured
    logger.removeHandler(handler)


def _empty_lines(records) -> list[str]:
    return [
        r.getMessage() for r in records if "web_search empty/failed" in r.getMessage()
    ]


class TestFailure:
    """Nothing escapes. Every fault is an empty list and one log line."""

    def test_raising_ddgs_returns_empty_and_logs(self, monkeypatch, fsm_logs):
        def _boom(*args, **kwargs):
            raise RuntimeError("rate limited")

        monkeypatch.setattr(web_search_module, "DDGS", _boom)

        assert web_search("Perseid meteor shower peak") == []
        lines = _empty_lines(fsm_logs)
        assert len(lines) == 1
        assert "RuntimeError: rate limited" in lines[0]

    def test_raising_text_method_returns_empty(self, monkeypatch, fsm_logs):
        class _Broken(_FakeDDGS):
            def text(self, query, max_results=None, **kwargs):
                raise TimeoutError("read timed out")

        monkeypatch.setattr(web_search_module, "DDGS", _Broken)

        assert web_search("anything") == []
        assert "TimeoutError" in _empty_lines(fsm_logs)[0]

    def test_empty_response_returns_empty_and_logs(self, monkeypatch, fsm_logs):
        class _Empty(_FakeDDGS):
            def text(self, query, max_results=None, **kwargs):
                return []

        monkeypatch.setattr(web_search_module, "DDGS", _Empty)

        assert web_search("no such thing") == []
        assert "no results for 'no such thing'" in _empty_lines(fsm_logs)[0]

    def test_blank_query_never_touches_the_network(self, monkeypatch, fsm_logs):
        monkeypatch.setattr(web_search_module, "DDGS", _FakeDDGS)

        assert web_search("   ") == []
        assert _FakeDDGS.instances == []
        assert "blank query" in _empty_lines(fsm_logs)[0]

    def test_unreadable_config_degrades_to_empty(self, monkeypatch, fsm_logs):
        monkeypatch.setattr(web_search_module, "DDGS", _FakeDDGS)
        monkeypatch.setattr(
            web_search_module,
            "get_node_config",
            lambda: (_ for _ in ()).throw(RuntimeError("config.yaml not found")),
        )

        assert web_search("query") == []
        assert "config.yaml not found" in _empty_lines(fsm_logs)[0]


class TestSuccess:
    """Rows are normalized to exactly {"title", "url", "snippet"}."""

    def test_rows_are_normalized(self, monkeypatch):
        monkeypatch.setattr(web_search_module, "DDGS", _FakeDDGS)

        results = web_search("Perseid meteor shower peak", 3)

        assert results == [
            {
                "title": "Perseids peak",
                "url": "https://example.org/perseids",
                "snippet": "The shower peaks around August 12.",
            },
            {
                "title": "Meteor guide",
                "url": "https://example.org/guide",
                "snippet": "Best viewed after midnight.",
            },
        ]

    def test_configured_timeout_reaches_the_client(self, monkeypatch):
        monkeypatch.setattr(web_search_module, "DDGS", _FakeDDGS)

        web_search("query", 2)

        assert _FakeDDGS.instances[0].timeout == 7
        assert _FakeDDGS.instances[0].calls == [("query", 2)]

    def test_alternate_field_names_are_accepted(self, monkeypatch):
        class _Alt(_FakeDDGS):
            def text(self, query, max_results=None, **kwargs):
                return [{"title": "T", "url": "https://e.org/a", "snippet": "S"}]

        monkeypatch.setattr(web_search_module, "DDGS", _Alt)

        assert web_search("q") == [
            {"title": "T", "url": "https://e.org/a", "snippet": "S"}
        ]

    def test_rows_without_a_url_are_dropped(self, monkeypatch, fsm_logs):
        class _NoUrl(_FakeDDGS):
            def text(self, query, max_results=None, **kwargs):
                return [{"title": "T", "body": "S"}, "junk"]

        monkeypatch.setattr(web_search_module, "DDGS", _NoUrl)

        assert web_search("q") == []
        assert _empty_lines(fsm_logs)

    def test_results_are_capped_at_max_results(self, monkeypatch):
        class _Many(_FakeDDGS):
            def text(self, query, max_results=None, **kwargs):
                return [
                    {"title": f"t{i}", "href": f"https://e.org/{i}", "body": "b"}
                    for i in range(10)
                ]

        monkeypatch.setattr(web_search_module, "DDGS", _Many)

        assert len(web_search("q", 3)) == 3

    def test_long_snippets_are_truncated(self, monkeypatch):
        class _Long(_FakeDDGS):
            def text(self, query, max_results=None, **kwargs):
                return [{"title": "t", "href": "https://e.org/x", "body": "z" * 5000}]

        monkeypatch.setattr(web_search_module, "DDGS", _Long)

        assert len(web_search("q")[0]["snippet"]) == 400


class TestRegistry:
    def test_tool_spec_describes_web_search(self):
        function = WEB_SEARCH_TOOL_SPEC["function"]
        assert WEB_SEARCH_TOOL_SPEC["type"] == "function"
        assert function["name"] == "web_search"
        assert set(function["parameters"]["properties"]) == {"query", "max_results"}
        assert function["parameters"]["required"] == ["query"]

    def test_registry_maps_the_one_v1_tool(self):
        assert TOOL_IMPLS == {"web_search": web_search}
