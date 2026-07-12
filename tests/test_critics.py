"""Tests for museai.fsm.nodes.critics.

``run_agent_loop`` is replaced in the node's namespace, so no test reaches an
endpoint or a search engine. The node's real work — rendering the prompt,
parsing the findings, tracking the best draft seen — runs for real.
"""

from __future__ import annotations

import pytest

from museai.core.stream_bus import bus
from museai.fsm.nodes import critics as critics_module
from museai.fsm.nodes.critics import adversarial_critics, critic_messages
from museai.fsm.nodes.deps import set_node_config
from museai.fsm.state import FSM_Pointer, FailureObject, make_initial_state
from museai.llm.structured import StructuredOutputError

DRAFT = "The sun stood at noon. Mara lied about the letter."

PACKAGE = {
    "beat": {
        "id": "arc-1-c01-b01",
        "ordering": 1,
        "intent": "Mara finds the letter.",
        "entry_state": "A routine morning.",
        "exit_state": "Mara is holding her own handwriting.",
        "focal_character_id": "char-mara",
    },
    "pad_constraint": "Energy with nowhere to go.",
    "chapter": {
        "id": "arc-1-c01",
        "description": "Mara catalogs the letters.",
        "obligations": ["Mara dates the earliest letter."],
    },
    "threads": [
        {"id": "t1", "status": "open", "description": "Who writes them?",
         "priority_score": 0.9}
    ],
    "characters": [
        {"id": "char-mara", "name": "Mara", "description": "She never lies.",
         "pad": {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0}}
    ],
    "recent_prose": ["The lamp turned through the fog at midnight."],
    "budget": {"budget": 8000, "tokens": 400, "tokens_before": 400,
               "dropped_prose_passages": 0, "dropped_threads": 0,
               "over_budget": False},
}

CLEAN_RESPONSE = "[]"

ONE_FAILURE_RESPONSE = """```json
[
  {
    "error_code": "CONTRADICTS_CHARACTER",
    "offending_text": "Mara lied about the letter.",
    "suggested_fix": "Mara never lies; have her say nothing at all.",
    "critic_source": "continuity_critic"
  }
]
```"""

TWO_FAILURE_RESPONSE = """```json
[
  {
    "error_code": "CONTRADICTS_CHARACTER",
    "offending_text": "Mara lied about the letter.",
    "suggested_fix": "Mara never lies.",
    "critic_source": "continuity_critic"
  },
  {
    "error_code": "CONTRADICTS_PRIOR_PROSE",
    "offending_text": "The sun stood at noon.",
    "suggested_fix": "The prior passage puts this at midnight.",
    "critic_source": "continuity_critic"
  }
]
```"""


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tool_calls: list[dict] = []
        self.finish_reason = "stop"


def programmatic_failure() -> FailureObject:
    return FailureObject(
        error_code="PASSIVE_VOICE_DENSITY",
        offending_text="The door was opened by Mara.",
        suggested_fix="Rewrite the passive clauses.",
        critic_source="programmatic_audit",
    )


def state_with(**overrides):
    return make_initial_state(
        "test-project",
        FSM_Pointer(arc_id="arc-1", chapter_id="arc-1-c01", beat_index=0),
        active_context_package=PACKAGE,
        current_draft_text=DRAFT,
        **overrides,
    )


@pytest.fixture(autouse=True)
def _config(config_factory):
    set_node_config(config_factory(revision_retry_cap=3, max_agent_iterations=6))


@pytest.fixture
def patched_loop(monkeypatch):
    calls: list[dict] = []

    def _install(text: str):
        async def fake_loop(endpoint, messages, tools, tool_impls, max_iterations, on_event=None, **kwargs):
            calls.append(
                {
                    "messages": messages,
                    "tools": tools,
                    "tool_impls": tool_impls,
                    "max_iterations": max_iterations,
                    "on_event": on_event,
                }
            )
            return _Response(text)

        monkeypatch.setattr(critics_module, "run_agent_loop", fake_loop)
        return calls

    return _install


class TestPrompt:
    def test_the_draft_and_context_reach_the_prompt(self):
        messages = critic_messages(DRAFT, PACKAGE)
        assert [m["role"] for m in messages] == ["system", "user"]
        body = messages[1]["content"]
        assert DRAFT in body
        assert "Mara dates the earliest letter." in body
        assert "She never lies." in body
        assert "The lamp turned through the fog at midnight." in body


class TestCleanPass:
    async def test_a_clean_critic_reports_no_failures(self, patched_loop):
        patched_loop(CLEAN_RESPONSE)

        delta = await adversarial_critics(state_with())

        # Omitted, not []: an explicit [] would reset the reducer.
        assert "critic_failures" not in delta
        assert delta["best_seen_draft"] == DRAFT
        assert delta["best_seen_failure_count"] == 0

    async def test_a_clean_critic_does_not_erase_programmatic_failures(self, patched_loop):
        """audit's finding must survive a clean continuity pass."""
        patched_loop(CLEAN_RESPONSE)
        state = state_with(critic_failures=[programmatic_failure()])

        delta = await adversarial_critics(state)

        assert "critic_failures" not in delta
        # The passive-voice breach still counts against the draft.
        assert delta["best_seen_failure_count"] == 1

    async def test_the_loop_gets_the_critic_roster_and_the_configured_budget(self, patched_loop):
        calls = patched_loop(CLEAN_RESPONSE)

        await adversarial_critics(state_with())

        call = calls[0]
        offered = [t["function"]["name"] for t in call["tools"]]
        assert offered == [
            "search_manuscript", "get_full_outline", "get_thread_status",
            "get_thread_history", "get_canonical_state",
            "get_current_pointer_context", "get_recent_commits",
            "find_repetition",
        ]
        assert "web_search" not in offered  # research_mode is off by default
        assert set(call["tool_impls"]) == set(offered)
        assert call["max_iterations"] == 6


class TestFailures:
    async def test_a_failure_array_is_parsed_and_returned(self, patched_loop):
        patched_loop(ONE_FAILURE_RESPONSE)

        delta = await adversarial_critics(state_with())

        failures = delta["critic_failures"]
        assert len(failures) == 1
        assert failures[0].error_code == "CONTRADICTS_CHARACTER"
        assert failures[0].critic_source == "continuity_critic"
        assert delta["best_seen_failure_count"] == 1

    async def test_programmatic_and_critic_failures_both_count(self, patched_loop):
        patched_loop(TWO_FAILURE_RESPONSE)
        state = state_with(critic_failures=[programmatic_failure()])

        delta = await adversarial_critics(state)

        # The node returns only its own findings; the reducer appends them.
        assert len(delta["critic_failures"]) == 2
        assert delta["best_seen_failure_count"] == 3

    async def test_an_unparseable_response_survives_the_run(self, patched_loop):
        """A broken critic must not destroy a run; it raises the streak instead.

        It is not laundered into a clean pass either: `critic_health` reports the
        streak, and the UI warns once it reaches the degrade threshold.
        """
        patched_loop("I could not find any problems, honestly.")

        delta = await adversarial_critics(state_with())

        assert "critic_failures" not in delta  # nothing found, nothing claimed
        assert delta["critic_parse_failure_streak"] == 1

    async def test_an_empty_response_is_not_a_clean_pass(self, patched_loop):
        patched_loop("   ")

        delta = await adversarial_critics(state_with())
        assert delta["critic_parse_failure_streak"] == 1


class TestBestSeen:
    async def test_the_first_draft_is_always_the_best_seen(self, patched_loop):
        patched_loop(TWO_FAILURE_RESPONSE)

        delta = await adversarial_critics(state_with())

        assert delta["best_seen_draft"] == DRAFT
        assert delta["best_seen_failure_count"] == 2

    async def test_a_lower_count_replaces_the_best_seen_draft(self, patched_loop):
        patched_loop(ONE_FAILURE_RESPONSE)
        state = state_with(best_seen_draft="an older draft", best_seen_failure_count=2)

        delta = await adversarial_critics(state)

        assert delta["best_seen_draft"] == DRAFT
        assert delta["best_seen_failure_count"] == 1

    async def test_a_worse_draft_does_not_replace_the_best_seen(self, patched_loop):
        patched_loop(TWO_FAILURE_RESPONSE)
        state = state_with(best_seen_draft="a better draft", best_seen_failure_count=1)

        delta = await adversarial_critics(state)

        assert delta["best_seen_draft"] == "a better draft"
        assert delta["best_seen_failure_count"] == 1

    async def test_an_equal_count_does_not_replace_the_best_seen(self, patched_loop):
        patched_loop(ONE_FAILURE_RESPONSE)
        state = state_with(best_seen_draft="an equal draft", best_seen_failure_count=1)

        delta = await adversarial_critics(state)

        assert delta["best_seen_draft"] == "an equal draft"


class TestEvents:
    async def test_a_tool_call_publishes_a_critic_tool_event(self, monkeypatch):
        async def fake_loop(endpoint, messages, tools, tool_impls, max_iterations, on_event=None, **kwargs):
            await on_event(
                {"tool": "web_search", "arguments": {"query": "perseids"},
                 "result_preview": "August 12"}
            )
            return _Response(CLEAN_RESPONSE)

        monkeypatch.setattr(critics_module, "run_agent_loop", fake_loop)

        queue = bus.subscribe()
        try:
            await adversarial_critics(state_with())
            events = [queue.get_nowait() for _ in range(queue.qsize())]
        finally:
            bus.unsubscribe(queue)

        by_type = {e["type"]: e["data"] for e in events}
        assert by_type["critic_tool"]["tool"] == "web_search"
        assert by_type["critic_tool"]["arguments"] == {"query": "perseids"}
        assert by_type["phase_change"]["phase"] == "Auditing"
        assert by_type["critic_reasoning"]["text"] == CLEAN_RESPONSE
        assert by_type["critic_summary"]["summary"] == "clean"

    async def test_the_summary_counts_every_issue(self, patched_loop):
        patched_loop(TWO_FAILURE_RESPONSE)
        state = state_with(critic_failures=[programmatic_failure()])

        queue = bus.subscribe()
        try:
            await adversarial_critics(state)
            events = [queue.get_nowait() for _ in range(queue.qsize())]
        finally:
            bus.unsubscribe(queue)

        summary = next(e["data"] for e in events if e["type"] == "critic_summary")
        assert summary["summary"] == "3 issues found"
        assert summary["total_failures"] == 3


async def test_critic_accepts_zero_revision_retry_cap(
    config_factory, patched_loop, monkeypatch
):
    """revision_retry_cap=0 is legal config (straight to review); the critic
    must still be able to parse its findings."""
    set_node_config(config_factory(revision_retry_cap=0, max_agent_iterations=6))
    patched_loop(CLEAN_RESPONSE)

    delta = await adversarial_critics(state_with())
    assert "critic_failures" not in delta
