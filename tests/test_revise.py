"""Tests for museai.fsm.nodes.revise.

The agent loop's ``call_llm`` is replaced with a fake. The locating logic, the
mode choice, the splice guard, and the budget tiers all run for real.
"""

from __future__ import annotations

import pytest

from museai.core.stream_bus import bus
from museai.fsm.tools import loop as loop_module
from museai.fsm.nodes.deps import DraftingError, set_node_config
from museai.fsm.nodes.revise import (
    fuzzy_find,
    locate,
    replacement_rejection,
    revise_prose,
)
from museai.fsm.state import FSM_Pointer, FailureObject, make_initial_state

DRAFT = (
    "The lamp turned through the fog. "
    "Mara lied about the letter. "
    "She climbed the stair and slept."
)
OFFENDING = "Mara lied about the letter."
REPLACEMENT = "Mara said nothing about the letter."

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
    "recent_prose": ["The keeper trimmed the wick."],
    "budget": {"budget": 8000, "tokens": 400, "tokens_before": 400,
               "dropped_prose_passages": 0, "dropped_threads": 0,
               "over_budget": False},
}


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text
        self.finish_reason = "stop"
        self.tool_calls = []


def failure(offending_text: str = OFFENDING, code: str = "CONTRADICTS_CHARACTER"):
    return FailureObject(
        error_code=code,
        offending_text=offending_text,
        suggested_fix="Mara never lies.",
        critic_source="continuity_critic",
    )


def state_with(failures, draft: str = DRAFT, retry_count: int = 0):
    return make_initial_state(
        "test-project",
        FSM_Pointer(arc_id="arc-1", chapter_id="arc-1-c01", beat_index=0),
        active_context_package=PACKAGE,
        current_draft_text=draft,
        critic_failures=failures,
        retry_count=retry_count,
    )


@pytest.fixture(autouse=True)
def _config(config_factory):
    set_node_config(config_factory(context_token_budget=8000))


@pytest.fixture
def patched_llm(monkeypatch):
    calls: list[list[dict]] = []

    def _install(*texts: str):
        replies = list(texts)

        async def fake_llm(endpoint, messages, **kwargs):
            calls.append([dict(m) for m in messages])
            return _Response(replies.pop(0) if replies else "revised prose")

        monkeypatch.setattr(loop_module, "call_llm", fake_llm)
        return calls

    return _install


class TestLocate:
    def test_an_exact_span_is_found(self):
        start = DRAFT.index(OFFENDING)
        assert locate(DRAFT, OFFENDING) == (start, start + len(OFFENDING))

    def test_surrounding_whitespace_is_ignored(self):
        assert locate(DRAFT, f"  {OFFENDING}  ") is not None

    def test_a_lightly_mangled_quote_is_found_fuzzily(self):
        # A critic that normalised the period and dropped a word.
        assert locate(DRAFT, "Mara lied about the letter") is not None

    def test_a_paraphrase_is_not_found(self):
        assert locate(DRAFT, "She was dishonest concerning the correspondence") is None

    def test_an_empty_needle_is_not_found(self):
        assert locate(DRAFT, "   ") is None

    def test_fuzzy_find_returns_the_best_window(self):
        span = fuzzy_find(DRAFT, "Mara lied about the lettre.")
        assert span is not None
        start, end = span
        assert DRAFT[start:end].startswith("Mara lied")


class TestSpanMode:
    async def test_a_located_span_is_replaced_in_place(self, patched_llm):
        patched_llm(REPLACEMENT)

        delta = await revise_prose(state_with([failure()]))

        revised = delta["current_draft_text"]
        assert OFFENDING not in revised
        assert REPLACEMENT in revised
        # The prose the critic did not fault is untouched.
        assert revised.startswith("The lamp turned through the fog.")
        assert revised.endswith("She climbed the stair and slept.")

    async def test_the_span_prompt_carries_only_that_span(self, patched_llm):
        calls = patched_llm(REPLACEMENT)

        await revise_prose(state_with([failure()]))

        assert len(calls) == 1
        body = calls[0][1]["content"]
        assert "<passage_to_rewrite>" in body
        assert OFFENDING in body

    async def test_two_located_spans_are_each_replaced(self, patched_llm):
        patched_llm("The lamp guttered in the fog.", REPLACEMENT)
        failures = [
            failure(),
            failure("The lamp turned through the fog.", "CONTRADICTS_PRIOR_PROSE"),
        ]

        delta = await revise_prose(state_with(failures))

        revised = delta["current_draft_text"]
        assert "The lamp guttered in the fog." in revised
        assert REPLACEMENT in revised
        assert revised.endswith("She climbed the stair and slept.")

    async def test_overlapping_spans_fall_back_to_a_full_rewrite(self, patched_llm):
        calls = patched_llm("A wholly rewritten beat.")
        failures = [
            failure("Mara lied about the letter."),
            failure("lied about the letter. She climbed"),
        ]

        delta = await revise_prose(state_with(failures))

        assert len(calls) == 1
        assert "<passage_to_rewrite>" not in calls[0][1]["content"]
        assert delta["current_draft_text"] == "A wholly rewritten beat."


class TestSpliceGuard:
    """A span rewrite padded with surrounding prose must never be spliced —
    that is exactly how duplicated paragraphs reached exported manuscripts."""

    def test_a_sane_replacement_is_not_rejected(self):
        span = (DRAFT.index(OFFENDING), DRAFT.index(OFFENDING) + len(OFFENDING))
        assert replacement_rejection(DRAFT, span, REPLACEMENT) is None

    def test_an_oversized_replacement_is_rejected(self):
        span = (DRAFT.index(OFFENDING), DRAFT.index(OFFENDING) + len(OFFENDING))
        bloated = "word " * 200
        assert "grew" in replacement_rejection(DRAFT, span, bloated)

    def test_a_replacement_echoing_surrounding_prose_is_rejected(self):
        span = (DRAFT.index(OFFENDING), DRAFT.index(OFFENDING) + len(OFFENDING))
        echoing = f"{REPLACEMENT} The lamp turned through the fog. She climbed"
        assert "repeats surrounding prose" in replacement_rejection(DRAFT, span, echoing)

    async def test_an_oversized_span_rewrite_falls_back_to_a_full_rewrite(
        self, patched_llm
    ):
        calls = patched_llm("word " * 200, "A wholly rewritten beat.")

        delta = await revise_prose(state_with([failure()]))

        # First call was the span attempt; the second is the full rewrite.
        assert len(calls) == 2
        assert "<passage_to_rewrite>" in calls[0][1]["content"]
        assert "<passage_to_rewrite>" not in calls[1][1]["content"]
        assert delta["current_draft_text"] == "A wholly rewritten beat."

    async def test_an_echoing_span_rewrite_falls_back_to_a_full_rewrite(
        self, patched_llm
    ):
        echoing = f"{REPLACEMENT} The lamp turned through the fog. She climbed"
        patched_llm(echoing, "A wholly rewritten beat.")

        delta = await revise_prose(state_with([failure()]))

        assert delta["current_draft_text"] == "A wholly rewritten beat."
        assert echoing not in delta["current_draft_text"]


class TestFullMode:
    async def test_a_missing_span_falls_back_to_a_full_rewrite(self, patched_llm):
        calls = patched_llm("A wholly rewritten beat.")
        failures = [failure("She was dishonest concerning the correspondence")]

        delta = await revise_prose(state_with(failures))

        assert delta["current_draft_text"] == "A wholly rewritten beat."
        assert len(calls) == 1
        body = calls[0][1]["content"]
        assert "<passage_to_rewrite>" not in body
        assert "<draft_beat>" in body

    async def test_the_full_prompt_carries_every_failure(self, patched_llm):
        calls = patched_llm("rewritten")
        failures = [
            failure("nowhere in the draft at all, truly"),
            failure("also absent from the draft entirely"),
        ]

        await revise_prose(state_with(failures))

        body = calls[0][1]["content"]
        assert "nowhere in the draft at all, truly" in body
        assert "also absent from the draft entirely" in body

    async def test_empty_prose_from_the_endpoint_is_a_hard_error(self, patched_llm):
        patched_llm("   ")

        with pytest.raises(DraftingError):
            await revise_prose(state_with([failure()]))


class TestBudget:
    async def test_an_over_budget_prompt_collapses_to_hard_constraints(
        self, patched_llm, config_factory
    ):
        set_node_config(config_factory(context_token_budget=1))
        calls = patched_llm(REPLACEMENT)

        await revise_prose(state_with([failure()]))

        body = calls[0][1]["content"]
        # The hard constraints survive.
        assert "Mara dates the earliest letter." in body
        assert "Energy with nowhere to go." in body
        assert "Mara finds the letter." in body
        # The droppable context does not.
        assert "Who writes them?" not in body
        assert "The keeper trimmed the wick." not in body

    async def test_a_within_budget_prompt_keeps_the_full_context(self, patched_llm):
        calls = patched_llm(REPLACEMENT)

        await revise_prose(state_with([failure()]))

        body = calls[0][1]["content"]
        assert "Who writes them?" in body
        assert "The keeper trimmed the wick." in body


class TestStateDelta:
    async def test_retry_count_increments(self, patched_llm):
        patched_llm(REPLACEMENT)

        delta = await revise_prose(state_with([failure()], retry_count=2))

        assert delta["retry_count"] == 3

    async def test_failures_are_cleared(self, patched_llm):
        patched_llm(REPLACEMENT)

        delta = await revise_prose(state_with([failure()]))

        # An explicit [] — the reducer resets, because the findings described
        # prose that no longer exists.
        assert delta["critic_failures"] == []

    async def test_revising_with_no_failures_is_a_hard_error(self, patched_llm):
        patched_llm(REPLACEMENT)

        with pytest.raises(DraftingError):
            await revise_prose(state_with([]))

    async def test_a_revision_event_is_published(self, patched_llm):
        patched_llm(REPLACEMENT)

        queue = bus.subscribe()
        try:
            await revise_prose(state_with([failure()]))
            events = [queue.get_nowait() for _ in range(queue.qsize())]
        finally:
            bus.unsubscribe(queue)

        revision = next(e["data"] for e in events if e["type"] == "revision")
        assert revision["mode"] == "span"
        assert revision["failures_fixed"] == 1
        assert revision["retry_count"] == 1
