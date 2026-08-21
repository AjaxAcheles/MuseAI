"""Tests for museai.fsm.nodes.revise.

The agent loop's ``call_llm`` is replaced with a fake. The locating logic, the
mode choice, the splice guard, and the budget tiers all run for real.
"""

from __future__ import annotations

import logging
import threading

import pytest

from museai.core.config import RevisionConfig
from museai.core.stream_bus import bus
from museai.fsm.nodes.audit import ERROR_CODE, POV_ERROR_CODE
from museai.fsm.nodes import revise as revise_module
from museai.fsm.tools import loop as loop_module
from museai.fsm.nodes.deps import DraftingError, set_node_config
from museai.fsm.nodes.revise import (
    fuzzy_find,
    locate,
    replacement_rejection,
    revise_prose,
)
from museai.fsm.state import (
    FSM_Pointer,
    FailureObject,
    failure_signature,
    make_initial_state,
)

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


def failure(
    offending_text: str = OFFENDING,
    code: str = "CONTRADICTS_CHARACTER",
    whole_draft: bool = False,
):
    return FailureObject(
        error_code=code,
        offending_text=offending_text,
        suggested_fix="Mara never lies.",
        critic_source="continuity_critic",
        whole_draft=whole_draft,
    )


def state_with(failures, draft: str = DRAFT, retry_count: int = 0, **overrides):
    return make_initial_state(
        "test-project",
        FSM_Pointer(arc_id="arc-1", chapter_id="arc-1-c01", beat_index=0),
        active_context_package=PACKAGE,
        current_draft_text=draft,
        critic_failures=failures,
        retry_count=retry_count,
        **overrides,
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

    def test_fuzzy_find_regressions_keep_the_selected_span(self):
        exact = "Mara lied about the letter."
        near_match = "Mara lied about the letter!"
        exact_start = DRAFT.index(exact)

        assert fuzzy_find(DRAFT, exact) == (exact_start, exact_start + len(exact))
        assert fuzzy_find(DRAFT, near_match) == (exact_start, exact_start + len(near_match))
        assert fuzzy_find(DRAFT, "nope") is None
        assert fuzzy_find("", exact) is None
        assert fuzzy_find(DRAFT, "A passage absent from this draft entirely.") is None

    async def test_location_work_runs_in_a_worker_thread(self, monkeypatch, patched_llm):
        observed = []

        def record_location_thread(draft, failures):
            observed.append(threading.current_thread())
            return [(item, locate(draft, item.offending_text)) for item in failures]

        patched_llm(REPLACEMENT)
        monkeypatch.setattr(revise_module, "_locate_all", record_location_thread)

        await revise_prose(state_with([failure()]))

        assert observed
        assert all(thread is not threading.main_thread() for thread in observed)


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

    async def test_multiple_located_pov_failures_use_span_mode(self, patched_llm):
        draft = (
            "I crossed the courtyard before dawn. "
            "Mara fastened the gate behind her. "
            "My hands shook around the key. "
            "We waited for the rain to stop."
        )
        failures = [
            failure("I crossed the courtyard before dawn.", POV_ERROR_CODE),
            failure("My hands shook around the key.", POV_ERROR_CODE),
            failure("We waited for the rain to stop.", POV_ERROR_CODE),
        ]
        patched_llm(
            "They waited for the rain to stop.",
            "Her hands shook around the key.",
            "Mara crossed the courtyard before dawn.",
        )
        queue = bus.subscribe()
        try:
            delta = await revise_prose(state_with(failures, draft=draft))
            events = [queue.get_nowait() for _ in range(queue.qsize())]
        finally:
            bus.unsubscribe(queue)

        revision = next(event["data"] for event in events if event["type"] == "revision")
        assert revision["mode"] == "span"
        assert delta["current_draft_text"] == (
            "Mara crossed the courtyard before dawn. "
            "Mara fastened the gate behind her. "
            "Her hands shook around the key. "
            "They waited for the rain to stop."
        )

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

    async def test_identical_pov_quotes_fall_back_to_full_mode(self, patched_llm):
        sentence = "I waited by the gate."
        draft = f"{sentence} Mara crossed the yard. {sentence}"
        failures = [
            failure(sentence, POV_ERROR_CODE),
            failure(sentence, POV_ERROR_CODE),
        ]
        calls = patched_llm("A wholly rewritten beat.")

        delta = await revise_prose(state_with(failures, draft=draft))

        assert [locate(draft, failure.offending_text) for failure in failures] == [
            (0, len(sentence)),
            (0, len(sentence)),
        ]
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
        calls = patched_llm(
            "word " * 200,
            "word " * 200,
            "A wholly rewritten beat.",
        )

        delta = await revise_prose(state_with([failure()]))

        # Two bounded span attempts are rejected; the third call is full mode.
        assert len(calls) == 3
        assert "<passage_to_rewrite>" in calls[0][1]["content"]
        assert "<passage_to_rewrite>" in calls[1][1]["content"]
        assert "<passage_to_rewrite>" not in calls[2][1]["content"]
        assert delta["current_draft_text"] == "A wholly rewritten beat."

    async def test_an_echoing_span_rewrite_falls_back_to_a_full_rewrite(
        self, patched_llm
    ):
        echoing = f"{REPLACEMENT} The lamp turned through the fog. She climbed"
        patched_llm(echoing, echoing, "A wholly rewritten beat.")

        delta = await revise_prose(state_with([failure()]))

        assert delta["current_draft_text"] == "A wholly rewritten beat."
        assert echoing not in delta["current_draft_text"]

    async def test_a_corrected_span_rewrite_stays_in_span_mode(self, patched_llm):
        echoing = f"{REPLACEMENT} The lamp turned through the fog. She climbed"
        calls = patched_llm(echoing, REPLACEMENT)

        delta = await revise_prose(state_with([failure()]))

        assert len(calls) == 2
        assert all("<passage_to_rewrite>" in call[1]["content"] for call in calls)
        assert delta["current_draft_text"] == DRAFT.replace(OFFENDING, REPLACEMENT)

    async def test_zero_span_rejection_retries_falls_back_immediately(
        self, patched_llm, config_factory
    ):
        set_node_config(
            config_factory(
                context_token_budget=8000,
                revision=RevisionConfig(span_rejection_retries=0),
            )
        )
        echoing = f"{REPLACEMENT} The lamp turned through the fog. She climbed"
        calls = patched_llm(echoing, "A wholly rewritten beat.")

        delta = await revise_prose(state_with([failure()]))

        assert len(calls) == 2
        assert "<passage_to_rewrite>" in calls[0][1]["content"]
        assert "<passage_to_rewrite>" not in calls[1][1]["content"]
        assert delta["current_draft_text"] == "A wholly rewritten beat."

    async def test_a_span_rejection_correction_quotes_the_guarded_echo(
        self, patched_llm
    ):
        echoed = "The lamp turned through the fog. She climbed"
        echoing = f"{REPLACEMENT} {echoed}"
        calls = patched_llm(echoing, REPLACEMENT)

        await revise_prose(state_with([failure()]))

        correction = calls[1][-1]["content"]
        assert echoed in correction
        assert "repeated prose from the surrounding beat" in correction


class TestFullMode:
    async def test_a_locatable_density_failure_uses_full_mode(
        self, patched_llm, caplog, monkeypatch
    ):
        calls = patched_llm("A wholly rewritten beat.")
        monkeypatch.setattr(logging.getLogger("museai"), "propagate", True)
        with caplog.at_level(logging.INFO, logger="museai.fsm"):
            await revise_prose(state_with([failure(OFFENDING, ERROR_CODE, True)]))

        revised = next(
            record.getMessage()
            for record in caplog.records
            if "node=revise event=revised" in record.getMessage()
        )
        assert len(calls) == 1
        assert "<passage_to_rewrite>" not in calls[0][1]["content"]
        assert "mode=full" in revised
        assert "whole_draft_failures=1" in revised

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

    async def test_a_truncated_rewrite_is_refused_rather_than_spliced(
        self, monkeypatch
    ):
        """A rewrite cut off at the token limit must not replace the beat."""

        async def truncated(endpoint, messages, **kwargs):
            reply = _Response("A rewrite that stops mid-sen")
            reply.finish_reason = "length"
            return reply

        monkeypatch.setattr(loop_module, "call_llm", truncated)

        with pytest.raises(DraftingError, match="max_output_tokens"):
            await revise_prose(state_with([failure("nowhere to be found in the draft")]))


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
    async def test_a_regressing_cycle_revises_the_best_seen_draft_and_findings(
        self, patched_llm
    ):
        """A worse rewrite must not become the next rewrite's starting point."""
        best_draft = "The earlier, better draft is still on the page."
        worse_draft = "The later draft regressed in six different ways."
        best_failures = [failure(f"best finding {number}") for number in range(3)]
        worse_failures = [failure(f"worse finding {number}") for number in range(6)]
        calls = patched_llm("A recovered rewrite.")

        delta = await revise_prose(
            state_with(
                worse_failures,
                draft=worse_draft,
                best_seen_draft=best_draft,
                best_seen_failures=best_failures,
                best_seen_failure_count=3,
            )
        )

        body = calls[0][1]["content"]
        assert best_draft in body
        assert worse_draft not in body
        assert "best finding 0" in body
        assert "worse finding 0" not in body
        # The next critics pass compares the recovered rewrite with the score of
        # the prose it really revised, rather than the discarded regression.
        assert delta["pre_revise_failure_count"] == 3
        assert delta["pre_revise_failure_signatures"] == [
            failure_signature(finding) for finding in best_failures
        ]

    @pytest.mark.parametrize(
        ("best_count", "current_count"),
        [(5, 5), (5, 3)],
        ids=("equal-count", "improving-count"),
    )
    async def test_equal_or_improving_cycles_do_not_roll_back(
        self, patched_llm, best_count, current_count
    ):
        current_draft = "The current draft must remain the reviser's input."
        calls = patched_llm("A continued rewrite.")

        await revise_prose(
            state_with(
                [failure(f"current finding {number}") for number in range(current_count)],
                draft=current_draft,
                best_seen_draft="An older draft should not replace this one.",
                best_seen_failures=[failure(f"best finding {number}") for number in range(best_count)],
                best_seen_failure_count=best_count,
            )
        )

        assert current_draft in calls[0][1]["content"]

    async def test_missing_best_seen_failures_never_rolls_back(self, patched_llm):
        current_draft = "The first cycle has no paired best findings yet."
        calls = patched_llm("A first rewrite.")

        await revise_prose(
            state_with(
                [failure("current finding")],
                draft=current_draft,
                best_seen_draft="An incomplete best-seen record.",
                best_seen_failure_count=0,
                best_seen_failures=None,
            )
        )

        assert current_draft in calls[0][1]["content"]

    async def test_a_locatable_span_local_failure_uses_span_mode(self, patched_llm):
        patched_llm(REPLACEMENT)
        queue = bus.subscribe()
        try:
            await revise_prose(state_with([failure()]))
            events = [queue.get_nowait() for _ in range(queue.qsize())]
        finally:
            bus.unsubscribe(queue)

        revision = next(event["data"] for event in events if event["type"] == "revision")
        assert revision["mode"] == "span"

    def test_a_legacy_failure_record_defaults_to_span_local(self):
        legacy = FailureObject.model_validate(
            {
                "error_code": "CONTRADICTS_CHARACTER",
                "offending_text": OFFENDING,
                "suggested_fix": "Mara never lies.",
                "critic_source": "continuity_critic",
            }
        )

        assert legacy.whole_draft is False

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


class TestToolRoster:
    async def test_the_loop_is_offered_the_reviser_roster(self, _config, monkeypatch):
        offered: list = []

        async def fake(endpoint, messages, **kwargs):
            offered.append(kwargs.get("tools"))
            return _Response(REPLACEMENT)

        monkeypatch.setattr(loop_module, "call_llm", fake)

        await revise_prose(state_with([failure()]))

        assert [t["function"]["name"] for t in offered[0]] == [
            "check_draft", "verify_replacement", "search_manuscript",
            "find_repetition",
        ]
