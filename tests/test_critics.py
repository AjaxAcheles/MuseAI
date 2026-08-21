"""Tests for museai.fsm.nodes.critics.

``run_agent_loop`` is replaced in the node's namespace, so no test reaches an
endpoint or a search engine. The node's real work — rendering the prompt,
parsing the findings, tracking the best draft seen — runs for real.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from museai.core.config import AppConfig, load_config
from museai.core.stream_bus import bus
from museai.fsm.nodes import critics as critics_module
from museai.fsm.nodes.critics import adversarial_critics, critic_messages
from museai.fsm.nodes.deps import set_node_config
from museai.fsm.state import (
    FSM_Pointer,
    FailureObject,
    failure_signature,
    make_initial_state,
)
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
        "thread_updates": [{"id": "t1", "status": "progressing"}],
    },
    "pad_constraint": "Energy with nowhere to go.",
    "project": {
        "genre": "mystery",
        "premise": "Letters in Mara's own hand keep arriving.",
        "setting": "A shrinking harbour town with one post office.",
    },
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


def critic_response(*findings: FailureObject) -> str:
    """Render only the fields the continuity critic is allowed to return."""
    return json.dumps(
        [
            {
                "error_code": finding.error_code,
                "offending_text": finding.offending_text,
                "suggested_fix": finding.suggested_fix,
                "critic_source": finding.critic_source,
            }
            for finding in findings
        ]
    )


def state_with(**overrides):
    defaults = {
        "active_context_package": PACKAGE,
        "current_draft_text": DRAFT,
    }
    defaults.update(overrides)
    return make_initial_state(
        "test-project",
        FSM_Pointer(arc_id="arc-1", chapter_id="arc-1-c01", beat_index=0),
        **defaults,
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
        messages = critic_messages(DRAFT, PACKAGE, 0)
        assert [m["role"] for m in messages] == ["system", "user"]
        body = messages[1]["content"]
        assert DRAFT in body
        assert "Mara dates the earliest letter." in body
        assert "She never lies." in body
        assert "The lamp turned through the fog at midnight." in body

    def test_the_beat_goal_reaches_the_prompt(self):
        """The critic can only enforce an exit state it has been shown."""
        body = critic_messages(DRAFT, PACKAGE, 0)[1]["content"]
        assert "<beat_goal>" in body
        assert "Mara finds the letter." in body
        assert "Mara is holding her own handwriting." in body
        assert '<update thread="t1" new_status="progressing"/>' in body

    def test_the_story_world_reaches_the_prompt(self):
        """The critic can only defend a premise and setting it has been shown."""
        body = critic_messages(DRAFT, PACKAGE, 0)[1]["content"]
        assert "<story_world>" in body
        assert "Letters in Mara's own hand keep arriving." in body
        assert "A shrinking harbour town with one post office." in body

    def test_a_package_without_a_project_still_renders(self):
        """Old context packages predate the story-world block; the prompt must
        render without it rather than crash mid-run."""
        package = {k: v for k, v in PACKAGE.items() if k != "project"}
        body = critic_messages(DRAFT, package, 0)[1]["content"]
        assert "<beat_goal>" in body

    def test_obligation_scope_names_this_beats_rendered_field(self):
        package = {
            **PACKAGE,
            "beat": {**PACKAGE["beat"], "discharges": ["Mara dates the earliest letter."]},
        }
        body = critic_messages(DRAFT, package, 0)[0]["content"]
        assert "<discharges_chapter_obligations>" in body
        assert "this beat's `discharges`" in body
        assert "list) or its planned thread movement" in body
        assert "other <chapter_obligations>" in body


class TestCriticConfig:
    def test_shipped_configs_boot_with_the_critic_cap(self, monkeypatch):
        root = Path(__file__).parents[1]
        monkeypatch.setenv("MUSEAI_API_KEY", "test-key")

        live = load_config(root / "config.yaml")
        example = load_config(root / "config.example.yaml")

        assert live.generation.critic_max_findings_per_code == 2
        assert example.generation.critic_max_findings_per_code == 2

    def test_missing_or_unknown_critic_cap_key_is_fatal(self, config_factory):
        missing = config_factory().model_dump()
        del missing["generation"]["critic_max_findings_per_code"]
        with pytest.raises(ValidationError, match="critic_max_findings_per_code"):
            AppConfig(**missing)

        unknown = config_factory().model_dump()
        unknown["generation"]["unexpected_critic_key"] = 1
        with pytest.raises(ValidationError, match="unexpected_critic_key"):
            AppConfig(**unknown)


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
        ]
        assert "web_search" not in offered  # research_mode is off by default
        assert set(call["tool_impls"]) == set(offered)
        assert call["max_iterations"] == 6


class TestRepetitionCheck:
    """audit's own repetition guard replaces find_repetition in the prompt.

    See museai/fsm/nodes/critics.py:critic_messages and CRITIC_ERROR_CODES in
    museai/llm/structured.py — the critic's error-code list never had a
    repetition category, so a find_repetition hit had nowhere to be reported.
    """

    def test_a_clean_repetition_check_says_so(self):
        body = critic_messages(DRAFT, PACKAGE, 0)[1]["content"]
        assert '<repetition_check status="clean"/>' in body
        assert "find_repetition" not in body

    def test_an_overlap_is_reported_with_an_instruction_not_to_repeat_it(self):
        """The one guard against double-counting: `total` in `adversarial_
        critics` sums `state["critic_failures"]` (which already carries
        audit's PARAGRAPH_OVERLAP entries) with whatever the critic reports.
        There is no code-level dedup, so this sentence surviving in the
        rendered prompt is what stops the critic from filing its own finding
        for the same paragraph and inflating the count."""
        body = critic_messages(DRAFT, PACKAGE, 3)[1]["content"]
        assert 'status="overlap_found"' in body
        assert 'count="3"' in body
        assert "do not report" in body.lower()

    async def test_a_critic_finding_lifted_from_committed_prose_is_discarded(
        self, patched_loop
    ):
        """A critic finding quoted only from committed prose is discarded.

        The audit finding remains counted, while the unlocatable critic reply
        leaves the critic unhealthy instead of adding a second issue.
        """
        restated_response = """```json
[
  {
    "error_code": "CONTRADICTS_PRIOR_PROSE",
    "offending_text": "The lamp turned through the fog at midnight.",
    "suggested_fix": "This repeats prose already committed earlier.",
    "critic_source": "continuity_critic"
  }
]
```"""
        patched_loop(restated_response)
        state = state_with(
            critic_failures=[
                FailureObject(
                    error_code="PARAGRAPH_OVERLAP",
                    offending_text="The lamp turned through the fog at midnight.",
                    suggested_fix="Write fresh prose.",
                    critic_source="programmatic_audit",
                )
            ],
            repetition_overlap_count=1,
        )

        delta = await adversarial_critics(state)

        # audit's one overlap + the critic restating the same paragraph: no
        # code-level dedup, so both count.
        assert delta["best_seen_failure_count"] == 1
        assert delta["critic_parse_failure_streak"] == 1


UNFULFILLED_RESPONSE = """```json
[
  {
    "error_code": "UNFULFILLED_OBLIGATION",
    "offending_text": "Mara lied about the letter.",
    "suggested_fix": "The exit state requires Mara holding her own handwriting; deliver it on the page.",
    "critic_source": "continuity_critic"
  }
]
```"""


class TestFailures:
    async def test_an_unfulfilled_obligation_is_parsed_and_returned(self, patched_loop):
        """The critic can report a beat that failed its own mandate."""
        patched_loop(UNFULFILLED_RESPONSE)

        delta = await adversarial_critics(state_with())

        failures = delta["critic_failures"]
        assert len(failures) == 1
        assert failures[0].error_code == "UNFULFILLED_OBLIGATION"

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

    async def test_repeated_error_codes_are_capped_in_order_and_logged(
        self, patched_loop, monkeypatch
    ):
        """One latched code cannot inflate routing's failure-count measurement."""
        response = json.dumps(
            [
                {
                    "error_code": "CONTRADICTS_CHARACTER",
                    "offending_text": "Mara lied about the letter.",
                    "suggested_fix": f"Fix number {number}.",
                    "critic_source": "continuity_critic",
                }
                for number in range(1, 4)
            ]
        )
        events: list[dict] = []
        original = critics_module.log_node_event

        def capture(node_name, **fields):
            if fields.get("event") == "critiqued":
                events.append(fields)
            original(node_name, **fields)

        monkeypatch.setattr(critics_module, "log_node_event", capture)
        patched_loop(response)

        delta = await adversarial_critics(state_with())

        failures = delta["critic_failures"]
        assert [finding.suggested_fix for finding in failures] == [
            "Fix number 1.", "Fix number 2."
        ]
        assert events[-1]["critic_findings_dropped_by_code_cap"] == 1

    async def test_mixed_findings_within_each_code_cap_are_untouched(self, patched_loop):
        response = json.dumps(
            [
                {
                    "error_code": "CONTRADICTS_CHARACTER",
                    "offending_text": "Mara lied about the letter.",
                    "suggested_fix": "First character fix.",
                    "critic_source": "continuity_critic",
                },
                {
                    "error_code": "CONTRADICTS_PRIOR_PROSE",
                    "offending_text": "The sun stood at noon.",
                    "suggested_fix": "First prose fix.",
                    "critic_source": "continuity_critic",
                },
                {
                    "error_code": "CONTRADICTS_CHARACTER",
                    "offending_text": "Mara lied about the letter.",
                    "suggested_fix": "Second character fix.",
                    "critic_source": "continuity_critic",
                },
            ]
        )
        patched_loop(response)

        delta = await adversarial_critics(state_with())

        assert [finding.suggested_fix for finding in delta["critic_failures"]] == [
            "First character fix.", "First prose fix.", "Second character fix."
        ]

    async def test_programmatic_audit_failures_are_not_subject_to_the_critic_cap(
        self, config_factory, patched_loop
    ):
        set_node_config(config_factory(critic_max_findings_per_code=1))
        patched_loop(CLEAN_RESPONSE)
        audit_failures = [programmatic_failure(), programmatic_failure()]

        delta = await adversarial_critics(state_with(critic_failures=audit_failures))

        assert "critic_failures" not in delta
        assert delta["best_seen_failure_count"] == len(audit_failures)


class TestBestSeen:
    async def test_the_first_draft_is_always_the_best_seen(self, patched_loop):
        patched_loop(TWO_FAILURE_RESPONSE)

        delta = await adversarial_critics(state_with())

        assert delta["best_seen_draft"] == DRAFT
        assert delta["best_seen_failure_count"] == 2
        assert len(delta["best_seen_failures"]) == delta["best_seen_failure_count"]

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


class TestRevisionProgress:
    async def test_a_replaced_finding_is_progress_at_the_same_count(
        self, patched_loop, monkeypatch
    ):
        """The 2026-08-20 first-beat park fixed the character finding before
        the critic found a new thread finding; that is ordinary iteration."""
        character = FailureObject(
            error_code="CONTRADICTS_CHARACTER",
            offending_text="Mara lied about the letter.",
            suggested_fix="Keep Mara from lying.",
        )
        thread = FailureObject(
            error_code="CONTRADICTS_THREAD",
            offending_text="The sun stood at noon.",
            suggested_fix="Advance the open thread.",
        )
        events: list[dict] = []
        original = critics_module.log_node_event

        def capture(node_name, **fields):
            if fields.get("event") == "critiqued":
                events.append(fields)
            original(node_name, **fields)

        monkeypatch.setattr(critics_module, "log_node_event", capture)
        patched_loop(critic_response(thread))

        delta = await adversarial_critics(
            state_with(
                pre_revise_failure_count=1,
                pre_revise_failure_signatures=[failure_signature(character)],
            )
        )

        assert delta["last_cycle_improved"] is True
        assert events[-1]["progressed_by_signature"] is True

    async def test_a_returning_finding_is_not_progress(self, patched_loop):
        finding = FailureObject(
            error_code="CONTRADICTS_CHARACTER",
            offending_text="Mara lied about the letter.",
            suggested_fix="Keep Mara from lying.",
        )
        patched_loop(critic_response(finding))

        delta = await adversarial_critics(
            state_with(
                pre_revise_failure_count=1,
                pre_revise_failure_signatures=[failure_signature(finding)],
            )
        )

        assert delta["last_cycle_improved"] is False

    async def test_whitespace_reflow_does_not_turn_a_returning_finding_new(
        self, patched_loop
    ):
        before = FailureObject(
            error_code="CONTRADICTS_CHARACTER",
            offending_text="Mara lied about the letter.",
            suggested_fix="Keep Mara from lying.",
        )
        reflowed = FailureObject(
            error_code="CONTRADICTS_CHARACTER",
            offending_text="Mara  lied\nabout the letter.",
            suggested_fix="Keep Mara from lying.",
        )
        patched_loop(critic_response(reflowed))

        delta = await adversarial_critics(
            state_with(
                current_draft_text=reflowed.offending_text,
                pre_revise_failure_count=1,
                pre_revise_failure_signatures=[failure_signature(before)],
            )
        )

        assert delta["last_cycle_improved"] is False

    async def test_a_partial_fix_at_the_same_count_is_not_progress(
        self, patched_loop
    ):
        character = FailureObject(
            error_code="CONTRADICTS_CHARACTER",
            offending_text="Mara lied about the letter.",
            suggested_fix="Keep Mara from lying.",
        )
        prior_prose = FailureObject(
            error_code="CONTRADICTS_PRIOR_PROSE",
            offending_text="The sun stood at noon.",
            suggested_fix="Match the earlier midnight.",
        )
        replacement = FailureObject(
            error_code="CONTRADICTS_THREAD",
            offending_text="The sun stood at noon.",
            suggested_fix="Advance the open thread.",
        )
        patched_loop(critic_response(character, replacement))

        delta = await adversarial_critics(
            state_with(
                pre_revise_failure_count=2,
                pre_revise_failure_signatures=[
                    failure_signature(character),
                    failure_signature(prior_prose),
                ],
            )
        )

        assert delta["last_cycle_improved"] is False

    async def test_a_count_drop_remains_progress_when_signatures_return(
        self, patched_loop
    ):
        character = FailureObject(
            error_code="CONTRADICTS_CHARACTER",
            offending_text="Mara lied about the letter.",
            suggested_fix="Keep Mara from lying.",
        )
        prior_prose = FailureObject(
            error_code="CONTRADICTS_PRIOR_PROSE",
            offending_text="The sun stood at noon.",
            suggested_fix="Match the earlier midnight.",
        )
        # The list preserves all three handed findings, including a duplicate;
        # both remaining identities still intersect it, but 3 -> 2 wins.
        patched_loop(critic_response(character, prior_prose))

        delta = await adversarial_critics(
            state_with(
                pre_revise_failure_count=3,
                pre_revise_failure_signatures=[
                    failure_signature(character),
                    failure_signature(character),
                    failure_signature(prior_prose),
                ],
            )
        )

        assert delta["last_cycle_improved"] is True

    async def test_missing_pre_revise_signatures_does_not_imply_a_clear(
        self, patched_loop
    ):
        patched_loop(ONE_FAILURE_RESPONSE)

        delta = await adversarial_critics(
            state_with(
                pre_revise_failure_count=None,
                pre_revise_failure_signatures=None,
            )
        )

        assert delta["last_cycle_improved"] is True


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
