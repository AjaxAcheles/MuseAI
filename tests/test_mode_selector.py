"""Tests for museai.fsm.routers.mode_selector.

Pure and synchronous: state in, node name out. Nothing is mutated.
"""

from __future__ import annotations

import pytest

from museai.fsm.nodes.deps import set_node_config
from museai.fsm.routers.mode_selector import (
    COMMIT,
    RETRY_CRITIC,
    REVIEW,
    REVISE,
    mode_selector,
)
from museai.fsm.state import FSM_Pointer, FailureObject, make_initial_state

RETRY_CAP = 3


def failure(code: str = "CONTRADICTS_CHARACTER") -> FailureObject:
    return FailureObject(
        error_code=code,
        offending_text="Mara lied about the letter.",
        suggested_fix="Mara never lies.",
        critic_source="continuity_critic",
    )


def state_with(
    failures, retry_count: int, *, parse_streak: int = 0, last_cycle_improved: bool = True
):
    return make_initial_state(
        "test-project",
        FSM_Pointer(arc_id="arc-1", chapter_id="arc-1-c01", beat_index=0),
        current_draft_text="prose",
        critic_failures=failures,
        retry_count=retry_count,
        critic_parse_failure_streak=parse_streak,
        last_cycle_improved=last_cycle_improved,
    )


@pytest.fixture(autouse=True)
def _config(config_factory):
    set_node_config(config_factory(revision_retry_cap=RETRY_CAP))


class TestRouting:
    def test_a_clean_draft_commits(self):
        assert mode_selector(state_with([], retry_count=0)) == COMMIT

    def test_a_clean_draft_commits_even_at_the_cap(self):
        """No failures wins over an exhausted budget: first match wins."""
        assert mode_selector(state_with([], retry_count=RETRY_CAP)) == COMMIT

    @pytest.mark.parametrize("retry_count", [0, 1, 2])
    def test_failures_under_the_cap_revise(self, retry_count):
        assert mode_selector(state_with([failure()], retry_count)) == REVISE

    def test_failures_at_the_cap_go_to_review(self):
        assert mode_selector(state_with([failure()], RETRY_CAP)) == REVIEW

    def test_failures_over_the_cap_go_to_review(self):
        assert mode_selector(state_with([failure()], RETRY_CAP + 1)) == REVIEW

    def test_a_programmatic_failure_alone_routes_to_revise(self):
        audit_failure = failure("PASSIVE_VOICE_DENSITY")
        assert mode_selector(state_with([audit_failure], retry_count=0)) == REVISE

    def test_the_cap_is_read_from_config(self, config_factory):
        set_node_config(config_factory(revision_retry_cap=1))
        assert mode_selector(state_with([failure()], retry_count=0)) == REVISE
        assert mode_selector(state_with([failure()], retry_count=1)) == REVIEW

    def test_an_unreadable_critic_reply_retries_the_critic(self):
        assert mode_selector(state_with([], 0, parse_streak=1)) == RETRY_CRITIC

    def test_persistently_unreadable_critic_reply_goes_to_review(self, config_factory):
        set_node_config(config_factory(critic_degrade_threshold=2))
        assert mode_selector(state_with([], 0, parse_streak=2)) == REVIEW


class TestNoProgress:
    """A revise cycle that already ran and did not improve stops early rather
    than spending another full critic pass on the same non-improving cycle."""

    def test_a_non_improving_retry_under_the_cap_goes_to_review(self):
        state = state_with([failure()], retry_count=1, last_cycle_improved=False)
        assert mode_selector(state) == REVIEW

    def test_an_improving_retry_under_the_cap_still_revises(self):
        state = state_with([failure()], retry_count=1, last_cycle_improved=True)
        assert mode_selector(state) == REVISE

    def test_the_first_cycle_is_never_no_progress(self):
        # retry_count == 0: no prior cycle to have failed to improve on, so
        # `last_cycle_improved` (its default, True) cannot route to REVIEW here.
        state = state_with([failure()], retry_count=0, last_cycle_improved=False)
        assert mode_selector(state) == REVISE

    def test_a_clean_draft_commits_even_after_a_non_improving_retry(self):
        # No outstanding failures wins over no-progress: first match wins.
        state = state_with([], retry_count=1, last_cycle_improved=False)
        assert mode_selector(state) == COMMIT

    def test_an_unreadable_critic_below_the_threshold_outranks_no_progress(self):
        """The re-score has not happened yet, so there is no verdict to act on.
        Retrying the critic first is what keeps a parser fault from being read as
        a failed revise cycle."""
        state = state_with([failure()], retry_count=1, parse_streak=1,
                           last_cycle_improved=False)
        assert mode_selector(state) == RETRY_CRITIC

    def test_a_degraded_critic_with_salvaged_findings_still_routes_to_review(
        self, config_factory
    ):
        """Both conditions are true at once and both send the beat to the human
        boundary. The log carries `critic_parse_failure_streak` alongside
        `no_progress`, so the operator can tell which one drove it."""
        set_node_config(config_factory(revision_retry_cap=RETRY_CAP,
                                       critic_degrade_threshold=2))
        state = state_with([failure()], retry_count=1, parse_streak=2,
                           last_cycle_improved=False)
        assert mode_selector(state) == REVIEW

    def test_a_degraded_critic_that_found_progress_still_revises(self, config_factory):
        """Degrading loosens validation; it does not spend the revision budget."""
        set_node_config(config_factory(revision_retry_cap=RETRY_CAP,
                                       critic_degrade_threshold=2))
        state = state_with([failure()], retry_count=1, parse_streak=2,
                           last_cycle_improved=True)
        assert mode_selector(state) == REVISE


class TestPurity:
    def test_the_router_mutates_nothing(self):
        state = state_with([failure()], retry_count=RETRY_CAP)
        before = dict(state)

        assert mode_selector(state) == REVIEW

        # The review branch decides nothing: no discard, no accept, no rollback.
        assert dict(state) == before
        assert state["best_seen_draft"] is None
        assert state["review_requested"] is False
