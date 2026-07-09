"""Tests for museai.fsm.routers.mode_selector.

Pure and synchronous: state in, node name out. Nothing is mutated.
"""

from __future__ import annotations

import pytest

from museai.fsm.nodes.deps import set_node_config
from museai.fsm.routers.mode_selector import COMMIT, REVIEW, REVISE, mode_selector
from museai.fsm.state import FSM_Pointer, FailureObject, make_initial_state

RETRY_CAP = 3


def failure(code: str = "CONTRADICTS_CHARACTER") -> FailureObject:
    return FailureObject(
        error_code=code,
        offending_text="Mara lied about the letter.",
        suggested_fix="Mara never lies.",
        critic_source="continuity_critic",
    )


def state_with(failures, retry_count: int):
    return make_initial_state(
        "test-project",
        FSM_Pointer(arc_id="arc-1", chapter_id="arc-1-c01", beat_index=0),
        current_draft_text="prose",
        critic_failures=failures,
        retry_count=retry_count,
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


class TestPurity:
    def test_the_router_mutates_nothing(self):
        state = state_with([failure()], retry_count=RETRY_CAP)
        before = dict(state)

        assert mode_selector(state) == REVIEW

        # The review branch decides nothing: no discard, no accept, no rollback.
        assert dict(state) == before
        assert state["best_seen_draft"] is None
        assert state["review_requested"] is False
