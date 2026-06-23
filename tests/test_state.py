"""Module: M01 (Coordinator & State Machine)
Synthetic tests for FSM state schemas and initial defaults.
"""

from typing import get_args

import pytest

from fsm.state import (
    FSM_Pointer,
    FailureObject,
    OrchestratorState,
    accumulate_or_reset,
    failure_object_json_schema,
    make_initial_state,
)


STATE_FIELDS = {
    "project_id",
    "fsm_pointer",
    "active_context_package",
    "current_draft_text",
    "streaming_buffer",
    "critic_failures",
    "stylometric_distance",
    "retry_count",
    "replan_count",
    "escalation_tier",
    "has_paradox",
    "transient_dc_override",
    "pause_requested",
    "hard_stop_asserted",
    "failed_beat_cache",
    "best_seen_draft",
}


def make_pointer() -> FSM_Pointer:
    return FSM_Pointer(
        arc_id="arc-1",
        chapter_id="chapter-1",
        scene_id="scene-1",
        beat_index=0,
    )


def make_failure() -> FailureObject:
    return FailureObject(
        error_code="PACING_ISSUE",
        offending_text="beat-marker",
        suggested_fix="slow down",
        critic_source="pacing",
    )


def test_pointer_schema_exposes_design_fields() -> None:
    pointer = make_pointer()

    assert pointer.model_dump() == {
        "arc_id": "arc-1",
        "chapter_id": "chapter-1",
        "scene_id": "scene-1",
        "beat_index": 0,
    }
    assert set(FSM_Pointer.model_json_schema()["properties"]) == {
        "arc_id",
        "chapter_id",
        "scene_id",
        "beat_index",
    }


def test_failure_schema_exposes_design_fields() -> None:
    failure = make_failure()

    assert failure.model_dump() == {
        "error_code": "PACING_ISSUE",
        "offending_text": "beat-marker",
        "suggested_fix": "slow down",
        "critic_source": "pacing",
    }
    assert set(failure_object_json_schema()["properties"]) == {
        "error_code",
        "offending_text",
        "suggested_fix",
        "critic_source",
    }


def test_initial_state_defaults_match_design() -> None:
    pointer = make_pointer()
    state = make_initial_state("project-1", pointer)

    assert set(state) == STATE_FIELDS
    assert set(OrchestratorState.__annotations__) == STATE_FIELDS
    assert state == {
        "project_id": "project-1",
        "fsm_pointer": pointer,
        "active_context_package": {},
        "current_draft_text": "",
        "streaming_buffer": "",
        "critic_failures": [],
        "stylometric_distance": 0.0,
        "retry_count": 0,
        "replan_count": 0,
        "escalation_tier": 0,
        "has_paradox": False,
        "transient_dc_override": None,
        "pause_requested": False,
        "hard_stop_asserted": False,
        "failed_beat_cache": [],
        "best_seen_draft": None,
    }


def test_initial_state_overrides_known_keys_and_rejects_unknown_keys() -> None:
    pointer = make_pointer()
    failure = make_failure()

    state = make_initial_state(
        "project-1",
        pointer,
        current_draft_text="synthetic draft",
        critic_failures=[failure],
        retry_count=2,
    )

    assert state["current_draft_text"] == "synthetic draft"
    assert state["critic_failures"] == [failure]
    assert state["retry_count"] == 2

    with pytest.raises(ValueError, match="Unknown OrchestratorState override key"):
        make_initial_state("project-1", pointer, unknown_field=True)


def test_initial_state_mutable_defaults_are_not_shared() -> None:
    pointer = make_pointer()
    first = make_initial_state("project-1", pointer)
    second = make_initial_state("project-2", pointer)

    first["active_context_package"]["marker"] = "first"
    first["critic_failures"].append(make_failure())
    first["failed_beat_cache"].append({"fingerprint": "first"})

    assert second["active_context_package"] == {}
    assert second["critic_failures"] == []
    assert second["failed_beat_cache"] == []
    assert first["active_context_package"] is not second["active_context_package"]
    assert first["critic_failures"] is not second["critic_failures"]
    assert first["failed_beat_cache"] is not second["failed_beat_cache"]


def test_accumulate_or_reset_appends_non_empty_contributions_across_calls() -> None:
    current = accumulate_or_reset([], [make_failure()])
    current = accumulate_or_reset(current, [make_failure()])

    assert len(current) == 2
    assert [failure.error_code for failure in current] == [
        "PACING_ISSUE",
        "PACING_ISSUE",
    ]


def test_accumulate_or_reset_explicit_empty_list_resets() -> None:
    current = accumulate_or_reset([make_failure()], [make_failure()])

    assert len(current) == 2
    assert accumulate_or_reset(current, []) == []


def test_failed_beat_cache_uses_accumulate_or_reset_reducer() -> None:
    reducer = get_args(OrchestratorState.__annotations__["failed_beat_cache"])[1]
    current = reducer([{"fingerprint": "first"}], [{"fingerprint": "second"}])

    assert reducer is accumulate_or_reset
    assert current == [{"fingerprint": "first"}, {"fingerprint": "second"}]
    assert reducer(current, []) == []
