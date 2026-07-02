"""Module: M05 (Hierarchical Planning Cascade)
Deterministic tests for planner decider seams and loop-outcome persistence.
"""

import asyncio

import pytest

from fsm.planning_actions import PlannerAction
from fsm.planning_loop import LoopOutcome
from fsm.planning_node_support import (
    PlannerDeciderError,
    make_planner_decider,
    persist_loop_outcome,
)
from fsm.planning_validators import ValidationResult
from llm.call_llm import LLMCallError
from memory.event_log import iter_events
from memory.sqlite_db import (
    create_planning_snapshot,
    get_planning_nodes,
    get_planning_snapshot,
    get_revisions_for_snapshot,
    upsert_planning_node,
)


class _Loader:
    def __init__(self):
        self.calls = []

    def render(self, node_name, context):
        self.calls.append((node_name, context))
        return f"rendered:{node_name}:{context['level']}:{context['loop_index']}"


class _Endpoint:
    name = "planner-endpoint"


class _Config:
    endpoints = type("Endpoints", (), {"planner": _Endpoint()})()
    runtime = type("Runtime", (), {"model_validate_retry_cap": 2})()


def _db(tmp_path):
    return tmp_path / "planning.db"


def _event_log(tmp_path):
    return tmp_path / "events" / "planning.jsonl"


def _validated_plan(snapshot_id="snap-1", node_id="scene-1"):
    return {
        "planning_nodes": [
            {
                "node_id": node_id,
                "snapshot_id": snapshot_id,
                "level": "scene",
                "status": "planned",
                "parent_id": "chapter-1",
                "ordering": 0,
                "title": "Scene 1",
                "summary": "synthetic scene",
                "purpose": '{"scene_function":"turn"}',
            }
        ]
    }


def _seed_snapshot(db_path):
    create_planning_snapshot(
        db_path,
        snapshot_id="snap-1",
        project_id="project-1",
        mode="macro_outline_before_draft",
    )
    return upsert_planning_node(
        db_path,
        node_id="chapter-1",
        snapshot_id="snap-1",
        level="chapter",
        status="planned",
        title="Chapter 1",
    )


def test_make_planner_decider_renders_node_and_returns_canned_action():
    loader = _Loader()
    calls = []
    canned = PlannerAction(
        action_type="finalize_plan",
        final_plan={"title": "plan"},
        self_check={"schema_valid": False},
    )

    async def _structured(messages, endpoint, *, schema_model, validate_retry_cap):
        calls.append(
            {
                "messages": messages,
                "endpoint": endpoint,
                "schema_model": schema_model,
                "validate_retry_cap": validate_retry_cap,
            }
        )
        return canned

    decider = make_planner_decider(
        "node_plan_scene",
        {"genre": "synthetic"},
        _Config(),
        loader=loader,
        call_structured=_structured,
    )
    action = asyncio.run(
        decider(
            {
                "level": "scene",
                "loop_index": 1,
                "current_plan": {"draft": "structure"},
                "best_valid_plan": None,
                "tool_results": [],
                "constraints": {"hard_annotations": []},
                "continuity": {"facts": []},
                "last_validation": None,
                "last_refusal": None,
            }
        )
    )

    assert action is canned
    assert loader.calls[0][0] == "node_plan_scene"
    assert loader.calls[0][1]["genre"] == "synthetic"
    assert loader.calls[0][1]["compiled_constraints"] == {"hard_annotations": []}
    assert calls[0]["messages"] == [
        {"role": "user", "content": "rendered:node_plan_scene:scene:1"}
    ]
    assert calls[0]["endpoint"] is _Config.endpoints.planner
    assert calls[0]["schema_model"] is PlannerAction
    assert calls[0]["validate_retry_cap"] == 2


def test_decider_structured_output_failure_is_recoverable_error():
    async def _failing_structured(*args, **kwargs):
        raise LLMCallError("cannot parse PlannerAction")

    decider = make_planner_decider(
        "node_plan_scene",
        {},
        _Config(),
        loader=_Loader(),
        call_structured=_failing_structured,
    )

    with pytest.raises(PlannerDeciderError, match="could not obtain a valid"):
        asyncio.run(decider({"level": "scene", "loop_index": 0}))


def test_persist_loop_outcome_finalized_writes_plan_revision_and_event(tmp_path):
    db_path = _db(tmp_path)
    event_log = _event_log(tmp_path)
    target = _seed_snapshot(db_path)
    prior_nodes = get_planning_nodes(db_path, "snap-1")
    calls = []

    def _persist_plan(plan):
        calls.append(plan)

    outcome = LoopOutcome(
        outcome="finalized",
        level="scene",
        plan=_validated_plan(),
        validation=ValidationResult(passes=True),
    )

    result = persist_loop_outcome(
        outcome,
        level="scene",
        snapshot_id="snap-1",
        target_node=target,
        persist_plan_fn=_persist_plan,
        db_path=db_path,
        event_log_path=event_log,
        prior_nodes=prior_nodes,
        annotation_outcomes={"ann-1": "applied"},
    )

    assert calls == [outcome.plan]
    assert result.persisted is True
    assert result.event_written is True
    assert result.revision_id is not None
    assert get_planning_snapshot(db_path, "snap-1")["active_revision_id"] == (
        result.revision_id
    )
    assert [node["node_id"] for node in get_planning_nodes(db_path, "snap-1")] == [
        "chapter-1",
        "scene-1",
    ]
    revisions = get_revisions_for_snapshot(db_path, "snap-1")
    assert [revision["revision_id"] for revision in revisions] == [result.revision_id]
    events = list(iter_events(event_log))
    assert len(events) == 1
    assert events[0]["event_type"] == "planning_commit"
    assert events[0]["revision_id"] == result.revision_id
    assert events[0]["validation_passes"] is True
    assert events[0]["diff"]["added_nodes"] == [
        {"node_id": "scene-1", "level": "scene", "title": "Scene 1"}
    ]


def test_persist_loop_outcome_needs_clarification_sets_block_without_plan(tmp_path):
    db_path = _db(tmp_path)
    event_log = _event_log(tmp_path)
    target = _seed_snapshot(db_path)
    calls = []
    outcome = LoopOutcome(
        outcome="needs_clarification",
        level="scene",
        conflicts=[{"annotation_ids": ["ann-pin", "ann-remove"]}],
    )

    result = persist_loop_outcome(
        outcome,
        level="scene",
        snapshot_id="snap-1",
        target_node=target,
        persist_plan_fn=lambda plan: calls.append(plan),
        db_path=db_path,
        event_log_path=event_log,
    )

    assert calls == []
    assert result.persisted is False
    assert result.planning_block_reason == "unresolved_hard_conflict"
    assert result.conflicts == [{"annotation_ids": ["ann-pin", "ann-remove"]}]
    assert get_planning_snapshot(db_path, "snap-1")["status"] == "revision_requested"
    assert get_revisions_for_snapshot(db_path, "snap-1") == []
    assert list(iter_events(event_log)) == []


def test_persist_loop_outcome_escalate_writes_nothing_and_returns_signal(tmp_path):
    db_path = _db(tmp_path)
    event_log = _event_log(tmp_path)
    target = _seed_snapshot(db_path)
    calls = []
    validation = ValidationResult(passes=False, failed_checks=["schema"])
    outcome = LoopOutcome(outcome="escalate", level="scene", validation=validation)

    result = persist_loop_outcome(
        outcome,
        level="scene",
        snapshot_id="snap-1",
        target_node=target,
        persist_plan_fn=lambda plan: calls.append(plan),
        db_path=db_path,
        event_log_path=event_log,
    )

    assert calls == []
    assert result.persisted is False
    assert result.validation is validation
    assert result.active_planning_revision_id is None
    assert get_revisions_for_snapshot(db_path, "snap-1") == []
    assert list(iter_events(event_log)) == []


def test_persist_loop_outcome_raises_instead_of_persisting_failed_plan(tmp_path):
    db_path = _db(tmp_path)
    event_log = _event_log(tmp_path)
    target = _seed_snapshot(db_path)
    calls = []
    outcome = LoopOutcome(
        outcome="finalized",
        level="scene",
        plan=_validated_plan(),
        validation=ValidationResult(passes=False, failed_checks=["schema"]),
    )

    with pytest.raises(ValueError, match="validator-passed plan"):
        persist_loop_outcome(
            outcome,
            level="scene",
            snapshot_id="snap-1",
            target_node=target,
            persist_plan_fn=lambda plan: calls.append(plan),
            db_path=db_path,
            event_log_path=event_log,
        )

    assert calls == []
    assert get_planning_nodes(db_path, "snap-1", level="scene") == []
    assert get_revisions_for_snapshot(db_path, "snap-1") == []
    assert list(iter_events(event_log)) == []
