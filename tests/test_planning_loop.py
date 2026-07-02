"""Module: M05 (Hierarchical Planning Cascade)
Deterministic tests for the bounded planner deliberation loop.
"""

import asyncio
from types import SimpleNamespace

from fsm.planning_actions import PlannerAction
from fsm.planning_loop import (
    OUTCOME_ESCALATE,
    OUTCOME_FALLBACK_BASELINE,
    OUTCOME_FINALIZED,
    OUTCOME_NEEDS_CLARIFICATION,
    run_planner_loop,
)
from fsm.planning_validators import run_validators


def _config(*, loops=3, tools=2):
    return SimpleNamespace(
        planning=SimpleNamespace(
            planner_max_deliberation_loops={"beat": loops},
            planner_max_tool_calls_per_loop={"beat": tools},
            planner_required_checks={
                "beat": ["schema", "draftability", "no_drafting"]
            },
            planner_max_accumulated_tool_results=4,
        )
    )


def _valid_plan(label="valid"):
    return {
        "plan_id": label,
        "immediate_objective": "cross the threshold",
        "physical_constraints": "locked gate and rain-slick stones",
    }


def _invalid_plan(label="invalid"):
    return {"plan_id": label, "immediate_objective": "cross the threshold"}


class ScriptedDecider:
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = []

    def __call__(self, state):
        self.calls.append(state)
        if self.actions:
            return self.actions.pop(0)
        return PlannerAction(action_type="continue_deliberation", rationale="hold")


class FakeRegistry:
    def __init__(self, permitted=True):
        self._permitted = permitted
        self.calls = []

    def permitted(self, level, tool_name):
        return bool(self._permitted)

    def call(self, level, tool_name, args, snapshot_id, loop_index):
        self.calls.append(
            {
                "level": level,
                "tool_name": tool_name,
                "args": dict(args),
                "snapshot_id": snapshot_id,
                "loop_index": loop_index,
            }
        )
        return {
            "tool": tool_name,
            "outcome": "executed",
            "available": True,
            "data": {"loop_index": loop_index},
            "reason": "ok",
            "trace_id": f"trace-{loop_index}",
        }


def _run(decider, *, config=None, registry=None, baseline=None):
    return asyncio.run(
        run_planner_loop(
            level="beat",
            snapshot_id="snap-1",
            target_node={"node_id": "beat-node"},
            constraints={},
            continuity={},
            registry=registry or FakeRegistry(),
            config=config or _config(),
            planner_decider=decider,
            deterministic_baseline=baseline,
        )
    )


def _assert_valid_if_non_escalate(outcome, config):
    if outcome.outcome in {OUTCOME_FINALIZED, OUTCOME_FALLBACK_BASELINE}:
        assert outcome.plan is not None
        assert run_validators("beat", outcome.plan, {}, {}, config).passes


def test_valid_finalize_returns_finalized_and_uses_injected_decider_only():
    config = _config()
    registry = FakeRegistry()
    decider = ScriptedDecider(
        [
            PlannerAction(
                action_type="finalize_plan",
                final_plan=_valid_plan("final"),
                self_check={"schema_valid": False},
            )
        ]
    )

    outcome = _run(decider, config=config, registry=registry)

    assert outcome.outcome == OUTCOME_FINALIZED
    assert outcome.plan == _valid_plan("final")
    assert outcome.validation.passes is True
    assert len(decider.calls) == 1
    assert registry.calls == []
    _assert_valid_if_non_escalate(outcome, config)


def test_invalid_finalization_is_not_accepted_and_falls_back_if_never_repaired():
    config = _config(loops=1)
    decider = ScriptedDecider(
        [PlannerAction(action_type="finalize_plan", final_plan=_invalid_plan())]
    )

    outcome = _run(
        decider,
        config=config,
        baseline=lambda level, target_node, constraints: _valid_plan("baseline"),
    )

    assert outcome.outcome == OUTCOME_FALLBACK_BASELINE
    assert outcome.baseline_source == "deterministic_baseline"
    assert outcome.plan == _valid_plan("baseline")
    assert outcome.records[0]["accepted"] is False
    assert outcome.records[0]["failed_checks"] == ["draftability"]
    _assert_valid_if_non_escalate(outcome, config)


def test_tool_call_caps_and_loop_caps_are_enforced():
    config = _config(loops=3, tools=1)
    registry = FakeRegistry()
    decider = ScriptedDecider(
        [
            PlannerAction(
                action_type="call_tool",
                tool_name="read_beats",
                tool_args={"scene_id": "scene-1"},
            ),
            PlannerAction(
                action_type="call_tool",
                tool_name="read_scenes",
                tool_args={"chapter_id": "chapter-1"},
            ),
            PlannerAction(action_type="continue_deliberation", rationale="one more turn"),
        ]
    )

    outcome = _run(
        decider,
        config=config,
        registry=registry,
        baseline=lambda level, target_node, constraints: _valid_plan("baseline"),
    )

    assert outcome.loops_used == 3
    assert len(decider.calls) == 3
    assert outcome.tool_call_count == 1
    assert [call["tool_name"] for call in registry.calls] == ["read_beats"]
    refused = outcome.records[1]
    assert refused["accepted"] is False
    assert refused["wasted"] is True
    assert "tool-call cap reached" in refused["reason"]
    assert outcome.outcome == OUTCOME_FALLBACK_BASELINE
    _assert_valid_if_non_escalate(outcome, config)


def test_fallback_prefers_best_valid_candidate_before_baseline():
    config = _config(loops=2)
    decider = ScriptedDecider(
        [
            PlannerAction(action_type="revise_plan", revised_plan=_valid_plan("best")),
            PlannerAction(action_type="continue_deliberation", rationale="consider"),
        ]
    )

    outcome = _run(
        decider,
        config=config,
        baseline=lambda level, target_node, constraints: _valid_plan("baseline"),
    )

    assert outcome.outcome == OUTCOME_FALLBACK_BASELINE
    assert outcome.baseline_source == "best_valid"
    assert outcome.plan == _valid_plan("best")
    _assert_valid_if_non_escalate(outcome, config)


def test_fallback_uses_valid_deterministic_baseline_then_escalates_without_one():
    baseline_config = _config(loops=1)
    baseline_decider = ScriptedDecider(
        [PlannerAction(action_type="revise_plan", revised_plan=_invalid_plan())]
    )

    baseline_outcome = _run(
        baseline_decider,
        config=baseline_config,
        baseline=lambda level, target_node, constraints: _valid_plan("baseline"),
    )

    assert baseline_outcome.outcome == OUTCOME_FALLBACK_BASELINE
    assert baseline_outcome.baseline_source == "deterministic_baseline"
    _assert_valid_if_non_escalate(baseline_outcome, baseline_config)

    escalate_config = _config(loops=1)
    escalate_decider = ScriptedDecider(
        [PlannerAction(action_type="finalize_plan", final_plan=_invalid_plan())]
    )
    escalate_outcome = _run(
        escalate_decider,
        config=escalate_config,
        baseline=lambda level, target_node, constraints: _invalid_plan("bad-baseline"),
    )

    assert escalate_outcome.outcome == OUTCOME_ESCALATE
    assert escalate_outcome.plan is None
    assert escalate_outcome.validation.passes is False


def test_no_non_escalate_outcome_returns_invalid_plan():
    config = _config(loops=2)
    outcomes = [
        _run(
            ScriptedDecider(
                [PlannerAction(action_type="finalize_plan", final_plan=_valid_plan("final"))]
            ),
            config=config,
        ),
        _run(
            ScriptedDecider(
                [
                    PlannerAction(
                        action_type="revise_plan", revised_plan=_valid_plan("best")
                    ),
                    PlannerAction(action_type="continue_deliberation", rationale="hold"),
                ]
            ),
            config=config,
            baseline=lambda level, target_node, constraints: _invalid_plan("unused"),
        ),
        _run(
            ScriptedDecider(
                [PlannerAction(action_type="finalize_plan", final_plan=_invalid_plan())]
            ),
            config=config,
            baseline=lambda level, target_node, constraints: _valid_plan("baseline"),
        ),
    ]

    assert [outcome.outcome for outcome in outcomes] == [
        OUTCOME_FINALIZED,
        OUTCOME_FALLBACK_BASELINE,
        OUTCOME_FALLBACK_BASELINE,
    ]
    for outcome in outcomes:
        _assert_valid_if_non_escalate(outcome, config)


def test_raise_conflict_returns_needs_clarification_and_skips_fallback():
    baseline_calls = []
    decider = ScriptedDecider(
        [
            PlannerAction(
                action_type="raise_conflict",
                conflicts=[
                    {
                        "conflict_type": "annotation_conflict",
                        "description": "pin and remove target the same node",
                        "requires_user_resolution": True,
                    }
                ],
            )
        ]
    )

    def _baseline(level, target_node, constraints):
        baseline_calls.append((level, target_node, constraints))
        return _valid_plan("baseline")

    outcome = _run(decider, baseline=_baseline)

    assert outcome.outcome == OUTCOME_NEEDS_CLARIFICATION
    assert outcome.plan is None
    assert outcome.conflicts[0]["conflict_type"] == "annotation_conflict"
    assert baseline_calls == []
