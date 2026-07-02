"""Module: M05 (Hierarchical Planning Cascade)
Deterministic full-cascade integration tests over real temp stores.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import fsm.nodes.node_plan_arc as node_plan_arc_mod
import fsm.nodes.node_plan_beat as node_plan_beat_mod
import fsm.nodes.node_plan_chapter as node_plan_chapter_mod
import fsm.nodes.node_plan_global as node_plan_global_mod
import fsm.nodes.node_plan_scene as node_plan_scene_mod
from fsm.pad_translation import compose_baseline_string
from fsm.planning_actions import PlannerAction
from fsm.planning_loop import LoopOutcome
from fsm.planning_node_support import persist_loop_outcome
from fsm.planning_validators import ValidationResult
from fsm.state import FSM_Pointer, make_initial_state
from memory.event_log import init_event_log, iter_events
from memory.provisional_store import init_provisional_store, upsert_claim
from memory.sqlite_db import (
    create_planning_snapshot,
    get_annotations_for_snapshot,
    get_beat,
    get_beats_for_scene_ordered,
    get_chapters_for_arc,
    get_planning_nodes,
    get_planning_nodes_by_parent,
    get_planning_snapshot,
    get_revisions_for_snapshot,
    get_scenes_for_chapter_ordered,
    init_db,
    insert_planning_annotation,
    upsert_planning_node,
)


PROJECT_ID = "project-full-cascade"
SNAPSHOT_ID = f"snap_{PROJECT_ID}"
GLOBAL_NODE_ID = f"{SNAPSHOT_ID}:global"
LEVELS = ("global", "arc", "chapter", "scene", "beat")


class ScriptedDecider:
    """Return one deterministic PlannerAction per deliberation turn."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = []

    def __call__(self, loop_state):
        self.calls.append(loop_state)
        if not self.actions:
            raise AssertionError("scripted decider exhausted")
        return self.actions.pop(0)


class ScriptedPlannerFactory:
    """Patch target for make_planner_decider that records each node base_context."""

    def __init__(self, scripts_by_node):
        self.scripts_by_node = {
            node_name: [list(script) for script in scripts]
            for node_name, scripts in scripts_by_node.items()
        }
        self.contexts = {node_name: [] for node_name in scripts_by_node}
        self.deciders = {node_name: [] for node_name in scripts_by_node}

    def __call__(self, node_name, base_context, config):
        del config
        queue = self.scripts_by_node.get(node_name)
        if not queue:
            raise AssertionError(f"no scripted decider queued for {node_name}")
        self.contexts.setdefault(node_name, []).append(base_context)
        decider = ScriptedDecider(queue.pop(0))
        self.deciders.setdefault(node_name, []).append(decider)
        return decider


class NoToolRegistry:
    """Registry that fails if a no-tool script accidentally asks for a tool."""

    def permitted(self, level, tool_name):
        raise AssertionError(f"unexpected permission check: {level}:{tool_name}")

    def call(self, level, tool_name, args, snapshot_id, loop_index):
        raise AssertionError(f"unexpected tool call: {level}:{tool_name}")


class FailingDecider:
    """Decider used to prove a clarification block happens before planner input."""

    def __init__(self):
        self.calls = []

    def __call__(self, loop_state):
        self.calls.append(loop_state)
        raise AssertionError("planner decider should not be called")


def _run(coro):
    return asyncio.run(coro)


def _finalize(plan):
    return PlannerAction(
        action_type="finalize_plan",
        final_plan=plan,
        self_check={"schema_valid": False},
    )


def _stores(tmp_path: Path):
    root = tmp_path / "cascade"
    db_path = root / "fictionwriter.sqlite3"
    provisional_path = root / "provisional.sqlite3"
    event_log_path = root / "events" / "planning.jsonl"
    init_db(db_path)
    init_provisional_store(provisional_path)
    init_event_log(None, event_log_path)
    upsert_claim(
        provisional_path,
        claim_id="claim-premise",
        source_ref="premise",
        subject_id=PROJECT_ID,
        claim_text="Synthetic premise: a keeper must honor a lighthouse vow.",
        confidence=0.8,
    )
    return SimpleNamespace(
        db_path=db_path,
        planning_db_path=db_path,
        provisional_path=provisional_path,
        event_log_path=event_log_path,
    )


@pytest.fixture
def stores(tmp_path):
    return _stores(tmp_path)


def _config(*, alpha=1.0, beats_per_scene_min=2):
    required_checks = {
        "global": [
            "schema",
            "no_drafting",
            "arc_coverage",
            "major_promise_payoff",
            "arc_diversity",
            "promise_payoff_distinct",
        ],
        "arc": ["schema", "no_drafting", "escalation", "thread_distribution"],
        "chapter": [
            "schema",
            "no_drafting",
            "depth",
            "chapter_function",
            "pacing",
            "annotation_satisfaction",
        ],
        "scene": [
            "schema",
            "no_drafting",
            "continuity",
            "depth",
            "scene_function",
            "entry_exit_state",
        ],
        "beat": ["schema", "no_drafting", "continuity", "draftability", "pad_grounding"],
    }
    return SimpleNamespace(
        planning=SimpleNamespace(
            execution_mode="macro_outline_before_draft",
            approval_mode="off",
            planner_max_deliberation_loops={level: 1 for level in LEVELS},
            planner_max_tool_calls_per_loop={level: 0 for level in LEVELS},
            planner_max_accumulated_tool_results=4,
            planner_required_checks=required_checks,
            baseline_act_count=1,
            baseline_word_weights=[1.0],
            creative_second_pass_enabled=False,
        ),
        runtime=SimpleNamespace(
            word_count_target=9000,
            beats_per_scene_min=beats_per_scene_min,
            model_validate_retry_cap=2,
        ),
        thresholds=SimpleNamespace(pad_ewma_alpha=alpha),
        endpoints=SimpleNamespace(planner=SimpleNamespace(name="unused-planner")),
    )


def _state(stores, config):
    state = make_initial_state(
        PROJECT_ID,
        FSM_Pointer(arc_id="", chapter_id="", scene_id="", beat_index=0),
        planning_execution_mode=config.planning.execution_mode,
        approval_mode=config.planning.approval_mode,
    )
    state["app_config"] = config
    state["sqlite_db_path"] = stores.db_path
    state["planning_db_path"] = stores.planning_db_path
    state["provisional_store_path"] = stores.provisional_path
    state["event_log_path"] = stores.event_log_path
    state["project_metadata"] = {
        "genre": "synthetic cascade",
        "target_word_count": config.runtime.word_count_target,
        "premise_seed": "A keeper must honor a lighthouse vow before the harbor fails.",
    }
    state["world_rules"] = ["The harbor light is the visible measure of public trust."]
    return state


def _global_plan():
    return {
        "premise": "A keeper must honor a lighthouse vow before the harbor fails.",
        "central_conflict": "The keeper must choose between a private vow and public safety.",
        "ending_target": "The harbor is saved when the vow's public cost is revealed.",
        "arcs": [
            {
                "arc_id": "arc-vow",
                "title": "The Vow Tightens",
                "function": "establish the vow, pressure, and first irreversible choice",
                "word_allocation": 9000,
            }
        ],
        "promises": [
            {
                "id": "promise-vow",
                "promise": "the lighthouse vow hides a public consequence",
                "payoff": "the consequence is revealed when the keeper saves the harbor",
            }
        ],
    }


def _arc_plan():
    return {
        "arcs": [
            {
                "arc_id": "arc-vow",
                "title": "The Vow Tightens",
                "function": "raise the vow pressure through two linked choices",
                "character_milestones": ["the keeper accepts the public cost"],
                "chapters": [
                    {
                        "chapter_id": "ch-choice",
                        "stub": "ch-choice turns the vow into a practical choice",
                    },
                    {
                        "chapter_id": "ch-cost",
                        "stub": "ch-cost makes the public cost impossible to ignore",
                    },
                ],
            }
        ],
        "milestones": [
            {"arc_id": "arc-vow", "label": "choice", "tension": 1},
            {"arc_id": "arc-vow", "label": "cost", "tension": 2},
        ],
        "thread_distribution": [
            {"thread_id": "main", "lifecycle": "open->progress", "in_arcs": ["arc-vow"]}
        ],
    }


def _chapter_plan(chapter_id):
    return {
        "chapter_id": chapter_id,
        "dramatic_function": f"{chapter_id} forces the keeper to act on the vow",
        "expected_emotional_shift": "guarded resolve toward alarm",
        "pacing": "measured escalation",
        "obligations": {
            "thread_obligations": [{"thread_id": "main", "required_progress": "raise"}],
            "causal_prerequisites": ["the vow is still secret"],
            "causal_deliverables": [f"{chapter_id} makes inaction impossible"],
        },
        "annotation_outcomes": {},
        "scene_planning_constraints": [
            f"{chapter_id} must stage the vow as a visible choice",
            f"{chapter_id} must close with a concrete cost",
        ],
    }


def _scene_plan(scene_id, *, word_budget=0, pad_target=None):
    plan = {
        "scene_id": scene_id,
        "scene_function": f"{scene_id} makes the vow cost visible",
        "setting": "the lighthouse threshold",
        "participants": ["keeper", "witness"],
        "entry_state": "the keeper can still hide the vow",
        "exit_state": f"{scene_id} leaves the keeper publicly committed",
        "conflict_turn": "the witness blocks the easy exit",
        "asserted_facts": [],
        "continuity_constraints": [],
        "word_budget": word_budget,
    }
    if pad_target is not None:
        plan["pad_target"] = pad_target
    return plan


def _beat_plan(beat_id):
    return {
        "beat_id": beat_id,
        "immediate_objective": f"{beat_id} forces the keeper to cross the threshold",
        "physical_constraints": "lamp, locked stair, and witness line of sight",
        "entry_condition": "the keeper hesitates at the threshold",
        "exit_condition": f"{beat_id} lands the threshold choice",
        "pad_target": {"pleasure": 9.0, "arousal": 9.0, "dominance": 9.0},
        "behavioral_constraint": "MODEL-SHOULD-BE-OVERWRITTEN",
        "asserted_facts": [],
    }


def _install_factory(monkeypatch, factory):
    for module in (
        node_plan_global_mod,
        node_plan_arc_mod,
        node_plan_chapter_mod,
        node_plan_scene_mod,
        node_plan_beat_mod,
    ):
        monkeypatch.setattr(module, "make_planner_decider", factory)


def _snapshot_active_revision(stores):
    snapshot = get_planning_snapshot(stores.db_path, SNAPSHOT_ID)
    assert snapshot is not None
    return snapshot["active_revision_id"]


def _assert_state_revision_matches_snapshot(stores, state):
    assert state["planning_snapshot_id"] == SNAPSHOT_ID
    assert state["active_planning_revision_id"] == _snapshot_active_revision(stores)


def _revision_count(stores):
    return len(get_revisions_for_snapshot(stores.db_path, SNAPSHOT_ID))


def _purpose_by_node(stores, node_id):
    node = next(
        node for node in get_planning_nodes(stores.db_path, SNAPSHOT_ID)
        if node["node_id"] == node_id
    )
    return json.loads(node["purpose"] or "{}")


def test_full_cascade_threads_real_nodes_stores_and_macro_readiness(
    stores, monkeypatch
):
    config = _config(alpha=1.0, beats_per_scene_min=2)
    factory = ScriptedPlannerFactory(
        {
            "node_plan_global": [[_finalize(_global_plan())]],
            "node_plan_arc": [[_finalize(_arc_plan())]],
            "node_plan_chapter": [
                [_finalize(_chapter_plan("ch-choice"))],
                [_finalize(_chapter_plan("ch-cost"))],
            ],
            "node_plan_scene": [
                [_finalize(_scene_plan("scene-one", word_budget=0))],
                [
                    _finalize(
                        _scene_plan(
                            "scene-two",
                            word_budget=0,
                            pad_target={
                                "pleasure": 1.0,
                                "arousal": 1.0,
                                "dominance": -1.0,
                            },
                        )
                    )
                ],
            ],
            "node_plan_beat": [
                [_finalize(_beat_plan("beat-one"))],
                [_finalize(_beat_plan("beat-two"))],
            ],
        }
    )
    _install_factory(monkeypatch, factory)

    state = _state(stores, config)
    revision_counts = []
    ready_values = []

    state = _run(node_plan_global_mod.node_plan_global(state, registry=NoToolRegistry()))
    _assert_state_revision_matches_snapshot(stores, state)
    revision_counts.append(_revision_count(stores))
    ready_values.append(state["macro_outline_ready"])
    assert get_planning_snapshot(stores.db_path, SNAPSHOT_ID) is not None

    state = _run(node_plan_arc_mod.node_plan_arc(state, registry=NoToolRegistry()))
    _assert_state_revision_matches_snapshot(stores, state)
    revision_counts.append(_revision_count(stores))
    ready_values.append(state["macro_outline_ready"])

    arc_context = factory.contexts["node_plan_arc"][0]
    assert arc_context["global_plan"]["premise"] == _global_plan()["premise"]
    arc_node = get_planning_nodes(stores.db_path, SNAPSHOT_ID, level="arc")[0]
    assert arc_node["parent_id"] == GLOBAL_NODE_ID
    assert _purpose_by_node(stores, GLOBAL_NODE_ID)["arcs"][0]["arc_id"] == "arc-vow"

    state = _run(node_plan_chapter_mod.node_plan_chapter(state, registry=NoToolRegistry()))
    _assert_state_revision_matches_snapshot(stores, state)
    revision_counts.append(_revision_count(stores))
    ready_values.append(state["macro_outline_ready"])

    state = _run(node_plan_chapter_mod.node_plan_chapter(state, registry=NoToolRegistry()))
    _assert_state_revision_matches_snapshot(stores, state)
    revision_counts.append(_revision_count(stores))
    ready_values.append(state["macro_outline_ready"])

    assert ready_values == [False, False, False, True]
    chapter_contexts = factory.contexts["node_plan_chapter"]
    assert [ctx["active_chapter"]["chapter_id"] for ctx in chapter_contexts] == [
        "ch-choice",
        "ch-cost",
    ]
    assert chapter_contexts[0]["active_chapter"]["stub"] == (
        "ch-choice turns the vow into a practical choice"
    )
    ch_cost_node = f"{SNAPSHOT_ID}:chapter:ch-cost"
    ch_cost_plan = _purpose_by_node(stores, ch_cost_node)
    assert ch_cost_plan["obligations"]["causal_deliverables"] == [
        "ch-cost makes inaction impossible"
    ]
    assert ch_cost_plan["scene_planning_constraints"] == [
        "ch-cost must stage the vow as a visible choice",
        "ch-cost must close with a concrete cost",
    ]
    assert "scenes" not in ch_cost_plan
    assert get_scenes_for_chapter_ordered(stores.db_path, "ch-choice") == []
    assert get_scenes_for_chapter_ordered(stores.db_path, "ch-cost") == []
    assert [row["id"] for row in get_chapters_for_arc(stores.db_path, "arc-vow")] == [
        "ch-choice",
        "ch-cost",
    ]

    state = _run(node_plan_scene_mod.node_plan_scene(state, registry=NoToolRegistry()))
    _assert_state_revision_matches_snapshot(stores, state)
    revision_counts.append(_revision_count(stores))

    state = _run(node_plan_scene_mod.node_plan_scene(state, registry=NoToolRegistry()))
    _assert_state_revision_matches_snapshot(stores, state)
    revision_counts.append(_revision_count(stores))

    scene_contexts = factory.contexts["node_plan_scene"]
    assert scene_contexts[0]["chapter_plan"]["scene_planning_constraints"] == (
        ch_cost_plan["scene_planning_constraints"]
    )
    assert scene_contexts[1]["planned_scenes"][0]["scene_id"] == "scene-one"
    scenes = get_scenes_for_chapter_ordered(stores.db_path, "ch-cost")
    assert [scene["id"] for scene in scenes] == ["scene-one", "scene-two"]
    assert [scene["ordering"] for scene in scenes] == [0, 1]

    state = _run(
        node_plan_beat_mod.node_plan_beat(
            state, registry=NoToolRegistry(), adapt_fn=None
        )
    )
    _assert_state_revision_matches_snapshot(stores, state)
    revision_counts.append(_revision_count(stores))
    assert state["fsm_pointer"].beat_id == "beat-one"
    assert state["fsm_pointer"].beat_index == 0
    assert state["scene_needs_more"] is True

    state = _run(
        node_plan_beat_mod.node_plan_beat(
            state, registry=NoToolRegistry(), adapt_fn=None
        )
    )
    _assert_state_revision_matches_snapshot(stores, state)
    revision_counts.append(_revision_count(stores))
    assert state["fsm_pointer"].beat_id == "beat-two"
    assert state["fsm_pointer"].beat_index == 1
    assert state["scene_needs_more"] is False

    beat_rows = get_beats_for_scene_ordered(stores.db_path, "scene-two")
    assert [beat["id"] for beat in beat_rows] == ["beat-one", "beat-two"]
    assert [beat["beat_index"] for beat in beat_rows] == [0, 1]
    expected_pad_constraint = compose_baseline_string("P+A+D-")
    beat_one_plan = _purpose_by_node(stores, f"{SNAPSHOT_ID}:beat:beat-one")
    assert beat_one_plan["behavioral_constraint"] == expected_pad_constraint
    assert beat_one_plan["behavioral_constraint"] != "MODEL-SHOULD-BE-OVERWRITTEN"
    assert get_beat(stores.db_path, "beat-one")["status"] == "planned"
    pad_trace = [
        item for item in state["planner_deliberation_trace"]
        if item.get("phase") == "pad_grounding"
    ]
    assert [item["rung"] for item in pad_trace] == ["static", "static"]

    assert revision_counts == list(range(1, 9))
    events = list(iter_events(stores.event_log_path))
    assert [event["level"] for event in events] == [
        "global",
        "arc",
        "chapter",
        "chapter",
        "scene",
        "scene",
        "beat",
        "beat",
    ]
    assert len({event["revision_id"] for event in events}) == len(events)
    assert state["active_planning_revision_id"] == events[-1]["revision_id"]


def test_hard_annotation_conflict_blocks_before_chapter_decider(stores, monkeypatch):
    config = _config()
    factory = ScriptedPlannerFactory(
        {
            "node_plan_global": [[_finalize(_global_plan())]],
            "node_plan_arc": [[_finalize(_arc_plan())]],
        }
    )
    _install_factory(monkeypatch, factory)
    state = _state(stores, config)
    state = _run(node_plan_global_mod.node_plan_global(state, registry=NoToolRegistry()))
    state = _run(node_plan_arc_mod.node_plan_arc(state, registry=NoToolRegistry()))
    revisions_before = _revision_count(stores)
    events_before = list(iter_events(stores.event_log_path))

    target_node_id = f"{SNAPSHOT_ID}:chapter:ch-choice"
    insert_planning_annotation(
        stores.db_path,
        annotation_id="ann-pin-choice",
        snapshot_id=SNAPSHOT_ID,
        target_node_id=target_node_id,
        target_level="chapter",
        note_type="pin",
        scope="this_node",
        priority="hard",
        text="Keep ch-choice exactly in this chapter slot.",
    )
    insert_planning_annotation(
        stores.db_path,
        annotation_id="ann-remove-choice",
        snapshot_id=SNAPSHOT_ID,
        target_node_id=target_node_id,
        target_level="chapter",
        note_type="remove",
        scope="this_node",
        priority="hard",
        text="Remove ch-choice from the outline.",
    )

    decider = FailingDecider()
    result = _run(
        node_plan_chapter_mod.node_plan_chapter(
            state, decider=decider, registry=NoToolRegistry()
        )
    )

    assert decider.calls == []
    statuses = {
        row["annotation_id"]: row["status"]
        for row in get_annotations_for_snapshot(stores.db_path, SNAPSHOT_ID)
    }
    assert statuses["ann-pin-choice"] == "needs_clarification"
    assert statuses["ann-remove-choice"] == "needs_clarification"
    assert result["planning_block_reason"] == "unresolved_hard_conflict"
    assert result["fsm_pointer"].chapter_id == ""
    assert _revision_count(stores) == revisions_before
    assert list(iter_events(stores.event_log_path)) == events_before
    assert get_scenes_for_chapter_ordered(stores.db_path, "ch-choice") == []
    chapter_plan = _purpose_by_node(stores, target_node_id)
    assert chapter_plan.get("dramatic_function") is None


def test_persist_loop_outcome_replay_keeps_canonical_rows_idempotent(stores):
    create_planning_snapshot(
        stores.db_path,
        snapshot_id=SNAPSHOT_ID,
        project_id=PROJECT_ID,
        mode="macro_outline_before_draft",
    )
    prior_nodes = get_planning_nodes(stores.db_path, SNAPSHOT_ID)
    plan = _global_plan()

    def persist_plan_fn(validated_plan):
        upsert_planning_node(
            stores.db_path,
            node_id=GLOBAL_NODE_ID,
            snapshot_id=SNAPSHOT_ID,
            level="global",
            status="planned",
            title=validated_plan["premise"],
            summary=validated_plan["central_conflict"],
            purpose=json.dumps(validated_plan, sort_keys=True),
        )

    outcome = LoopOutcome(
        outcome="finalized",
        level="global",
        plan=plan,
        validation=ValidationResult(passes=True),
        best_valid_plan=plan,
    )

    first = persist_loop_outcome(
        outcome,
        level="global",
        snapshot_id=SNAPSHOT_ID,
        target_node=GLOBAL_NODE_ID,
        persist_plan_fn=persist_plan_fn,
        db_path=stores.db_path,
        event_log_path=stores.event_log_path,
        prior_nodes=prior_nodes,
    )
    second = persist_loop_outcome(
        outcome,
        level="global",
        snapshot_id=SNAPSHOT_ID,
        target_node=GLOBAL_NODE_ID,
        persist_plan_fn=persist_plan_fn,
        db_path=stores.db_path,
        event_log_path=stores.event_log_path,
        prior_nodes=prior_nodes,
    )

    assert first.event_written is True
    assert second.event_written is True
    assert first.revision_id == second.revision_id
    assert get_planning_snapshot(stores.db_path, SNAPSHOT_ID)["active_revision_id"] == (
        first.revision_id
    )
    assert [node["node_id"] for node in get_planning_nodes(stores.db_path, SNAPSHOT_ID)] == [
        GLOBAL_NODE_ID
    ]
    revisions = get_revisions_for_snapshot(stores.db_path, SNAPSHOT_ID)
    assert [revision["revision_id"] for revision in revisions] == [first.revision_id]
    events = list(iter_events(stores.event_log_path))
    assert [event["revision_id"] for event in events] == [
        first.revision_id,
        first.revision_id,
    ]
    assert events[0]["diff"] == events[1]["diff"]
