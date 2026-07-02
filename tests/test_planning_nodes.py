"""Module: M05 (Hierarchical Planning Cascade)
Deterministic, no-network tests for the five first-class planner nodes.
"""

import asyncio
import copy
import inspect
import json
from types import SimpleNamespace

import pytest

import fsm.nodes.node_plan_arc as node_plan_arc_mod
import fsm.nodes.node_plan_beat as node_plan_beat_mod
import fsm.nodes.node_plan_chapter as node_plan_chapter_mod
import fsm.nodes.node_plan_global as node_plan_global_mod
import fsm.nodes.node_plan_scene as node_plan_scene_mod
from fsm.pad_translation import compose_baseline_string
from fsm.planning_actions import PlannerAction
from fsm.state import FSM_Pointer, make_initial_state
from memory.event_log import init_event_log, iter_events
from memory.provisional_store import init_provisional_store
from memory.sqlite_db import (
    connect_db,
    create_planning_snapshot,
    get_arc,
    get_beat,
    get_beats_for_scene_ordered,
    get_chapters_for_arc,
    get_planning_nodes,
    get_planning_nodes_by_parent,
    get_planning_snapshot,
    get_revisions_for_snapshot,
    get_scenes_for_chapter_ordered,
    init_db,
    upsert_arc_plan,
    upsert_beat_commit,
    upsert_chapter_plan,
    upsert_planning_node,
    upsert_scene_plan,
)


PROJECT_ID = "project-1"
SNAPSHOT_ID = f"snap_{PROJECT_ID}"
INVALID_MARKER = "INVALID_MODEL_PLAN"

_LEVELS = ("global", "arc", "chapter", "scene", "beat")
_ALLOWED_M05_STATE_CHANGES = {
    "fsm_pointer",
    "planning_snapshot_id",
    "active_planning_revision_id",
    "planner_deliberation_trace",
    "planning_block_reason",
    "macro_outline_ready",
    "awaiting_planning_approval",
    "scene_needs_more",
}


class ScriptedDecider:
    """Return one scripted PlannerAction per loop turn."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = []

    def __call__(self, loop_state):
        self.calls.append(loop_state)
        if self.actions:
            return self.actions.pop(0)
        return PlannerAction(action_type="continue_deliberation", rationale="done")


class NoToolRegistry:
    """Registry that fails the test if a synthetic decider asks for a tool."""

    def permitted(self, level, tool_name):
        raise AssertionError(f"unexpected tool permission check: {level}:{tool_name}")

    def call(self, level, tool_name, args, snapshot_id, loop_index):
        raise AssertionError(f"unexpected tool call: {level}:{tool_name}")


def _make_stores(tmp_path, name="case"):
    root = tmp_path / name
    planning_db = root / "planning.sqlite3"
    provisional_db = root / "provisional.sqlite3"
    event_log = root / "events" / "planning.jsonl"
    init_db(planning_db)
    init_provisional_store(provisional_db)
    init_event_log(None, event_log)
    return SimpleNamespace(
        db=planning_db,
        planning_db=planning_db,
        provisional_db=provisional_db,
        event_log=event_log,
    )


@pytest.fixture
def stores(tmp_path):
    return _make_stores(tmp_path)


def _planning_config(
    *,
    loops=1,
    tools=0,
    alpha=0.25,
    beats_per_scene_min=3,
    execution_mode="macro_outline_before_draft",
    approval_mode="off",
):
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
            execution_mode=execution_mode,
            approval_mode=approval_mode,
            planner_max_deliberation_loops={level: loops for level in _LEVELS},
            planner_max_tool_calls_per_loop={level: tools for level in _LEVELS},
            planner_max_accumulated_tool_results=4,
            planner_required_checks=required_checks,
            baseline_act_count=2,
            baseline_word_weights=[0.55, 0.45],
            creative_second_pass_enabled=False,
        ),
        runtime=SimpleNamespace(
            word_count_target=12000,
            beats_per_scene_min=beats_per_scene_min,
            model_validate_retry_cap=2,
        ),
        thresholds=SimpleNamespace(pad_ewma_alpha=alpha),
        endpoints=SimpleNamespace(planner=SimpleNamespace(name="unused-planner")),
    )


def _state(
    stores,
    *,
    config=None,
    pointer=None,
    planning_execution_mode="macro_outline_before_draft",
    approval_mode="off",
):
    state = make_initial_state(
        PROJECT_ID,
        pointer
        or FSM_Pointer(arc_id="", chapter_id="", scene_id="", beat_index=0),
        planning_execution_mode=planning_execution_mode,
        approval_mode=approval_mode,
    )
    state["sqlite_db_path"] = stores.db
    state["planning_db_path"] = stores.planning_db
    state["provisional_store_path"] = stores.provisional_db
    state["event_log_path"] = stores.event_log
    state["app_config"] = config or _planning_config()
    state["project_metadata"] = {
        "genre": "synthetic",
        "target_word_count": 12000,
        "premise_seed": "a deterministic planning premise",
    }
    state["world_rules"] = []
    return state


def _run(coro):
    return asyncio.run(coro)


def _finalize(plan):
    return PlannerAction(
        action_type="finalize_plan",
        final_plan=plan,
        self_check={"schema_valid": False},
    )


def _invalid_plan():
    return {
        "draft_text": INVALID_MARKER,
        "prose": INVALID_MARKER,
    }


def _valid_global_plan():
    return {
        "premise": "synthetic premise",
        "central_conflict": "the unresolved pressure intensifies",
        "ending_target": "the central pressure resolves cleanly",
        "arcs": [
            {
                "arc_id": "arc-a",
                "title": "Opening pressure",
                "function": "establish and complicate the central pressure",
                "word_allocation": 6600,
            },
            {
                "arc_id": "arc-b",
                "title": "Resolution pressure",
                "function": "resolve the pressure through a distinct final reversal",
                "word_allocation": 5400,
            },
        ],
        "promises": [
            {
                "id": "promise-a",
                "promise": "the source of the pressure is knowable",
                "payoff": "the source is revealed by the final reversal",
            }
        ],
    }


def _valid_arc_plan(chapter_ids=("ch-1", "ch-2")):
    chapters = [
        {"chapter_id": chapter_id, "stub": f"{chapter_id} advances arc-a"}
        for chapter_id in chapter_ids
    ]
    return {
        "arcs": [
            {
                "arc_id": "arc-a",
                "title": "Opening pressure",
                "function": "raise the pressure through irreversible choices",
                "character_milestones": ["the protagonist accepts the cost"],
                "chapters": chapters,
            }
        ],
        "milestones": [
            {"arc_id": "arc-a", "label": "choice", "tension": 1},
            {"arc_id": "arc-a", "label": "cost", "tension": 2},
        ],
        "thread_distribution": [
            {"thread_id": "main", "lifecycle": "open->progress", "in_arcs": ["arc-a"]}
        ],
    }


def _valid_chapter_plan(chapter_id="ch-1"):
    return {
        "chapter_id": chapter_id,
        "dramatic_function": "force the protagonist to choose a cost",
        "expected_emotional_shift": "guarded resolve toward alarm",
        "pacing": "measured escalation",
        "obligations": {
            "thread_obligations": [{"thread_id": "main", "required_progress": "raise"}],
            "causal_prerequisites": ["the prior promise remains open"],
            "causal_deliverables": ["the choice becomes unavoidable"],
        },
        "annotation_outcomes": {},
        "scene_planning_constraints": [
            "one scene must make the cost visible",
            "the chapter closes after the choice is made",
        ],
    }


def _valid_scene_plan(scene_id="scene-1", *, word_budget=120):
    return {
        "scene_id": scene_id,
        "scene_function": "make the cost visible through a concrete choice",
        "setting": "a lit threshold room",
        "participants": ["protagonist", "witness"],
        "entry_state": "the protagonist can still walk away",
        "exit_state": "the protagonist has accepted the cost",
        "conflict_turn": "the witness blocks the easy exit",
        "asserted_facts": [],
        "continuity_constraints": [],
        "word_budget": word_budget,
    }


def _scene_plan_with_pad(scene_id="scene-1", *, word_budget=10):
    plan = _valid_scene_plan(scene_id, word_budget=word_budget)
    plan["pad_target"] = {"pleasure": 1.0, "arousal": 0.0, "dominance": -1.0}
    return plan


def _valid_beat_plan(beat_id="beat-model-id"):
    return {
        "beat_id": beat_id,
        "immediate_objective": "reach the locked threshold",
        "physical_constraints": "rain, lock, and witness line of sight",
        "entry_condition": "the scene has opened on the threshold",
        "exit_condition": "the lock is confronted",
        "pad_target": {"pleasure": 9.0, "arousal": 9.0, "dominance": 9.0},
        "behavioral_constraint": "MODEL-INVENTED",
        "asserted_facts": [],
    }


def _seed_snapshot(stores, *, mode="macro_outline_before_draft"):
    return create_planning_snapshot(
        stores.db,
        snapshot_id=SNAPSHOT_ID,
        project_id=PROJECT_ID,
        mode=mode,
    )


def _seed_global_node(stores, *, mode="macro_outline_before_draft"):
    _seed_snapshot(stores, mode=mode)
    return upsert_planning_node(
        stores.db,
        node_id=f"{SNAPSHOT_ID}:global",
        snapshot_id=SNAPSHOT_ID,
        level="global",
        status="planned",
        title="Global",
        summary="synthetic premise",
        purpose=json.dumps(_valid_global_plan(), sort_keys=True),
    )


def _seed_arc_with_chapter_stubs(
    stores,
    *,
    chapter_ids=("ch-1", "ch-2"),
    mode="macro_outline_before_draft",
):
    _seed_global_node(stores, mode=mode)
    arc_plan = _valid_arc_plan(chapter_ids=chapter_ids)
    arc = arc_plan["arcs"][0]
    arc_id = arc["arc_id"]
    arc_node_id = f"{SNAPSHOT_ID}:arc:{arc_id}"
    upsert_arc_plan(stores.db, arc_id=arc_id, description=arc["function"])
    upsert_planning_node(
        stores.db,
        node_id=arc_node_id,
        snapshot_id=SNAPSHOT_ID,
        level="arc",
        status="planned",
        parent_id=f"{SNAPSHOT_ID}:global",
        ordering=0,
        title=arc["title"],
        summary=arc["function"],
        purpose=json.dumps(arc, sort_keys=True),
    )
    for ordering, chapter in enumerate(arc["chapters"]):
        upsert_chapter_plan(
            stores.db,
            chapter_id=chapter["chapter_id"],
            arc_id=arc_id,
            description=chapter["stub"],
            snapshot_id=SNAPSHOT_ID,
            node_id=f"{SNAPSHOT_ID}:chapter:{chapter['chapter_id']}",
            parent_node_id=arc_node_id,
            ordering=ordering,
            title=chapter["stub"],
        )
    return arc_plan


def _seed_planned_chapter(
    stores,
    *,
    chapter_id="ch-1",
    mode="macro_outline_before_draft",
):
    _seed_arc_with_chapter_stubs(stores, chapter_ids=(chapter_id,), mode=mode)
    chapter_plan = _valid_chapter_plan(chapter_id)
    node_id = f"{SNAPSHOT_ID}:chapter:{chapter_id}"
    upsert_chapter_plan(
        stores.db,
        chapter_id=chapter_id,
        arc_id="arc-a",
        description=chapter_plan["dramatic_function"],
        snapshot_id=SNAPSHOT_ID,
        node_id=node_id,
        parent_node_id=f"{SNAPSHOT_ID}:arc:arc-a",
        ordering=0,
        dramatic_function=chapter_plan["dramatic_function"],
        expected_emotional_shift=chapter_plan["expected_emotional_shift"],
        required_thread_progress=json.dumps(chapter_plan["obligations"], sort_keys=True),
        scene_planning_constraints=json.dumps(
            chapter_plan["scene_planning_constraints"], sort_keys=True
        ),
    )
    upsert_planning_node(
        stores.db,
        node_id=node_id,
        snapshot_id=SNAPSHOT_ID,
        level="chapter",
        status="planned",
        parent_id=f"{SNAPSHOT_ID}:arc:arc-a",
        ordering=0,
        title=chapter_id,
        summary=chapter_plan["dramatic_function"],
        purpose=json.dumps(chapter_plan, sort_keys=True),
    )
    return chapter_plan


def _seed_planned_scene(
    stores,
    *,
    chapter_id="ch-1",
    scene_id="scene-1",
    word_budget=10,
    mode="macro_outline_before_draft",
):
    _seed_planned_chapter(stores, chapter_id=chapter_id, mode=mode)
    scene_plan = _scene_plan_with_pad(scene_id, word_budget=word_budget)
    upsert_scene_plan(
        stores.db,
        scene_id=scene_id,
        chapter_id=chapter_id,
        description=scene_plan["scene_function"],
        ordering=0,
        word_budget=word_budget,
    )
    upsert_planning_node(
        stores.db,
        node_id=f"{SNAPSHOT_ID}:scene:{scene_id}",
        snapshot_id=SNAPSHOT_ID,
        level="scene",
        status="planned",
        parent_id=f"{SNAPSHOT_ID}:chapter:{chapter_id}",
        ordering=0,
        title=scene_plan["setting"],
        summary=scene_plan["scene_function"],
        purpose=json.dumps(scene_plan, sort_keys=True),
    )
    return scene_plan


def _seed_character(stores, character_id="char-1", name="Synthetic Character"):
    with connect_db(stores.db) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO Characters (id, name) VALUES (?, ?)",
            (character_id, name),
        )


def _assert_only_m05_state_changed(before, after, *, allowed_extra=frozenset()):
    changed = {key for key in set(before) | set(after) if before.get(key) != after.get(key)}
    assert changed <= (_ALLOWED_M05_STATE_CHANGES | set(allowed_extra))


def _assert_revision_and_event(stores, expected_outcome="finalized"):
    revisions = get_revisions_for_snapshot(stores.db, SNAPSHOT_ID)
    events = list(iter_events(stores.event_log))
    assert len(revisions) >= 1
    assert len(events) >= 1
    assert events[-1]["event_type"] == "planning_commit"
    assert events[-1]["outcome"] == expected_outcome
    assert events[-1]["revision_id"] == revisions[0]["revision_id"]
    return revisions[0]


def _all_purposes(stores):
    return [
        node["purpose"] or ""
        for node in get_planning_nodes(stores.db, SNAPSHOT_ID)
    ]


def test_global_finalize_persists_story_node_revision_and_snapshot(stores):
    state = _state(stores, config=_planning_config())
    before = copy.deepcopy(state)
    decider = ScriptedDecider([_finalize(_valid_global_plan())])

    result = _run(
        node_plan_global_mod.node_plan_global(
            state, decider=decider, registry=NoToolRegistry()
        )
    )

    assert result["planning_snapshot_id"] == SNAPSHOT_ID
    assert get_planning_snapshot(stores.db, SNAPSHOT_ID)["mode"] == (
        "macro_outline_before_draft"
    )
    nodes = get_planning_nodes(stores.db, SNAPSHOT_ID, level="global")
    assert [node["node_id"] for node in nodes] == [f"{SNAPSHOT_ID}:global"]
    persisted = json.loads(nodes[0]["purpose"])
    assert persisted["premise"] == "synthetic premise"
    assert persisted["arcs"][0]["arc_id"] == "arc-a"
    revision = _assert_revision_and_event(stores, expected_outcome="finalized")
    assert result["active_planning_revision_id"] == revision["revision_id"]
    assert len(decider.calls) == 1
    _assert_only_m05_state_changed(before, result)


def test_arc_finalize_persists_arcs_chapter_stubs_revision_and_pointer(stores):
    _seed_global_node(stores)
    state = _state(stores, config=_planning_config())
    before = copy.deepcopy(state)
    decider = ScriptedDecider([_finalize(_valid_arc_plan())])

    result = _run(
        node_plan_arc_mod.node_plan_arc(state, decider=decider, registry=NoToolRegistry())
    )

    assert get_arc(stores.db, "arc-a")["status"] == "planned"
    arc_nodes = get_planning_nodes(stores.db, SNAPSHOT_ID, level="arc")
    assert [node["node_id"] for node in arc_nodes] == [f"{SNAPSHOT_ID}:arc:arc-a"]
    assert json.loads(arc_nodes[0]["purpose"])["chapters"][0]["chapter_id"] == "ch-1"
    chapter_children = get_planning_nodes_by_parent(
        stores.db, SNAPSHOT_ID, f"{SNAPSHOT_ID}:arc:arc-a"
    )
    assert [node["node_id"] for node in chapter_children] == [
        f"{SNAPSHOT_ID}:chapter:ch-1",
        f"{SNAPSHOT_ID}:chapter:ch-2",
    ]
    assert [node["ordering"] for node in chapter_children] == [0, 1]
    assert [row["id"] for row in get_chapters_for_arc(stores.db, "arc-a")] == [
        "ch-1",
        "ch-2",
    ]
    assert result["fsm_pointer"].arc_id == "arc-a"
    _assert_revision_and_event(stores, expected_outcome="finalized")
    _assert_only_m05_state_changed(before, result)


def test_chapter_finalize_persists_obligations_revision_and_no_scene_rows(stores):
    _seed_arc_with_chapter_stubs(stores, chapter_ids=("ch-1",), mode="rolling")
    pointer = FSM_Pointer(arc_id="arc-a", chapter_id="ch-1", scene_id="", beat_index=0)
    state = _state(
        stores,
        config=_planning_config(execution_mode="rolling"),
        pointer=pointer,
        planning_execution_mode="rolling",
    )
    before = copy.deepcopy(state)
    decider = ScriptedDecider([_finalize(_valid_chapter_plan("ch-1"))])

    result = _run(
        node_plan_chapter_mod.node_plan_chapter(
            state, decider=decider, registry=NoToolRegistry()
        )
    )

    chapter_node = get_planning_nodes(stores.db, SNAPSHOT_ID, level="chapter")[0]
    chapter_plan = json.loads(chapter_node["purpose"])
    assert chapter_plan["obligations"]["causal_deliverables"] == [
        "the choice becomes unavoidable"
    ]
    assert chapter_plan["scene_planning_constraints"] == [
        "one scene must make the cost visible",
        "the chapter closes after the choice is made",
    ]
    assert "scenes" not in chapter_plan
    assert get_scenes_for_chapter_ordered(stores.db, "ch-1") == []
    assert result["fsm_pointer"].chapter_id == "ch-1"
    _assert_revision_and_event(stores, expected_outcome="finalized")
    _assert_only_m05_state_changed(before, result)


def test_scene_finalize_persists_gapless_scene_ordering_revision_and_pointer(stores):
    _seed_planned_chapter(stores, chapter_id="ch-1")
    existing_plan = _valid_scene_plan("scene-existing", word_budget=80)
    upsert_scene_plan(
        stores.db,
        scene_id="scene-existing",
        chapter_id="ch-1",
        description=existing_plan["scene_function"],
        ordering=0,
        word_budget=80,
    )
    upsert_planning_node(
        stores.db,
        node_id=f"{SNAPSHOT_ID}:scene:scene-existing",
        snapshot_id=SNAPSHOT_ID,
        level="scene",
        status="planned",
        parent_id=f"{SNAPSHOT_ID}:chapter:ch-1",
        ordering=0,
        title=existing_plan["setting"],
        summary=existing_plan["scene_function"],
        purpose=json.dumps(existing_plan, sort_keys=True),
    )
    pointer = FSM_Pointer(arc_id="arc-a", chapter_id="ch-1", scene_id="", beat_index=0)
    state = _state(stores, config=_planning_config(), pointer=pointer)
    before = copy.deepcopy(state)
    decider = ScriptedDecider([_finalize(_valid_scene_plan("scene-new", word_budget=90))])

    result = _run(
        node_plan_scene_mod.node_plan_scene(
            state, decider=decider, registry=NoToolRegistry()
        )
    )

    scenes = get_scenes_for_chapter_ordered(stores.db, "ch-1")
    assert [scene["id"] for scene in scenes] == ["scene-existing", "scene-new"]
    assert [scene["ordering"] for scene in scenes] == [0, 1]
    new_node = next(
        node
        for node in get_planning_nodes(stores.db, SNAPSHOT_ID, level="scene")
        if node["node_id"] == f"{SNAPSHOT_ID}:scene:scene-new"
    )
    assert new_node["ordering"] == 1
    assert json.loads(new_node["purpose"])["ordering"] == 1
    assert result["fsm_pointer"].scene_id == "scene-new"
    _assert_revision_and_event(stores, expected_outcome="finalized")
    _assert_only_m05_state_changed(before, result)


def test_beat_finalize_smooths_pad_persists_constraint_and_scene_stop(stores):
    config = _planning_config(alpha=0.25, beats_per_scene_min=3)
    _seed_planned_scene(stores, scene_id="scene-1", word_budget=10)
    _seed_character(stores, "char-1")
    upsert_beat_commit(
        stores.db,
        beat_id="committed-0",
        scene_id="scene-1",
        beat_index=0,
        prose="ten committed words seed the scene volume now",
        word_count=10,
        committed_at="2026-07-02T00:00:00+00:00",
        pad_states={
            "char-1": {"pleasure": -1.0, "arousal": 1.0, "dominance": 0.0}
        },
    )
    pointer = FSM_Pointer(
        arc_id="arc-a", chapter_id="ch-1", scene_id="scene-1", beat_index=0
    )
    state = _state(stores, config=config, pointer=pointer)
    before = copy.deepcopy(state)

    first = _run(
        node_plan_beat_mod.node_plan_beat(
            state,
            decider=ScriptedDecider([_finalize(_valid_beat_plan("beat-model-id"))]),
            registry=NoToolRegistry(),
            adapt_fn=None,
        )
    )

    assert first["fsm_pointer"].beat_id == "beat-model-id"
    assert first["fsm_pointer"].beat_index == 1
    assert first["scene_needs_more"] is True
    expected_constraint = compose_baseline_string("P-A+D-")
    first_node = next(
        node
        for node in get_planning_nodes(stores.db, SNAPSHOT_ID, level="beat")
        if node["node_id"] == f"{SNAPSHOT_ID}:beat:beat-model-id"
    )
    first_plan = json.loads(first_node["purpose"])
    assert first_plan["pad_target"] == pytest.approx(
        {"pleasure": -0.5, "arousal": 0.75, "dominance": -0.25}
    )
    assert first_plan["behavioral_constraint"] == expected_constraint
    assert first_plan["behavioral_constraint"] != "MODEL-INVENTED"
    persisted_beat = get_beat(stores.db, "beat-model-id")
    assert persisted_beat["prose"] is None
    assert persisted_beat["word_count"] == 0
    assert persisted_beat["status"] == "planned"
    pad_trace = next(
        item for item in first["planner_deliberation_trace"] if item.get("phase") == "pad_grounding"
    )
    assert pad_trace["rung"] == "static"

    second = _run(
        node_plan_beat_mod.node_plan_beat(
            first,
            decider=ScriptedDecider([_finalize(_valid_beat_plan("beat-model-id"))]),
            registry=NoToolRegistry(),
            adapt_fn=None,
        )
    )

    beats = get_beats_for_scene_ordered(stores.db, "scene-1")
    assert [beat["beat_index"] for beat in beats] == [0, 1, 2]
    assert second["fsm_pointer"].beat_index == 2
    assert second["scene_needs_more"] is False
    assert len(get_revisions_for_snapshot(stores.db, SNAPSHOT_ID)) == 2
    _assert_only_m05_state_changed(before, second)


def test_always_invalid_deciders_fallback_or_escalate_without_persisting_invalid(tmp_path):
    cases = [
        (
            "global",
            node_plan_global_mod.node_plan_global,
            lambda stores: _state(stores, config=_planning_config(loops=1)),
            {},
        ),
        (
            "arc",
            node_plan_arc_mod.node_plan_arc,
            lambda stores: (
                _seed_global_node(stores),
                _state(stores, config=_planning_config(loops=1)),
            )[1],
            {},
        ),
        (
            "chapter",
            node_plan_chapter_mod.node_plan_chapter,
            lambda stores: (
                _seed_arc_with_chapter_stubs(stores, chapter_ids=("ch-1",), mode="rolling"),
                _state(
                    stores,
                    config=_planning_config(loops=1, execution_mode="rolling"),
                    pointer=FSM_Pointer(
                        arc_id="arc-a", chapter_id="ch-1", scene_id="", beat_index=0
                    ),
                    planning_execution_mode="rolling",
                ),
            )[1],
            {},
        ),
        (
            "scene",
            node_plan_scene_mod.node_plan_scene,
            lambda stores: (
                _seed_planned_chapter(stores, chapter_id="ch-1"),
                _state(
                    stores,
                    config=_planning_config(loops=1),
                    pointer=FSM_Pointer(
                        arc_id="arc-a", chapter_id="ch-1", scene_id="", beat_index=0
                    ),
                ),
            )[1],
            {},
        ),
        (
            "beat",
            node_plan_beat_mod.node_plan_beat,
            lambda stores: (
                _seed_planned_scene(stores, scene_id="scene-1", word_budget=0),
                _state(
                    stores,
                    config=_planning_config(loops=1, beats_per_scene_min=1),
                    pointer=FSM_Pointer(
                        arc_id="arc-a",
                        chapter_id="ch-1",
                        scene_id="scene-1",
                        beat_index=0,
                    ),
                ),
            )[1],
            {"adapt_fn": None},
        ),
    ]

    for name, node_fn, build_state, extra_kwargs in cases:
        stores = _make_stores(tmp_path, f"invalid-{name}")
        state = build_state(stores)
        result = _run(
            node_fn(
                state,
                decider=ScriptedDecider([_finalize(_invalid_plan())]),
                registry=NoToolRegistry(),
                **extra_kwargs,
            )
        )
        events = list(iter_events(stores.event_log))
        if events:
            assert events[-1]["outcome"] == "fallback_baseline"
        else:
            assert result["planning_block_reason"] == "planning_escalation"
        assert INVALID_MARKER not in "\n".join(_all_purposes(stores))


def test_planning_nodes_use_decider_seam_and_do_not_call_llm_directly():
    modules = [
        node_plan_global_mod,
        node_plan_arc_mod,
        node_plan_chapter_mod,
        node_plan_scene_mod,
        node_plan_beat_mod,
    ]

    for module in modules:
        source = inspect.getsource(module)
        assert "make_planner_decider" in source
        assert "call_llm(" not in source
        assert "call_llm_structured" not in source
