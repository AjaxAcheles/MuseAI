"""Module: M05 (Hierarchical Planning Cascade)
Deterministic tests for macro-vs-rolling planner state behavior.
"""

import copy

import fsm.nodes.node_plan_beat as node_plan_beat_mod
import fsm.nodes.node_plan_chapter as node_plan_chapter_mod
import fsm.nodes.node_plan_scene as node_plan_scene_mod
from fsm.state import FSM_Pointer
from memory.event_log import iter_events
from memory.sqlite_db import (
    get_annotations_for_snapshot,
    get_beats_for_scene_ordered,
    get_planning_nodes,
    get_planning_snapshot,
    get_revisions_for_snapshot,
    get_scenes_for_chapter_ordered,
    insert_planning_annotation,
)
from tests.test_planning_nodes import (
    SNAPSHOT_ID,
    NoToolRegistry,
    ScriptedDecider,
    _assert_only_m05_state_changed,
    _finalize,
    _invalid_plan,
    _make_stores,
    _planning_config,
    _run,
    _seed_arc_with_chapter_stubs,
    _seed_planned_chapter,
    _state,
    _valid_beat_plan,
    _valid_chapter_plan,
    _valid_scene_plan,
)


def _chapter_pointer(chapter_id="ch-1"):
    return FSM_Pointer(arc_id="arc-a", chapter_id=chapter_id, scene_id="", beat_index=0)


def _scene_pointer(scene_id):
    return FSM_Pointer(
        arc_id="arc-a", chapter_id="ch-1", scene_id=scene_id, beat_index=0
    )


def _add_hard_conflict(stores, target_node_id, *, target_level):
    insert_planning_annotation(
        stores.db,
        annotation_id="ann-hard-pin",
        snapshot_id=SNAPSHOT_ID,
        target_node_id=target_node_id,
        target_level=target_level,
        note_type="pin",
        scope="this_node",
        priority="hard",
        text="pin the target",
    )
    insert_planning_annotation(
        stores.db,
        annotation_id="ann-hard-remove",
        snapshot_id=SNAPSHOT_ID,
        target_node_id=target_node_id,
        target_level=target_level,
        note_type="remove",
        scope="this_node",
        priority="hard",
        text="remove the target",
    )


def test_macro_mode_with_macro_approval_sets_approval_state(tmp_path):
    stores = _make_stores(tmp_path, "macro-approval")
    _seed_arc_with_chapter_stubs(stores, chapter_ids=("ch-1",))
    config = _planning_config(execution_mode="rolling", approval_mode="off")
    state = _state(
        stores,
        config=config,
        pointer=_chapter_pointer(),
        planning_execution_mode="macro_outline_before_draft",
        approval_mode="macro_outline",
    )
    before = copy.deepcopy(state)

    result = _run(
        node_plan_chapter_mod.node_plan_chapter(
            state,
            decider=ScriptedDecider([_finalize(_valid_chapter_plan("ch-1"))]),
            registry=NoToolRegistry(),
        )
    )

    assert result["macro_outline_ready"] is True
    assert result["awaiting_planning_approval"] is True
    assert result["planning_block_reason"] == "awaiting_macro_approval"
    snapshot = get_planning_snapshot(stores.db, SNAPSHOT_ID)
    assert snapshot["status"] == "draft"
    assert result["pause_requested"] is False
    assert result["hard_stop_asserted"] is False
    _assert_only_m05_state_changed(before, result)


def test_macro_mode_without_approval_marks_ready_without_wait(tmp_path):
    stores = _make_stores(tmp_path, "macro-no-approval")
    _seed_arc_with_chapter_stubs(stores, chapter_ids=("ch-1",))
    config = _planning_config(execution_mode="rolling", approval_mode="macro_outline")
    state = _state(
        stores,
        config=config,
        pointer=_chapter_pointer(),
        planning_execution_mode="macro_outline_before_draft",
        approval_mode="off",
    )
    before = copy.deepcopy(state)

    result = _run(
        node_plan_chapter_mod.node_plan_chapter(
            state,
            decider=ScriptedDecider([_finalize(_valid_chapter_plan("ch-1"))]),
            registry=NoToolRegistry(),
        )
    )

    assert result["macro_outline_ready"] is True
    assert result["awaiting_planning_approval"] is False
    assert result["planning_block_reason"] is None
    assert result["pause_requested"] is False
    assert result["hard_stop_asserted"] is False
    _assert_only_m05_state_changed(before, result)


def test_rolling_mode_does_not_pause_and_scene_beat_plan_just_in_time(tmp_path):
    stores = _make_stores(tmp_path, "rolling")
    _seed_arc_with_chapter_stubs(stores, chapter_ids=("ch-1",), mode="rolling")
    config = _planning_config(
        beats_per_scene_min=1,
        execution_mode="macro_outline_before_draft",
        approval_mode="macro_outline",
    )
    state = _state(
        stores,
        config=config,
        pointer=_chapter_pointer(),
        planning_execution_mode="rolling",
        approval_mode="macro_outline",
    )
    before = copy.deepcopy(state)

    after_chapter = _run(
        node_plan_chapter_mod.node_plan_chapter(
            state,
            decider=ScriptedDecider([_finalize(_valid_chapter_plan("ch-1"))]),
            registry=NoToolRegistry(),
        )
    )

    assert after_chapter["macro_outline_ready"] is False
    assert after_chapter["awaiting_planning_approval"] is False
    assert after_chapter["planning_block_reason"] is None

    after_scene = _run(
        node_plan_scene_mod.node_plan_scene(
            after_chapter,
            decider=ScriptedDecider([_finalize(_valid_scene_plan("scene-rolling", word_budget=0))]),
            registry=NoToolRegistry(),
        )
    )
    after_beat = _run(
        node_plan_beat_mod.node_plan_beat(
            after_scene,
            decider=ScriptedDecider([_finalize(_valid_beat_plan("beat-rolling"))]),
            registry=NoToolRegistry(),
            adapt_fn=None,
        )
    )

    assert after_beat["macro_outline_ready"] is False
    assert after_beat["awaiting_planning_approval"] is False
    assert after_beat["planning_block_reason"] is None
    assert [scene["id"] for scene in get_scenes_for_chapter_ordered(stores.db, "ch-1")] == [
        "scene-rolling"
    ]
    assert [beat["id"] for beat in get_beats_for_scene_ordered(stores.db, "scene-rolling")] == [
        "beat-rolling"
    ]
    assert after_beat["fsm_pointer"].scene_id == "scene-rolling"
    assert after_beat["fsm_pointer"].beat_id == "beat-rolling"
    _assert_only_m05_state_changed(before, after_beat)


def test_needs_clarification_sets_block_and_persists_no_scene_plan(tmp_path):
    stores = _make_stores(tmp_path, "needs-clarification")
    _seed_planned_chapter(stores, chapter_id="ch-1")
    target_node_id = f"{SNAPSHOT_ID}:chapter:ch-1"
    _add_hard_conflict(stores, target_node_id, target_level="chapter")
    decider = ScriptedDecider([_finalize(_valid_scene_plan("scene-should-not-write"))])
    state = _state(
        stores,
        config=_planning_config(),
        pointer=_chapter_pointer(),
        planning_execution_mode="macro_outline_before_draft",
    )
    before = copy.deepcopy(state)

    result = _run(
        node_plan_scene_mod.node_plan_scene(
            state, decider=decider, registry=NoToolRegistry()
        )
    )

    assert result["planning_block_reason"] == "unresolved_hard_conflict"
    assert decider.calls == []
    assert get_scenes_for_chapter_ordered(stores.db, "ch-1") == []
    assert get_revisions_for_snapshot(stores.db, SNAPSHOT_ID) == []
    assert list(iter_events(stores.event_log)) == []
    statuses = {
        row["annotation_id"]: row["status"]
        for row in get_annotations_for_snapshot(stores.db, SNAPSHOT_ID)
    }
    assert statuses == {
        "ann-hard-pin": "needs_clarification",
        "ann-hard-remove": "needs_clarification",
    }
    _assert_only_m05_state_changed(before, result)


def test_escalate_from_invalid_decider_and_invalid_baseline_persists_nothing(
    tmp_path, monkeypatch
):
    stores = _make_stores(tmp_path, "escalate")
    _seed_planned_chapter(stores, chapter_id="ch-1")
    state = _state(
        stores,
        config=_planning_config(loops=1),
        pointer=_chapter_pointer(),
        planning_execution_mode="rolling",
    )
    before = copy.deepcopy(state)

    monkeypatch.setattr(
        node_plan_scene_mod,
        "_build_scene_baseline",
        lambda base_context: (
            lambda level, target_node, constraints: _invalid_plan()
        ),
    )

    result = _run(
        node_plan_scene_mod.node_plan_scene(
            state,
            decider=ScriptedDecider([_finalize(_invalid_plan())]),
            registry=NoToolRegistry(),
        )
    )

    assert result["planning_block_reason"] == "planning_escalation"
    assert get_scenes_for_chapter_ordered(stores.db, "ch-1") == []
    assert get_planning_nodes(stores.db, SNAPSHOT_ID, level="scene") == []
    assert get_revisions_for_snapshot(stores.db, SNAPSHOT_ID) == []
    assert list(iter_events(stores.event_log)) == []
    _assert_only_m05_state_changed(before, result)
