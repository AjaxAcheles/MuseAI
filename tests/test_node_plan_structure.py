"""Module: M05 (Hierarchical Planning Cascade) — local end-to-end scaffold
Synthetic tests for the minimal structure planner (`fsm/nodes/node_plan_structure.py`).
Exercises it against a real temp SQLite hub: beat count derives from the word target,
Scenes/Beats rows are persisted, and the pointer lands on the first planned beat. No
model calls (the node is deterministic and model-free).
"""

import asyncio
from types import SimpleNamespace

from fsm.nodes.node_plan_structure import node_plan_structure
from fsm.state import FSM_Pointer
from memory import sqlite_db


def _config(beats_per_scene_min=3, word_count_target=2000):
    return SimpleNamespace(
        runtime=SimpleNamespace(
            beats_per_scene_min=beats_per_scene_min, word_count_target=word_count_target
        )
    )


def _state(db, **meta):
    return {
        "project_id": "p1",
        "app_config": _config(),
        "sqlite_db_path": db,
        "planning_snapshot_id": "snap_p1",
        "project_metadata": meta,
        "fsm_pointer": FSM_Pointer(arc_id="arc_1", chapter_id="", scene_id="", beat_index=0),
    }


def test_beat_count_derives_from_word_target_and_persists_structure(tmp_path):
    db = tmp_path / "hub.db"
    sqlite_db.init_db(db)
    # 750 words / 250-per-beat = 3 beats, at the 3-beats-per-scene floor -> 1 scene.
    state = asyncio.run(
        node_plan_structure(_state(db, target_word_count=750, beat_word_target=250))
    )

    beat_order = state["beat_order"]
    assert len(beat_order) == 3
    assert set(state["beat_plan_by_id"]) == set(beat_order)

    pointer = state["fsm_pointer"]
    assert pointer.beat_id == beat_order[0]
    assert pointer.scene_id  # pointer advanced onto the first scene/beat

    scene_id = state["beat_plan_by_id"][beat_order[0]]["scene_id"]
    persisted = sqlite_db.get_beats_for_scene_ordered(db, scene_id)
    assert len(persisted) == 3
    # Beat indices within the scene are gapless and ordered.
    assert [b["beat_index"] for b in persisted] == [0, 1, 2]


def test_larger_target_splits_into_multiple_scenes(tmp_path):
    db = tmp_path / "hub.db"
    sqlite_db.init_db(db)
    # 2000 / 250 = 8 beats, 3 per scene -> 3 scenes (3+3+2).
    state = asyncio.run(
        node_plan_structure(_state(db, target_word_count=2000, beat_word_target=250))
    )
    assert len(state["beat_order"]) == 8
    scene_ids = {p["scene_id"] for p in state["beat_plan_by_id"].values()}
    assert len(scene_ids) == 3
