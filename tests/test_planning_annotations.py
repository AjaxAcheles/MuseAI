"""Module: M05 (Hierarchical Planning Cascade)
Deterministic tests for annotation-to-constraint compilation and revision diffs.
"""

import copy
import json

from fsm.planning_annotations import compile_planning_constraints, compute_revision_diff
from memory.sqlite_db import (
    create_planning_snapshot,
    get_annotations_for_snapshot,
    insert_planning_annotation,
    upsert_planning_node,
)


def _db(tmp_path):
    return tmp_path / "planning.db"


def _seed_hierarchy(db_path):
    create_planning_snapshot(
        db_path,
        snapshot_id="snap-1",
        project_id="project-1",
        mode="macro_outline_before_draft",
    )
    upsert_planning_node(
        db_path,
        node_id="global",
        snapshot_id="snap-1",
        level="global",
        status="planned",
        title="Global",
    )
    upsert_planning_node(
        db_path,
        node_id="arc-a",
        snapshot_id="snap-1",
        level="arc",
        status="planned",
        parent_id="global",
        ordering=0,
        title="Arc A",
    )
    upsert_planning_node(
        db_path,
        node_id="arc-b",
        snapshot_id="snap-1",
        level="arc",
        status="planned",
        parent_id="global",
        ordering=1,
        title="Arc B",
    )
    upsert_planning_node(
        db_path,
        node_id="chapter-a",
        snapshot_id="snap-1",
        level="chapter",
        status="planned",
        parent_id="arc-a",
        locked_pinned=True,
        title="Chapter A",
    )
    upsert_planning_node(
        db_path,
        node_id="chapter-b",
        snapshot_id="snap-1",
        level="chapter",
        status="planned",
        parent_id="arc-b",
        title="Chapter B",
    )
    upsert_planning_node(
        db_path,
        node_id="scene-a",
        snapshot_id="snap-1",
        level="scene",
        status="planned",
        parent_id="chapter-a",
        title="Scene A",
    )
    return upsert_planning_node(
        db_path,
        node_id="beat-a",
        snapshot_id="snap-1",
        level="beat",
        status="planned",
        parent_id="scene-a",
        title="Beat A",
    )


def _ann(
    db_path,
    annotation_id,
    *,
    target_node_id,
    target_level,
    note_type="constraint",
    scope="subtree",
    priority="normal",
    text=None,
    created_at=None,
):
    return insert_planning_annotation(
        db_path,
        annotation_id=annotation_id,
        snapshot_id="snap-1",
        target_node_id=target_node_id,
        target_level=target_level,
        note_type=note_type,
        scope=scope,
        priority=priority,
        text=text or annotation_id,
        created_at=created_at or f"2026-07-02T00:00:{annotation_id[-1]}+00:00",
    )


def test_compiler_inherits_global_arc_and_chapter_constraints_for_descendant(tmp_path):
    db_path = _db(tmp_path)
    _seed_hierarchy(db_path)
    _ann(
        db_path,
        "ann-global",
        target_node_id="global",
        target_level="global",
        scope="global",
        priority="hard",
        text="global rule",
    )
    _ann(
        db_path,
        "ann-arc-a",
        target_node_id="arc-a",
        target_level="arc",
        scope="subtree",
        priority="normal",
        text="arc A rule",
    )
    _ann(
        db_path,
        "ann-chapter-a",
        target_node_id="chapter-a",
        target_level="chapter",
        scope="subtree",
        priority="high",
        text="chapter A rule",
    )
    _ann(
        db_path,
        "ann-arc-b",
        target_node_id="arc-b",
        target_level="arc",
        scope="subtree",
        priority="high",
        text="wrong arc rule",
    )

    package = compile_planning_constraints("snap-1", "beat-a", db_path=db_path)

    assert package["needs_clarification"] is False
    assert [entry["annotation_id"] for entry in package["hard_annotations"]] == [
        "ann-global"
    ]
    assert [entry["annotation_id"] for entry in package["soft_preferences"]] == [
        "ann-chapter-a",
        "ann-arc-a",
    ]
    assert all(entry["origin"] == "inherited" for entry in package["soft_preferences"])
    assert "ann-arc-b" not in {
        entry["annotation_id"]
        for entry in package["hard_annotations"] + package["soft_preferences"]
    }


def test_precedence_keeps_hard_requirements_and_flags_pinned_constraints(tmp_path):
    db_path = _db(tmp_path)
    _seed_hierarchy(db_path)
    _ann(
        db_path,
        "ann-pin",
        target_node_id="chapter-a",
        target_level="chapter",
        note_type="pin",
        scope="this_node",
        priority="hard",
        text="keep this chapter",
    )
    _ann(
        db_path,
        "ann-soft-remove",
        target_node_id="chapter-a",
        target_level="chapter",
        note_type="remove",
        scope="this_node",
        priority="normal",
        text="prefer dropping this chapter",
    )

    package = compile_planning_constraints("snap-1", "chapter-a", db_path=db_path)

    assert package["needs_clarification"] is False
    assert package["target_locked_pinned"] is True
    assert [entry["annotation_id"] for entry in package["hard_annotations"]] == [
        "ann-pin"
    ]
    assert [entry["annotation_id"] for entry in package["soft_preferences"]] == [
        "ann-soft-remove"
    ]
    assert [entry["annotation_id"] for entry in package["pinned_annotations"]] == [
        "ann-pin",
        "ann-soft-remove",
    ]
    assert all(entry["from_pinned_node"] for entry in package["pinned_annotations"])


def test_hard_conflicts_mark_annotations_and_emit_no_contradictory_constraints(tmp_path):
    db_path = _db(tmp_path)
    _seed_hierarchy(db_path)
    _ann(
        db_path,
        "ann-hard-pin",
        target_node_id="scene-a",
        target_level="scene",
        note_type="pin",
        scope="this_node",
        priority="hard",
        text="pin scene",
    )
    _ann(
        db_path,
        "ann-hard-remove",
        target_node_id="scene-a",
        target_level="scene",
        note_type="remove",
        scope="this_node",
        priority="hard",
        text="remove scene",
    )

    package = compile_planning_constraints("snap-1", "scene-a", db_path=db_path)

    assert package["needs_clarification"] is True
    assert package["block_reason"] == "unresolved_hard_conflict"
    assert package["hard_annotations"] == []
    assert package["soft_preferences"] == []
    assert package["unresolved_conflict_annotation_ids"] == [
        "ann-hard-pin",
        "ann-hard-remove",
    ]
    assert package["hard_conflicts"][0]["annotation_ids"] == [
        "ann-hard-pin",
        "ann-hard-remove",
    ]
    statuses = {
        row["annotation_id"]: row["status"]
        for row in get_annotations_for_snapshot(db_path, "snap-1")
    }
    assert statuses == {
        "ann-hard-pin": "needs_clarification",
        "ann-hard-remove": "needs_clarification",
    }

    rerun = compile_planning_constraints("snap-1", "scene-a", db_path=db_path)

    assert rerun["needs_clarification"] is True
    assert rerun["hard_annotations"] == []
    assert rerun["hard_conflicts"] == []
    assert rerun["unresolved_conflict_annotation_ids"] == [
        "ann-hard-pin",
        "ann-hard-remove",
    ]


def test_compute_revision_diff_is_serializable_and_preserves_prior_state():
    previous = [
        {
            "node_id": "chapter-a",
            "level": "chapter",
            "title": "Old chapter",
            "summary": "old summary",
            "purpose": "old purpose",
            "status": "planned",
            "parent_id": "arc-a",
            "ordering": 0,
            "locked_pinned": 0,
        },
        {
            "node_id": "scene-removed",
            "level": "scene",
            "title": "Removed scene",
            "parent_id": "chapter-a",
            "ordering": 0,
        },
    ]
    new = [
        {
            "node_id": "chapter-a",
            "level": "chapter",
            "title": "New chapter",
            "summary": "new summary",
            "purpose": "old purpose",
            "status": "planned",
            "parent_id": "arc-b",
            "ordering": 1,
            "locked_pinned": 0,
        },
        {
            "node_id": "scene-added",
            "level": "scene",
            "title": "Added scene",
            "parent_id": "chapter-a",
            "ordering": 0,
        },
    ]
    previous_before = copy.deepcopy(previous)

    diff = compute_revision_diff(
        previous, new, {"ann-1": "applied", "ann-2": "rejected"}
    )

    assert json.loads(json.dumps(diff)) == diff
    assert previous == previous_before
    assert diff["added_nodes"] == [
        {"node_id": "scene-added", "level": "scene", "title": "Added scene"}
    ]
    assert diff["removed_nodes"] == [
        {"node_id": "scene-removed", "level": "scene", "title": "Removed scene"}
    ]
    assert diff["modified_nodes"] == [
        {
            "node_id": "chapter-a",
            "changed_fields": {
                "title": {"from": "Old chapter", "to": "New chapter"},
                "summary": {"from": "old summary", "to": "new summary"},
            },
        }
    ]
    assert diff["moved_nodes"] == [
        {
            "node_id": "chapter-a",
            "position_change": {
                "parent_id": {"from": "arc-a", "to": "arc-b"},
                "ordering": {"from": 0, "to": 1},
            },
        }
    ]
    assert diff["annotation_outcomes"] == [
        {"annotation_id": "ann-1", "outcome": "applied"},
        {"annotation_id": "ann-2", "outcome": "rejected"},
    ]
