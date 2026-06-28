"""Module: M02 (Persistent Memory Stores & Interfaces)
Synthetic round-trip tests for the planning-persistence surface in
``memory/sqlite_db.py``: the five §2.7 planning proposal-surface tables
(PlanningSnapshot/PlanningNode/PlanningAnnotation/PlanningRevision/
PlannerToolCallTrace) and the plan-time narrative outline writers
(Arcs/Chapters/Scenes/Beats).

These are M02 schema/helpers added as the build-07 planning-persistence prerequisite;
the planning tables are written by the M05 planner nodes later. All fixtures are
file-local under ``tmp_path`` — these tests never touch the real ``data/`` directory.
No LLM, no network, no planner-loop/validator/annotation-compiler logic is exercised:
these are persistence primitives only.
"""

import json

from memory.sqlite_db import (
    create_planning_snapshot,
    get_annotations_for_node,
    get_annotations_for_snapshot,
    get_beat,
    get_beats_for_scene_ordered,
    get_chapters_for_arc,
    get_planning_nodes,
    get_planning_nodes_by_parent,
    get_planning_snapshot,
    get_revisions_for_snapshot,
    get_scenes_for_chapter_ordered,
    get_traces_for_snapshot,
    get_arc,
    insert_planner_tool_call_trace,
    insert_planning_annotation,
    insert_planning_revision,
    set_snapshot_active_revision,
    transition_snapshot_status,
    upsert_arc_plan,
    upsert_beat_plan,
    upsert_chapter_plan,
    upsert_planning_node,
    upsert_scene_plan,
)


def _db(tmp_path):
    """A nested temp DB path, proving parent-dir creation on first init."""
    return tmp_path / "nested" / "planning.db"


# --- (a) planning proposal-surface round-trip -----------------------------------


def test_planning_surface_round_trip(tmp_path):
    """snapshot -> node -> annotation -> revision -> trace, all read back."""
    db = _db(tmp_path)

    snap = create_planning_snapshot(
        db,
        snapshot_id="snap1",
        project_id="proj1",
        mode="macro_outline_before_draft",
    )
    assert snap["status"] == "draft"
    assert snap["active_revision_id"] is None
    assert get_planning_snapshot(db, "snap1")["project_id"] == "proj1"

    node = upsert_planning_node(
        db,
        node_id="pn-arc1",
        snapshot_id="snap1",
        level="arc",
        status="proposed",
        ordering=1,
        title="Arc One",
    )
    assert node["level"] == "arc"
    assert node["locked_pinned"] == 0
    assert [n["node_id"] for n in get_planning_nodes(db, "snap1", level="arc")] == [
        "pn-arc1"
    ]

    ann = insert_planning_annotation(
        db,
        annotation_id="ann1",
        snapshot_id="snap1",
        target_node_id="pn-arc1",
        target_level="arc",
        note_type="concern",
        scope="this_node",
        priority="high",
        text="pacing feels rushed",
    )
    assert ann["status"] == "pending"
    assert [a["annotation_id"] for a in get_annotations_for_node(db, "pn-arc1")] == [
        "ann1"
    ]
    assert [a["annotation_id"] for a in get_annotations_for_snapshot(db, "snap1")] == [
        "ann1"
    ]

    rev = insert_planning_revision(
        db,
        revision_id="rev1",
        snapshot_id="snap1",
        diff_json=json.dumps({"added": [], "annotation_outcomes": {"ann1": "applied"}}),
        change_summary="address pacing concern",
    )
    assert rev["revision_id"] == "rev1"
    # The revision insert advances the owning snapshot's active_revision_id.
    assert get_planning_snapshot(db, "snap1")["active_revision_id"] == "rev1"
    assert [r["revision_id"] for r in get_revisions_for_snapshot(db, "snap1")] == [
        "rev1"
    ]

    trace = insert_planner_tool_call_trace(
        db,
        trace_id="tr1",
        snapshot_id="snap1",
        planner_level="arc",
        loop_index=0,
        tool_name="propose_arcs",
        success=True,
    )
    assert trace["success"] == 1
    assert [
        t["trace_id"]
        for t in get_traces_for_snapshot(db, "snap1", planner_level="arc")
    ] == ["tr1"]


def test_planning_surface_replay_is_idempotent(tmp_path):
    """Re-running the same logical surface writes does not duplicate rows."""
    db = _db(tmp_path)

    def _write_all():
        create_planning_snapshot(
            db, snapshot_id="s1", project_id="p1", mode="rolling"
        )
        upsert_planning_node(
            db, node_id="n1", snapshot_id="s1", level="arc", status="proposed"
        )
        insert_planning_annotation(
            db,
            annotation_id="a1",
            snapshot_id="s1",
            target_node_id="n1",
            target_level="arc",
            note_type="preference",
            scope="this_node",
            priority="normal",
            text="keep it tight",
        )
        insert_planning_revision(
            db, revision_id="r1", snapshot_id="s1", diff_json="{}"
        )
        insert_planner_tool_call_trace(
            db,
            trace_id="t1",
            snapshot_id="s1",
            planner_level="arc",
            loop_index=0,
            tool_name="propose_arcs",
            success=True,
        )

    _write_all()
    _write_all()  # replay

    assert len(get_planning_nodes(db, "s1")) == 1
    assert len(get_annotations_for_snapshot(db, "s1")) == 1
    assert len(get_revisions_for_snapshot(db, "s1")) == 1
    assert len(get_traces_for_snapshot(db, "s1")) == 1
    # The revision was not silently overwritten and the pointer stays put.
    assert get_planning_snapshot(db, "s1")["active_revision_id"] == "r1"


def test_revision_history_is_preserved_not_overwritten(tmp_path):
    """A second revision preserves the first; active_revision_id re-advances."""
    db = _db(tmp_path)
    create_planning_snapshot(db, snapshot_id="s1", project_id="p1", mode="rolling")
    insert_planning_revision(
        db,
        revision_id="r1",
        snapshot_id="s1",
        diff_json="{}",
        created_at="2026-06-27T00:00:00+00:00",
    )
    insert_planning_revision(
        db,
        revision_id="r2",
        snapshot_id="s1",
        parent_revision_id="r1",
        diff_json="{}",
        created_at="2026-06-27T01:00:00+00:00",
    )

    newest_first = [r["revision_id"] for r in get_revisions_for_snapshot(db, "s1")]
    assert newest_first == ["r2", "r1"]
    oldest_first = [
        r["revision_id"]
        for r in get_revisions_for_snapshot(db, "s1", newest_first=False)
    ]
    assert oldest_first == ["r1", "r2"]
    assert get_planning_snapshot(db, "s1")["active_revision_id"] == "r2"


def test_replaying_older_revision_does_not_demote_active_pointer(tmp_path):
    """A resumed older revision insert cannot move active_revision_id backward."""
    db = _db(tmp_path)
    create_planning_snapshot(db, snapshot_id="s1", project_id="p1", mode="rolling")
    insert_planning_revision(
        db,
        revision_id="r1",
        snapshot_id="s1",
        diff_json="{}",
        created_at="2026-06-27T00:00:00+00:00",
    )
    insert_planning_revision(
        db,
        revision_id="r2",
        snapshot_id="s1",
        parent_revision_id="r1",
        diff_json="{}",
        created_at="2026-06-27T01:00:00+00:00",
    )

    insert_planning_revision(
        db,
        revision_id="r1",
        snapshot_id="s1",
        diff_json='{"replayed": true}',
        created_at="2026-06-27T00:00:00+00:00",
    )

    assert get_planning_snapshot(db, "s1")["active_revision_id"] == "r2"
    assert [r["revision_id"] for r in get_revisions_for_snapshot(db, "s1")] == [
        "r2",
        "r1",
    ]


def test_approved_gate_blocks_on_unresolved_conflict(tmp_path):
    """Approval is persisted only when no unresolved conflict is flagged."""
    db = _db(tmp_path)
    create_planning_snapshot(db, snapshot_id="s1", project_id="p1", mode="rolling")

    approved = transition_snapshot_status(db, "s1", status="approved")
    assert approved["status"] == "approved"
    assert approved["approved_at"] is not None

    try:
        transition_snapshot_status(
            db, "s1", status="approved", has_unresolved_conflict=True
        )
    except ValueError:
        pass
    else:  # pragma: no cover - the gate must raise
        raise AssertionError("expected ValueError when approving with a conflict")


def test_snapshot_create_replay_preserves_advanced_lifecycle(tmp_path):
    """Replaying create cannot revert status, approval stamp, or active revision."""
    db = _db(tmp_path)
    create_planning_snapshot(
        db,
        snapshot_id="s1",
        project_id="p1",
        mode="rolling",
        created_at="2026-06-27T00:00:00+00:00",
    )
    transition_snapshot_status(
        db,
        "s1",
        status="approved",
        approved_at="2026-06-27T00:05:00+00:00",
    )
    insert_planning_revision(
        db,
        revision_id="r1",
        snapshot_id="s1",
        diff_json="{}",
        created_at="2026-06-27T00:10:00+00:00",
    )
    set_snapshot_active_revision(db, "s1", "r1")

    replayed = create_planning_snapshot(
        db,
        snapshot_id="s1",
        project_id="p1-refreshed",
        mode="rolling",
        created_at="2026-06-27T00:00:00+00:00",
    )

    assert replayed["project_id"] == "p1-refreshed"
    assert replayed["status"] == "approved"
    assert replayed["approved_at"] == "2026-06-27T00:05:00+00:00"
    assert replayed["active_revision_id"] == "r1"


def test_reapproving_snapshot_preserves_first_approval_stamp(tmp_path):
    """Approving twice is a replay-safe no-op for approved_at."""
    db = _db(tmp_path)
    create_planning_snapshot(db, snapshot_id="s1", project_id="p1", mode="rolling")

    first = transition_snapshot_status(
        db,
        "s1",
        status="approved",
        approved_at="2026-06-27T00:05:00+00:00",
    )
    second = transition_snapshot_status(db, "s1", status="approved")

    assert first["approved_at"] == "2026-06-27T00:05:00+00:00"
    assert second["approved_at"] == "2026-06-27T00:05:00+00:00"


def test_leaving_approved_clears_approval_stamp(tmp_path):
    """Rejected/superseded snapshots do not retain stale approval timestamps."""
    db = _db(tmp_path)
    create_planning_snapshot(db, snapshot_id="s1", project_id="p1", mode="rolling")
    transition_snapshot_status(
        db,
        "s1",
        status="approved",
        approved_at="2026-06-27T00:05:00+00:00",
    )

    rejected = transition_snapshot_status(db, "s1", status="rejected")

    assert rejected["status"] == "rejected"
    assert rejected["approved_at"] is None


def test_schema_init_guard_recovers_after_db_file_is_removed(tmp_path):
    """The once-per-process init guard reinitializes if the DB file disappears."""
    db = _db(tmp_path)
    create_planning_snapshot(db, snapshot_id="s1", project_id="p1", mode="rolling")
    db.unlink()

    recreated = create_planning_snapshot(
        db, snapshot_id="s2", project_id="p1", mode="rolling"
    )

    assert recreated["snapshot_id"] == "s2"
    assert get_planning_snapshot(db, "s1") is None


# --- (b) plan-time narrative outline round-trip ----------------------------------


def test_outline_round_trip(tmp_path):
    """arc-plan -> chapter-plan (obligations) -> scene-plan (ordering) -> beat-plan (PAD)."""
    db = _db(tmp_path)

    # A snapshot is needed for the chapter/beat PlanningNode proposal rows.
    create_planning_snapshot(
        db, snapshot_id="snap1", project_id="proj1", mode="macro_outline_before_draft"
    )

    arc = upsert_arc_plan(db, arc_id="arc1", description="Rise and fall")
    assert arc["status"] == "planned"
    assert get_arc(db, "arc1")["description"] == "Rise and fall"

    chapter = upsert_chapter_plan(
        db,
        chapter_id="ch1",
        arc_id="arc1",
        description="The summons",
        snapshot_id="snap1",
        node_id="pn-ch1",
        ordering=1,
        dramatic_function="introduce the antagonist",
        expected_emotional_shift="calm -> dread",
        required_thread_progress="open the missing-heir thread",
        scene_planning_constraints="at least two scenes; end on a cliff",
    )
    # Narrative row holds only minimal structural columns.
    chapters = get_chapters_for_arc(db, "arc1")
    assert [c["id"] for c in chapters] == ["ch1"]
    assert chapters[0]["arc_id"] == "arc1"
    # Obligations live in the paired PlanningNode (proposal surface), not the Chapters row.
    assert "dramatic_function" not in chapters[0]
    obligations = json.loads(chapter["planning_node"]["purpose"])
    assert obligations["dramatic_function"] == "introduce the antagonist"
    assert obligations["scene_planning_constraints"] == "at least two scenes; end on a cliff"
    ch_node = get_planning_nodes(db, "snap1", level="chapter")[0]
    assert ch_node["node_id"] == "pn-ch1"
    assert json.loads(ch_node["purpose"])["required_thread_progress"] == (
        "open the missing-heir thread"
    )

    # Two scenes written out of order, read back by ordering ASC.
    upsert_scene_plan(
        db, scene_id="sc2", chapter_id="ch1", description="confrontation", ordering=2
    )
    upsert_scene_plan(
        db, scene_id="sc1", chapter_id="ch1", description="arrival", ordering=1
    )
    scenes = get_scenes_for_chapter_ordered(db, "ch1")
    assert [s["id"] for s in scenes] == ["sc1", "sc2"]
    assert [s["ordering"] for s in scenes] == [1, 2]

    beat = upsert_beat_plan(
        db,
        beat_id="b1",
        scene_id="sc1",
        beat_index=0,
        snapshot_id="snap1",
        node_id="pn-b1",
        pad_constraint="tense, guarded, low dominance — clipped speech",
        immediate_objective="get past the gate",
        physical_constraints="raining; locked portcullis",
    )
    # Beat narrative row carries no prose at plan time.
    persisted_beat = get_beat(db, "b1")
    assert persisted_beat["status"] == "planned"
    assert persisted_beat["prose"] is None
    assert persisted_beat["word_count"] == 0
    assert [b["id"] for b in get_beats_for_scene_ordered(db, "sc1")] == ["b1"]
    # The tailored PAD behavioral-constraint string lives in the paired PlanningNode.
    beat_detail = json.loads(beat["planning_node"]["purpose"])
    assert beat_detail["pad_constraint"] == "tense, guarded, low dominance — clipped speech"
    b_node = get_planning_nodes(db, "snap1", level="beat")[0]
    assert b_node["node_id"] == "pn-b1"
    assert json.loads(b_node["purpose"])["immediate_objective"] == "get past the gate"


def test_outline_writers_are_idempotent_and_prose_safe(tmp_path):
    """Replaying plan writes does not duplicate rows or clobber committed prose."""
    db = _db(tmp_path)
    create_planning_snapshot(
        db, snapshot_id="snap1", project_id="proj1", mode="rolling"
    )

    def _write_outline():
        upsert_arc_plan(db, arc_id="arc1", description="Rise and fall")
        upsert_chapter_plan(
            db,
            chapter_id="ch1",
            arc_id="arc1",
            description="The summons",
            snapshot_id="snap1",
            node_id="pn-ch1",
            dramatic_function="introduce the antagonist",
        )
        upsert_scene_plan(
            db, scene_id="sc1", chapter_id="ch1", description="arrival", ordering=1
        )
        upsert_beat_plan(
            db,
            beat_id="b1",
            scene_id="sc1",
            beat_index=0,
            snapshot_id="snap1",
            node_id="pn-b1",
            pad_constraint="tense",
        )

    _write_outline()

    # A later commit fills in prose on the beat structural row.
    from memory.sqlite_db import upsert_beat_commit

    upsert_beat_commit(
        db,
        beat_id="b1",
        scene_id="sc1",
        beat_index=0,
        prose="The portcullis groaned upward.",
    )

    _write_outline()  # replay the plan-time writes AFTER commit

    assert len(get_chapters_for_arc(db, "arc1")) == 1
    assert len(get_beats_for_scene_ordered(db, "sc1")) == 1
    assert len(get_planning_nodes(db, "snap1", level="chapter")) == 1
    assert len(get_planning_nodes(db, "snap1", level="beat")) == 1
    # The plan-time replay must NOT wipe the committed prose.
    committed = get_beat(db, "b1")
    assert committed["prose"] == "The portcullis groaned upward."
    assert committed["word_count"] == 4


def test_obligations_require_planning_node_context(tmp_path):
    """Obligations / PAD detail without a snapshot+node raise, never silently dropped."""
    db = _db(tmp_path)

    try:
        upsert_chapter_plan(
            db,
            chapter_id="ch1",
            arc_id="arc1",
            description="The summons",
            dramatic_function="introduce the antagonist",
        )
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for chapter obligations without node")

    try:
        upsert_beat_plan(
            db,
            beat_id="b1",
            scene_id="sc1",
            beat_index=0,
            pad_constraint="tense",
        )
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for beat PAD detail without node")


def test_planning_node_parent_child_traversal(tmp_path):
    """Chapter plan-node parents a beat plan-node; parent lookup returns the child."""
    db = _db(tmp_path)
    create_planning_snapshot(
        db, snapshot_id="snap1", project_id="proj1", mode="rolling"
    )
    # Seed the narrative FK chain (Chapters->Arcs, Scenes->Chapters, Beats->Scenes)
    # that the plan-time writers also persist into.
    upsert_arc_plan(db, arc_id="arc1", description="Rise and fall")
    upsert_chapter_plan(
        db,
        chapter_id="ch1",
        arc_id="arc1",
        description="The summons",
        snapshot_id="snap1",
        node_id="pn-ch1",
        dramatic_function="introduce the antagonist",
    )
    upsert_scene_plan(
        db, scene_id="sc1", chapter_id="ch1", description="arrival", ordering=1
    )
    upsert_beat_plan(
        db,
        beat_id="b1",
        scene_id="sc1",
        beat_index=0,
        snapshot_id="snap1",
        node_id="pn-b1",
        parent_node_id="pn-ch1",
        pad_constraint="tense",
    )
    children = get_planning_nodes_by_parent(db, "snap1", "pn-ch1")
    assert [n["node_id"] for n in children] == ["pn-b1"]
    assert [n["node_id"] for n in get_planning_nodes_by_parent(db, "snap1", None)] == [
        "pn-ch1"
    ]


def test_root_planning_node_traversal_is_snapshot_scoped(tmp_path):
    """Root traversal for one snapshot cannot mix in another snapshot's roots."""
    db = _db(tmp_path)
    create_planning_snapshot(db, snapshot_id="snap1", project_id="proj1", mode="rolling")
    create_planning_snapshot(db, snapshot_id="snap2", project_id="proj1", mode="rolling")
    upsert_planning_node(
        db, node_id="snap1-root", snapshot_id="snap1", level="arc", status="proposed"
    )
    upsert_planning_node(
        db, node_id="snap2-root", snapshot_id="snap2", level="arc", status="proposed"
    )

    assert [
        n["node_id"] for n in get_planning_nodes_by_parent(db, "snap1", None)
    ] == ["snap1-root"]
    assert [
        n["node_id"] for n in get_planning_nodes_by_parent(db, "snap2", None)
    ] == ["snap2-root"]
