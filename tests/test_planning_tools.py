"""Module: M05 (Hierarchical Planning Cascade)
Deterministic tests for the safe planning-tool registry and trace surface.
"""

import json

from fsm import planning_tools
from fsm.planning_tools import DISPATCH, PlanningToolRegistry
from memory.sqlite_db import (
    connect_db,
    create_planning_snapshot,
    get_traces_for_snapshot,
    upsert_arc_plan,
    upsert_beat_plan,
    upsert_chapter_plan,
    upsert_scene_plan,
)


def _seed_planning_db(tmp_path):
    db_path = tmp_path / "planning.db"
    create_planning_snapshot(
        db_path,
        snapshot_id="snap-1",
        project_id="project-1",
        mode="macro_outline_before_draft",
    )
    upsert_arc_plan(db_path, arc_id="arc-1", description="A seeded arc")
    upsert_chapter_plan(
        db_path,
        chapter_id="chapter-1",
        arc_id="arc-1",
        description="A seeded chapter",
    )
    upsert_scene_plan(
        db_path,
        scene_id="scene-1",
        chapter_id="chapter-1",
        description="A seeded scene",
        ordering=0,
    )
    upsert_beat_plan(db_path, beat_id="beat-1", scene_id="scene-1", beat_index=0)
    conn = connect_db(db_path)
    try:
        conn.execute(
            "INSERT INTO Threads (id, description, status, priority_score) "
            "VALUES (?, ?, ?, ?)",
            ("thread-1", "recover the map", "open", 1.0),
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def test_permission_matrix_representative_behavior(tmp_path):
    registry = PlanningToolRegistry(_seed_planning_db(tmp_path))

    assert registry.permitted("global", "read_arcs") is True
    assert registry.permitted("global", "read_chapters") is False
    assert registry.permitted("chapter", "read_scenes") is True
    assert registry.permitted("scene", "write_beat_plan") is False
    assert registry.permitted("beat", "write_beat_plan") is True
    assert registry.permitted("global", "query_chroma_flavour") is False
    assert registry.access_for("beat", "query_raptor_summaries") == "limited"


def test_permitted_read_tools_return_seeded_data_and_write_traces(tmp_path):
    db_path = _seed_planning_db(tmp_path)
    registry = PlanningToolRegistry(db_path)

    arc = registry.call("global", "read_arcs", {"arc_id": "arc-1"}, "snap-1", 0)
    chapters = registry.call(
        "arc", "read_chapters", {"arc_id": "arc-1"}, "snap-1", 1
    )
    scenes = registry.call(
        "chapter", "read_scenes", {"chapter_id": "chapter-1"}, "snap-1", 2
    )
    beats = registry.call("scene", "read_beats", {"scene_id": "scene-1"}, "snap-1", 3)
    threads = registry.call("beat", "read_open_threads", {}, "snap-1", 4)

    assert arc["outcome"] == "executed"
    assert arc["data"]["id"] == "arc-1"
    assert chapters["data"][0]["id"] == "chapter-1"
    assert scenes["data"][0]["id"] == "scene-1"
    assert beats["data"][0]["id"] == "beat-1"
    assert threads["data"][0]["id"] == "thread-1"
    traces = get_traces_for_snapshot(db_path, "snap-1")
    assert [trace["tool_name"] for trace in traces] == [
        "read_arcs",
        "read_chapters",
        "read_scenes",
        "read_beats",
        "read_open_threads",
    ]
    assert all(trace["success"] == 1 for trace in traces)
    assert json.loads(traces[0]["tool_args_json"]) == {"arc_id": "arc-1"}


def test_disallowed_and_unknown_tools_are_rejected_without_execution(tmp_path, monkeypatch):
    db_path = _seed_planning_db(tmp_path)
    registry = PlanningToolRegistry(db_path)

    def _must_not_execute(db_path, args):  # pragma: no cover - failure path only
        raise AssertionError("disallowed tool executed")

    monkeypatch.setitem(planning_tools.DISPATCH, "read_beats", _must_not_execute)

    disallowed = registry.call(
        "global", "read_beats", {"scene_id": "scene-1"}, "snap-1", 0
    )
    unknown = registry.call("global", "open_socket", {"host": "example.test"}, "snap-1", 1)

    assert disallowed["outcome"] == "rejected"
    assert disallowed["available"] is False
    assert "not permitted" in disallowed["reason"]
    assert unknown["outcome"] == "rejected"
    assert unknown["available"] is False
    assert "unknown tool" in unknown["reason"]
    traces = get_traces_for_snapshot(db_path, "snap-1")
    assert [trace["tool_name"] for trace in traces] == ["read_beats", "open_socket"]
    assert all(trace["success"] == 0 for trace in traces)
    assert all(trace["error"].startswith("rejected:") for trace in traces)


def test_deferred_store_query_tools_degrade_as_unavailable(tmp_path):
    db_path = _seed_planning_db(tmp_path)
    registry = PlanningToolRegistry(db_path)

    continuity = registry.call(
        "scene", "query_continuity_facts", {"subject": "gate"}, "snap-1", 0
    )
    raptor = registry.call(
        "beat", "query_raptor_summaries", {"scope": "chapter"}, "snap-1", 1
    )
    chroma = registry.call(
        "scene", "query_chroma_flavour", {"query": "rain on stone"}, "snap-1", 2
    )

    for result in (continuity, raptor, chroma):
        assert result["outcome"] == "degraded"
        assert result["available"] is False
        assert result["data"] == []
        assert "not yet implemented" in result["reason"]
    traces = get_traces_for_snapshot(db_path, "snap-1")
    assert [trace["success"] for trace in traces] == [0, 0, 0]
    assert all(trace["error"].startswith("unavailable:") for trace in traces)


def test_dispatch_surface_has_no_shell_filesystem_or_network_tools():
    unsafe_tokens = (
        "shell",
        "command",
        "subprocess",
        "filesystem",
        "file",
        "path",
        "network",
        "socket",
        "http",
        "url",
        "request",
    )

    assert DISPATCH
    assert not [
        tool_name
        for tool_name in DISPATCH
        if any(token in tool_name for token in unsafe_tokens)
    ]
