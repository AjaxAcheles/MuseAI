"""Tests for the SQLite-grounded commit router."""

from __future__ import annotations

from museai.fsm.nodes.deps import set_node_config
from museai.fsm.routers.commit_router import (
    ASSEMBLE,
    EXPORT,
    PLAN_BEAT,
    PLAN_CHAPTER,
    commit_router,
)
from museai.fsm.state import FSM_Pointer, make_initial_state
from museai.memory import db


PROJECT_ID = "project"


def _config(config_factory):
    config = config_factory(project_id=PROJECT_ID)
    set_node_config(config)
    db.init_db(config.db_path)
    return config


def _state(arc_id="arc-1", chapter_id="arc-1-c01", beat_index=0):
    return make_initial_state(
        PROJECT_ID,
        FSM_Pointer(arc_id=arc_id, chapter_id=chapter_id, beat_index=beat_index),
    )


def test_planned_beat_in_active_chapter_routes_to_assemble(config_factory):
    config = _config(config_factory)
    conn = db.connect_db(config.db_path)
    with conn:
        db.upsert_project(conn, id=PROJECT_ID, genre="g", premise="p",
                          word_count_target=1000)
        db.upsert_arc(conn, id="arc-1", project_id=PROJECT_ID, ordering=1,
                      description="arc", status="active")
        db.upsert_chapter(conn, id="arc-1-c01", arc_id="arc-1", ordering=1,
                          description="chapter", status="active")
        db.upsert_beat(conn, id="b1", chapter_id="arc-1-c01", ordering=1,
                       status="completed", word_count=10)
        db.upsert_beat(conn, id="b2", chapter_id="arc-1-c01", ordering=2,
                       status="planned")
    conn.close()

    state = _state()
    assert commit_router(state) == ASSEMBLE
    assert state["fsm_pointer"] == FSM_Pointer(
        arc_id="arc-1", chapter_id="arc-1-c01", beat_index=1
    )
    conn = db.connect_db(config.db_path)
    assert conn.execute("SELECT status FROM Beats WHERE id='b2'").fetchone()["status"] == "active"
    conn.close()


def test_planned_chapter_with_no_beats_routes_to_plan_beat(config_factory):
    config = _config(config_factory)
    conn = db.connect_db(config.db_path)
    with conn:
        db.upsert_project(conn, id=PROJECT_ID, genre="g", premise="p",
                          word_count_target=1000)
        db.upsert_arc(conn, id="arc-1", project_id=PROJECT_ID, ordering=1,
                      description="arc", status="active")
        db.upsert_chapter(conn, id="arc-1-c01", arc_id="arc-1", ordering=1,
                          description="done", status="completed")
        db.upsert_beat(conn, id="b1", chapter_id="arc-1-c01", ordering=1,
                       status="completed", word_count=10)
        db.upsert_chapter(conn, id="arc-1-c02", arc_id="arc-1", ordering=2,
                          description="next", status="planned")
    conn.close()

    state = _state()
    assert commit_router(state) == PLAN_BEAT
    assert state["fsm_pointer"] == FSM_Pointer(
        arc_id="arc-1", chapter_id="arc-1-c02", beat_index=0
    )
    conn = db.connect_db(config.db_path)
    assert conn.execute(
        "SELECT status FROM Chapters WHERE id='arc-1-c02'"
    ).fetchone()["status"] == "active"
    conn.close()


def test_planned_arc_routes_to_plan_chapter(config_factory):
    config = _config(config_factory)
    conn = db.connect_db(config.db_path)
    with conn:
        db.upsert_project(conn, id=PROJECT_ID, genre="g", premise="p",
                          word_count_target=1000)
        db.upsert_arc(conn, id="arc-1", project_id=PROJECT_ID, ordering=1,
                      description="done", status="completed")
        db.upsert_chapter(conn, id="arc-1-c01", arc_id="arc-1", ordering=1,
                          description="done", status="completed")
        db.upsert_beat(conn, id="b1", chapter_id="arc-1-c01", ordering=1,
                       status="completed", word_count=10)
        db.upsert_arc(conn, id="arc-2", project_id=PROJECT_ID, ordering=2,
                      description="next", status="planned")
    conn.close()

    state = _state()
    assert commit_router(state) == PLAN_CHAPTER
    assert state["fsm_pointer"] == FSM_Pointer(arc_id="arc-2", chapter_id="", beat_index=0)
    conn = db.connect_db(config.db_path)
    assert conn.execute("SELECT status FROM Arcs WHERE id='arc-2'").fetchone()["status"] == "active"
    conn.close()


def test_completed_outline_or_target_routes_to_export(config_factory):
    config = _config(config_factory)
    conn = db.connect_db(config.db_path)
    with conn:
        db.upsert_project(conn, id=PROJECT_ID, genre="g", premise="p",
                          word_count_target=10)
        db.upsert_arc(conn, id="arc-1", project_id=PROJECT_ID, ordering=1,
                      description="done", status="completed")
        db.upsert_chapter(conn, id="arc-1-c01", arc_id="arc-1", ordering=1,
                          description="done", status="completed")
        db.upsert_beat(conn, id="b1", chapter_id="arc-1-c01", ordering=1,
                       status="completed", word_count=10)
    conn.close()

    state = _state()
    assert commit_router(state) == EXPORT