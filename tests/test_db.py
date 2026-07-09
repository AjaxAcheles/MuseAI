"""Tests for the SQLite persistence layer."""

from __future__ import annotations

import sqlite3

import pytest

from museai.memory import db


@pytest.fixture
def conn(tmp_path):
    path = tmp_path / "museai.db"
    db.init_db(path)
    c = db.connect_db(path)
    yield c
    c.close()


def _seed_project(conn, project_id="p1"):
    with conn:
        db.upsert_project(conn, id=project_id, genre="g", premise="pr",
                          word_count_target=1000)
    return project_id


def test_init_db_is_idempotent(tmp_path):
    path = tmp_path / "museai.db"
    db.init_db(path)
    db.init_db(path)  # second call must not raise
    c = db.connect_db(path)
    tables = {
        r["name"]
        for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    c.close()
    assert {
        "Projects", "Arcs", "Chapters", "Beats", "Threads",
        "Characters", "CharacterEmotions", "CommitIntent",
    } <= tables
    assert "Scenes" not in tables


def test_upsert_round_trip_and_overwrite(conn):
    _seed_project(conn)
    row = db.get_project(conn, "p1")
    assert row["genre"] == "g"
    assert row["word_count_target"] == 1000

    # Re-upsert same id updates in place (idempotent, no duplicate).
    with conn:
        db.upsert_project(conn, id="p1", genre="g2", premise="pr2",
                          word_count_target=2000)
    row = db.get_project(conn, "p1")
    assert row["genre"] == "g2"
    assert row["word_count_target"] == 2000
    count = conn.execute("SELECT COUNT(*) AS n FROM Projects").fetchone()["n"]
    assert count == 1


def test_ordered_reads_sort_by_ordering(conn):
    pid = _seed_project(conn)
    with conn:
        # Insert arcs out of order.
        db.upsert_arc(conn, id="a2", project_id=pid, ordering=2,
                      description="second", status="planned")
        db.upsert_arc(conn, id="a0", project_id=pid, ordering=0,
                      description="zeroth", status="active")
        db.upsert_arc(conn, id="a1", project_id=pid, ordering=1,
                      description="first", status="planned")
    arcs = db.get_arcs(conn, pid)
    assert [a["id"] for a in arcs] == ["a0", "a1", "a2"]

    with conn:
        db.upsert_chapter(conn, id="c1", arc_id="a0", ordering=1,
                          description="ch1", status="planned")
        db.upsert_chapter(conn, id="c0", arc_id="a0", ordering=0,
                          description="ch0", status="active")
    chapters = db.get_chapters_for_arc(conn, "a0")
    assert [c["id"] for c in chapters] == ["c0", "c1"]

    with conn:
        db.upsert_beat(conn, id="b1", chapter_id="c0", ordering=1)
        db.upsert_beat(conn, id="b0", chapter_id="c0", ordering=0)
    beats = db.get_beats_for_chapter(conn, "c0")
    assert [b["id"] for b in beats] == ["b0", "b1"]


def test_pad_check_rejects_out_of_range(conn):
    pid = _seed_project(conn)
    with conn:
        db.upsert_character(conn, id="ch1", project_id=pid, name="Name")
    with pytest.raises(sqlite3.IntegrityError):
        with conn:
            db.upsert_character_emotions(
                conn, character_id="ch1", pleasure=2.0, arousal=0.0, dominance=0.0
            )
    # In-range value is accepted and upserts in place.
    with conn:
        db.upsert_character_emotions(
            conn, character_id="ch1", pleasure=0.5, arousal=-0.5, dominance=0.0
        )
    emo = db.get_character_emotions(conn, "ch1")
    assert emo["pleasure"] == 0.5


def test_get_open_threads_priority_order(conn):
    pid = _seed_project(conn)
    with conn:
        db.upsert_thread(conn, id="t1", project_id=pid, description="low",
                         status="open", priority_score=0.2)
        db.upsert_thread(conn, id="t2", project_id=pid, description="high",
                         status="open", priority_score=0.9)
        db.upsert_thread(conn, id="t3", project_id=pid, description="closed",
                         status="closed", priority_score=1.0)
    threads = db.get_open_threads(conn, pid)
    assert [t["id"] for t in threads] == ["t2", "t1"]


def test_recent_committed_beats_newest_last(conn):
    pid = _seed_project(conn)
    with conn:
        db.upsert_arc(conn, id="a0", project_id=pid, ordering=0,
                      description="arc", status="active")
        db.upsert_chapter(conn, id="c0", arc_id="a0", ordering=0,
                          description="ch", status="active")
        for i in range(3):
            db.upsert_beat(conn, id=f"b{i}", chapter_id="c0", ordering=i,
                           prose=f"prose {i}", word_count=10, status="completed")
        # A planned beat must be excluded.
        db.upsert_beat(conn, id="b3", chapter_id="c0", ordering=3,
                       status="planned")
    recent = db.get_recent_committed_beats(conn, pid, limit=2)
    # Newest-last: the two most recent completed beats, oldest first.
    assert [b["id"] for b in recent] == ["b1", "b2"]
