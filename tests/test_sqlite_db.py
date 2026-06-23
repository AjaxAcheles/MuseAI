"""Module: M02 (Persistent Memory Stores & Interfaces)
Synthetic tests for the relational hub (`memory/sqlite_db.py`): schema
initialization, idempotent re-init, hard CHECK constraints (status vocabularies +
PAD bounds), and deterministic `ordering ASC` scene reads.

All fixtures are file-local under ``tmp_path`` — these tests never write to the
real ``data/`` directory. Event log, provisional claims, Graphiti, RAPTOR, Chroma,
the style store, runtime lifecycle, and graph nodes are out of scope here.
"""

import sqlite3

import pytest

from memory.sqlite_db import (
    connect_db,
    get_beat,
    get_latest_pad_for_character,
    get_latest_pad_for_scene,
    get_open_threads,
    get_pending_commit_intents,
    get_scenes_for_chapter_ordered,
    init_db,
    upsert_beat_commit,
)

DOCUMENTED_TABLES = {
    "Arcs",
    "Chapters",
    "Scenes",
    "Beats",
    "Threads",
    "Characters",
    "CharacterEmotions",
    "CommitIntent",
    "RaptorNodes",
}


def _db(tmp_path):
    """A nested temp DB path — proves init_db creates parents, stays in tmp_path."""
    return tmp_path / "memory" / "fictionwriter.db"


def _table_names(db_path) -> set[str]:
    conn = connect_db(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        return {row["name"] for row in rows}
    finally:
        conn.close()


# --- Synthetic parent-row seeders (FK enforcement is ON per connection) ----------
def _seed_arc(conn, arc_id="a1", status="planned"):
    conn.execute(
        "INSERT INTO Arcs (id, description, status) VALUES (?, ?, ?)",
        (arc_id, "synthetic arc", status),
    )


def _seed_chapter(conn, chapter_id="c1", arc_id="a1", status="planned"):
    conn.execute(
        "INSERT INTO Chapters (id, arc_id, description, status) VALUES (?, ?, ?, ?)",
        (chapter_id, arc_id, "synthetic chapter", status),
    )


def _seed_scene(conn, scene_id, chapter_id, ordering, status="planned"):
    conn.execute(
        "INSERT INTO Scenes (id, chapter_id, description, word_budget, ordering, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (scene_id, chapter_id, "synthetic scene", 0, ordering, status),
    )


def _seed_pad_parent_chain(conn):
    """Arc -> Chapter -> Scene -> Beat plus a Character, so CharacterEmotions FKs hold."""
    _seed_arc(conn)
    _seed_chapter(conn)
    _seed_scene(conn, "s1", "c1", ordering=0)
    conn.execute(
        "INSERT INTO Beats (id, scene_id, beat_index, status) VALUES (?, ?, ?, ?)",
        ("b1", "s1", 0, "completed"),
    )
    conn.execute(
        "INSERT INTO Characters (id, name) VALUES (?, ?)", ("char1", "Mara")
    )


# --- Schema init -----------------------------------------------------------------
def test_init_db_creates_documented_schema_in_tmp(tmp_path):
    db = _db(tmp_path)
    assert not db.exists()

    init_db(db)

    assert db.exists()
    assert DOCUMENTED_TABLES <= _table_names(db)


def test_init_db_twice_is_safe_and_preserves_schema_and_data(tmp_path):
    db = _db(tmp_path)
    init_db(db)

    conn = connect_db(db)
    try:
        _seed_arc(conn, "a_keep", status="active")
        conn.commit()
    finally:
        conn.close()

    # A second init must not raise, drop tables, or wipe existing rows.
    init_db(db)

    assert DOCUMENTED_TABLES <= _table_names(db)
    conn = connect_db(db)
    try:
        row = conn.execute(
            "SELECT status FROM Arcs WHERE id = ?", ("a_keep",)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row["status"] == "active"


# --- Status CHECK constraints ----------------------------------------------------
@pytest.mark.parametrize(
    "sql, params",
    [
        # Planning-status vocabulary: only planned/active/completed are valid.
        (
            "INSERT INTO Arcs (id, description, status) VALUES (?, ?, ?)",
            ("a_bad", "d", "bogus"),
        ),
        # Thread-status vocabulary: only open/progressing/closed are valid.
        (
            "INSERT INTO Threads (id, description, status, priority_score) "
            "VALUES (?, ?, ?, ?)",
            ("t_bad", "d", "reopened", 1.0),
        ),
        # CommitIntent status: only pending/committed are valid.
        (
            "INSERT INTO CommitIntent "
            "(beat_id, arc_id, chapter_id, scene_id, beat_index, status, initiated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("b", "a", "c", "s", 0, "in_progress", "2026-01-01T00:00:00+00:00"),
        ),
    ],
)
def test_status_check_constraint_rejects_invalid_enum(tmp_path, sql, params):
    db = _db(tmp_path)
    init_db(db)

    conn = connect_db(db)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql, params)
    finally:
        conn.close()


def test_status_check_constraint_accepts_documented_enum(tmp_path):
    db = _db(tmp_path)
    init_db(db)

    conn = connect_db(db)
    try:
        for status in ("planned", "active", "completed"):
            _seed_arc(conn, f"a_{status}", status=status)
        conn.commit()
        count = conn.execute("SELECT COUNT(*) AS n FROM Arcs").fetchone()["n"]
    finally:
        conn.close()
    assert count == 3


# --- CharacterEmotions PAD bound CHECK constraints -------------------------------
@pytest.mark.parametrize(
    "pleasure, arousal, dominance",
    [
        (1.01, 0.0, 0.0),   # pleasure just above the inclusive max
        (0.0, -1.01, 0.0),  # arousal just below the inclusive min
        (0.0, 0.0, 999.0),  # dominance wildly out of range (the hallucinated score)
    ],
)
def test_pad_bounds_check_constraint_rejects_out_of_range(
    tmp_path, pleasure, arousal, dominance
):
    db = _db(tmp_path)
    init_db(db)

    conn = connect_db(db)
    try:
        _seed_pad_parent_chain(conn)
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO CharacterEmotions "
                "(beat_id, character_id, pleasure, arousal, dominance) "
                "VALUES (?, ?, ?, ?, ?)",
                ("b1", "char1", pleasure, arousal, dominance),
            )
    finally:
        conn.close()


def test_pad_bounds_check_constraint_allows_inclusive_edges(tmp_path):
    db = _db(tmp_path)
    init_db(db)

    conn = connect_db(db)
    try:
        _seed_pad_parent_chain(conn)
        # The documented range is inclusive: exactly -1.00 and 1.00 must be accepted.
        conn.execute(
            "INSERT INTO CharacterEmotions "
            "(beat_id, character_id, pleasure, arousal, dominance) "
            "VALUES (?, ?, ?, ?, ?)",
            ("b1", "char1", 1.00, -1.00, 0.0),
        )
        conn.commit()
        row = conn.execute(
            "SELECT pleasure, arousal, dominance FROM CharacterEmotions "
            "WHERE beat_id = ? AND character_id = ?",
            ("b1", "char1"),
        ).fetchone()
    finally:
        conn.close()
    assert (row["pleasure"], row["arousal"], row["dominance"]) == (1.00, -1.00, 0.0)


# --- Deterministic ordered reads -------------------------------------------------
def test_scenes_read_by_ordering_asc_not_insertion_order(tmp_path):
    db = _db(tmp_path)
    init_db(db)

    conn = connect_db(db)
    try:
        _seed_arc(conn)
        _seed_chapter(conn)
        # Insert in an order that deliberately does NOT match `ordering`.
        _seed_scene(conn, "s_mid", "c1", ordering=2)
        _seed_scene(conn, "s_first", "c1", ordering=1)
        _seed_scene(conn, "s_last", "c1", ordering=3)
        conn.commit()
    finally:
        conn.close()

    ids = [scene["id"] for scene in get_scenes_for_chapter_ordered(db, "c1")]

    assert ids == ["s_first", "s_mid", "s_last"]  # sorted by `ordering ASC`
    assert ids != ["s_mid", "s_first", "s_last"]  # i.e. not insertion order


# --- Idempotent beat commit (write + read round-trip) ----------------------------
# A synthetic committed beat: metadata + prose, one thread transition, and PAD
# snapshots for two characters bundled into the same commit (matching the design's
# "PAD travels inside the beat commit" rule).
BEAT_COMMIT = {
    "beat_id": "b1",
    "scene_id": "s1",
    "beat_index": 0,
    "prose": "The door clicked shut behind her.",
    "pad_states": {
        "char1": {"pleasure": 0.4, "arousal": -0.2, "dominance": 0.7},
        "char2": {"pleasure": -0.8, "arousal": 0.9, "dominance": -0.6},
    },
    "thread_updates": {"t1": "progressing"},
}


def _seed_beat_commit_parents(db_path):
    """Seed the parent rows a beat commit's FKs require, plus two open threads.

    upsert_beat_commit writes a Beat (FK -> Scenes), per-character PAD rows
    (FK -> Beats, Characters), and thread status UPDATEs (the planner owns thread
    creation). ``t1`` is the thread the commit progresses; ``t2`` stays open so the
    open-thread read has something to return after ``t1`` transitions out.
    """
    conn = connect_db(db_path)
    try:
        _seed_arc(conn)
        _seed_chapter(conn)
        _seed_scene(conn, "s1", "c1", ordering=0)
        conn.execute("INSERT INTO Characters (id, name) VALUES (?, ?)", ("char1", "Mara"))
        conn.execute("INSERT INTO Characters (id, name) VALUES (?, ?)", ("char2", "Tom"))
        conn.execute(
            "INSERT INTO Threads (id, description, status, priority_score) "
            "VALUES (?, ?, ?, ?)",
            ("t1", "the locked door", "open", 0.8),
        )
        conn.execute(
            "INSERT INTO Threads (id, description, status, priority_score) "
            "VALUES (?, ?, ?, ?)",
            ("t2", "the missing letter", "open", 0.5),
        )
        conn.commit()
    finally:
        conn.close()


def test_beat_commit_written_twice_has_single_beat_row(tmp_path):
    db = _db(tmp_path)
    init_db(db)
    _seed_beat_commit_parents(db)

    upsert_beat_commit(db, **BEAT_COMMIT)
    upsert_beat_commit(db, **BEAT_COMMIT)  # replay (e.g. crash-recovery)

    conn = connect_db(db)
    try:
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM Beats WHERE id = ?", ("b1",)
        ).fetchone()["n"]
    finally:
        conn.close()
    assert count == 1


def test_beat_commit_pad_and_thread_updates_are_idempotent(tmp_path):
    db = _db(tmp_path)
    init_db(db)
    _seed_beat_commit_parents(db)

    upsert_beat_commit(db, **BEAT_COMMIT)
    upsert_beat_commit(db, **BEAT_COMMIT)

    conn = connect_db(db)
    try:
        pad_total = conn.execute(
            "SELECT COUNT(*) AS n FROM CharacterEmotions WHERE beat_id = ?", ("b1",)
        ).fetchone()["n"]
        per_char = {
            char_id: conn.execute(
                "SELECT COUNT(*) AS n FROM CharacterEmotions "
                "WHERE beat_id = ? AND character_id = ?",
                ("b1", char_id),
            ).fetchone()["n"]
            for char_id in ("char1", "char2")
        }
        thread_count = conn.execute(
            "SELECT COUNT(*) AS n FROM Threads"
        ).fetchone()["n"]
        t1_status = conn.execute(
            "SELECT status FROM Threads WHERE id = ?", ("t1",)
        ).fetchone()["status"]
    finally:
        conn.close()

    # One PAD row per (beat_id, character_id) — replay updates in place, never appends.
    assert pad_total == 2
    assert per_char == {"char1": 1, "char2": 1}
    # The thread transition mutates the canonical row; it never creates a duplicate.
    assert thread_count == 2
    assert t1_status == "progressing"


def test_beat_commit_read_helpers_and_empty_result_behavior(tmp_path):
    db = _db(tmp_path)
    init_db(db)
    _seed_beat_commit_parents(db)
    upsert_beat_commit(db, **BEAT_COMMIT)

    # Committed beat is retrievable with its committed prose/state.
    beat = get_beat(db, "b1")
    assert beat is not None
    assert beat["prose"] == BEAT_COMMIT["prose"]
    assert beat["beat_index"] == 0
    assert beat["status"] == "completed"

    # Open threads: t1 progressed out, t2 still open.
    assert [t["id"] for t in get_open_threads(db)] == ["t2"]

    # Latest PAD state per character and across the scene.
    pad1 = get_latest_pad_for_character(db, "char1")
    assert pad1 is not None
    assert (pad1["pleasure"], pad1["arousal"], pad1["dominance"]) == (0.4, -0.2, 0.7)
    scene_pad = get_latest_pad_for_scene(db, "s1")
    assert {p["character_id"] for p in scene_pad} == {"char1", "char2"}

    # The relational helper writes no CommitIntent (that is M10) -> documented [].
    assert get_pending_commit_intents(db) == []

    # Documented empty-result behavior on misses.
    assert get_beat(db, "missing") is None
    with pytest.raises(KeyError):
        get_beat(db, "missing", strict=True)
    assert get_latest_pad_for_character(db, "ghost") is None
    assert get_latest_pad_for_scene(db, "missing_scene") == []
