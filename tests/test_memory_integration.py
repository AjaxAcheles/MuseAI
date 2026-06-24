"""Module: M14 (Configuration, Startup & Observability) x M02 (Persistent Memory)
INT-A·B integration tests: the resource lifecycle brings up a real SQLite store,
and the relational write/read helpers round-trip against the database that
`core.runtime.init_resources` creates.

All fixtures are file-local under ``tmp_path`` (runtime paths are monkeypatched) —
these tests never write to the real ``data/`` or ``logs/`` directories. Scope is
Config/Startup + SQLite only: event log and provisional store have their own
isolated tests, and Graphiti/RAPTOR/Chroma/style remain deferred stubs. No
LangGraph nodes, LLM calls, or branch replay here.
"""

import pytest

from core import runtime
from memory.sqlite_db import (
    connect_db,
    get_beat,
    get_latest_pad_for_character,
    get_open_threads,
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

# A synthetic committed beat: metadata + prose, one thread transition, and PAD for
# two characters bundled into the same commit.
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


@pytest.fixture
def config():
    """Synthetic config: init_db ignores it; the remaining store stubs no-op on it."""
    return object()


@pytest.fixture
def temp_runtime(tmp_path, monkeypatch):
    """Point every runtime artifact path at a temp root, so init/reset stay local."""
    data = tmp_path / "data"
    logs = tmp_path / "logs"
    monkeypatch.setattr(runtime, "DATA_DIR", data)
    monkeypatch.setattr(runtime, "LOG_DIR", logs)
    monkeypatch.setattr(runtime, "SQLITE_DB_PATH", data / "fictionwriter.db")
    monkeypatch.setattr(runtime, "GRAPHITI_DB_PATH", data / "graphiti.db")
    monkeypatch.setattr(runtime, "CHROMA_STORE_DIR", data / "chroma")
    monkeypatch.setattr(runtime, "STYLE_STORE_DIR", data / "styles")
    monkeypatch.setattr(runtime, "SNAPSHOT_DIR", data / "snapshots")
    monkeypatch.setattr(runtime, "EVENT_LOG_PATH", data / "events.jsonl")
    return tmp_path


def _table_names(db_path) -> set[str]:
    conn = connect_db(db_path)
    try:
        return {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        conn.close()


def _seed_beat_commit_parents(db_path):
    """Seed FK parents a beat commit requires, plus two open threads."""
    conn = connect_db(db_path)
    try:
        conn.execute(
            "INSERT INTO Arcs (id, description, status) VALUES (?, ?, ?)",
            ("a1", "arc", "planned"),
        )
        conn.execute(
            "INSERT INTO Chapters (id, arc_id, description, status) VALUES (?, ?, ?, ?)",
            ("c1", "a1", "chap", "planned"),
        )
        conn.execute(
            "INSERT INTO Scenes (id, chapter_id, description, word_budget, ordering, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("s1", "c1", "scene", 0, 0, "planned"),
        )
        conn.execute("INSERT INTO Characters (id, name) VALUES (?, ?)", ("char1", "Mara"))
        conn.execute("INSERT INTO Characters (id, name) VALUES (?, ?)", ("char2", "Tom"))
        conn.execute(
            "INSERT INTO Threads (id, description, status, priority_score) VALUES (?, ?, ?, ?)",
            ("t1", "the locked door", "open", 0.8),
        )
        conn.execute(
            "INSERT INTO Threads (id, description, status, priority_score) VALUES (?, ?, ?, ?)",
            ("t2", "the missing letter", "open", 0.5),
        )
        conn.commit()
    finally:
        conn.close()


def test_init_resources_creates_real_relational_schema(temp_runtime, config):
    runtime.init_resources(config)

    db = runtime.SQLITE_DB_PATH
    assert db.is_file()
    # A genuine SQLite database, not the old empty placeholder.
    assert db.read_bytes().startswith(b"SQLite format 3\x00")
    assert DOCUMENTED_TABLES <= _table_names(db)
    # Stayed entirely inside the temp root.
    assert str(db).startswith(str(temp_runtime))


def test_beat_commit_round_trips_through_init_resources_db(temp_runtime, config):
    runtime.init_resources(config)
    db = runtime.SQLITE_DB_PATH
    _seed_beat_commit_parents(db)

    upsert_beat_commit(db, **BEAT_COMMIT)

    beat = get_beat(db, "b1")
    assert beat is not None
    assert beat["prose"] == BEAT_COMMIT["prose"]
    assert beat["status"] == "completed"
    # Thread transition applied; t1 progressed out, t2 still open.
    assert [t["id"] for t in get_open_threads(db)] == ["t2"]
    # PAD snapshot readable through the public helper.
    pad = get_latest_pad_for_character(db, "char1")
    assert pad is not None
    assert (pad["pleasure"], pad["arousal"], pad["dominance"]) == (0.4, -0.2, 0.7)


def test_beat_commit_is_idempotent_through_init_resources_db(temp_runtime, config):
    runtime.init_resources(config)
    db = runtime.SQLITE_DB_PATH
    _seed_beat_commit_parents(db)

    upsert_beat_commit(db, **BEAT_COMMIT)
    upsert_beat_commit(db, **BEAT_COMMIT)  # replay

    conn = connect_db(db)
    try:
        beat_rows = conn.execute(
            "SELECT COUNT(*) AS n FROM Beats WHERE id = ?", ("b1",)
        ).fetchone()["n"]
        pad_rows = conn.execute(
            "SELECT COUNT(*) AS n FROM CharacterEmotions WHERE beat_id = ?", ("b1",)
        ).fetchone()["n"]
    finally:
        conn.close()
    assert beat_rows == 1
    assert pad_rows == 2  # one row per (beat_id, character_id), updated in place


def test_reset_removes_db_artifact_and_reinit_recreates_cleanly(temp_runtime, config):
    runtime.init_resources(config)
    db = runtime.SQLITE_DB_PATH
    _seed_beat_commit_parents(db)
    upsert_beat_commit(db, **BEAT_COMMIT)
    assert get_beat(db, "b1") is not None

    # reset deletes the relational artifact and re-runs init_resources to recreate it.
    runtime.reset_resources(config)

    assert db.is_file()
    assert db.read_bytes().startswith(b"SQLite format 3\x00")
    assert DOCUMENTED_TABLES <= _table_names(db)
    # The prior committed beat is gone — the artifact was truly removed, not appended.
    assert get_beat(db, "b1") is None

    # A second explicit init over the fresh DB is a clean no-op (idempotent).
    runtime.init_resources(config)
    assert DOCUMENTED_TABLES <= _table_names(db)
