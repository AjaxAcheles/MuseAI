"""Module: M10 (Commit, Consolidation & Crash Immunity)
Synthetic tests for `node_commit_transaction`: it persists the drafted beat to the
canonical Beats row, appends a beat_commit event to the append-only log, advances the
pointer to the next planned beat, and flags completion on the last beat. Real temp
SQLite + event log; no model calls.
"""

import asyncio

from fsm.nodes.node_commit_transaction import node_commit_transaction
from fsm.state import FSM_Pointer
from memory import sqlite_db
from memory.event_log import iter_events


def _seed_parents(db):
    """Materialize the arc -> chapter -> scene chain so the beat commit FK holds
    (the real flow builds these in node_plan_arc / node_plan_structure)."""
    sqlite_db.upsert_arc_plan(db, arc_id="a1", description="")
    sqlite_db.upsert_chapter_plan(db, chapter_id="c1", arc_id="a1", description="")
    sqlite_db.upsert_scene_plan(db, scene_id="s1", chapter_id="c1", description="", ordering=0)


def _state(db, log, beat_id):
    return {
        "project_id": "p1",
        "sqlite_db_path": db,
        "event_log_path": log,
        "fsm_pointer": FSM_Pointer(
            arc_id="a1", chapter_id="c1", scene_id="s1", beat_index=0, beat_id=beat_id
        ),
        "beat_order": ["b1", "b2"],
        "beat_plan_by_id": {
            "b1": {"scene_id": "s1", "beat_index": 0},
            "b2": {"scene_id": "s1", "beat_index": 1},
        },
        "current_draft_text": "The rain fell on the empty pier.",
        "committed_word_count": 0,
    }


def test_commit_persists_beat_and_event_and_advances_pointer(tmp_path):
    db = tmp_path / "hub.db"
    log = tmp_path / "events.jsonl"
    sqlite_db.init_db(db)
    _seed_parents(db)
    state = _state(db, log, "b1")

    delta = asyncio.run(node_commit_transaction(state))

    # Canonical relational write.
    beat = sqlite_db.get_beat(db, "b1")
    assert beat is not None
    assert beat["prose"] == "The rain fell on the empty pier."
    assert beat["word_count"] == 6

    # Append-only event record.
    events = list(iter_events(log))
    assert len(events) == 1
    assert events[0]["event"] == "beat_commit"
    assert events[0]["beat_id"] == "b1"

    # Pointer advanced to the next planned beat; not yet complete.
    assert delta["generation_complete"] is False
    assert delta["fsm_pointer"].beat_id == "b2"
    assert delta["last_committed_prose"] == "The rain fell on the empty pier."
    assert delta["committed_word_count"] == 6


def test_commit_of_last_beat_flags_completion(tmp_path):
    db = tmp_path / "hub.db"
    log = tmp_path / "events.jsonl"
    sqlite_db.init_db(db)
    _seed_parents(db)
    state = _state(db, log, "b2")  # b2 is the last in beat_order

    delta = asyncio.run(node_commit_transaction(state))

    assert delta["generation_complete"] is True
    assert "fsm_pointer" not in delta  # nothing left to advance to


def test_commit_is_idempotent_on_replay(tmp_path):
    db = tmp_path / "hub.db"
    log = tmp_path / "events.jsonl"
    sqlite_db.init_db(db)
    _seed_parents(db)

    asyncio.run(node_commit_transaction(_state(db, log, "b1")))
    asyncio.run(node_commit_transaction(_state(db, log, "b1")))

    # Beat row upserts by id — a single row, not a duplicate.
    beat = sqlite_db.get_beat(db, "b1")
    assert beat["prose"] == "The rain fell on the empty pier."
