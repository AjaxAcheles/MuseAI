"""Tests for the crash-safe commit node."""

from __future__ import annotations

import json

import pytest

from museai.core.stream_bus import bus
from museai.fsm.nodes import commit as commit_module
from museai.fsm.nodes.commit import commit_transaction
from museai.fsm.nodes.deps import set_node_config
from museai.fsm.state import FSM_Pointer, FailureObject, make_initial_state
from museai.memory import db
from museai.memory.reconcile import scan_and_recover


PROJECT_ID = "project"
ARC_ID = "arc-1"
CHAPTER_ID = "arc-1-c01"
BEAT_ID = "arc-1-c01-b01"
CHAR_ID = "char-mara"
THREAD_ID = "thread-letters"


def _config(config_factory):
    config = config_factory(project_id=PROJECT_ID)
    set_node_config(config)
    bus.last_snapshot.clear()
    return config


def _seed(config, *, planned_word_count: int = 900) -> None:
    db.init_db(config.db_path)
    conn = db.connect_db(config.db_path)
    spec = {
        "intent": "Mara accepts the impossible letter.",
        "target_pad": {"pleasure": -0.4, "arousal": 0.7, "dominance": -0.2},
        "focal_character_id": CHAR_ID,
        "thread_updates": [{"id": THREAD_ID, "status": "progressing", "priority_score": 0.8}],
    }
    with conn:
        db.upsert_project(
            conn, id=PROJECT_ID, genre="mystery", premise="letters", word_count_target=100
        )
        db.upsert_arc(conn, id=ARC_ID, project_id=PROJECT_ID, ordering=1,
                      description="arc", status="active")
        db.upsert_chapter(conn, id=CHAPTER_ID, arc_id=ARC_ID, ordering=1,
                          description="chapter", status="active")
        db.upsert_beat(conn, id="previous", chapter_id=CHAPTER_ID, ordering=0,
                       prose="ten " * 10, word_count=10, status="completed")
        db.upsert_beat(conn, id=BEAT_ID, chapter_id=CHAPTER_ID, ordering=1,
                       beat_spec=json.dumps(spec), pad_constraint="constraint", status="active")
        db.upsert_beat(conn, id="future", chapter_id=CHAPTER_ID, ordering=2,
                       word_count=planned_word_count, status="planned")
        db.upsert_character(conn, id=CHAR_ID, project_id=PROJECT_ID, name="Mara")
        db.upsert_character_emotions(conn, character_id=CHAR_ID,
                                     pleasure=0.0, arousal=0.0, dominance=0.0)
        db.upsert_thread(conn, id=THREAD_ID, project_id=PROJECT_ID,
                         description="Who writes the letters?", status="open",
                         priority_score=0.5)
    conn.close()


def _state(draft: str = "The lamp turns twice."):
    return make_initial_state(
        PROJECT_ID,
        FSM_Pointer(arc_id=ARC_ID, chapter_id=CHAPTER_ID, beat_index=0),
        current_draft_text=draft,
        streaming_buffer="live token stream not committed",
        retry_count=2,
        critic_failures=[
            FailureObject(
                error_code="X",
                offending_text="lamp",
                suggested_fix="lantern",
                critic_source="test",
            )
        ],
        best_seen_draft="older draft",
        best_seen_failure_count=1,
        review_requested=True,
    )


@pytest.mark.asyncio
async def test_commit_writes_pending_intent_then_flips_to_committed(config_factory):
    config = _config(config_factory)
    _seed(config)

    delta = await commit_transaction(_state())

    conn = db.connect_db(config.db_path)
    intent = conn.execute("SELECT * FROM CommitIntent").fetchone()
    assert intent["beat_id"] == BEAT_ID
    assert intent["status"] == "committed"
    assert intent["completed_at"] is not None

    beat = conn.execute("SELECT * FROM Beats WHERE id=?", (BEAT_ID,)).fetchone()
    assert beat["status"] == "completed"
    assert beat["prose"] == "The lamp turns twice."
    assert beat["word_count"] == 4

    thread = conn.execute("SELECT * FROM Threads WHERE id=?", (THREAD_ID,)).fetchone()
    assert thread["status"] == "progressing"
    assert thread["priority_score"] == 0.8

    pad = db.get_character_emotions(conn, CHAR_ID)
    assert pad["pleasure"] == -0.4
    assert pad["arousal"] == 0.7
    assert pad["dominance"] == -0.2
    conn.close()

    assert delta == {
        "retry_count": 0,
        "critic_failures": [],
        "best_seen_draft": None,
        "best_seen_failure_count": None,
        "current_draft_text": "",
        "streaming_buffer": "",
        "review_requested": False,
    }


@pytest.mark.asyncio
async def test_pending_intent_with_appended_event_is_recoverable(config_factory, monkeypatch):
    config = _config(config_factory)
    _seed(config)

    def crash_before_flip(*args, **kwargs):
        raise RuntimeError("crash before intent flip")

    monkeypatch.setattr(commit_module, "mark_commit_committed", crash_before_flip)
    with pytest.raises(RuntimeError, match="crash before intent flip"):
        await commit_module.commit_transaction(_state())

    conn = db.connect_db(config.db_path)
    pending = conn.execute("SELECT * FROM CommitIntent").fetchone()
    assert pending["status"] == "pending"
    conn.close()

    summary = scan_and_recover(config.db_path, config.event_log_path)
    assert summary["pending_found"] == 1
    assert summary["recovered"] == [BEAT_ID]

    conn = db.connect_db(config.db_path)
    recovered = conn.execute("SELECT * FROM CommitIntent").fetchone()
    assert recovered["status"] == "committed"
    beat = conn.execute("SELECT * FROM Beats WHERE id=?", (BEAT_ID,)).fetchone()
    assert json.loads(beat["beat_spec"])["focal_character_id"] == CHAR_ID
    assert beat["pad_constraint"] == "constraint"
    assert beat["ordering"] == 1
    conn.close()


@pytest.mark.asyncio
async def test_word_count_event_uses_committed_beats_only(config_factory):
    config = _config(config_factory)
    _seed(config, planned_word_count=900)

    await commit_transaction(_state(draft="Only committed words count"))

    assert bus.last_snapshot["word_count"] == {
        "project_id": PROJECT_ID,
        "word_count": 14,
        "target": 100,
    }