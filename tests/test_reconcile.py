"""Tests for crash reconciliation."""

from __future__ import annotations

from museai.core.events import append_event
from museai.memory import db
from museai.memory.reconcile import scan_and_recover


def _build_graph(config):
    """Create project -> arc -> chapter -> beat + character + thread."""
    db.init_db(config.db_path)
    conn = db.connect_db(config.db_path)
    with conn:
        db.upsert_project(conn, id="p1", genre="g", premise="pr",
                          word_count_target=1000)
        db.upsert_arc(conn, id="a0", project_id="p1", ordering=0,
                      description="arc", status="active")
        db.upsert_chapter(conn, id="c0", arc_id="a0", ordering=0,
                          description="ch", status="active")
        db.upsert_beat(conn, id="b0", chapter_id="c0", ordering=0,
                       status="active")
        db.upsert_character(conn, id="char0", project_id="p1", name="Mara")
        db.upsert_character_emotions(conn, character_id="char0",
                                     pleasure=0.0, arousal=0.0, dominance=0.0)
        db.upsert_thread(conn, id="t0", project_id="p1", description="thread",
                         status="open", priority_score=0.5)
    return conn


def test_pending_intent_with_matching_event_recovers(config_factory):
    config = config_factory()
    conn = _build_graph(config)
    with conn:
        db.create_commit_intent(conn, beat_id="b0", arc_id="a0",
                                chapter_id="c0", beat_index=0)
    conn.close()

    append_event(config.event_log_path, {
        "type": "beat_commit",
        "beat_id": "b0",
        "fsm_pointer": {"arc_id": "a0", "chapter_id": "c0", "beat_index": 0},
        "prose_delta": "The lamp turned in the fog.",
        "thread_updates": [{"id": "t0", "status": "progressing",
                            "priority_score": 0.7}],
        "pad_states": [{"character_id": "char0", "pleasure": 0.2,
                        "arousal": -0.1, "dominance": 0.3}],
        "word_count": 6,
    })

    summary = scan_and_recover(config.db_path, config.event_log_path)
    assert summary["pending_found"] == 1
    assert summary["recovered"] == ["b0"]
    assert summary["cleared"] == []

    conn = db.connect_db(config.db_path)
    beat = db.get_beats_for_chapter(conn, "c0")[0]
    assert beat["status"] == "completed"
    assert beat["prose"] == "The lamp turned in the fog."
    assert beat["word_count"] == 6

    intent = conn.execute("SELECT * FROM CommitIntent").fetchone()
    assert intent["status"] == "committed"

    thread = conn.execute("SELECT * FROM Threads WHERE id='t0'").fetchone()
    assert thread["status"] == "progressing"
    assert thread["priority_score"] == 0.7

    emo = db.get_character_emotions(conn, "char0")
    assert emo["pleasure"] == 0.2
    conn.close()


def test_pending_intent_without_event_is_cleared(config_factory):
    config = config_factory()
    conn = _build_graph(config)
    with conn:
        db.create_commit_intent(conn, beat_id="b0", arc_id="a0",
                                chapter_id="c0", beat_index=0)
    conn.close()

    # Event log exists but has no matching beat_commit.
    append_event(config.event_log_path, {"type": "note", "text": "unrelated"})

    summary = scan_and_recover(config.db_path, config.event_log_path)
    assert summary["pending_found"] == 1
    assert summary["recovered"] == []
    assert summary["cleared"] == ["b0"]

    conn = db.connect_db(config.db_path)
    beat = db.get_beats_for_chapter(conn, "c0")[0]
    assert beat["status"] == "planned"  # left for the FSM to re-draft
    remaining = conn.execute("SELECT COUNT(*) AS n FROM CommitIntent").fetchone()
    assert remaining["n"] == 0
    conn.close()


def test_clean_db_is_no_op(config_factory):
    config = config_factory()
    db.init_db(config.db_path)
    summary = scan_and_recover(config.db_path, config.event_log_path)
    assert summary == {"pending_found": 0, "recovered": [], "cleared": []}
