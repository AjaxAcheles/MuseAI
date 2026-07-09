"""Crash reconciliation for MuseAI v1.

A beat commit is guarded by a ``CommitIntent`` row written *before* the writes
and flipped to ``committed`` *after* the ``beat_commit`` event is durably
appended. If the process dies in between, a ``pending`` intent is left behind.

``scan_and_recover`` resolves every pending intent:

* If a matching ``beat_commit`` event exists in the log, the commit's writes are
  re-applied idempotently and the intent is flipped to ``committed``.
* If no matching event exists, the commit never completed: the pending intent is
  deleted and the beat is left ``planned`` so the FSM re-drafts it.

Safe on a clean DB — no pending rows means an empty summary, which is a real
result, not a placeholder.
"""

from __future__ import annotations

from pathlib import Path

from museai.core.events import replay_events
from museai.core.logging_setup import get_fsm_logger
from museai.memory.db import (
    connect_db,
    delete_commit_intent,
    get_pending_intents,
    mark_commit_committed,
    set_beat_status,
    upsert_beat,
    upsert_character_emotions,
)


def _index_beat_commits(event_log_path: str | Path) -> dict[str, dict]:
    """Map beat_id -> its most recent ``beat_commit`` event payload."""
    index: dict[str, dict] = {}
    for event in replay_events(event_log_path):
        if event.get("type") == "beat_commit" and event.get("beat_id"):
            index[event["beat_id"]] = event
    return index


def _apply_beat_commit(conn, event: dict) -> None:
    """Re-apply a beat_commit event's writes idempotently."""
    pointer = event.get("fsm_pointer") or {}
    existing = conn.execute(
        "SELECT * FROM Beats WHERE id=?", (event["beat_id"],)
    ).fetchone()
    chapter_id = existing["chapter_id"] if existing is not None else pointer.get("chapter_id")
    if chapter_id is None:
        raise ValueError(f"beat_commit event for {event['beat_id']!r} has no chapter_id")
    upsert_beat(
        conn,
        id=event["beat_id"],
        chapter_id=str(chapter_id),
        ordering=(
            existing["ordering"]
            if existing is not None
            else int(pointer.get("beat_index", 0)) + 1
        ),
        beat_spec=(existing["beat_spec"] if existing is not None else None),
        pad_constraint=(existing["pad_constraint"] if existing is not None else None),
        word_target=(existing["word_target"] if existing is not None else None),
        prose=event.get("prose_delta"),
        word_count=event.get("word_count", 0),
        status="completed",
    )

    for pad in event.get("pad_states", []) or []:
        upsert_character_emotions(
            conn,
            character_id=pad["character_id"],
            pleasure=pad["pleasure"],
            arousal=pad["arousal"],
            dominance=pad["dominance"],
            updated_at=pad.get("updated_at"),
        )

    for update in event.get("thread_updates", []) or []:
        fields = {
            key: update[key]
            for key in ("status", "priority_score", "description")
            if key in update
        }
        if not fields:
            continue
        assignments = ", ".join(f"{key}=?" for key in fields)
        conn.execute(
            f"UPDATE Threads SET {assignments} WHERE id=?",
            (*fields.values(), update["id"]),
        )


def scan_and_recover(db_path: str | Path, event_log_path: str | Path) -> dict:
    """Resolve every pending CommitIntent. Returns a summary dict."""
    summary = {"pending_found": 0, "recovered": [], "cleared": []}
    logger = get_fsm_logger()

    conn = connect_db(db_path)
    try:
        pending = get_pending_intents(conn)
        summary["pending_found"] = len(pending)
        if not pending:
            logger.info("recovery_scan clean pending=0")
            return summary

        logger.info("recovery_scan pending=%d", len(pending))
        commits = _index_beat_commits(event_log_path)

        with conn:
            for intent in pending:
                beat_id = intent["beat_id"]
                event = commits.get(beat_id)
                if event is not None:
                    _apply_beat_commit(conn, event)
                    mark_commit_committed(conn, intent["id"])
                    summary["recovered"].append(beat_id)
                    logger.info(
                        "recovery_recovered beat_id=%s intent_id=%s word_count=%s",
                        beat_id,
                        intent["id"],
                        event.get("word_count"),
                    )
                else:
                    delete_commit_intent(conn, intent["id"])
                    if beat_id is not None:
                        set_beat_status(conn, beat_id, "planned")
                    summary["cleared"].append(beat_id)
                    logger.warning(
                        "recovery_cleared beat_id=%s intent_id=%s "
                        "no beat_commit event; beat reset to planned",
                        beat_id,
                        intent["id"],
                    )
    finally:
        conn.close()

    logger.info(
        "recovery_complete pending=%d recovered=%d cleared=%d",
        summary["pending_found"],
        len(summary["recovered"]),
        len(summary["cleared"]),
    )
    return summary
