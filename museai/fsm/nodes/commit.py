"""Crash-safe beat commit node.

The commit boundary is deliberately narrow for MuseAI v1: SQLite plus the
append-only event log. A ``CommitIntent`` row is written before any commit writes,
the beat/chapter/arc/thread/PAD rows are updated idempotently, the durable
``beat_commit`` event is appended, and only then is the intent flipped to
``committed``. If the process dies after the event append but before the flip,
``museai.memory.reconcile.scan_and_recover`` can replay the event and complete
the intent.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

from museai.core.events import append_event
from museai.core.logging_setup import log_node_event
from museai.core.stream_bus import bus
from museai.fsm.nodes.deps import get_node_config
from museai.fsm.pad import PAD_AXES
from museai.fsm.state import UNFULFILLED_OBLIGATION, OrchestratorState
from museai.memory.db import (
    connect_db,
    create_commit_intent,
    get_arcs,
    get_beats_for_chapter,
    get_chapters_for_arc,
    get_project,
    mark_commit_committed,
    upsert_arc,
    upsert_beat,
    upsert_chapter,
    upsert_character_emotions,
    upsert_thread,
)

PHASE = "Committing"

_WORD = re.compile(r"\S+")
_THREAD_ORDER = {"open": 0, "progressing": 1, "closed": 2}


class CommitError(RuntimeError):
    """The active beat could not be committed from SQLite ground truth."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _word_count(text: str) -> int:
    return len(_WORD.findall(text))


def _pointer_payload(state: OrchestratorState) -> dict[str, Any]:
    pointer = state["fsm_pointer"]
    return {
        "arc_id": pointer.arc_id,
        "chapter_id": pointer.chapter_id,
        "beat_index": pointer.beat_index,
    }


def _resolve_chapter(conn: sqlite3.Connection, pointer) -> sqlite3.Row:
    chapters = get_chapters_for_arc(conn, pointer.arc_id)
    for row in chapters:
        if row["id"] == pointer.chapter_id:
            return row
    for row in chapters:
        if row["status"] == "active":
            return row
    raise CommitError(
        f"arc {pointer.arc_id!r} has no chapter {pointer.chapter_id!r} and no active chapter"
    )


def _resolve_beat(
    conn: sqlite3.Connection, chapter_id: str, beat_index: int
) -> sqlite3.Row:
    ordering = beat_index + 1
    beats = get_beats_for_chapter(conn, chapter_id)
    for row in beats:
        if row["ordering"] == ordering:
            return row
    for row in beats:
        if row["status"] == "active":
            return row
    raise CommitError(
        f"chapter {chapter_id!r} has no beat at index {beat_index} and no active beat"
    )


def _beat_spec(beat: sqlite3.Row) -> dict[str, Any]:
    if not beat["beat_spec"]:
        return {}
    try:
        parsed = json.loads(beat["beat_spec"])
    except json.JSONDecodeError as exc:
        raise CommitError(f"beat {beat['id']!r} has invalid beat_spec JSON") from exc
    if not isinstance(parsed, dict):
        raise CommitError(f"beat {beat['id']!r} beat_spec must be a JSON object")
    return parsed


def _pad_states(spec: dict[str, Any], committed_at: str) -> list[dict[str, Any]]:
    focal_character_id = str(spec.get("focal_character_id") or "").strip()
    target_pad = spec.get("target_pad") or {}
    if not focal_character_id:
        return []
    if not isinstance(target_pad, dict):
        raise CommitError("beat target_pad must be a JSON object")

    pad: dict[str, float] = {}
    for axis in PAD_AXES:
        try:
            value = float(target_pad[axis])
        except (KeyError, TypeError, ValueError) as exc:
            raise CommitError(f"beat target_pad.{axis} is required for PAD commit") from exc
        pad[axis] = max(-1.0, min(1.0, value))

    return [{"character_id": focal_character_id, **pad, "updated_at": committed_at}]


def _requested_thread_updates(spec: dict[str, Any]) -> list[dict[str, Any]]:
    raw = spec.get("thread_updates", spec.get("threads", []))
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise CommitError("beat thread_updates must be a JSON array when present")
    updates: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise CommitError("each beat thread update must be a JSON object")
        thread_id = str(item.get("id") or item.get("thread_id") or "").strip()
        status = str(item.get("status") or "").strip()
        if not thread_id or status not in _THREAD_ORDER:
            continue
        update = {"id": thread_id, "status": status}
        if "priority_score" in item:
            update["priority_score"] = item["priority_score"]
        if "description" in item:
            update["description"] = str(item["description"])
        updates.append(update)
    return updates


def _apply_thread_updates(
    conn: sqlite3.Connection, project_id: str, spec: dict[str, Any]
) -> list[dict[str, Any]]:
    applied: list[dict[str, Any]] = []
    for requested in _requested_thread_updates(spec):
        current = conn.execute(
            "SELECT * FROM Threads WHERE id=? AND project_id=?",
            (requested["id"], project_id),
        ).fetchone()
        if current is None:
            continue

        current_rank = _THREAD_ORDER[current["status"]]
        requested_rank = _THREAD_ORDER[requested["status"]]
        if requested_rank < current_rank:
            continue

        priority = requested.get("priority_score", current["priority_score"])
        description = requested.get("description", current["description"])
        upsert_thread(
            conn,
            id=current["id"],
            project_id=project_id,
            description=description,
            status=requested["status"],
            priority_score=priority,
        )
        applied.append(
            {
                "id": current["id"],
                "status": requested["status"],
                "priority_score": priority,
            }
        )
    return applied


def _chapter_status_after_commit(conn: sqlite3.Connection, chapter_id: str) -> str:
    remaining = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM Beats
        WHERE chapter_id=? AND status != 'completed'
        """,
        (chapter_id,),
    ).fetchone()["n"]
    return "completed" if remaining == 0 else "active"


def _arc_status_after_commit(conn: sqlite3.Connection, arc_id: str) -> str:
    unfinished_chapters = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM Chapters
        WHERE arc_id=? AND status != 'completed'
        """,
        (arc_id,),
    ).fetchone()["n"]
    unfinished_beats = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM Beats
        JOIN Chapters ON Beats.chapter_id = Chapters.id
        WHERE Chapters.arc_id=? AND Beats.status != 'completed'
        """,
        (arc_id,),
    ).fetchone()["n"]
    return "completed" if unfinished_chapters == 0 and unfinished_beats == 0 else "active"


def _project_total_words(conn: sqlite3.Connection, project_id: str) -> int:
    row = conn.execute(
        """
        SELECT COALESCE(SUM(Beats.word_count), 0) AS total
        FROM Beats
        JOIN Chapters ON Beats.chapter_id = Chapters.id
        JOIN Arcs ON Chapters.arc_id = Arcs.id
        WHERE Arcs.project_id=? AND Beats.status='completed'
        """,
        (project_id,),
    ).fetchone()
    return int(row["total"] or 0)


async def commit_transaction(state: OrchestratorState) -> dict:
    """Commit the current draft at the active beat boundary.

    Story-state advancement is gated on the beat having earned it. The prose
    itself always commits — a multi-hour run must not die at the boundary —
    but the beat's planned ``thread_updates`` are applied only when the critic
    neither reported an unfulfilled obligation against this draft nor was
    itself unreliable (its output unreadable after every retry). A withheld
    advance is recorded in the durable event, so canon never claims a change
    the prose did not deliver.
    """
    config = get_node_config()
    pointer = state["fsm_pointer"]
    project_id = state["project_id"]
    prose = state["current_draft_text"]
    count = _word_count(prose)
    committed_at = _utc_now()

    unfulfilled = [
        failure
        for failure in state["critic_failures"]
        if failure.error_code == UNFULFILLED_OBLIGATION
    ]
    critic_unreliable = state["critic_parse_failure_streak"] > 0
    withheld_reason = ""
    if unfulfilled:
        withheld_reason = "unfulfilled_obligation"
    elif critic_unreliable:
        withheld_reason = "critic_unreliable"

    log_node_event(
        "commit_transaction",
        event="start",
        arc_id=pointer.arc_id,
        chapter_id=pointer.chapter_id,
        beat_index=pointer.beat_index,
    )

    conn = connect_db(config.db_path)
    try:
        project = get_project(conn, project_id)
        if project is None:
            raise CommitError(f"project {project_id!r} is not in the database")

        chapter = _resolve_chapter(conn, pointer)
        beat = _resolve_beat(conn, chapter["id"], pointer.beat_index)
        spec = _beat_spec(beat)

        with conn:
            intent_id = create_commit_intent(
                conn,
                beat_id=beat["id"],
                arc_id=pointer.arc_id,
                chapter_id=chapter["id"],
                beat_index=pointer.beat_index,
                initiated_at=committed_at,
            )

        pad_states = _pad_states(spec, committed_at)
        with conn:
            upsert_beat(
                conn,
                id=beat["id"],
                chapter_id=beat["chapter_id"],
                ordering=beat["ordering"],
                beat_spec=beat["beat_spec"],
                pad_constraint=beat["pad_constraint"],
                prose=prose,
                word_count=count,
                status="completed",
            )
            for pad in pad_states:
                upsert_character_emotions(conn, **pad)
            if withheld_reason:
                thread_updates = []
                withheld_updates = _requested_thread_updates(spec)
            else:
                thread_updates = _apply_thread_updates(conn, project_id, spec)
                withheld_updates = []

            chapter_status = _chapter_status_after_commit(conn, chapter["id"])
            upsert_chapter(
                conn,
                id=chapter["id"],
                arc_id=chapter["arc_id"],
                ordering=chapter["ordering"],
                description=chapter["description"],
                obligations=chapter["obligations"],
                status=chapter_status,
            )
            arc = next(row for row in get_arcs(conn, project_id) if row["id"] == pointer.arc_id)
            arc_status = _arc_status_after_commit(conn, pointer.arc_id)
            upsert_arc(
                conn,
                id=arc["id"],
                project_id=arc["project_id"],
                ordering=arc["ordering"],
                description=arc["description"],
                status=arc_status,
            )

        event = {
            "type": "beat_commit",
            "beat_id": beat["id"],
            "fsm_pointer": _pointer_payload(state),
            "prose_delta": prose,
            "thread_updates": thread_updates,
            "pad_states": pad_states,
            "word_count": count,
        }
        if withheld_reason:
            # The advance the plan requested but the prose did not earn. Kept
            # out of "thread_updates" so a reconcile replay stays honest.
            event["thread_updates_withheld"] = {
                "reason": withheld_reason,
                "requested": withheld_updates,
                "unfulfilled_findings": [f.model_dump() for f in unfulfilled],
            }
        append_event(config.event_log_path, event)

        with conn:
            mark_commit_committed(conn, intent_id, completed_at=_utc_now())

        total_words = _project_total_words(conn, project_id)
    finally:
        conn.close()

    log_node_event(
        "commit_transaction",
        event="committed",
        beat_id=beat["id"],
        intent_id=intent_id,
        word_count=count,
        project_total=total_words,
    )
    if withheld_reason:
        # WARNING: the prose is committed but the story state did not advance.
        # A run where this repeats is producing prose that misses its plan.
        log_node_event(
            "commit_transaction",
            level=logging.WARNING,
            event="thread_updates_withheld",
            beat_id=beat["id"],
            reason=withheld_reason,
            requested=len(withheld_updates),
            unfulfilled=len(unfulfilled),
        )
        await bus.publish(
            "commit_gate",
            {
                "beat_id": beat["id"],
                "reason": withheld_reason,
                "requested": withheld_updates,
                "unfulfilled_findings": [f.model_dump() for f in unfulfilled],
            },
        )
    log_node_event("commit_transaction", event="phase_change", phase=PHASE)
    await bus.publish(
        "phase_change",
        {"phase": PHASE, "node": "commit_transaction", "project_id": project_id},
    )
    await bus.publish(
        "word_count",
        {
            "project_id": project_id,
            "word_count": total_words,
            "target": project["word_count_target"],
        },
    )
    await bus.publish(
        "pointer_update",
        {"project_id": project_id, "beat_id": beat["id"], "fsm_pointer": _pointer_payload(state)},
    )

    return {
        "retry_count": 0,
        "critic_failures": [],
        "best_seen_draft": None,
        "best_seen_failure_count": None,
        # The next beat has no revise history: a baseline carried over from this
        # one would make its first retry compare against the wrong number.
        "pre_revise_failure_count": None,
        "last_cycle_improved": True,
        "current_draft_text": "",
        "streaming_buffer": "",
        "review_requested": False,
    }
