"""Module: M10 (Commit, Consolidation & Crash Immunity)

Persist the current beat's committed prose and advance the FSM pointer.

Minimal-but-real implementation of the commit boundary: it writes the drafted
prose to the canonical ``Beats`` row via the idempotent ``upsert_beat_commit``
(SQLite) and appends a ``beat_commit`` record to the append-only event log, then
advances ``fsm_pointer`` to the next planned beat. The prose is carried forward as
``last_committed_prose`` so the next draft continues seamlessly.

Both writes are replay-safe: ``upsert_beat_commit`` keys by ``beat_id`` and the
event log is append-only, so re-running a committed beat neither duplicates nor
corrupts state. The Graphiti temporal-graph write and RAPTOR consolidation are
part of the full commit path (M09/M10) and are intentionally skipped here — they
degrade cleanly to "not yet wired" for the local end-to-end path.

Returns a state *delta*; the driver merges it. ``generation_complete`` is set true
when the just-committed beat was the last planned beat.
"""

from __future__ import annotations

from typing import Any

import core.runtime as runtime
from memory import sqlite_db
from memory.event_log import write_event


def _word_count(text: str) -> int:
    """Whitespace word count — the canonical Beats.word_count sizing."""
    return len(text.split())


async def node_commit_transaction(state: dict[str, Any]) -> dict[str, Any]:
    """Commit the current beat's prose, advance the pointer, and return a delta."""
    db_path = state.get("sqlite_db_path", runtime.SQLITE_DB_PATH)
    event_log_path = state.get("event_log_path", runtime.EVENT_LOG_PATH)
    pointer = state["fsm_pointer"]
    beat_id = getattr(pointer, "beat_id", "") or ""
    prose = state.get("current_draft_text", "") or ""

    beat_plan = (state.get("beat_plan_by_id") or {}).get(beat_id, {})
    scene_id = beat_plan.get("scene_id") or getattr(pointer, "scene_id", "") or ""
    beat_index = int(beat_plan.get("beat_index", getattr(pointer, "beat_index", 0)))
    word_count = _word_count(prose)

    # 1. Canonical relational write (idempotent by beat_id).
    sqlite_db.upsert_beat_commit(
        db_path,
        beat_id=beat_id,
        scene_id=scene_id,
        beat_index=beat_index,
        prose=prose,
        word_count=word_count,
        status="completed",
    )

    # 2. Append-only audit/replay record.
    write_event(
        event_log_path,
        {
            "event": "beat_commit",
            "project_id": state.get("project_id", ""),
            "beat_id": beat_id,
            "scene_id": scene_id,
            "beat_index": beat_index,
            "word_count": word_count,
            "prose": prose,
        },
    )

    # 3. Advance the pointer to the next planned beat (if any).
    beat_order: list[str] = list(state.get("beat_order") or [])
    delta: dict[str, Any] = {
        "current_draft_text": "",
        "last_committed_prose": prose,
        "committed_word_count": state.get("committed_word_count", 0) + word_count,
    }
    try:
        position = beat_order.index(beat_id)
    except ValueError:
        position = -1

    if position == -1 or position + 1 >= len(beat_order):
        delta["generation_complete"] = True
    else:
        next_beat_id = beat_order[position + 1]
        next_plan = (state.get("beat_plan_by_id") or {}).get(next_beat_id, {})
        delta["fsm_pointer"] = pointer.model_copy(
            update={
                "beat_id": next_beat_id,
                "scene_id": next_plan.get("scene_id", scene_id),
                "beat_index": int(next_plan.get("beat_index", 0)),
            }
        )
        delta["generation_complete"] = False

    return delta
