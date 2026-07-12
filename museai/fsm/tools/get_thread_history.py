"""One narrative thread's status and every committed beat that advanced it.

A thread advance is a fact the planner declared: each beat's stored
``beat_spec`` carries the ``thread_updates`` the commit node applied. Reading
those back is exact — no inference about what a beat "probably" advanced.
"""

from __future__ import annotations

import json
from typing import Any

from museai.fsm.tools.project_db import project_connection

GET_THREAD_HISTORY_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_thread_history",
        "description": (
            "One narrative thread's description and current status, plus every "
            "committed beat that advanced it and the status each set. Use it "
            "before planning a chapter that pays a thread off, to see how far "
            "the story has actually carried it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "thread_id": {
                    "type": "string",
                    "description": "The thread's id, exactly as it appears in your context.",
                },
            },
            "required": ["thread_id"],
        },
    },
}


def _declared_updates(spec_text: str | None, thread_id: str) -> list[dict]:
    """The updates this beat's spec declared for ``thread_id``, if any."""
    if not spec_text:
        return []
    try:
        spec = json.loads(spec_text)
    except json.JSONDecodeError:
        return []
    updates = spec.get("thread_updates") or []
    return [
        {"intent": str(spec.get("intent") or ""), "status_set": update.get("status")}
        for update in updates
        if isinstance(update, dict) and update.get("id") == thread_id
    ]


def get_thread_history(thread_id: str) -> dict:
    """The thread row plus its committed advances, in narrative order."""
    wanted = (thread_id or "").strip()
    with project_connection() as (conn, project_id):
        thread = conn.execute(
            "SELECT * FROM Threads WHERE id=? AND project_id=?",
            (wanted, project_id),
        ).fetchone()
        if thread is None:
            known = [
                row["id"]
                for row in conn.execute(
                    "SELECT id FROM Threads WHERE project_id=? ORDER BY id",
                    (project_id,),
                ).fetchall()
            ]
            return {
                "error": f"no thread {wanted!r} in this project",
                "known_threads": known,
            }

        rows = conn.execute(
            """
            SELECT Beats.id AS beat_id, Beats.chapter_id, Beats.beat_spec
            FROM Beats
            JOIN Chapters ON Beats.chapter_id = Chapters.id
            JOIN Arcs ON Chapters.arc_id = Arcs.id
            WHERE Arcs.project_id = ?
              AND Beats.status = 'completed'
              AND Beats.beat_spec IS NOT NULL
            ORDER BY Arcs.ordering ASC, Chapters.ordering ASC, Beats.ordering ASC
            """,
            (project_id,),
        ).fetchall()

    advanced_by = [
        {
            "beat_id": row["beat_id"],
            "chapter_id": row["chapter_id"],
            "intent": update["intent"],
            "status_set": update["status_set"],
        }
        for row in rows
        for update in _declared_updates(row["beat_spec"], wanted)
    ]
    return {
        "thread": {
            "id": thread["id"],
            "description": thread["description"],
            "status": thread["status"],
            "priority_score": thread["priority_score"],
        },
        "advanced_by": advanced_by,
    }
