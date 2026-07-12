"""One chapter's structural context: what it owes, holds, and has delivered.

Deliberately *not* the chapter's full prose — dumping whole chapters into a
model's context floods it and feeds the style-echo loop. A beat planner needing
a callback gets the chapter's obligations and each beat's intent and exit
state; a specific line is found with ``search_manuscript``.
"""

from __future__ import annotations

import json
from typing import Any

from museai.fsm.tools.project_db import project_connection
from museai.memory.db import get_beats_for_chapter

GET_CHAPTER_CONTEXT_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_chapter_context",
        "description": (
            "One chapter's structural context: description, obligations, "
            "status, and each of its beats with intent, exit state, status, "
            "and word count. Use it when a beat builds on an earlier chapter "
            "beyond the recent prose in your context. For exact wording, use "
            "search_manuscript instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chapter_id": {
                    "type": "string",
                    "description": "The chapter's id, e.g. from the outline.",
                },
            },
            "required": ["chapter_id"],
        },
    },
}


def _obligations(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return [raw]
    return [str(item) for item in parsed] if isinstance(parsed, list) else [str(parsed)]


def _spec_field(spec_text: str | None, field: str) -> str:
    if not spec_text:
        return ""
    try:
        spec = json.loads(spec_text)
    except json.JSONDecodeError:
        return ""
    return str(spec.get(field) or "")


def get_chapter_context(chapter_id: str) -> dict:
    """The chapter's structure, or an error naming the real chapter ids."""
    wanted = (chapter_id or "").strip()
    with project_connection() as (conn, project_id):
        chapter = conn.execute(
            """
            SELECT Chapters.*
            FROM Chapters
            JOIN Arcs ON Chapters.arc_id = Arcs.id
            WHERE Chapters.id = ? AND Arcs.project_id = ?
            """,
            (wanted, project_id),
        ).fetchone()
        if chapter is None:
            known = [
                row["id"]
                for row in conn.execute(
                    """
                    SELECT Chapters.id
                    FROM Chapters
                    JOIN Arcs ON Chapters.arc_id = Arcs.id
                    WHERE Arcs.project_id = ?
                    ORDER BY Arcs.ordering ASC, Chapters.ordering ASC
                    """,
                    (project_id,),
                ).fetchall()
            ]
            return {
                "error": f"no chapter {wanted!r} in this project",
                "known_chapters": known,
            }
        beats = get_beats_for_chapter(conn, wanted)

    return {
        "chapter_id": chapter["id"],
        "description": chapter["description"],
        "status": chapter["status"],
        "obligations": _obligations(chapter["obligations"]),
        "beats": [
            {
                "beat_id": row["id"],
                "ordering": row["ordering"],
                "intent": _spec_field(row["beat_spec"], "intent"),
                "exit_state": _spec_field(row["beat_spec"], "exit_state"),
                "status": row["status"],
                "word_count": row["word_count"],
            }
            for row in beats
        ],
        "committed_word_count": sum(
            row["word_count"] for row in beats if row["status"] == "completed"
        ),
    }
