"""Canonical database records, by scope, optionally filtered by id.

The generic read over the project's source of truth. Beats deliberately come
back as *metadata* — spec fields, status, word count — never their prose:
whole-prose dumps flood the context, and exact wording is what
``search_manuscript`` is for.
"""

from __future__ import annotations

import json
from typing import Any

from museai.fsm.pad import PAD_AXES
from museai.fsm.tools.project_db import project_connection
from museai.memory.db import get_character_emotions

_SCOPES = ("arcs", "chapters", "beats", "threads", "characters")

GET_CANONICAL_STATE_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_canonical_state",
        "description": (
            "Canonical records for one scope of this project: arcs, chapters, "
            "beats (metadata only, never prose), threads, or characters (with "
            "current emotional state). Optionally filter to specific ids. Use "
            "it to check the recorded truth before asserting it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": list(_SCOPES),
                    "description": "Which records to fetch.",
                },
                "ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: only these ids.",
                },
            },
            "required": ["scope"],
        },
    },
}


def _spec(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _obligations(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return [raw]
    return [str(item) for item in parsed] if isinstance(parsed, list) else [str(parsed)]


def _rows(conn, scope: str, project_id: str) -> list:
    if scope == "arcs":
        return conn.execute(
            "SELECT * FROM Arcs WHERE project_id=? ORDER BY ordering", (project_id,)
        ).fetchall()
    if scope == "chapters":
        return conn.execute(
            """
            SELECT Chapters.* FROM Chapters
            JOIN Arcs ON Chapters.arc_id = Arcs.id
            WHERE Arcs.project_id = ?
            ORDER BY Arcs.ordering, Chapters.ordering
            """,
            (project_id,),
        ).fetchall()
    if scope == "beats":
        return conn.execute(
            """
            SELECT Beats.* FROM Beats
            JOIN Chapters ON Beats.chapter_id = Chapters.id
            JOIN Arcs ON Chapters.arc_id = Arcs.id
            WHERE Arcs.project_id = ?
            ORDER BY Arcs.ordering, Chapters.ordering, Beats.ordering
            """,
            (project_id,),
        ).fetchall()
    if scope == "threads":
        return conn.execute(
            "SELECT * FROM Threads WHERE project_id=? ORDER BY id", (project_id,)
        ).fetchall()
    return conn.execute(
        "SELECT * FROM Characters WHERE project_id=? ORDER BY name", (project_id,)
    ).fetchall()


def _record(conn, scope: str, row) -> dict:
    if scope == "arcs":
        return {"id": row["id"], "ordering": row["ordering"],
                "description": row["description"], "status": row["status"]}
    if scope == "chapters":
        return {"id": row["id"], "arc_id": row["arc_id"],
                "ordering": row["ordering"], "description": row["description"],
                "obligations": _obligations(row["obligations"]),
                "status": row["status"]}
    if scope == "beats":
        spec = _spec(row["beat_spec"])
        return {"id": row["id"], "chapter_id": row["chapter_id"],
                "ordering": row["ordering"],
                "intent": str(spec.get("intent") or ""),
                "exit_state": str(spec.get("exit_state") or ""),
                "status": row["status"], "word_count": row["word_count"]}
    if scope == "threads":
        return {"id": row["id"], "description": row["description"],
                "status": row["status"], "priority_score": row["priority_score"]}
    emotions = get_character_emotions(conn, row["id"])
    return {
        "id": row["id"], "name": row["name"],
        "description": row["description"] or "",
        "current_pad": {
            axis: (emotions[axis] if emotions is not None else 0.0)
            for axis in PAD_AXES
        },
    }


def get_canonical_state(scope: str, ids: list[str] | None = None) -> dict:
    """The scope's records, filtered to ``ids`` when given."""
    wanted_scope = (scope or "").strip().lower()
    if wanted_scope not in _SCOPES:
        return {
            "error": f"unknown scope {scope!r}",
            "supported_scopes": list(_SCOPES),
        }

    with project_connection() as (conn, project_id):
        rows = _rows(conn, wanted_scope, project_id)
        records = [_record(conn, wanted_scope, row) for row in rows]

    if ids:
        wanted_ids = {str(i) for i in ids}
        found = [record for record in records if record["id"] in wanted_ids]
        missing = sorted(wanted_ids - {record["id"] for record in found})
        result = {"scope": wanted_scope, "records": found}
        if missing:
            result["missing_ids"] = missing
        return result
    return {"scope": wanted_scope, "records": records}
