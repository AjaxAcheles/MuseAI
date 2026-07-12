"""Where the run stands right now: the active arc, chapter, and beat.

Read from the canonical ``status='active'`` rows, which the FSM maintains as it
advances — not from a cached pointer — so what this returns is exactly what the
graph will draft next.
"""

from __future__ import annotations

import json
from typing import Any

from museai.fsm.tools.project_db import project_connection

GET_CURRENT_POINTER_CONTEXT_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_current_pointer_context",
        "description": (
            "Where the story stands right now: the active arc, the active "
            "chapter with its obligations, and the active beat with its "
            "intent, entry and exit states, and emotional constraint. Use it "
            "to orient before planning or writing."
        ),
        "parameters": {"type": "object", "properties": {}},
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


def _spec(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def get_current_pointer_context() -> dict:
    """The active arc/chapter/beat rows, each ``None`` when nothing is active."""
    with project_connection() as (conn, project_id):
        arc = conn.execute(
            """
            SELECT * FROM Arcs
            WHERE project_id = ? AND status = 'active'
            ORDER BY ordering ASC LIMIT 1
            """,
            (project_id,),
        ).fetchone()

        chapter = None
        if arc is not None:
            chapter = conn.execute(
                """
                SELECT * FROM Chapters
                WHERE arc_id = ? AND status = 'active'
                ORDER BY ordering ASC LIMIT 1
                """,
                (arc["id"],),
            ).fetchone()

        beat = None
        if chapter is not None:
            beat = conn.execute(
                """
                SELECT * FROM Beats
                WHERE chapter_id = ? AND status = 'active'
                ORDER BY ordering ASC LIMIT 1
                """,
                (chapter["id"],),
            ).fetchone()

    spec = _spec(beat["beat_spec"]) if beat is not None else {}
    return {
        "arc": None if arc is None else {
            "id": arc["id"],
            "ordering": arc["ordering"],
            "description": arc["description"],
        },
        "chapter": None if chapter is None else {
            "id": chapter["id"],
            "ordering": chapter["ordering"],
            "description": chapter["description"],
            "obligations": _obligations(chapter["obligations"]),
        },
        "beat": None if beat is None else {
            "id": beat["id"],
            "ordering": beat["ordering"],
            "intent": str(spec.get("intent") or ""),
            "entry_state": str(spec.get("entry_state") or ""),
            "exit_state": str(spec.get("exit_state") or ""),
            "target_pad": spec.get("target_pad") or {},
            "focal_character_id": str(spec.get("focal_character_id") or ""),
            "pad_constraint": beat["pad_constraint"] or "",
        },
    }
