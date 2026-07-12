"""The chapter planner's map of what is actually planned and dramatized.

Returns every arc and chapter of the active project with status, description,
obligations, and beat progress, so a planner writing new obligations builds on
what the outline already owns instead of re-covering it.
"""

from __future__ import annotations

import json
from typing import Any

from museai.fsm.tools.project_db import project_connection
from museai.memory.db import get_arcs, get_beats_for_chapter, get_chapters_for_arc

GET_FULL_OUTLINE_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_full_outline",
        "description": (
            "The complete outline of this project: every arc and chapter with "
            "its status, description, obligations, and how many of its beats "
            "are already written. Use it to see what the story has planned and "
            "delivered before adding chapters that would repeat it."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


def _obligations(raw: str | None) -> list[str]:
    """The obligations column, which ``plan_chapter`` writes as a JSON list."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return [raw]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def get_full_outline() -> list[dict]:
    """Arcs in order, each with its chapters in order. ``[]`` before planning."""
    with project_connection() as (conn, project_id):
        outline: list[dict] = []
        for arc in get_arcs(conn, project_id):
            chapters = []
            for chapter in get_chapters_for_arc(conn, arc["id"]):
                beats = get_beats_for_chapter(conn, chapter["id"])
                chapters.append(
                    {
                        "id": chapter["id"],
                        "ordering": chapter["ordering"],
                        "description": chapter["description"],
                        "status": chapter["status"],
                        "obligations": _obligations(chapter["obligations"]),
                        "beats_completed": sum(
                            1 for beat in beats if beat["status"] == "completed"
                        ),
                        "beats_planned": len(beats),
                    }
                )
            outline.append(
                {
                    "id": arc["id"],
                    "ordering": arc["ordering"],
                    "description": arc["description"],
                    "status": arc["status"],
                    "chapters": chapters,
                }
            )
    return outline
