"""A character's PAD trajectory across the committed manuscript.

Each committed beat's stored ``beat_spec`` names a focal character and the
``target_pad`` the planner set for them, so the trajectory is read straight
back from the plans that were actually dramatized. The beat planner uses it to
shape a varied intensity arc against real data instead of guessing — the guess
is what used to trigger ``intensity_reprompt``.
"""

from __future__ import annotations

import json
from typing import Any

from museai.fsm.pad import PAD_AXES
from museai.fsm.tools.project_db import project_connection
from museai.memory.db import get_character_emotions, get_characters

GET_CHARACTER_EMOTION_HISTORY_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_character_emotion_history",
        "description": (
            "One character's emotional trajectory: their current "
            "pleasure/arousal/dominance state and the target PAD of every "
            "committed beat that focused on them, in story order. Use it to "
            "plan intensity that varies against where the character has "
            "actually been."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "character_id": {
                    "type": "string",
                    "description": "The character's id (their name also resolves).",
                },
            },
            "required": ["character_id"],
        },
    },
}


def _resolve(rows: list, wanted: str):
    """The character row ``wanted`` names, matching id first, then name."""
    lowered = wanted.casefold()
    for row in rows:
        if row["id"] == wanted or row["id"].casefold() == lowered:
            return row
    for row in rows:
        if row["name"].strip().casefold() == lowered:
            return row
    return None


def get_character_emotion_history(character_id: str) -> dict:
    """Current PAD plus the committed focal-beat trajectory, oldest first."""
    wanted = (character_id or "").strip()
    with project_connection() as (conn, project_id):
        characters = get_characters(conn, project_id)
        character = _resolve(characters, wanted)
        if character is None:
            return {
                "error": f"no character {wanted!r} in this project",
                "known_characters": [
                    {"id": row["id"], "name": row["name"]} for row in characters
                ],
            }

        emotions = get_character_emotions(conn, character["id"])
        current = {
            axis: (emotions[axis] if emotions is not None else 0.0)
            for axis in PAD_AXES
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

    history: list[dict] = []
    for row in rows:
        try:
            spec = json.loads(row["beat_spec"])
        except json.JSONDecodeError:
            continue
        if spec.get("focal_character_id") != character["id"]:
            continue
        history.append(
            {
                "beat_id": row["beat_id"],
                "chapter_id": row["chapter_id"],
                "intent": str(spec.get("intent") or ""),
                "target_pad": spec.get("target_pad") or {},
            }
        )

    return {
        "character_id": character["id"],
        "name": character["name"],
        "current_pad": current,
        "history": history,
    }
