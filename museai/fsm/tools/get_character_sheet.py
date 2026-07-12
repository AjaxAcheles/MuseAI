"""One character's description and how they actually talk on the page.

The seeded description says who a character is; their committed dialogue says
how they sound. The samples are lines quoted in paragraphs that name the
character — a paragraph-level attribution, which is a heuristic, so the samples
are voice reference, not a transcript of everything they said.
"""

from __future__ import annotations

import re
from typing import Any

from museai.fsm.nodes.audit import split_paragraphs
from museai.fsm.tools.project_db import project_connection
from museai.memory.db import get_characters, get_committed_beats

# Enough lines to hear a voice; spread across the book so early and late
# chapters both contribute.
_MAX_SAMPLES = 8
_MAX_SAMPLE_CHARS = 200

# Straight or curly double quotes, non-greedy across one quoted run.
_QUOTED = re.compile(r"[\"“]([^\"“”]+)[\"”]")

GET_CHARACTER_SHEET_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_character_sheet",
        "description": (
            "One character's description plus sampled lines of dialogue they "
            "have spoken in the committed manuscript. Use it to keep a "
            "character's voice and details consistent with what is already on "
            "the page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The character's name (their id also resolves).",
                },
            },
            "required": ["name"],
        },
    },
}


def _resolve(rows: list, wanted: str):
    """The character row ``wanted`` names, matching name first, then id."""
    lowered = wanted.casefold()
    for row in rows:
        if row["name"].strip().casefold() == lowered:
            return row
    for row in rows:
        if row["id"] == wanted or row["id"].casefold() == lowered:
            return row
    # A first name alone resolves when exactly one character carries it.
    partial = [
        row for row in rows if lowered in {w.casefold() for w in row["name"].split()}
    ]
    if len(partial) == 1:
        return partial[0]
    return None


def _dialogue_samples(prose_passages: list[str], name: str) -> list[str]:
    """Quoted lines from paragraphs that mention ``name``, spread evenly."""
    mention = re.compile(rf"\b{re.escape(name.split()[0])}\b", re.IGNORECASE)
    lines: list[str] = []
    seen: set[str] = set()
    for passage in prose_passages:
        for paragraph in split_paragraphs(passage):
            if not mention.search(paragraph):
                continue
            for quoted in _QUOTED.findall(paragraph):
                line = quoted.strip()[:_MAX_SAMPLE_CHARS]
                if line and line not in seen:
                    seen.add(line)
                    lines.append(line)
    if len(lines) <= _MAX_SAMPLES:
        return lines
    step = len(lines) / _MAX_SAMPLES
    return [lines[int(i * step)] for i in range(_MAX_SAMPLES)]


def get_character_sheet(name: str) -> dict:
    """Description and sampled committed dialogue, or an error naming the cast."""
    wanted = (name or "").strip()
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
        passages = [row["prose"] for row in get_committed_beats(conn, project_id)]

    return {
        "id": character["id"],
        "name": character["name"],
        "description": character["description"] or "",
        "dialogue_samples": _dialogue_samples(passages, character["name"]),
    }
