"""Has the manuscript already said this?

Takes a passage (or a suspect phrase) and finds where the committed prose
already carries it: near-duplicate paragraphs by the audit's own similarity
machinery, plus verbatim hits for short phrases. This is the "have I written
this before?" check for the style-echo problem — a drafter can test a line it
suspects it is recycling, and a critic can ground a repetition claim.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from museai.fsm.nodes.audit import _normalize_for_compare, split_paragraphs
from museai.fsm.nodes.deps import get_node_config
from museai.fsm.tools.project_db import project_connection
from museai.memory.db import get_committed_beats, get_recent_committed_beats

_SCOPES = ("project", "recent")

FIND_REPETITION_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "find_repetition",
        "description": (
            "Check whether the committed manuscript already contains this "
            "text: near-duplicate paragraphs, or verbatim hits for a short "
            "phrase. Returns where each repetition lives. An empty list means "
            "the text is fresh. Use it before repeating an image, a line, or "
            "a paragraph you suspect the story has already used."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text_or_query": {
                    "type": "string",
                    "description": "The passage or phrase to check.",
                },
                "scope": {
                    "type": "string",
                    "enum": list(_SCOPES),
                    "description": (
                        "'project' checks every committed beat; 'recent' only "
                        "the recent-prose window."
                    ),
                    "default": "project",
                },
            },
            "required": ["text_or_query"],
        },
    },
}


def _snippet(text: str, limit: int) -> str:
    return text[:limit] + ("…" if len(text) > limit else "")


def find_repetition(text_or_query: str, scope: str = "project") -> dict:
    """Where the committed prose already says this, best match first."""
    wanted_scope = (scope or "").strip().lower()
    if wanted_scope not in _SCOPES:
        return {"error": f"unknown scope {scope!r}", "supported_scopes": list(_SCOPES)}

    text = (text_or_query or "").strip()
    if not text:
        return {"error": "no text was given to check"}

    config = get_node_config()
    threshold = config.generation.repetition_threshold
    tools = config.tools
    snippet_chars = tools.repetition_snippet_chars

    with project_connection() as (conn, project_id):
        if wanted_scope == "recent":
            rows = get_recent_committed_beats(
                conn, project_id, config.generation.recent_prose_beats
            )
            beats = [(row["id"], row["chapter_id"], row["prose"]) for row in rows]
        else:
            rows = get_committed_beats(conn, project_id)
            beats = [(row["beat_id"], row["chapter_id"], row["prose"]) for row in rows]

    input_norm = _normalize_for_compare(text)
    input_paragraphs = [
        (paragraph, _normalize_for_compare(paragraph))
        for paragraph in split_paragraphs(text) or [text]
    ]
    is_phrase = len(input_norm.split()) <= tools.repetition_phrase_max_words

    matches: list[dict] = []
    for beat_id, chapter_id, prose in beats:
        for committed in split_paragraphs(prose or ""):
            committed_norm = _normalize_for_compare(committed)
            if not committed_norm:
                continue
            if is_phrase:
                if input_norm and input_norm in committed_norm:
                    matches.append(
                        {
                            "beat_id": beat_id,
                            "chapter_id": chapter_id,
                            "similarity": 1.0,
                            "committed_text": _snippet(committed, snippet_chars),
                        }
                    )
                continue
            best = max(
                (
                    SequenceMatcher(None, norm, committed_norm).ratio()
                    for _, norm in input_paragraphs
                    if norm
                ),
                default=0.0,
            )
            if best >= threshold:
                matches.append(
                    {
                        "beat_id": beat_id,
                        "chapter_id": chapter_id,
                        "similarity": round(best, 3),
                        "committed_text": _snippet(committed, snippet_chars),
                    }
                )

    matches.sort(key=lambda m: -m["similarity"])
    return {"matches": matches[: tools.repetition_max_matches]}
