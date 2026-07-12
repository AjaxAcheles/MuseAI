"""Full-text search over the committed manuscript.

The single highest-value agent tool: every agent's injected context stops at the
last ``recent_prose_beats`` beats, which is how a character once got referenced
before ever being introduced. This tool lets any agent check the *whole* book.

The index is the ``Beats`` table itself — committed beats number in the
hundreds, so a scored scan is instant and needs no FTS table to migrate or fall
out of sync. Scoring is whole-word term frequency with a large bonus when the
query appears verbatim, so "the lantern oath" finds the beat where the oath is
sworn before every beat that merely mentions a lantern.
"""

from __future__ import annotations

import re
from typing import Any

from museai.fsm.tools.project_db import project_connection
from museai.memory.db import get_committed_beats

# Enough of the matching passage to quote or verify against; more is context
# spend, and the model can ask for the whole chapter if it needs it.
_SNIPPET_CHARS = 300

# A verbatim phrase hit outranks any pile of scattered term hits.
_PHRASE_BONUS = 25

_WORD = re.compile(r"[\w']+")

SEARCH_MANUSCRIPT_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_manuscript",
        "description": (
            "Search the full committed manuscript — every beat of prose written "
            "so far, not just the recent excerpt in your context. Returns the "
            "best-matching beats, each with a snippet around the match. Use it "
            "to verify an earlier event, name, or detail before relying on it. "
            "An empty list means nothing in the manuscript matches."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Words or a phrase to find, e.g. a character name, an "
                        "object, or a line you think was written before."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of matching beats to return.",
                    "default": 5,
                    "minimum": 1,
                },
                "scope": {
                    "type": "string",
                    "description": (
                        "What to search. Only committed prose exists to search."
                    ),
                    "enum": ["committed_only"],
                    "default": "committed_only",
                },
            },
            "required": ["query"],
        },
    },
}


def _terms(query: str) -> list[str]:
    return [t for t in _WORD.findall((query or "").lower()) if t]


def _snippet(prose: str, needle: str) -> str:
    """~300 chars of ``prose`` around the first occurrence of ``needle``."""
    lowered = prose.lower()
    at = lowered.find(needle)
    if at == -1:
        at = 0
    start = max(0, at - _SNIPPET_CHARS // 3)
    end = min(len(prose), start + _SNIPPET_CHARS)
    # Snap to word boundaries so the quote never opens or closes mid-word.
    while start > 0 and not prose[start - 1].isspace():
        start -= 1
    while end < len(prose) and not prose[end].isspace():
        end += 1
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(prose) else ""
    return f"{prefix}{prose[start:end].strip()}{suffix}"


def search_manuscript(
    query: str, limit: int = 5, scope: str = "committed_only"
) -> list[dict] | dict:
    """Return up to ``limit`` committed beats matching ``query``, best first.

    Each result carries ``beat_id``, ``chapter_id``, and a ``snippet`` around
    the match — never a whole beat's prose. ``[]`` when the query is blank or
    nothing matches.
    """
    if scope != "committed_only":
        return {
            "error": f"unsupported scope {scope!r}",
            "supported_scopes": ["committed_only"],
        }
    terms = _terms(query)
    if not terms:
        return []
    try:
        limit = max(1, int(limit))
    except (TypeError, ValueError):
        limit = 5

    phrase = " ".join(terms)
    patterns = [re.compile(rf"\b{re.escape(term)}\b") for term in terms]

    with project_connection() as (conn, project_id):
        rows = get_committed_beats(conn, project_id)

    scored: list[tuple[int, int, dict]] = []
    for position, row in enumerate(rows):
        prose = row["prose"] or ""
        haystack = " ".join(_WORD.findall(prose.lower()))
        score = sum(len(pattern.findall(haystack)) for pattern in patterns)
        if score == 0:
            continue
        if len(terms) > 1 and phrase in haystack:
            score += _PHRASE_BONUS
        # The snippet centres on the verbatim phrase when the prose has it,
        # else on the first term that occurs.
        needle = phrase if phrase in prose.lower() else ""
        if not needle:
            needle = next((t for t in terms if t in prose.lower()), terms[0])
        scored.append(
            (
                -score,
                position,  # ties resolve in narrative order
                {
                    "beat_id": row["beat_id"],
                    "chapter_id": row["chapter_id"],
                    "snippet": _snippet(prose, needle),
                },
            )
        )

    scored.sort(key=lambda item: (item[0], item[1]))
    return [result for _, _, result in scored[:limit]]
