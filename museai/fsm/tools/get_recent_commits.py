"""The last few committed beats, as metadata plus their closing words.

What just happened, without a prose dump: each entry carries the beat's intent
and exit state and only the final words of its prose — enough to continue from,
small enough not to feed the style-echo loop. Exact wording lives behind
``search_manuscript``.
"""

from __future__ import annotations

import json
from typing import Any

from museai.fsm.nodes.deps import get_node_config
from museai.fsm.tools.project_db import project_connection
from museai.memory.db import get_recent_committed_beats

GET_RECENT_COMMITS_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_recent_commits",
        "description": (
            "The most recently committed beats, oldest first: each with its "
            "intent, exit state, word count, and the closing words of its "
            "prose. Use it to see exactly where the story left off."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "How many recent beats to return.",
                    "default": 5,
                    "minimum": 1,
                },
            },
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


def get_recent_commits(limit: int = 5) -> list[dict]:
    """Up to ``limit`` recent committed beats, oldest first. ``[]`` before any."""
    try:
        limit = max(1, int(limit))
    except (TypeError, ValueError):
        limit = 5

    with project_connection() as (conn, project_id):
        rows = get_recent_committed_beats(conn, project_id, limit)

    # The tail of each beat is what the drafter continues from.
    closing_words = get_node_config().tools.recent_commit_closing_words

    commits: list[dict] = []
    for row in rows:
        spec = _spec(row["beat_spec"])
        words = (row["prose"] or "").split()
        commits.append(
            {
                "beat_id": row["id"],
                "chapter_id": row["chapter_id"],
                "intent": str(spec.get("intent") or ""),
                "exit_state": str(spec.get("exit_state") or ""),
                "word_count": row["word_count"],
                "closing_words": " ".join(words[-closing_words:]),
            }
        )
    return commits
