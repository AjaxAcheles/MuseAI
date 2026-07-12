"""The seeded contract: the durable authorial commitments of this project.

Genre, premise, target, the seeded cast, and the seeded threads — the facts
every plan and every beat must stay true to. Read from the canonical database
rows the seed loader wrote, which outlive the seed file itself.
"""

from __future__ import annotations

from typing import Any

from museai.fsm.tools.project_db import project_connection
from museai.memory.db import get_characters, get_project, get_threads_for_project

GET_SEED_CONTRACT_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_seed_contract",
        "description": (
            "The project's seeded contract: genre, premise, setting, word-count "
            "target, every character with their description, and every narrative "
            "thread. These are the commitments plans and prose must honour. "
            "Use it to ground a plan in what the author actually asked for."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


def get_seed_contract() -> dict:
    """Project row, cast, and threads, or an error when nothing is seeded."""
    with project_connection() as (conn, project_id):
        project = get_project(conn, project_id)
        if project is None:
            return {"error": f"no project {project_id!r} is seeded in the database"}
        characters = get_characters(conn, project_id)
        threads = get_threads_for_project(conn, project_id)

    return {
        "project": {
            "id": project["id"],
            "genre": project["genre"] or "",
            "premise": project["premise"] or "",
            "setting": project["setting"] or "",
            "word_count_target": project["word_count_target"],
        },
        "characters": [
            {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"] or "",
            }
            for row in characters
        ],
        "threads": [
            {
                "id": row["id"],
                "description": row["description"],
                "status": row["status"],
                "priority_score": row["priority_score"],
            }
            for row in threads
        ],
    }
