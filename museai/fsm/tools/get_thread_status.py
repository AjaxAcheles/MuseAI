"""Every narrative thread of the active project, with status and priority.

The critic's grounding for a thread-conflict claim: before faulting a draft for
advancing a closed thread or dropping an open one, look the threads up rather
than trusting the excerpt in context.
"""

from __future__ import annotations

from typing import Any

from museai.fsm.tools.project_db import project_connection
from museai.memory.db import get_threads_for_project

GET_THREAD_STATUS_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_thread_status",
        "description": (
            "Every narrative thread in this project with its current status "
            "(open, progressing, or closed) and priority. Use it to ground a "
            "claim about a thread before reporting it as a continuity problem."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


def get_thread_status() -> list[dict]:
    """All threads, unresolved first. ``[]`` when the project has none."""
    with project_connection() as (conn, project_id):
        rows = get_threads_for_project(conn, project_id)
    return [
        {
            "id": row["id"],
            "description": row["description"],
            "status": row["status"],
            "priority_score": row["priority_score"],
        }
        for row in rows
    ]
