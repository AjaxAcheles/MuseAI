"""Chapter planning node.

Breaks the pointer's active arc into an ordered sequence of chapters, writes them
to SQLite, activates the first one, and advances the pointer to it.

Chapter ids are derived from the arc id and the chapter's position in the
returned array, so re-planning an arc overwrites the same rows rather than
accumulating duplicates — the upserts stay idempotent under an event-log replay.

The LLM call sends no ``max_tokens``: a reasoning-style endpoint bills its hidden
reasoning against that budget, and a plan truncated mid-array parses to nothing.
"""

from __future__ import annotations

import json
import sqlite3

from museai.core.logging_setup import log_node_event
from museai.core.stream_bus import bus
from museai.fsm.nodes.deps import PlanningError, get_node_config
from museai.fsm.state import FSM_Pointer, OrchestratorState
from museai.llm.client import call_llm
from museai.llm.prompts import render_messages
from museai.llm.structured import parse_json_array
from museai.memory.db import (
    connect_db,
    get_arcs,
    get_characters,
    get_open_threads,
    get_project,
    upsert_arc,
    upsert_chapter,
)

PHASE = "Planning"


def chapter_id_for(arc_id: str, ordering: int) -> str:
    """The stable id of the ``ordering``-th chapter of an arc (1-based)."""
    return f"{arc_id}-c{ordering:02d}"


def _thread_context(rows: list[sqlite3.Row]) -> list[dict]:
    return [
        {"id": row["id"], "status": row["status"], "description": row["description"]}
        for row in rows
    ]


def _character_context(rows: list[sqlite3.Row]) -> list[dict]:
    return [
        {"id": row["id"], "name": row["name"], "description": row["description"] or ""}
        for row in rows
    ]


def _normalise_obligations(value: object, ordering: int) -> list[str]:
    """Coerce a chapter's obligations to a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise PlanningError(
            f"chapter {ordering}: obligations must be a JSON array of strings, "
            f"got {type(value).__name__}"
        )
    return [str(item).strip() for item in value if str(item).strip()]


async def plan_chapter(state: OrchestratorState) -> dict:
    """Plan the chapters of the pointer's arc and advance the pointer to the first.

    Returns the state delta: a pointer aimed at the newly-active chapter, beat
    index reset to 0.
    """
    config = get_node_config()
    pointer = state["fsm_pointer"]
    project_id = state["project_id"]

    log_node_event("plan_chapter", event="start", project_id=project_id,
                   arc_id=pointer.arc_id)

    conn = connect_db(config.db_path)
    try:
        project = get_project(conn, project_id)
        if project is None:
            raise PlanningError(f"project {project_id!r} is not in the database")

        arc = next(
            (row for row in get_arcs(conn, project_id) if row["id"] == pointer.arc_id),
            None,
        )
        if arc is None:
            raise PlanningError(
                f"arc {pointer.arc_id!r} is not an arc of project {project_id!r}"
            )

        threads = get_open_threads(conn, project_id)
        characters = get_characters(conn, project_id)
        log_node_event(
            "plan_chapter",
            event="context_assembled",
            arc_id=arc["id"],
            open_threads=len(threads),
            characters=len(characters),
        )

        messages = render_messages(
            "chapter_planner",
            project={"genre": project["genre"] or "", "premise": project["premise"] or ""},
            arc={"description": arc["description"]},
            threads=_thread_context(threads),
            characters=_character_context(characters),
        )

        response = await call_llm(config.endpoint, messages, agent="chapter_planner", stream=True)
        planned = parse_json_array(response.text, what="chapters")

        chapters: list[dict] = []
        for ordering, item in enumerate(planned, start=1):
            description = str(item.get("description") or "").strip()
            if not description:
                raise PlanningError(f"chapter {ordering} has no description")
            chapters.append(
                {
                    "id": chapter_id_for(arc["id"], ordering),
                    "arc_id": arc["id"],
                    "ordering": ordering,
                    "description": description,
                    "obligations": json.dumps(
                        _normalise_obligations(item.get("obligations"), ordering),
                        ensure_ascii=False,
                    ),
                }
            )

        active = chapters[0]
        with conn:
            for chapter in chapters:
                upsert_chapter(conn, status="planned", **chapter)
            upsert_chapter(conn, status="active", **active)
            upsert_arc(
                conn,
                id=arc["id"],
                project_id=project_id,
                ordering=arc["ordering"],
                description=arc["description"],
                status="active",
            )
    finally:
        conn.close()

    log_node_event(
        "plan_chapter",
        event="chapters_planned",
        arc_id=arc["id"],
        chapters=len(chapters),
        active_chapter_id=active["id"],
    )
    log_node_event("plan_chapter", event="phase_change", phase=PHASE, arc_id=arc["id"])
    await bus.publish(
        "phase_change",
        {"phase": PHASE, "node": "plan_chapter", "project_id": project_id,
         "arc_id": arc["id"]},
    )
    await bus.publish(
        "chapters_planned",
        {
            "arc_id": arc["id"],
            "chapter_count": len(chapters),
            "active_chapter_id": active["id"],
            "chapters": [
                {"id": c["id"], "ordering": c["ordering"], "description": c["description"]}
                for c in chapters
            ],
        },
    )

    return {
        "fsm_pointer": FSM_Pointer(
            arc_id=arc["id"], chapter_id=active["id"], beat_index=0
        )
    }
