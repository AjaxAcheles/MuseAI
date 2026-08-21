"""Chapter planning node.

Breaks the pointer's active arc into an ordered sequence of chapters, writes them
to SQLite, activates the first unfinished one, and advances the pointer to it.

**An arc is planned once.** If the arc already has chapters this node reuses
them and never calls the model. That is not an optimisation. Every run enters at
this node, including the one you start after a crash, and re-planning would hand
back a *different* set of chapters whose descriptions no longer describe the
prose already drafted from the old ones — while resetting their status and
walking the graph back over committed beats. Re-planning is what `Reset` is for.

The LLM call sends no ``max_tokens``: a reasoning-style endpoint bills its hidden
reasoning against that budget, and a plan truncated mid-array parses to nothing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3

from museai.core.logging_setup import log_node_event
from museai.core.stream_bus import bus
from museai.fsm.nodes.context_budget import pop_back, prune_to_budget, window_budget
from museai.fsm.nodes.deps import PlanningError, get_node_config
from museai.fsm.plan_validation import validate_concrete_obligation
from museai.fsm.state import FSM_Pointer, OrchestratorState
from museai.fsm.tools.registry import tool_impls_for, tool_specs_for
from museai.llm.planning import call_llm_for_json_array
from museai.llm.structured import StructuredOutputError
from museai.llm.prompts import render_messages
from museai.memory.db import (
    connect_db,
    get_arcs,
    get_chapters_for_arc,
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


def _stored_chapters(rows: list[sqlite3.Row]) -> list[dict]:
    """Rebuild the planner's chapter dicts from rows this node wrote earlier."""
    return [
        {
            "id": row["id"],
            "arc_id": row["arc_id"],
            "ordering": row["ordering"],
            "description": row["description"],
            "obligations": row["obligations"] or "[]",
        }
        for row in rows
    ]


def _first_unfinished(rows: list[sqlite3.Row]) -> int:
    """Index of the first chapter still to write, or the last one if the arc is done."""
    for index, row in enumerate(rows):
        if row["status"] != "completed":
            return index
    return len(rows) - 1


def _read_chapter_plan_context(db_path, project_id: str, pointer: FSM_Pointer) -> dict:
    """Read planner inputs with a connection owned by this worker thread."""
    conn = connect_db(db_path)
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
        return {
            "project": dict(project),
            "arc": dict(arc),
            "existing": [dict(row) for row in get_chapters_for_arc(conn, arc["id"])],
            "threads": _thread_context(get_open_threads(conn, project_id)),
            "characters": _character_context(get_characters(conn, project_id)),
        }
    finally:
        conn.close()


def _write_chapter_plan(
    db_path, project_id: str, arc: dict, chapters: list[dict], active: dict,
    reused: bool, arc_done: bool,
) -> None:
    """Persist a chapter plan without moving its transaction onto the event loop."""
    conn = connect_db(db_path)
    try:
        with conn:
            if not reused:
                for chapter in chapters:
                    upsert_chapter(conn, status="planned", **chapter)
            # On the reuse path only the active chapter is touched, and only when
            # there is still one to write. Marking a finished chapter 'active'
            # would walk the graph back over prose it already committed.
            if not arc_done:
                upsert_chapter(conn, status="active", **active)
            upsert_arc(
                conn,
                id=arc["id"],
                project_id=project_id,
                ordering=arc["ordering"],
                description=arc["description"],
                status="completed" if arc_done else "active",
            )
    finally:
        conn.close()


def _normalise_obligations(value: object, ordering: int) -> list[str]:
    """Normalize a chapter's already-validated concrete obligations."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise PlanningError(
            f"chapter {ordering}: obligations must be a JSON array of strings, "
            f"got {type(value).__name__}"
        )
    normalized: list[str] = []
    for position, item in enumerate(value, start=1):
        text, problem = validate_concrete_obligation(item)
        if problem:
            raise PlanningError(f"chapter {ordering} obligation {position} {problem}")
        assert text is not None
        normalized.append(text)
    return normalized


_CHAPTER_PLAN_KEYS = {"ordering", "description", "obligations"}


def _validate_chapter_item(item: dict, ordering: int) -> dict:
    """Validate one model-produced chapter before the planner retry ladder exits."""
    extra = set(item) - _CHAPTER_PLAN_KEYS
    if extra:
        raise StructuredOutputError(
            f"chapter {ordering} has unexpected fields: {', '.join(sorted(extra))}"
        )
    declared_order = item.get("ordering")
    if isinstance(declared_order, bool) or not isinstance(declared_order, int):
        raise StructuredOutputError(f"chapter {ordering} ordering must be an integer")
    if declared_order != ordering:
        raise StructuredOutputError(
            f"chapter {ordering} declares ordering {declared_order}; order must be contiguous"
        )
    description = item.get("description")
    if not isinstance(description, str) or not description.strip():
        raise StructuredOutputError(f"chapter {ordering} description must be a non-empty string")
    obligations = item.get("obligations")
    if not isinstance(obligations, list) or not obligations:
        raise StructuredOutputError(
            f"chapter {ordering} obligations must be a non-empty array of concrete strings"
        )
    for position, value in enumerate(obligations, start=1):
        _, problem = validate_concrete_obligation(value)
        if problem:
            raise StructuredOutputError(f"chapter {ordering} obligation {position} {problem}")
    return item


async def plan_chapter(state: OrchestratorState) -> dict:
    """Plan the chapters of the pointer's arc, or reuse the ones already planned.

    Returns the state delta: a pointer aimed at the active chapter, beat index
    reset to 0.
    """
    config = get_node_config()
    pointer = state["fsm_pointer"]
    project_id = state["project_id"]

    log_node_event("plan_chapter", event="start", project_id=project_id,
                   arc_id=pointer.arc_id)

    context = await asyncio.to_thread(
        _read_chapter_plan_context, config.db_path, project_id, pointer
    )
    try:
        project = context["project"]
        arc = context["arc"]
        existing = context["existing"]
        reused = bool(existing)

        if reused:
            chapters = _stored_chapters(existing)
            index = _first_unfinished(existing)
            active = chapters[index]
            arc_done = existing[index]["status"] == "completed"
        else:
            threads = context["threads"]
            characters = context["characters"]
            log_node_event(
                "plan_chapter",
                event="context_assembled",
                arc_id=arc["id"],
                open_threads=len(threads),
                characters=len(characters),
            )

            thread_ctx = threads
            character_ctx = characters

            def _render() -> list[dict]:
                return render_messages(
                    "chapter_planner",
                    project={
                        "genre": project["genre"] or "",
                        "premise": project["premise"] or "",
                        "setting": project["setting"] or "",
                    },
                    arc={"description": arc["description"]},
                    threads=thread_ctx,
                    characters=character_ctx,
                    research_mode=config.generation.research_mode,
                )

            # Trim to leave generation room when the endpoint declares a window.
            # The project, arc, and cast are the instructions; only the
            # lowest-priority open threads are shed.
            chapter_endpoint = config.endpoint_for("chapter_planner")
            budget = window_budget(chapter_endpoint)
            if budget is not None:
                report = prune_to_budget(
                    budget=budget,
                    render=_render,
                    tokenizer_family=chapter_endpoint.tokenizer_family,
                    model_name=chapter_endpoint.model_name,
                    drops=[("threads", lambda: pop_back(thread_ctx))],
                )
                log_node_event(
                    "plan_chapter",
                    level=logging.WARNING if report["over_budget"] else logging.INFO,
                    event="context_pruned",
                    arc_id=arc["id"],
                    budget=report["budget"],
                    tokens_before=report["tokens_before"],
                    tokens=report["tokens"],
                    over_budget=report["over_budget"],
                    dropped_threads=report["dropped"]["threads"],
                )
            messages = _render()
            async def on_tool_call(event: dict) -> None:
                log_node_event(
                    "plan_chapter",
                    event="tool_call",
                    arc_id=arc["id"],
                    tool=event["tool"],
                    args=event["arguments"],
                )
                await bus.publish(
                    "planner_tool",
                    {"node": "plan_chapter", "arc_id": arc["id"], **event},
                )

            planned = await call_llm_for_json_array(
                config.endpoint_for("chapter_planner"),
                messages,
                what="chapters",
                agent="chapter_planner",
                node="plan_chapter",
                retries=config.generation.planner_parse_retries,
                tools=tool_specs_for("chapter_planner"),
                tool_impls=tool_impls_for("chapter_planner"),
                max_tool_iterations=config.generation.max_agent_iterations,
                on_tool_event=on_tool_call,
                tool_call_cap=config.generation.tool_call_cap,
                tool_timeout=config.generation.tool_timeout,
                # A real chapter always has a description; a truncated reply's
                # inner array (obligations) never does.
                element_keys=("description",),
                item_validator=_validate_chapter_item,
            )

            chapters = []
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
            arc_done = False

        await asyncio.to_thread(
            _write_chapter_plan,
            config.db_path,
            project_id,
            arc,
            chapters,
            active,
            reused,
            arc_done,
        )
    finally:
        pass

    log_node_event(
        "plan_chapter",
        event="chapters_planned",
        arc_id=arc["id"],
        chapters=len(chapters),
        active_chapter_id=active["id"],
        reused=reused,
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
            "reused": reused,
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
