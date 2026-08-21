"""Context assembly node.

Model-free and SQLite-only. Reads everything the drafter needs for the active
beat, prunes it to fit a token budget, and writes ``active_context_package``.

**The budget.** ``generation.context_token_budget`` is a soft ceiling on the
rendered drafter prompt, measured with the endpoint's own tokenizer. Over it,
context is dropped by priority, cheapest-to-lose first:

1. Recent committed prose, oldest passage first. The newest passage is the one
   the drafter must continue from, so it is the last to go.
2. Open threads, lowest ``priority_score`` first.

The beat spec, the ``pad_constraint``, and the chapter's obligations are never
dropped. They are the beat's instructions — a draft written without them is not
a shorter draft, it is the wrong one. If the protected core alone exceeds the
budget, the node proceeds over budget and logs a WARNING rather than silently
mutilating the prompt.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

from museai.core.logging_setup import get_fsm_logger, log_node_event
from museai.core.stream_bus import bus
from museai.fsm.nodes.context_budget import (
    pop_back,
    pop_front,
    prune_to_budget,
    window_budget,
)
from museai.fsm.nodes.deps import PlanningError, get_node_config
from museai.fsm.pad import PAD_AXES
from museai.fsm.state import FSM_Pointer, OrchestratorState
from museai.llm.prompts import render_messages
from museai.memory.db import (
    connect_db,
    get_beats_for_chapter,
    get_character_emotions,
    get_characters,
    get_chapters_for_arc,
    get_committed_beats,
    get_open_threads,
    get_project,
    get_recent_committed_beats,
)

PHASE = "Drafting"


def drafter_messages(package: dict) -> list[dict]:
    """Render the drafter prompt from an assembled context package.

    Lives here rather than in ``draft_prose`` so the budget is measured against
    the exact messages the drafter will later send: one renderer, so what was
    counted and what is transmitted cannot diverge.
    """
    return render_messages(
        "drafter",
        beat=package["beat"],
        pad_constraint=package["pad_constraint"],
        project=package.get("project") or {},
        chapter=package["chapter"],
        threads=package["threads"],
        characters=package["characters"],
        recent_prose=package["recent_prose"],
        research_mode=get_node_config().generation.research_mode,
    )


def _resolve_chapter(conn: sqlite3.Connection, pointer: FSM_Pointer) -> sqlite3.Row:
    chapters = get_chapters_for_arc(conn, pointer.arc_id)
    if not chapters:
        raise PlanningError(f"arc {pointer.arc_id!r} has no planned chapters")
    for row in chapters:
        if row["id"] == pointer.chapter_id:
            return row
    for row in chapters:
        if row["status"] == "active":
            return row
    raise PlanningError(
        f"arc {pointer.arc_id!r} has no chapter {pointer.chapter_id!r} and no "
        f"active chapter"
    )


def _resolve_beat(
    conn: sqlite3.Connection, chapter_id: str, beat_index: int
) -> sqlite3.Row:
    """The beat the pointer names. ``beat_index`` is 0-based; ``ordering`` is 1-based."""
    beats = get_beats_for_chapter(conn, chapter_id)
    if not beats:
        raise PlanningError(f"chapter {chapter_id!r} has no planned beats")

    for row in beats:
        if row["ordering"] == beat_index + 1:
            return row
    for row in beats:
        if row["status"] == "active":
            return row
    raise PlanningError(
        f"chapter {chapter_id!r} has no beat at index {beat_index} and no active beat"
    )


def _intended_refrain(raw: object) -> list[str]:
    """Normalize a planner's ``intended_refrain`` to a list of non-empty strings.

    The planner may emit a single phrase or a list; either way the audit wants a
    flat list of phrases that this beat is permitted to repeat verbatim.
    """
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def _thread_updates(raw: object) -> list[dict]:
    """Normalize a beat spec's ``thread_updates`` to ``{id, status}`` dicts.

    The planner validated these at planning time; they are surfaced here so the
    drafter is told which thread movement the beat must earn and the critic can
    check that the prose actually delivered it.
    """
    if not isinstance(raw, list):
        return []
    updates = []
    for item in raw:
        if isinstance(item, dict) and item.get("id") and item.get("status"):
            updates.append({"id": str(item["id"]), "status": str(item["status"])})
    return updates


def _obligations(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return [raw]
    return [str(item) for item in parsed] if isinstance(parsed, list) else [str(parsed)]


def _characters(conn: sqlite3.Connection, project_id: str) -> list[dict]:
    characters = []
    for row in get_characters(conn, project_id):
        emotions = get_character_emotions(conn, row["id"])
        characters.append(
            {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"] or "",
                "pad": {
                    axis: (emotions[axis] if emotions is not None else 0.0)
                    for axis in PAD_AXES
                },
            }
        )
    return characters


def _read_context_rows(
    db_path, project_id: str, pointer: FSM_Pointer, recent_prose_beats: int
) -> dict:
    """Read the package inputs with a connection owned by this worker thread."""
    conn = connect_db(db_path)
    try:
        project = get_project(conn, project_id)
        chapter = _resolve_chapter(conn, pointer)
        beat = _resolve_beat(conn, chapter["id"], pointer.beat_index)
        return {
            "project": dict(project) if project is not None else None,
            "chapter": dict(chapter),
            "beat": dict(beat),
            "threads": [dict(row) for row in get_open_threads(conn, project_id)],
            "characters": _characters(conn, project_id),
            "recent_prose": [
                row["prose"]
                for row in get_recent_committed_beats(
                    conn, project_id, recent_prose_beats
                )
            ],
            "committed_prose": [
                row["prose"] for row in get_committed_beats(conn, project_id)
            ],
        }
    finally:
        conn.close()


def _prune_to_budget(package: dict, config) -> dict:
    """Drop context until the rendered drafter prompt fits the budget.

    The budget is ``context_token_budget``, tightened to the endpoint's declared
    context window (leaving output_reservation free) when one is set — so a small
    model never leaves zero room to generate. Drop order, cheapest first: oldest
    recent prose (the newest passage is what the beat continues), then the
    lowest-priority open threads (``get_open_threads`` sorts them descending, so
    the tail is cheapest). Returns a report of what was dropped.
    """
    endpoint = config.endpoint_for("drafter")
    budget = window_budget(endpoint, fallback=config.generation.context_token_budget)
    report = prune_to_budget(
        budget=budget,
        render=lambda: drafter_messages(package),
        tokenizer_family=endpoint.tokenizer_family,
        model_name=endpoint.model_name,
        drops=[
            ("prose", lambda: pop_front(package["recent_prose"])),
            ("threads", lambda: pop_back(package["threads"])),
        ],
    )
    if report["over_budget"]:
        get_fsm_logger().warning(
            "node=assemble_context protected context exceeds budget: "
            "tokens=%d budget=%d beat_id=%s; drafting over budget",
            report["tokens"],
            budget,
            package["beat"]["id"],
        )

    return {
        "budget": report["budget"],
        "tokens_before": report["tokens_before"],
        "tokens": report["tokens"],
        "dropped_prose_passages": report["dropped"]["prose"],
        "dropped_threads": report["dropped"]["threads"],
        "over_budget": report["over_budget"],
    }


async def assemble_context(state: OrchestratorState) -> dict:
    """Build the drafting context package for the active beat.

    Returns the state delta: ``active_context_package``.
    """
    config = get_node_config()
    pointer = state["fsm_pointer"]
    project_id = state["project_id"]

    log_node_event(
        "assemble_context",
        event="start",
        arc_id=pointer.arc_id,
        chapter_id=pointer.chapter_id,
        beat_index=pointer.beat_index,
    )

    rows = await asyncio.to_thread(
        _read_context_rows,
        config.db_path,
        project_id,
        pointer,
        config.generation.recent_prose_beats,
    )
    project = rows["project"]
    chapter = rows["chapter"]
    beat = rows["beat"]
    spec = json.loads(beat["beat_spec"]) if beat["beat_spec"] else {}

    if not beat["pad_constraint"]:
        raise PlanningError(
            f"beat {beat['id']!r} has no pad_constraint; it was never planned"
        )

    package = {
            "beat": {
                "id": beat["id"],
                "ordering": beat["ordering"],
                "intent": spec.get("intent", ""),
                "entry_state": spec.get("entry_state", ""),
                "exit_state": spec.get("exit_state", ""),
                # The plot mandate: the change this beat must produce, the
                # on-page event that carries it, its structural function, and
                # the chapter obligations it discharges. Older specs read back
                # as empty — the templates render nothing for them.
                "required_change": spec.get("required_change", ""),
                "observable_event": spec.get("observable_event", ""),
                "beat_function": spec.get("beat_function", ""),
                "discharges": [str(o) for o in (spec.get("discharges") or []) if str(o).strip()],
                "focal_character_id": spec.get("focal_character_id", ""),
                # Phrases this beat's planner declared may recur verbatim. The
                # repetition audit exempts a paragraph matching one of these, so a
                # deliberate refrain is not faulted as a copy. Only the planner
                # writes this — never the drafter.
                "intended_refrain": _intended_refrain(spec.get("intended_refrain")),
                # The thread movement this beat was planned to produce. Shown to
                # the drafter as a deliverable and to the critic as a check.
                "thread_updates": _thread_updates(spec.get("thread_updates")),
            },
            "pad_constraint": beat["pad_constraint"],
            # The story world: premise, genre, and setting are part of the
            # protected core — a beat written blind to them treats the world as
            # interchangeable background. Small and never pruned.
            "project": {
                "genre": (project["genre"] if project else "") or "",
                "premise": (project["premise"] if project else "") or "",
                "setting": (project["setting"] if project else "") or "",
            },
            "chapter": {
                "id": chapter["id"],
                "description": chapter["description"],
                "obligations": _obligations(chapter["obligations"]),
            },
            "threads": [
                {"id": row["id"], "status": row["status"],
                 "description": row["description"],
                 "priority_score": row["priority_score"]}
                for row in rows["threads"]
            ],
            "characters": rows["characters"],
            "recent_prose": rows["recent_prose"],
            # The whole committed manuscript, for the audit's repetition guard.
            # Never rendered into a prompt and never pruned: it costs no tokens,
            # and it lets the guard catch a beat that copies a distant chapter,
            # not just one inside the drafter's recent-prose window.
            "committed_prose": rows["committed_prose"],
        }

    package["budget"] = _prune_to_budget(package, config)

    log_node_event(
        "assemble_context",
        event="context_assembled",
        beat_id=package["beat"]["id"],
        tokens=package["budget"]["tokens"],
        budget=package["budget"]["budget"],
        dropped_prose=package["budget"]["dropped_prose_passages"],
        dropped_threads=package["budget"]["dropped_threads"],
        threads=len(package["threads"]),
        recent_prose=len(package["recent_prose"]),
    )
    log_node_event("assemble_context", event="phase_change", phase=PHASE)
    await bus.publish(
        "phase_change",
        {
            "phase": PHASE,
            "node": "assemble_context",
            "project_id": project_id,
            "beat_id": package["beat"]["id"],
        },
    )

    return {"active_context_package": package}
