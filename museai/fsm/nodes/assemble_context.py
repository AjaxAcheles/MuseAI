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

import json
import sqlite3

from museai.core.logging_setup import get_fsm_logger, log_node_event
from museai.core.stream_bus import bus
from museai.fsm.nodes.deps import PlanningError, get_node_config
from museai.fsm.pad import PAD_AXES
from museai.fsm.state import FSM_Pointer, OrchestratorState
from museai.llm.prompts import render_messages
from museai.llm.tokenizer import count_message_tokens
from museai.memory.db import (
    connect_db,
    get_beats_for_chapter,
    get_character_emotions,
    get_characters,
    get_chapters_for_arc,
    get_open_threads,
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
        chapter=package["chapter"],
        threads=package["threads"],
        characters=package["characters"],
        recent_prose=package["recent_prose"],
    )


def _count(package: dict, config) -> int:
    endpoint = config.endpoint
    return count_message_tokens(
        drafter_messages(package), endpoint.tokenizer_family, endpoint.model_name
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


def _prune_to_budget(package: dict, config) -> dict:
    """Drop context until the rendered prompt fits ``context_token_budget``.

    Returns a report of what was dropped and the final token count.
    """
    budget = config.generation.context_token_budget
    before = _count(package, config)
    dropped_prose = 0
    dropped_threads = 0

    # 1. Oldest recent prose first — the newest passage is what the beat continues.
    while _count(package, config) > budget and package["recent_prose"]:
        package["recent_prose"].pop(0)
        dropped_prose += 1

    # 2. Then the lowest-priority open threads. ``get_open_threads`` returns them
    #    already sorted by priority_score descending, so the tail is cheapest.
    while _count(package, config) > budget and package["threads"]:
        package["threads"].pop()
        dropped_threads += 1

    after = _count(package, config)
    if after > budget:
        get_fsm_logger().warning(
            "node=assemble_context protected context exceeds budget: "
            "tokens=%d budget=%d beat_id=%s; drafting over budget",
            after,
            budget,
            package["beat"]["id"],
        )

    return {
        "budget": budget,
        "tokens_before": before,
        "tokens": after,
        "dropped_prose_passages": dropped_prose,
        "dropped_threads": dropped_threads,
        "over_budget": after > budget,
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

    conn = connect_db(config.db_path)
    try:
        chapter = _resolve_chapter(conn, pointer)
        beat = _resolve_beat(conn, chapter["id"], pointer.beat_index)
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
                "focal_character_id": spec.get("focal_character_id", ""),
                # Phrases this beat's planner declared may recur verbatim. The
                # repetition audit exempts a paragraph matching one of these, so a
                # deliberate refrain is not faulted as a copy. Only the planner
                # writes this — never the drafter.
                "intended_refrain": _intended_refrain(spec.get("intended_refrain")),
            },
            "pad_constraint": beat["pad_constraint"],
            "chapter": {
                "id": chapter["id"],
                "description": chapter["description"],
                "obligations": _obligations(chapter["obligations"]),
            },
            "threads": [
                {"id": row["id"], "status": row["status"],
                 "description": row["description"],
                 "priority_score": row["priority_score"]}
                for row in get_open_threads(conn, project_id)
            ],
            "characters": _characters(conn, project_id),
            "recent_prose": [
                row["prose"]
                for row in get_recent_committed_beats(
                    conn, project_id, config.generation.recent_prose_beats
                )
            ],
        }
    finally:
        conn.close()

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
