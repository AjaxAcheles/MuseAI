"""Beat planning node.

Decomposes the pointer's active chapter — and only that chapter — into an ordered
sequence of beats, then activates the first one.

Each beat's ``pad_constraint`` is a pure deterministic lookup: the target PAD
coordinate quantizes into one of 27 regions and
:func:`museai.fsm.pad.resolve_pad_constraint` returns that region's authored
behavioural constraint, which the drafter injects verbatim. No model is involved
in the translation, so the same coordinate always produces the same constraint.

The one LLM call this node makes is the beat plan itself. It sends no
``max_tokens``: a reasoning-style endpoint bills its hidden reasoning against
that budget, and a plan truncated mid-array parses to nothing.
"""

from __future__ import annotations

import json
import sqlite3

from museai.core.logging_setup import get_fsm_logger, log_node_event
from museai.core.stream_bus import bus
from museai.fsm.nodes.deps import PlanningError, get_node_config
from museai.fsm.pad import PAD_AXES, resolve_pad_constraint
from museai.fsm.state import FSM_Pointer, OrchestratorState
from museai.llm.client import call_llm
from museai.llm.prompts import render_messages
from museai.llm.structured import parse_json_array
from museai.memory.db import (
    connect_db,
    get_character_emotions,
    get_characters,
    get_chapters_for_arc,
    get_open_threads,
    get_recent_committed_beats,
    upsert_beat,
)


def beat_id_for(chapter_id: str, ordering: int) -> str:
    """The stable id of the ``ordering``-th beat of a chapter (1-based)."""
    return f"{chapter_id}-b{ordering:02d}"


def _resolve_chapter(conn: sqlite3.Connection, pointer: FSM_Pointer) -> sqlite3.Row:
    """The chapter the pointer names, or the arc's active chapter."""
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


def _chapter_obligations(raw: str | None) -> list[str]:
    """Read back the obligations column, which ``plan_chapter`` writes as JSON."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return [raw]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def _character_context(conn: sqlite3.Connection, project_id: str) -> list[dict]:
    """Characters with their current PAD; an unrecorded PAD reads as the origin."""
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


def _target_pad(item: dict, ordering: int) -> dict[str, float]:
    """Validate a beat's target PAD, clamped to the axes' [-1.0, 1.0] range."""
    raw = item.get("target_pad") or {}
    if not isinstance(raw, dict):
        raise PlanningError(
            f"beat {ordering}: target_pad must be a JSON object, "
            f"got {type(raw).__name__}"
        )

    target: dict[str, float] = {}
    for axis in PAD_AXES:
        value = raw.get(axis, 0.0)
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise PlanningError(
                f"beat {ordering}: target_pad.{axis} is not a number: {value!r}"
            ) from exc
        target[axis] = max(-1.0, min(1.0, value))
    return target


def _word_target(item: dict, default: int) -> int:
    """A beat's word target, falling back to the configured beat target."""
    try:
        value = int(item.get("word_target") or default)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _row(beat: dict) -> dict:
    """The beat's column values, minus the spec kept for the emitted events."""
    return {key: value for key, value in beat.items() if not key.startswith("_")}


async def plan_beat(state: OrchestratorState) -> dict:
    """Plan the beats of the active chapter and activate the first one.

    Returns the state delta: a pointer at beat index 0 of that chapter.
    """
    config = get_node_config()
    pointer = state["fsm_pointer"]
    project_id = state["project_id"]
    logger = get_fsm_logger()

    log_node_event("plan_beat", event="start", arc_id=pointer.arc_id,
                   chapter_id=pointer.chapter_id or "(active)")

    conn = connect_db(config.db_path)
    try:
        chapter = _resolve_chapter(conn, pointer)
        characters = _character_context(conn, project_id)
        recent = get_recent_committed_beats(
            conn, project_id, config.generation.recent_prose_beats
        )
        log_node_event(
            "plan_beat",
            event="context_assembled",
            chapter_id=chapter["id"],
            characters=len(characters),
            recent_prose_beats=len(recent),
        )

        messages = render_messages(
            "beat_planner",
            chapter={
                "description": chapter["description"],
                "obligations": _chapter_obligations(chapter["obligations"]),
            },
            threads=[
                {"id": row["id"], "status": row["status"], "description": row["description"]}
                for row in get_open_threads(conn, project_id)
            ],
            characters=characters,
            recent_prose=[row["prose"] for row in recent],
            beat_word_target=config.generation.beat_word_target,
        )

        response = await call_llm(config.endpoint, messages)
        planned = parse_json_array(response.text, what="beats")

        known_character_ids = {character["id"] for character in characters}
        beats: list[dict] = []
        for ordering, item in enumerate(planned, start=1):
            intent = str(item.get("intent") or "").strip()
            if not intent:
                raise PlanningError(f"beat {ordering} has no intent")

            target_pad = _target_pad(item, ordering)
            focal = str(item.get("focal_character_id") or "").strip()
            if focal not in known_character_ids:
                focal = next(iter(sorted(known_character_ids)), "")

            spec = {
                "intent": intent,
                "entry_state": str(item.get("entry_state") or "").strip(),
                "exit_state": str(item.get("exit_state") or "").strip(),
                "target_pad": target_pad,
                "focal_character_id": focal,
            }
            beats.append(
                {
                    "id": beat_id_for(chapter["id"], ordering),
                    "chapter_id": chapter["id"],
                    "ordering": ordering,
                    "beat_spec": json.dumps(spec, ensure_ascii=False),
                    "pad_constraint": resolve_pad_constraint(
                        *(target_pad[axis] for axis in PAD_AXES)
                    ),
                    "word_target": _word_target(item, config.generation.beat_word_target),
                    "_spec": spec,
                }
            )

        active = beats[0]
        with conn:
            for beat in beats:
                upsert_beat(conn, status="planned", **_row(beat))
            upsert_beat(conn, status="active", **_row(active))
    finally:
        conn.close()

    log_node_event(
        "plan_beat",
        event="beats_planned",
        chapter_id=chapter["id"],
        beats=len(beats),
        active_beat_id=active["id"],
        word_target_total=sum(beat["word_target"] for beat in beats),
    )
    await bus.publish(
        "beats_planned",
        {
            "chapter_id": chapter["id"],
            "beat_count": len(beats),
            "active_beat_id": active["id"],
            "beats": [
                {
                    "id": beat["id"],
                    "ordering": beat["ordering"],
                    "intent": beat["_spec"]["intent"],
                    "word_target": beat["word_target"],
                }
                for beat in beats
            ],
        },
    )

    focal_character_id = active["_spec"]["focal_character_id"]
    if focal_character_id:
        log_node_event(
            "plan_beat",
            event="pad_update",
            beat_id=active["id"],
            character_id=focal_character_id,
            **active["_spec"]["target_pad"],
        )
        await bus.publish(
            "pad_update",
            {
                "character_id": focal_character_id,
                "beat_id": active["id"],
                "beat_index": 0,
                "target_pad": active["_spec"]["target_pad"],
            },
        )
    else:
        logger.warning(
            "node=plan_beat no character to attribute the target PAD of beat %s to",
            active["id"],
        )

    return {
        "fsm_pointer": FSM_Pointer(
            arc_id=pointer.arc_id, chapter_id=chapter["id"], beat_index=0
        )
    }
