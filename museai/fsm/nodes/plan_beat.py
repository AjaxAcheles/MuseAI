"""Beat planning node.

Decomposes the pointer's active chapter — and only that chapter — into an ordered
sequence of beats, then activates the first one still to be written.

**A chapter is decomposed once.** If it already has beats this node reuses them
and never calls the model, for the same reason ``plan_chapter`` reuses chapters:
every run enters through the planners, and re-planning a chapter that already
holds committed prose would overwrite the specs that prose was written from and
send the drafter back over beats it had finished. ``beat_spec`` stores the plan
verbatim as JSON, so the reuse path reconstitutes it exactly rather than asking
the model to reinvent it.

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
import logging
import sqlite3

from museai.core.logging_setup import get_fsm_logger, log_node_event
from museai.core.stream_bus import bus
from museai.fsm.nodes.deps import PlanningError, get_node_config
from museai.fsm.pad import PAD_AXES, resolve_pad_constraint
from museai.fsm.state import FSM_Pointer, OrchestratorState
from museai.fsm.tools.web_search import TOOL_IMPLS, WEB_SEARCH_TOOL_SPEC
from museai.llm.planning import call_llm_for_json_array
from museai.llm.prompts import render_messages
from museai.memory.db import (
    connect_db,
    get_beats_for_chapter,
    get_character_emotions,
    get_characters,
    get_chapters_for_arc,
    get_recent_committed_beats,
    get_threads_for_project,
    upsert_beat,
)


def beat_id_for(chapter_id: str, ordering: int) -> str:
    """The stable id of the ``ordering``-th beat of a chapter (1-based)."""
    return f"{chapter_id}-b{ordering:02d}"


# Prefixes models glue onto the value when they echo the schema key back at us,
# e.g. `"focal_character_id": "character-id=lantern-keeper-char-1"`.
_FOCAL_PREFIXES = ("focal_character_id", "character_id", "character-id", "id")


def resolve_focal_character(raw: str, characters: list) -> str:
    """Map whatever the planner called the focal character onto a real id.

    Models answer with the seeded id, the character's *name*, a lowercased id, or
    the schema's placeholder fused to the value. All of those are recoverable.

    Returns ``""`` when nothing resolves **or when more than one character could
    match**: attributing a beat's target PAD to the wrong character silently
    corrupts that character's emotional state for the rest of the book, which is
    strictly worse than attributing it to nobody.
    """
    candidate = (raw or "").strip().strip("\"'").strip()
    if not candidate:
        return ""

    # Strip a `key=` / `key:` prefix, however the model spelled the key.
    lowered = candidate.casefold()
    for prefix in _FOCAL_PREFIXES:
        for separator in ("=", ":"):
            token = f"{prefix}{separator}"
            if lowered.startswith(token):
                candidate = candidate[len(token) :].strip().strip("\"'").strip()
                lowered = candidate.casefold()
                break

    if not candidate:
        return ""

    known_ids = [character["id"] for character in characters]
    if candidate in known_ids:
        return candidate

    by_id = {character["id"].casefold(): character["id"] for character in characters}
    if lowered in by_id:
        return by_id[lowered]

    by_name = {character["name"].strip().casefold(): character["id"] for character in characters}
    if lowered in by_name:
        return by_name[lowered]

    # Last resort: the real id is buried in a longer string. Only safe when
    # exactly one known id is in there.
    embedded = [known for known in known_ids if known.casefold() in lowered]
    if len(embedded) == 1:
        return embedded[0]

    return ""


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


def _arc_description(conn: sqlite3.Connection, arc_id: str) -> str:
    """The active arc's one-line description, for the beat planner's orientation."""
    row = conn.execute(
        "SELECT description FROM Arcs WHERE id=?", (arc_id,)
    ).fetchone()
    return (row["description"] if row and row["description"] else "").strip()


def _sibling_chapters(chapters: list[sqlite3.Row], active_id: str) -> list[dict]:
    """Every chapter of the arc, so the planner sees what its neighbours cover.

    Without this the beat planner sees only its own one-line chapter and re-covers
    ground a sibling already owns — the pacing loop. Each entry says whether it is
    the chapter being planned now and whether it is already finished.
    """
    return [
        {
            "ordering": row["ordering"],
            "description": row["description"] or "",
            "obligations": _chapter_obligations(row["obligations"]),
            "is_current": row["id"] == active_id,
            "status": row["status"],
        }
        for row in chapters
    ]


def _already_dramatized(
    conn: sqlite3.Connection, chapters: list[sqlite3.Row], active_id: str
) -> list[dict]:
    """Intents of beats already planned in earlier chapters.

    A compact "here is what the story has already dramatized" list, so the
    planner does not re-stage a scene (e.g. a confession) that an earlier chapter
    already delivered. Reads intents straight from ``beat_spec``; no re-derivation.
    """
    dramatized: list[dict] = []
    for row in chapters:
        if row["id"] == active_id:
            continue
        beats = get_beats_for_chapter(conn, row["id"])
        intents = []
        for beat in beats:
            if not beat["beat_spec"]:
                continue
            try:
                spec = json.loads(beat["beat_spec"])
            except json.JSONDecodeError:
                continue
            intent = str(spec.get("intent") or "").strip()
            if intent:
                intents.append(intent)
        if intents:
            dramatized.append(
                {
                    "ordering": row["ordering"],
                    "description": row["description"] or "",
                    "intents": intents,
                }
            )
    return dramatized


def _thread_context(rows: list[sqlite3.Row]) -> list[dict]:
    """All threads with status, so the planner can advance and stop re-opening them."""
    return [
        {
            "id": row["id"],
            "status": row["status"],
            "description": row["description"],
        }
        for row in rows
    ]


def _intended_refrain(raw: object) -> list[str]:
    """Normalize a planner's ``intended_refrain`` (string or list) to a phrase list."""
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def _thread_updates(item: dict) -> list[dict]:
    """Validate a beat's ``thread_updates`` into ``{id, status}`` dicts.

    The commit node's ``_apply_thread_updates`` reads exactly this shape off the
    stored ``beat_spec``. It already rejects backwards transitions and unknown
    ids, so here we only keep well-formed entries with a legal status.
    """
    raw = item.get("thread_updates")
    if not raw or not isinstance(raw, list):
        return []
    updates: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        thread_id = str(entry.get("id") or entry.get("thread_id") or "").strip()
        status = str(entry.get("status") or "").strip()
        if not thread_id or status not in ("open", "progressing", "closed"):
            continue
        update = {"id": thread_id, "status": status}
        if "priority_score" in entry:
            update["priority_score"] = entry["priority_score"]
        if "description" in entry:
            update["description"] = str(entry["description"])
        updates.append(update)
    return updates


# Sent when a plan comes back with almost every beat at high arousal. It keeps
# the beats and their work — only the emotional shaping is asked to change — so a
# varied arc is a re-weighting, not a re-plan.
_INTENSITY_CORRECTION = (
    "Your previous plan puts nearly every beat at high emotional arousal. Holding "
    "the intensity at maximum flattens the chapter and leaves the climax nowhere "
    "to rise to. Revise so the intensity varies: give the chapter quieter, "
    "lower-arousal beats between its peaks, and reserve the highest arousal for "
    "the single turning point. Keep the same beats, intents, and obligations — "
    "only reshape the target_pad arousal values into an arc. Return the full JSON "
    "array again."
)


def _beat_arousal(item: dict) -> float:
    """A planned beat's target arousal, or 0.0 when it is missing or unreadable."""
    pad = item.get("target_pad")
    if not isinstance(pad, dict):
        return 0.0
    try:
        return float(pad.get("arousal", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _is_flat_hot(planned: list[dict], *, hot_threshold: float, flat_fraction: float) -> bool:
    """True when so many beats sit at high arousal that the arc has no valleys.

    A chapter of one or two beats has no arc to shape, so it is never flagged.
    """
    if len(planned) < 3:
        return False
    hot = sum(1 for item in planned if abs(_beat_arousal(item)) >= hot_threshold)
    return hot / len(planned) > flat_fraction


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


def _row(beat: dict) -> dict:
    """The beat's column values, minus the spec kept for the emitted events."""
    return {key: value for key, value in beat.items() if not key.startswith("_")}


def _stored_beats(rows: list[sqlite3.Row]) -> list[dict]:
    """Rebuild the planner's beat dicts from rows this node wrote earlier.

    ``beat_spec`` is the exact dict that was planned, serialised — intent, entry
    and exit state, target PAD, focal character — so nothing is re-derived.
    """
    beats: list[dict] = []
    for row in rows:
        beats.append(
            {
                "id": row["id"],
                "chapter_id": row["chapter_id"],
                "ordering": row["ordering"],
                "beat_spec": row["beat_spec"],
                "pad_constraint": row["pad_constraint"],
                "_spec": json.loads(row["beat_spec"]) if row["beat_spec"] else {},
            }
        )
    return beats


def _first_unfinished(rows: list[sqlite3.Row]) -> int:
    """Index of the first beat still to write, or the last one if the chapter is done."""
    for index, row in enumerate(rows):
        if row["status"] != "completed":
            return index
    return len(rows) - 1


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
        existing = get_beats_for_chapter(conn, chapter["id"])
        reused = bool(existing)

        if reused:
            beats = _stored_beats(existing)
            active_index = _first_unfinished(existing)
            active = beats[active_index]
            chapter_done = existing[active_index]["status"] == "completed"
        else:
            active_index = 0
            chapter_done = False
            characters = _character_context(conn, project_id)
            recent = get_recent_committed_beats(
                conn, project_id, config.generation.recent_prose_beats
            )
            all_chapters = get_chapters_for_arc(conn, chapter["arc_id"])
            siblings = _sibling_chapters(all_chapters, chapter["id"])
            dramatized = _already_dramatized(conn, all_chapters, chapter["id"])
            threads = _thread_context(get_threads_for_project(conn, project_id))
            log_node_event(
                "plan_beat",
                event="context_assembled",
                chapter_id=chapter["id"],
                characters=len(characters),
                recent_prose_beats=len(recent),
                sibling_chapters=len(siblings),
                dramatized_chapters=len(dramatized),
                threads=len(threads),
            )

            messages = render_messages(
                "beat_planner",
                chapter={
                    "description": chapter["description"],
                    "obligations": _chapter_obligations(chapter["obligations"]),
                },
                story_position={
                    "arc_description": _arc_description(conn, chapter["arc_id"]),
                    "chapter_ordering": chapter["ordering"],
                    "chapter_count": len(all_chapters),
                },
                sibling_chapters=siblings,
                already_dramatized=dramatized,
                threads=threads,
                characters=characters,
                recent_prose=[row["prose"] for row in recent],
            )
            planned = await call_llm_for_json_array(
                config.endpoint,
                messages,
                what="beats",
                agent="beat_planner",
                node="plan_beat",
                retries=config.generation.planner_parse_retries,
                tools=[WEB_SEARCH_TOOL_SPEC],
                tool_impls=TOOL_IMPLS,
                max_tool_iterations=config.generation.max_agent_iterations,
            )

            # If the plan comes back with the emotional register pinned at
            # maximum, re-prompt once (bounded) for a varied arc, then accept
            # whatever comes back. This is a semantic retry, separate from the
            # JSON-parse ladder inside call_llm_for_json_array.
            for _ in range(config.generation.planner_intensity_retries):
                if not _is_flat_hot(
                    planned,
                    hot_threshold=config.generation.intensity_hot_threshold,
                    flat_fraction=config.generation.intensity_flat_fraction,
                ):
                    break
                log_node_event(
                    "plan_beat",
                    level=logging.WARNING,
                    event="intensity_reprompt",
                    chapter_id=chapter["id"],
                    beats=len(planned),
                )
                await bus.publish(
                    "planner_intensity",
                    {"chapter_id": chapter["id"], "beats": len(planned)},
                )
                messages = [
                    *messages,
                    {"role": "assistant", "content": json.dumps(planned, ensure_ascii=False)},
                    {"role": "user", "content": _INTENSITY_CORRECTION},
                ]
                planned = await call_llm_for_json_array(
                    config.endpoint,
                    messages,
                    what="beats",
                    agent="beat_planner",
                    node="plan_beat",
                    retries=config.generation.planner_parse_retries,
                    tools=[WEB_SEARCH_TOOL_SPEC],
                    tool_impls=TOOL_IMPLS,
                    max_tool_iterations=config.generation.max_agent_iterations,
                )

            beats = []
            for ordering, item in enumerate(planned, start=1):
                intent = str(item.get("intent") or "").strip()
                if not intent:
                    raise PlanningError(f"beat {ordering} has no intent")

                target_pad = _target_pad(item, ordering)
                raw_focal = str(item.get("focal_character_id") or "").strip()
                focal = resolve_focal_character(raw_focal, characters)
                if raw_focal and not focal:
                    logger.warning(
                        "node=plan_beat beat %d names unknown focal character %r; "
                        "PAD will not be attributed",
                        ordering,
                        raw_focal,
                    )

                spec = {
                    "intent": intent,
                    "entry_state": str(item.get("entry_state") or "").strip(),
                    "exit_state": str(item.get("exit_state") or "").strip(),
                    "target_pad": target_pad,
                    "focal_character_id": focal,
                }
                # Thread advances the beat declares — consumed unchanged by the
                # commit node. Refrains the beat is permitted to repeat verbatim —
                # read by the repetition audit. Both stored only when non-empty,
                # so a plain beat's spec is unchanged from before.
                thread_updates = _thread_updates(item)
                if thread_updates:
                    spec["thread_updates"] = thread_updates
                intended_refrain = _intended_refrain(item.get("intended_refrain"))
                if intended_refrain:
                    spec["intended_refrain"] = intended_refrain
                beats.append(
                    {
                        "id": beat_id_for(chapter["id"], ordering),
                        "chapter_id": chapter["id"],
                        "ordering": ordering,
                        "beat_spec": json.dumps(spec, ensure_ascii=False),
                        "pad_constraint": resolve_pad_constraint(
                            *(target_pad[axis] for axis in PAD_AXES)
                        ),
                        "_spec": spec,
                    }
                )
            active = beats[0]

        with conn:
            if not reused:
                for beat in beats:
                    upsert_beat(conn, status="planned", **_row(beat))
            if not chapter_done:
                upsert_beat(conn, status="active", **_row(active))
    finally:
        conn.close()

    log_node_event(
        "plan_beat",
        event="beats_planned",
        chapter_id=chapter["id"],
        beats=len(beats),
        active_beat_id=active["id"],
        active_beat_index=active_index,
        reused=reused,
    )
    await bus.publish(
        "beats_planned",
        {
            "chapter_id": chapter["id"],
            "beat_count": len(beats),
            "active_beat_id": active["id"],
            "reused": reused,
            "beats": [
                {
                    "id": beat["id"],
                    "ordering": beat["ordering"],
                    "intent": beat["_spec"]["intent"],
                }
                for beat in beats
            ],
        },
    )

    # A refrain the planner declared is a licence to repeat prose verbatim, which
    # the repetition audit will honour. It must never be silent: surface every one
    # so a human can see it and, if it is being abused to launder copied prose,
    # remove it. Only the planner writes these, and only on a fresh plan.
    if not reused:
        for beat in beats:
            refrains = beat["_spec"].get("intended_refrain") or []
            for phrase in refrains:
                log_node_event(
                    "plan_beat",
                    event="refrain_declared",
                    beat_id=beat["id"],
                    phrase=phrase[:120],
                )
                await bus.publish(
                    "planner_refrain",
                    {"beat_id": beat["id"], "chapter_id": chapter["id"], "phrase": phrase},
                )

    # The PAD target of a beat is applied when that beat is first planned. Reusing
    # a stored plan must not re-apply it: the character has since lived through
    # the beats that followed, and re-publishing an old target would drag their
    # emotional state backwards to where the run had already left it.
    focal_character_id = active["_spec"].get("focal_character_id")
    if not reused and focal_character_id:
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
                "beat_index": active_index,
                "target_pad": active["_spec"]["target_pad"],
            },
        )
    elif not reused:
        logger.warning(
            "node=plan_beat no character to attribute the target PAD of beat %s to",
            active["id"],
        )

    return {
        "fsm_pointer": FSM_Pointer(
            arc_id=pointer.arc_id, chapter_id=chapter["id"], beat_index=active_index
        )
    }
