"""Module: M05 (Hierarchical Planning Cascade) — local end-to-end scaffold

Deterministic minimal structure planner: lays down chapters -> scenes -> beats
between the real arc planner (``node_plan_arc``) and drafting, so the local
end-to-end path has concrete beats to draft.

THIS IS A SCAFFOLD, NOT THE PRODUCTION CASCADE. The full ``node_plan_chapter`` and
``node_plan_beat`` deliberation planners (bounded loops, validators, PAD
translation) remain stubs; this node stands in for them with a fixed, model-free
structure so the pipeline can run before those are built. It:

  1. reads the active arc (``fsm_pointer.arc_id``) and its planned chapters;
  2. sizes the beat count from the word-count target and a per-beat word budget;
  3. persists Scenes and (planning) Beats rows via the plan-time SQLite writers;
  4. populates ``beat_plan_by_id`` / ``beat_order`` (the draft/commit contract) and
     points ``fsm_pointer`` at the first beat.

It writes no prose and makes no model calls. Returns the mutated state.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import core.runtime as runtime
from memory import sqlite_db

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

# Fallback per-beat word budget when project metadata does not override it. Drives
# how many beats a given word-count target is split into.
_DEFAULT_BEAT_WORD_TARGET = 250


def _resolve_config(state: dict[str, Any]) -> Any:
    """Return the active typed config (mirrors the planner nodes' resolution)."""
    config = state.get("app_config") or state.get("config")
    if config is not None:
        return config
    from core.config_loader import load_config

    return load_config(state.get("config_path", CONFIG_PATH))


def _resolve_chapters(db_path: Any, arc_id: str) -> list[dict]:
    """Return planned chapters under ``arc_id``, synthesizing the parent rows if none.

    Foreign keys are enforced (``Chapters.arc_id -> Arcs.id``,
    ``Scenes.chapter_id -> Chapters.id``), so when the arc has no planned chapters
    yet (the arc planner ran a minimal path, or this node is driven standalone) we
    must materialize an Arc row and a Chapter row before scenes can reference them.
    All writers are idempotent; the real arc-planned chapters are used untouched
    when they exist (we never clobber a real arc's description).
    """
    try:
        chapters = sqlite_db.get_chapters_for_arc(db_path, arc_id)
    except Exception:  # noqa: BLE001 - degrade if the store is unavailable
        chapters = []
    if chapters:
        return chapters
    sqlite_db.upsert_arc_plan(
        db_path, arc_id=arc_id, description="(local scaffold arc)", status="planned"
    )
    chapter_id = f"{arc_id}_ch1"
    sqlite_db.upsert_chapter_plan(
        db_path, chapter_id=chapter_id, arc_id=arc_id, description="", status="planned"
    )
    return [{"id": chapter_id, "description": ""}]


def _beat_objective(index: int, total: int, chapter_desc: str) -> str:
    """A generic-but-directed objective for beat ``index`` of ``total``."""
    focus = chapter_desc.strip() or "advance the story"
    if index == 0:
        return f"Open the story: establish the situation and hook, and {focus}."
    if index == total - 1:
        return f"Bring this stretch to a satisfying close while you {focus}."
    return f"Continue and escalate: {focus} (beat {index + 1} of {total})."


async def node_plan_structure(state: dict[str, Any]) -> dict[str, Any]:
    """Lay down a minimal chapters/scenes/beats structure and set the first beat."""
    config = _resolve_config(state)
    db_path = state.get("sqlite_db_path", runtime.SQLITE_DB_PATH)
    snapshot_id = state.get("planning_snapshot_id") or f"snap_{state['project_id']}"
    pointer = state["fsm_pointer"]
    arc_id = getattr(pointer, "arc_id", "") or "arc_1"
    project_metadata = state.get("project_metadata") or {}

    beat_target = int(project_metadata.get("beat_word_target") or _DEFAULT_BEAT_WORD_TARGET)
    target_words = int(
        project_metadata.get("target_word_count")
        or getattr(config.runtime, "word_count_target", 2000)
    )
    beats_per_scene = max(1, int(getattr(config.runtime, "beats_per_scene_min", 3)))
    total_beats = max(beats_per_scene, round(target_words / max(1, beat_target)))

    chapters = _resolve_chapters(db_path, arc_id)
    scene_count = max(1, math.ceil(total_beats / beats_per_scene))

    # Distribute scenes across the available chapters round-robin, then fill each
    # scene with beats until the total beat budget is spent.
    beat_plan_by_id: dict[str, dict[str, Any]] = {}
    beat_order: list[str] = []
    first_scene_id = ""
    remaining = total_beats
    scene_ordinal = 0
    for s in range(scene_count):
        if remaining <= 0:
            break
        chapter = chapters[s % len(chapters)]
        chapter_id = chapter.get("id") or chapter.get("chapter_id") or f"{arc_id}_ch1"
        chapter_desc = str(chapter.get("description") or chapter.get("stub") or "")
        scene_id = f"{chapter_id}_sc{scene_ordinal + 1}"
        scene_ordinal += 1
        if not first_scene_id:
            first_scene_id = scene_id
        this_scene_beats = min(beats_per_scene, remaining)
        sqlite_db.upsert_scene_plan(
            db_path,
            scene_id=scene_id,
            chapter_id=chapter_id,
            description=chapter_desc or f"Scene {scene_ordinal}",
            ordering=s,
            word_budget=this_scene_beats * beat_target,
        )
        for b in range(this_scene_beats):
            global_index = len(beat_order)
            beat_id = f"{scene_id}_b{b + 1}"
            sqlite_db.upsert_beat_plan(
                db_path,
                beat_id=beat_id,
                scene_id=scene_id,
                beat_index=b,
                snapshot_id=snapshot_id,
                node_id=f"{snapshot_id}:beat:{beat_id}",
                immediate_objective=_beat_objective(global_index, total_beats, chapter_desc),
            )
            beat_plan_by_id[beat_id] = {
                "scene_id": scene_id,
                "chapter_id": chapter_id,
                "beat_index": b,
                "objective": _beat_objective(global_index, total_beats, chapter_desc),
                "scene_description": chapter_desc,
                "pad_constraint": "",
                "physical_constraints": "",
                "target_words": beat_target,
            }
            beat_order.append(beat_id)
        remaining -= this_scene_beats

    state["beat_plan_by_id"] = beat_plan_by_id
    state["beat_order"] = beat_order
    if beat_order:
        first = beat_plan_by_id[beat_order[0]]
        state["fsm_pointer"] = pointer.model_copy(
            update={
                "chapter_id": first["chapter_id"],
                "scene_id": first["scene_id"],
                "beat_index": 0,
                "beat_id": beat_order[0],
            }
        )
    return state
