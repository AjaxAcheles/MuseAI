"""Module: M05 (Hierarchical Planning Cascade)

Level-4 (scene) planner node — the first-class runtime Scene Planner, new in the
five-level cascade. Consumes the active chapter's obligations + scene-planning
constraints (from the chapter ``PlanningNode``'s full-plan ``purpose``) and realizes ONE
scene per invocation as a structural plan: setting, participants, scene function,
entry/exit state, conflict turn, continuity constraints, and word budget — never prose,
never beats (the Beat Planner partitions the scene). Runs just-in-time for the active
chapter in BOTH execution modes (macro mode does not pre-generate scenes).

Flow per invocation:
  1. resolve the run's ``PlanningSnapshot`` and the scene-planning anchor — the ACTIVE
     chapter's ``PlanningNode`` (``fsm_pointer.chapter_id``), whose ``purpose`` carries
     the approved chapter plan; if that chapter has no planned node yet, record a trace
     and return (the chapter planner has not run — routing is Build 14);
  2. ``compile_planning_constraints`` against the chapter anchor — if two hard
     annotations contradict, set the clarification block and return WITHOUT the loop;
  3. assemble the scene ``base_context`` — exactly the four P1-template variables
     (``chapter_plan`` / ``planned_scenes`` / ``genre`` / ``continuity_facts``),
     degrading gracefully for not-yet-built stores;
  4. build an M04-backed decider via ``make_planner_decider`` and run
     ``run_planner_loop`` with a deterministic minimal-valid scene baseline as the
     loop's floor;
  5. map the ``LoopOutcome`` through ``persist_loop_outcome`` — whose ``persist_plan_fn``
     writes the ``Scenes`` row via the 07.00 ``upsert_scene_plan`` plus a ``level='scene'``
     ``PlanningNode`` (full validated plan JSON in ``purpose``). **``ordering`` is
     allocated explicitly and race-safely at write time**: the chapter's existing scenes
     are re-read through ``get_scenes_for_chapter_ordered`` and the new scene gets
     ``max(existing) + 1`` (``0`` when none exist), so repeated just-in-time planning
     stays strictly monotonic and gapless. Scene identity is harness-owned: a missing or
     colliding plan ``scene_id`` is replaced with a fresh derived id (an upsert may
     never silently overwrite an existing scene). On success advance
     ``fsm_pointer.scene_id`` and fold snapshot/revision/block state back.

The node makes **no** direct ``call_llm``: all model contact is through the injected
decider seam, so it is fully testable by injecting a synthetic decider. It writes no
prose or beats and wires no graph route (that is Build 14).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import core.runtime as runtime
from fsm.planning_annotations import compile_planning_constraints
from fsm.planning_loop import run_planner_loop
from fsm.planning_node_support import make_planner_decider, persist_loop_outcome
from fsm.planning_tools import PlanningToolRegistry
from memory import sqlite_db

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
_LEVEL = "scene"
_NODE_NAME = "node_plan_scene"

# Human-readable, provisional planning-block reason for a loop that exhausted its caps and
# fallback ladder without a valid plan. The actual route to failure recovery is Build 14.
_ESCALATION_BLOCK_REASON = "planning_escalation"


def _resolve_config(state: dict[str, Any]) -> Any:
    """Return the active typed config (mirrors the other planner nodes' resolution)."""
    config = state.get("app_config") or state.get("config")
    if config is not None:
        return config
    from core.config_loader import load_config

    return load_config(state.get("config_path", CONFIG_PATH))


def _parse_purpose(node: dict | None) -> dict:
    """Parse a PlanningNode's ``purpose`` JSON into a dict ({} on absent/invalid)."""
    if not node:
        return {}
    try:
        parsed = json.loads(node.get("purpose") or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _continuity_facts(db_path: Any) -> list[dict]:
    """Established continuity facts for the scene to honor.

    The design reads these from the temporal knowledge graph (Graphiti); that store is
    not built yet, so this degrades to an empty list (the template renders its
    documented empty-state branch, and the ``continuity`` validator passes trivially on
    zero facts). Wired to the real store when that module lands.
    """
    del db_path  # no continuity-fact store exists to read yet
    return []


def _planned_scenes(db_path: Any, snapshot_id: str, chapter_node_id: str) -> list[dict]:
    """The chapter's already-planned scenes, in order, as lean plan summaries.

    Reads the scene ``PlanningNode`` children of the chapter node (their ``purpose``
    carries each scene's full plan) so the next scene can chain its ``entry_state`` from
    the last scene's ``exit_state`` and avoid duplicating functions.
    """
    scenes: list[dict] = []
    for node in sqlite_db.get_planning_nodes_by_parent(db_path, snapshot_id, chapter_node_id):
        if node.get("level") != _LEVEL:
            continue
        plan = _parse_purpose(node)
        scenes.append(
            {
                "scene_id": plan.get("scene_id") or node.get("title") or node["node_id"],
                "scene_function": plan.get("scene_function") or node.get("summary") or "",
                "exit_state": plan.get("exit_state") or "",
                "ordering": node.get("ordering", 0),
            }
        )
    return scenes


def _build_scene_baseline(base_context: dict[str, Any]) -> Any:
    """Build the loop's deterministic baseline: a minimal, single-pass valid scene plan.

    The fallback floor when the model never produces a validating plan. Its shape
    matches the authoritative scene validators (``schema``/``no_drafting``/
    ``continuity``/``scene_function``/``entry_exit_state``) and the depth contract
    (non-empty ``entry_state``/``exit_state``): the entry state chains from the last
    planned scene's exit state (or the chapter's causal prerequisites for the first
    scene), the exit state differs by delivering the chapter's next obligation (no dead
    scene), and ``asserted_facts`` stays empty — the baseline asserts no new continuity
    it cannot verify.
    """
    chapter_plan = base_context.get("chapter_plan") or {}
    planned = base_context.get("planned_scenes") or []

    def _baseline(level: str, target_node: Any, constraints: dict) -> dict:
        obligations = chapter_plan.get("obligations") or {}
        if planned:
            entry = f"as the previous scene ended: {planned[-1].get('exit_state') or 'its outcome stands'}"
        else:
            prereqs = obligations.get("causal_prerequisites") or []
            entry = prereqs[0] if prereqs else "the chapter opens on its established prerequisites"
        function = (
            chapter_plan.get("dramatic_function")
            or "advance the active chapter's obligations"
        )
        scene_constraints = chapter_plan.get("scene_planning_constraints") or []
        return {
            "scene_function": f"advance the chapter: {function}",
            "setting": "the chapter's established location, continuing directly",
            "participants": ["the chapter's focal character"],
            "entry_state": entry,
            "exit_state": (
                "the scene's turn has landed: the chapter is one concrete step closer to "
                "its causal deliverables"
            ),
            "conflict_turn": "the focal character's immediate goal is advanced or blocked on the page",
            "asserted_facts": [],
            "continuity_constraints": [str(c) for c in scene_constraints],
            "word_budget": 0,
        }

    return _baseline


def _scene_persist_plan_fn(
    db_path: Any,
    snapshot_id: str,
    chapter_id: str,
    chapter_node_id: str,
    allocated: dict[str, Any],
) -> Any:
    """Return the level-specific writer: the Scenes row + the scene PlanningNode.

    ``ordering`` is derived at write time from the chapter's existing scenes
    (``max + 1``, or ``0`` for the first) — never hardcoded, never colliding — so
    repeated just-in-time planning stays strictly monotonic and gapless. Scene identity
    is harness-owned: the plan's ``scene_id`` is kept only when it does not collide with
    an existing scene; otherwise a fresh ``{chapter_id}_s{n}`` id is derived (the 07.00
    writer upserts by id, and overwriting a previously planned scene would be silent
    data loss). The persisted PlanningNode ``purpose`` carries the full validated plan
    JSON with the harness-final ``scene_id``/``ordering`` folded in (the global/arc/
    chapter convention). The allocated id is reported back through ``allocated`` so the
    node can advance ``fsm_pointer.scene_id``.
    """

    def _persist(plan: dict) -> None:
        existing = sqlite_db.get_scenes_for_chapter_ordered(db_path, chapter_id)
        orderings = [s["ordering"] for s in existing]
        ordering = (max(orderings) + 1) if orderings else 0
        existing_ids = {s["id"] for s in existing}
        scene_id = plan.get("scene_id")
        if not scene_id or scene_id in existing_ids:
            n = ordering + 1
            scene_id = f"{chapter_id}_s{n}"
            while scene_id in existing_ids:
                n += 1
                scene_id = f"{chapter_id}_s{n}"
        try:
            word_budget = max(0, int(plan.get("word_budget") or 0))
        except (TypeError, ValueError):
            word_budget = 0
        final_plan = {**plan, "scene_id": scene_id, "ordering": ordering}
        sqlite_db.upsert_scene_plan(
            db_path,
            scene_id=scene_id,
            chapter_id=chapter_id,
            description=(plan.get("scene_function") or scene_id),
            ordering=ordering,
            word_budget=word_budget,
            status="planned",
        )
        sqlite_db.upsert_planning_node(
            db_path,
            node_id=f"{snapshot_id}:scene:{scene_id}",
            snapshot_id=snapshot_id,
            level=_LEVEL,
            status="planned",
            parent_id=chapter_node_id,
            ordering=ordering,
            title=plan.get("setting") or scene_id,
            summary=plan.get("scene_function"),
            purpose=json.dumps(final_plan, sort_keys=True),
        )
        allocated["scene_id"] = scene_id
        allocated["ordering"] = ordering

    return _persist


async def node_plan_scene(
    state: dict[str, Any],
    *,
    decider: Any = None,
    registry: Any = None,
) -> dict[str, Any]:
    """Plan the active chapter's next scene and persist the validated structural plan.

    Returns the (mutated) orchestrator state. ``decider`` and ``registry`` are injectable
    seams; when ``decider`` is None the production M04-backed decider is built via
    ``make_planner_decider`` and awaited by the loop.
    """
    config = _resolve_config(state)
    db_path = state.get("sqlite_db_path", runtime.SQLITE_DB_PATH)
    event_log_path = state.get("event_log_path", runtime.EVENT_LOG_PATH)
    project_id = state["project_id"]
    trace = state.setdefault("planner_deliberation_trace", [])

    # 1. resolve the snapshot (idempotent) + the active chapter anchor.
    execution_mode = state.get("planning_execution_mode") or "macro_outline_before_draft"
    snapshot_id = state.get("planning_snapshot_id") or f"snap_{project_id}"
    sqlite_db.create_planning_snapshot(
        db_path, snapshot_id=snapshot_id, project_id=project_id, mode=execution_mode
    )
    state["planning_snapshot_id"] = snapshot_id

    pointer = state.get("fsm_pointer")
    chapter_id = getattr(pointer, "chapter_id", None) if pointer is not None else None
    chapter_node_id = f"{snapshot_id}:chapter:{chapter_id}" if chapter_id else None
    chapter_node = None
    if chapter_node_id:
        chapter_node = next(
            (
                n
                for n in sqlite_db.get_planning_nodes(db_path, snapshot_id, level="chapter")
                if n["node_id"] == chapter_node_id
            ),
            None,
        )
    chapter_plan = _parse_purpose(chapter_node)
    if not chapter_plan.get("dramatic_function"):
        # The active chapter has not been planned (no node, or still an arc-written
        # stub): the scene planner has nothing to realize. Record and return — routing
        # back through node_plan_chapter is Build 14.
        trace.append(
            {
                "level": _LEVEL,
                "phase": "resolve_anchor",
                "outcome": "no_planned_chapter",
                "chapter_id": chapter_id,
            }
        )
        return state

    # 2. compile constraints against the chapter anchor (the scene node does not exist
    # yet; scene planning inherits the chapter's compiled package) — bail to
    # clarification on a hard-vs-hard contradiction.
    constraints = compile_planning_constraints(snapshot_id, chapter_node_id, db_path=db_path)
    if constraints.get("needs_clarification"):
        state["planning_block_reason"] = constraints.get(
            "block_reason", "unresolved_hard_conflict"
        )
        trace.append(
            {
                "level": _LEVEL,
                "phase": "compile_constraints",
                "outcome": "needs_clarification",
                "hard_conflicts": constraints.get("hard_conflicts", []),
            }
        )
        return state

    # 3. assemble the scene base_context — exactly the four P1-template variables
    # (degrade gracefully for not-yet-built stores).
    project_metadata = state.get("project_metadata") or {}
    base_context = {
        "chapter_plan": chapter_plan,
        "planned_scenes": _planned_scenes(db_path, snapshot_id, chapter_node_id),
        "genre": project_metadata.get("genre", ""),
        "continuity_facts": _continuity_facts(db_path),
    }
    continuity = {"continuity_facts": base_context["continuity_facts"]}

    # 4. registry + decider (model contact only through the injected decider seam).
    registry = registry or PlanningToolRegistry(db_path)
    if decider is None:
        decider = make_planner_decider(_NODE_NAME, base_context, config)

    # 5. run the bounded deliberation loop with a deterministic baseline floor.
    outcome = await run_planner_loop(
        level=_LEVEL,
        snapshot_id=snapshot_id,
        target_node=chapter_node_id,
        constraints=constraints,
        continuity=continuity,
        registry=registry,
        config=config,
        planner_decider=decider,
        deterministic_baseline=_build_scene_baseline(base_context),
    )
    if outcome.records:
        trace.extend(outcome.records)

    # 6. map the outcome to persistence + state (never persists an invalid plan).
    allocated: dict[str, Any] = {}
    prior_nodes = sqlite_db.get_planning_nodes(db_path, snapshot_id)
    result = persist_loop_outcome(
        outcome,
        level=_LEVEL,
        snapshot_id=snapshot_id,
        target_node=chapter_node_id,
        persist_plan_fn=_scene_persist_plan_fn(
            db_path, snapshot_id, chapter_id, chapter_node_id, allocated
        ),
        db_path=db_path,
        event_log_path=event_log_path,
        prior_nodes=prior_nodes,
    )

    state["planning_snapshot_id"] = result.planning_snapshot_id
    state["active_planning_revision_id"] = result.active_planning_revision_id
    if result.outcome == "needs_clarification":
        state["planning_block_reason"] = result.planning_block_reason
    elif result.outcome == "escalate":
        # Loop exhausted caps + fallback ladder with no valid plan: signal recovery.
        # Expected to be rare — the deterministic baseline is the validated floor.
        state["planning_block_reason"] = _ESCALATION_BLOCK_REASON
    else:
        # finalized / fallback_baseline: advance the pointer to the newly planned scene
        # so the cascade can proceed to beat planning.
        if allocated.get("scene_id") and pointer is not None:
            state["fsm_pointer"] = pointer.model_copy(
                update={"scene_id": allocated["scene_id"]}
            )
    return state
