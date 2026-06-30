"""Module: M05 (Hierarchical Planning Cascade)

Level-2 (arc) planner node. Expands the approved global plan into multi-chapter arcs by
running the shared bounded deliberation loop and persisting only a validator-passed arc
plan.

Flow per invocation:
  1. resolve the run's ``PlanningSnapshot`` and the arc-planning anchor (the global
     ``PlanningNode``, whose ``purpose`` carries the approved global plan to realize);
  2. ``compile_planning_constraints`` — if two hard annotations contradict, set the
     clarification block and return WITHOUT running the loop;
  3. assemble the arc ``base_context`` from the approved global plan + permitted arc-level
     reads (open-thread priority queue, RAPTOR root summary), degrading gracefully for
     not-yet-built stores;
  4. build an M04-backed decider via ``make_planner_decider`` and run ``run_planner_loop``
     with a deterministic minimal-valid arc baseline as the loop's floor;
  5. map the ``LoopOutcome`` through ``persist_loop_outcome`` — whose ``persist_plan_fn``
     writes the ``Arcs`` rows, the chapter stubs (each with a monotonic, gapless ordering
     within its arc), and the arc/chapter ``PlanningNode`` proposal-surface rows — then,
     on success, advance ``fsm_pointer.arc_id`` and fold snapshot/revision/block state back.

The node makes **no** direct ``call_llm``: all model contact is through the injected decider
seam, so it is fully testable by injecting a synthetic decider. It writes no prose and wires
no graph route / approval edge (that is Build 14).
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
_LEVEL = "arc"
_NODE_NAME = "node_plan_arc"

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


def _raptor_root_summary(db_path: Any) -> str:
    """Best-effort RAPTOR root summary; "" while clustering (memory/raptor) is a stub."""
    try:
        roots = sqlite_db.get_raptor_nodes_by_parent(db_path, None)
    except Exception:  # noqa: BLE001 - degrade to empty when the store is unavailable
        return ""
    for node in roots:
        summary = node.get("summary")
        if summary:
            return str(summary)
    return ""


def _build_arc_baseline(base_context: dict[str, Any]) -> Any:
    """Build the loop's deterministic baseline: a minimal, single-pass valid arc plan.

    The fallback floor when the model never produces a validating plan. Its shape matches the
    authoritative arc validators (``schema``/``escalation``/``thread_distribution`` in
    ``fsm/planning_validators.py``): a top-level ``milestones`` list with non-decreasing
    integer ``tension`` and a non-empty ``thread_distribution``. It realizes the approved
    global plan's arc slots, giving each arc a DISTINCT function and chapter stubs.
    """
    global_plan = base_context.get("global_plan") or {}
    global_arcs = global_plan.get("arcs") or []
    open_threads = base_context.get("open_threads") or []
    if not global_arcs:
        global_arcs = [{"arc_id": "arc_1"}]

    def _baseline(level: str, target_node: Any, constraints: dict) -> dict:
        arcs: list[dict] = []
        milestones: list[dict] = []
        for i, ga in enumerate(global_arcs):
            arc_id = ga.get("arc_id") or ga.get("id") or f"arc_{i + 1}"
            arcs.append(
                {
                    "arc_id": arc_id,
                    "title": ga.get("title") or f"Arc {i + 1}",
                    "function": f"advance the story through stage {i + 1} of {len(global_arcs)}",
                    "character_milestones": [f"the protagonist is changed by arc {i + 1}"],
                    "chapters": [
                        {"chapter_id": f"{arc_id}_ch1", "stub": f"open and develop {arc_id}"},
                        {"chapter_id": f"{arc_id}_ch2", "stub": f"turn and hand off {arc_id}"},
                    ],
                }
            )
            milestones.append(
                {"arc_id": arc_id, "label": f"turning point of arc {i + 1}", "tension": i + 1}
            )
        arc_ids = [a["arc_id"] for a in arcs]
        if open_threads:
            thread_distribution = [
                {
                    "thread_id": (t.get("id") or t.get("thread_id") or f"thread_{k + 1}")
                    if isinstance(t, dict)
                    else str(t),
                    "lifecycle": "progress",
                    "in_arcs": arc_ids,
                }
                for k, t in enumerate(open_threads)
            ]
        else:
            thread_distribution = [
                {"thread_id": "main", "lifecycle": "open->progress->close", "in_arcs": arc_ids}
            ]
        return {"arcs": arcs, "milestones": milestones, "thread_distribution": thread_distribution}

    return _baseline


def _arc_persist_plan_fn(db_path: Any, snapshot_id: str, global_node_id: str) -> Any:
    """Return the level-specific writer: Arcs rows + chapter stubs + arc/chapter PlanningNodes.

    Each arc becomes an ``Arcs`` row and a ``level='arc'`` ``PlanningNode`` parented to the
    global node, carrying the validated arc content as JSON in ``purpose``. Each of the arc's
    chapter stubs becomes a minimal ``Chapters`` row plus a ``level='chapter'`` PlanningNode
    (obligations left for ``node_plan_chapter``), assigned a monotonic, gapless ``ordering``
    within its arc. ``persist_loop_outcome`` reads the node set back to compute the revision.
    """

    def _persist(plan: dict) -> None:
        for i, arc in enumerate(plan.get("arcs", [])):
            arc_id = arc.get("arc_id")
            if not arc_id:
                continue
            arc_node_id = f"{snapshot_id}:arc:{arc_id}"
            sqlite_db.upsert_arc_plan(
                db_path,
                arc_id=arc_id,
                description=(arc.get("function") or arc.get("title") or ""),
                status="planned",
            )
            sqlite_db.upsert_planning_node(
                db_path,
                node_id=arc_node_id,
                snapshot_id=snapshot_id,
                level=_LEVEL,
                status="planned",
                parent_id=global_node_id,
                ordering=i,
                title=arc.get("title"),
                summary=arc.get("function"),
                purpose=json.dumps(arc, sort_keys=True),
            )
            for j, chapter in enumerate(arc.get("chapters", [])):
                chapter_id = chapter.get("chapter_id")
                if not chapter_id:
                    continue
                sqlite_db.upsert_chapter_plan(
                    db_path,
                    chapter_id=chapter_id,
                    arc_id=arc_id,
                    description=(chapter.get("stub") or ""),
                    status="planned",
                    snapshot_id=snapshot_id,
                    node_id=f"{snapshot_id}:chapter:{chapter_id}",
                    parent_node_id=arc_node_id,
                    ordering=j,  # monotonic, gapless within the arc
                )

    return _persist


async def node_plan_arc(
    state: dict[str, Any],
    *,
    decider: Any = None,
    registry: Any = None,
) -> dict[str, Any]:
    """Plan the book's arcs from the approved global plan and persist the validated arc plan.

    Returns the (mutated) orchestrator state. ``decider`` and ``registry`` are injectable
    seams; when ``decider`` is None the production M04-backed decider is built via
    ``make_planner_decider`` and awaited by the loop.
    """
    config = _resolve_config(state)
    db_path = state.get("sqlite_db_path", runtime.SQLITE_DB_PATH)
    event_log_path = state.get("event_log_path", runtime.EVENT_LOG_PATH)
    project_id = state["project_id"]
    trace = state.setdefault("planner_deliberation_trace", [])

    # 1. resolve the snapshot + the arc-planning anchor (the global PlanningNode).
    snapshot_id = state.get("planning_snapshot_id") or f"snap_{project_id}"
    mode = state.get("planning_execution_mode") or getattr(
        config.planning, "execution_mode", "macro_outline_before_draft"
    )
    sqlite_db.create_planning_snapshot(
        db_path, snapshot_id=snapshot_id, project_id=project_id, mode=mode
    )
    state["planning_snapshot_id"] = snapshot_id

    global_nodes = sqlite_db.get_planning_nodes(db_path, snapshot_id, level="global")
    if global_nodes:
        global_node_id = global_nodes[0]["node_id"]
        approved_global_plan = json.loads(global_nodes[0].get("purpose") or "{}")
    else:
        # Global not planned yet — create a minimal anchor so arc planning can still proceed
        # (and the annotation compiler has a target). Does not fabricate a global plan.
        global_node_id = f"{snapshot_id}:global"
        sqlite_db.upsert_planning_node(
            db_path, node_id=global_node_id, snapshot_id=snapshot_id, level="global", status="planning"
        )
        approved_global_plan = {}
    arc_target = global_node_id

    # 2. compile constraints — bail to clarification on a hard-vs-hard contradiction.
    constraints = compile_planning_constraints(snapshot_id, arc_target, db_path=db_path)
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

    # 3. assemble the arc base_context (degrade gracefully for not-yet-built stores).
    project_metadata = state.get("project_metadata") or {}
    try:
        open_threads = sqlite_db.get_open_threads(db_path)
    except Exception:  # noqa: BLE001 - degrade to empty if the store is unavailable
        open_threads = []
    base_context = {
        "global_plan": approved_global_plan,
        "genre": project_metadata.get("genre", ""),
        "raptor_root_summary": _raptor_root_summary(db_path),
        "open_threads": list(open_threads),
    }
    continuity: dict[str, Any] = {}

    # 4. registry + decider (model contact only through the injected decider seam).
    registry = registry or PlanningToolRegistry(db_path)
    if decider is None:
        decider = make_planner_decider(_NODE_NAME, base_context, config)

    # 5. run the bounded deliberation loop with a deterministic baseline floor.
    outcome = await run_planner_loop(
        level=_LEVEL,
        snapshot_id=snapshot_id,
        target_node=arc_target,
        constraints=constraints,
        continuity=continuity,
        registry=registry,
        config=config,
        planner_decider=decider,
        deterministic_baseline=_build_arc_baseline(base_context),
    )
    if outcome.records:
        trace.extend(outcome.records)

    # 6. map the outcome to persistence + state (never persists an invalid plan).
    prior_nodes = sqlite_db.get_planning_nodes(db_path, snapshot_id)
    result = persist_loop_outcome(
        outcome,
        level=_LEVEL,
        snapshot_id=snapshot_id,
        target_node=arc_target,
        persist_plan_fn=_arc_persist_plan_fn(db_path, snapshot_id, global_node_id),
        db_path=db_path,
        event_log_path=event_log_path,
        prior_nodes=prior_nodes,
    )

    state["planning_snapshot_id"] = result.planning_snapshot_id
    state["active_planning_revision_id"] = result.active_planning_revision_id
    if result.outcome == "needs_clarification":
        state["planning_block_reason"] = result.planning_block_reason
    elif result.outcome == "escalate":
        # Loop exhausted caps + fallback ladder with no valid plan: signal recovery. Expected
        # to be rare — the deterministic baseline is the validated floor.
        state["planning_block_reason"] = _ESCALATION_BLOCK_REASON
    else:
        # finalized / fallback_baseline: advance the pointer to the first planned arc so the
        # cascade can proceed to chapter planning.
        plan = result.plan or {}
        arcs = plan.get("arcs") or []
        pointer = state.get("fsm_pointer")
        if arcs and pointer is not None:
            first_arc_id = arcs[0].get("arc_id")
            if first_arc_id:
                state["fsm_pointer"] = pointer.model_copy(update={"arc_id": first_arc_id})
    return state
