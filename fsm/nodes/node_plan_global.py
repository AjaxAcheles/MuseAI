"""Module: M05 (Hierarchical Planning Cascade)

Level-1 (global) planner node. Creates the master timeline and thematic boundaries for
the whole book by running the shared bounded deliberation loop and persisting only a
validator-passed global plan.

Flow per invocation:
  1. resolve/create the run's ``PlanningSnapshot`` and the global target node id;
  2. ``compile_planning_constraints`` — if two hard annotations contradict, set the
     clarification block and return WITHOUT running the loop (never deliberate over
     contradictory hard constraints);
  3. assemble the global ``base_context`` from permitted global-level reads (project
     metadata / world rules), degrading gracefully for not-yet-built stores;
  4. build an M04-backed decider via ``make_planner_decider`` and run
     ``run_planner_loop`` with a deterministic single-pass baseline as the loop's floor;
  5. map the ``LoopOutcome`` through ``persist_loop_outcome`` — which enforces the
     never-persist-invalid invariant and writes the revision + event — and fold the
     resulting snapshot / revision / block state back into the orchestrator state.

The node makes **no** direct ``call_llm``: all model contact is through the injected
decider seam, so it is fully testable by injecting a synthetic decider. It writes no prose
and wires no graph route / approval edge (that is Build 14).
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import core.runtime as runtime
from fsm.planning_annotations import compile_planning_constraints
from fsm.planning_loop import run_planner_loop
from fsm.planning_node_support import make_planner_decider, persist_loop_outcome
from fsm.planning_tools import PlanningToolRegistry
from memory import sqlite_db

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
_LEVEL = "global"
_NODE_NAME = "node_plan_global"

# Human-readable, provisional planning-block reason for a loop that exhausted its caps and
# fallback ladder without a valid plan. The actual route to failure recovery is Build 14.
_ESCALATION_BLOCK_REASON = "planning_escalation"


def _resolve_config(state: dict[str, Any]) -> Any:
    """Return the active typed config (mirrors node_assemble_context's resolution)."""
    config = state.get("app_config") or state.get("config")
    if config is not None:
        return config
    from core.config_loader import load_config

    return load_config(state.get("config_path", CONFIG_PATH))


def _sync_decider_from_async(async_decider: Any) -> Any:
    """Adapt an async decider to the synchronous seam ``run_planner_loop`` expects.

    Temporary shim (07.07 carryover): ``run_planner_loop`` calls ``planner_decider(state)``
    synchronously, but ``make_planner_decider`` returns an async decider (it awaits the M04
    inference boundary). Until the loop is made async, bridge by running the coroutine to
    completion on a dedicated worker thread + event loop — safe even though this node runs
    inside an event loop, because the work is offloaded off the running loop's thread. Any
    exception (e.g. ``PlannerDeciderError``) propagates to the loop, which treats it as a
    non-finalizing wasted turn.
    """

    def _call(loop_state: Any) -> Any:
        box: dict[str, Any] = {}

        def _runner() -> None:
            worker_loop = asyncio.new_event_loop()
            try:
                box["value"] = worker_loop.run_until_complete(async_decider(loop_state))
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
                box["error"] = exc
            finally:
                worker_loop.close()

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if "error" in box:
            raise box["error"]
        return box["value"]

    return _call


def _build_global_baseline(base_context: dict[str, Any]) -> Any:
    """Build the loop's deterministic baseline: a minimal, single-pass valid global plan.

    The fallback floor when the model never produces a validating plan. Its shape matches
    the authoritative global validators (``schema``/``arc_coverage``/``major_promise_payoff``
    in ``fsm/planning_validators.py``): a non-empty ``arcs`` list whose entries carry an
    ``arc_id``, and a non-empty ``promises`` list whose entries carry a ``payoff``. A plain
    deterministic three-act scaffold — not a tunable threshold.
    """
    premise = base_context.get("premise_seed") or "Untitled story"
    try:
        total_words = int(base_context.get("target_word_count") or 0)
    except (TypeError, ValueError):
        total_words = 0
    acts = (
        ("arc_1", "Setup", "establish the premise, characters, and stakes"),
        ("arc_2", "Confrontation", "escalate the central conflict toward its crisis"),
        ("arc_3", "Resolution", "resolve the central conflict and honor the promises"),
    )
    per_arc = total_words // len(acts) if total_words else 0

    def _baseline(level: str, target_node: Any, constraints: dict) -> dict:
        return {
            "premise": premise,
            "central_conflict": f"The unresolved tension at the heart of: {premise}",
            "ending_target": "Resolve the central conflict and pay off the story's promises.",
            "arcs": [
                {"arc_id": aid, "title": title, "function": function, "word_allocation": per_arc}
                for aid, title, function in acts
            ],
            "promises": [
                {
                    "id": "promise_1",
                    "promise": "The dramatic question the premise raises will be answered.",
                    "payoff": "Answered as the resolution arc closes.",
                }
            ],
        }

    return _baseline


def _global_persist_plan_fn(db_path: Any, snapshot_id: str, global_target: str) -> Any:
    """Return the level-specific writer for the global plan (the top-level story node).

    There is no narrative "Story" table — the narrative tables start at ``Arcs`` (written
    later by ``node_plan_arc``). The validated global plan is therefore persisted as the
    global ``PlanningNode`` (the proposal-surface representation of the top-level story
    node), with the full plan content carried as JSON in ``purpose`` (the same convention
    07.00 used for chapter obligations / beat PAD strings). ``persist_loop_outcome`` reads
    the node set back after this writer runs to compute the revision diff.
    """

    def _persist(plan: dict) -> None:
        sqlite_db.upsert_planning_node(
            db_path,
            node_id=global_target,
            snapshot_id=snapshot_id,
            level=_LEVEL,
            status="planned",
            title=plan.get("premise"),
            summary=plan.get("premise"),
            purpose=json.dumps(plan, sort_keys=True),
        )

    return _persist


async def node_plan_global(
    state: dict[str, Any],
    *,
    decider: Any = None,
    registry: Any = None,
) -> dict[str, Any]:
    """Plan the global story structure and persist the validated global plan.

    Returns the (mutated) orchestrator state. ``decider`` and ``registry`` are injectable
    seams (the synthetic decider keeps the node testable with no model/network); when
    ``decider`` is None the production M04-backed decider is built and bridged to the loop's
    synchronous seam.
    """
    config = _resolve_config(state)
    db_path = state.get("sqlite_db_path", runtime.SQLITE_DB_PATH)
    event_log_path = state.get("event_log_path", runtime.EVENT_LOG_PATH)
    project_id = state["project_id"]
    trace = state.setdefault("planner_deliberation_trace", [])

    # 1. resolve/create the PlanningSnapshot + the global target node id.
    snapshot_id = state.get("planning_snapshot_id") or f"snap_{project_id}"
    mode = state.get("planning_execution_mode") or getattr(
        config.planning, "execution_mode", "macro_outline_before_draft"
    )
    sqlite_db.create_planning_snapshot(
        db_path, snapshot_id=snapshot_id, project_id=project_id, mode=mode
    )
    state["planning_snapshot_id"] = snapshot_id
    global_target = f"{snapshot_id}:global"
    # Ensure the global target PlanningNode exists (idempotent) so the annotation compiler
    # can resolve it and any node-attached annotations apply. persist_plan_fn later updates
    # this row in place with the validated plan content and final status.
    sqlite_db.upsert_planning_node(
        db_path, node_id=global_target, snapshot_id=snapshot_id, level=_LEVEL, status="planning"
    )

    # 2. compile constraints — bail to clarification on a hard-vs-hard contradiction.
    constraints = compile_planning_constraints(snapshot_id, global_target, db_path=db_path)
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

    # 3. assemble the global base_context from permitted reads (degrade gracefully).
    project_metadata = state.get("project_metadata") or {}
    base_context = {
        "genre": project_metadata.get("genre", ""),
        "target_word_count": project_metadata.get(
            "target_word_count", getattr(config.runtime, "word_count_target", 0)
        ),
        "premise_seed": project_metadata.get("premise_seed", ""),
        "world_rules": list(state.get("world_rules") or []),
        "existing_arcs": list(state.get("existing_arcs") or []),
    }
    continuity: dict[str, Any] = {}

    # 4. registry + decider (model contact only through the injected/bridged decider seam).
    registry = registry or PlanningToolRegistry(db_path)
    if decider is None:
        decider = _sync_decider_from_async(
            make_planner_decider(_NODE_NAME, base_context, config)
        )

    # 5. run the bounded deliberation loop with a deterministic baseline floor.
    outcome = run_planner_loop(
        level=_LEVEL,
        snapshot_id=snapshot_id,
        target_node=global_target,
        constraints=constraints,
        continuity=continuity,
        registry=registry,
        config=config,
        planner_decider=decider,
        deterministic_baseline=_build_global_baseline(base_context),
    )
    if outcome.records:
        trace.extend(outcome.records)

    # 6. map the outcome to persistence + state (never persists an invalid plan).
    prior_nodes = sqlite_db.get_planning_nodes(db_path, snapshot_id)
    result = persist_loop_outcome(
        outcome,
        level=_LEVEL,
        snapshot_id=snapshot_id,
        target_node=global_target,
        persist_plan_fn=_global_persist_plan_fn(db_path, snapshot_id, global_target),
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
        state["planning_block_reason"] = _ESCALATION_BLOCK_REASON
    return state
