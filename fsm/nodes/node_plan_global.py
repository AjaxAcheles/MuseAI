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

import dataclasses
import inspect
import json
from pathlib import Path
from typing import Any

import core.runtime as runtime
from fsm.planning_annotations import compile_planning_constraints
from fsm.planning_loop import run_planner_loop
from fsm.planning_node_support import (
    make_planner_decider,
    persist_loop_outcome,
    run_creative_consult,
)
from fsm.planning_tools import PlanningToolRegistry
from fsm.planning_validators import run_validators
from memory import sqlite_db

# Outcomes that carry a validator-passed plan eligible for the optional craft-consultant pass.
_REVISABLE_OUTCOMES = frozenset({"finalized", "fallback_baseline"})

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


# Distinct rising-action functions the baseline draws on (kept distinct so the scaffold
# also satisfies the arc_diversity quality-proxy validator). The final act is always a
# resolution; middles are taken from this list and synthesized beyond it.
_BASELINE_RISING_ACTS = (
    ("Setup", "establish the premise, characters, and stakes"),
    ("Inciting complication", "disrupt the status quo and commit the protagonist"),
    ("Rising action", "complicate the situation and deepen the conflict"),
    ("Midpoint turn", "reframe the stakes at the story's pivot"),
    ("Escalation", "escalate the central conflict toward its crisis"),
    ("Crisis", "force the decisive confrontation"),
)
_BASELINE_RESOLUTION_ACT = ("Resolution", "resolve the central conflict and pay off the promises")


def _baseline_act_specs(act_count: int) -> list[tuple[str, str]]:
    """Return ``act_count`` distinct (title, function) pairs; the last is the resolution."""
    if act_count <= 1:
        return [_BASELINE_RESOLUTION_ACT]
    middles = list(_BASELINE_RISING_ACTS[: act_count - 1])
    while len(middles) < act_count - 1:
        i = len(middles)
        middles.append((f"Development {i}", f"develop subplot strand {i} and sustain momentum"))
    return middles + [_BASELINE_RESOLUTION_ACT]


def _baseline_word_allocations(total_words: int, weights: list[float] | None, n: int) -> list[int]:
    """Distribute ``total_words`` across ``n`` acts by ``weights`` (equal split if absent/mismatched)."""
    if not total_words or n <= 0:
        return [0] * max(n, 0)
    if not weights or len(weights) != n or sum(weights) <= 0:
        weights = [1.0] * n
    norm = sum(weights)
    return [int(round(total_words * (w / norm))) for w in weights]


def _build_global_baseline(base_context: dict[str, Any], config: Any) -> Any:
    """Build the loop's deterministic baseline: a config-shaped, single-pass valid global plan.

    The fallback floor when the model never produces a validating plan. Its shape matches the
    authoritative global validators (``schema``/``arc_coverage``/``major_promise_payoff``): a
    non-empty ``arcs`` list whose entries carry a unique ``arc_id``, and a non-empty
    ``promises`` list whose entries carry a non-empty, non-tautological ``payoff``. Act count
    and per-act pacing weights are read from config (``planning.baseline_act_count`` /
    ``planning.baseline_word_weights``) — no hardcoded scaffold size — and the act functions
    are kept distinct so the scaffold also passes the quality-proxy validators.
    """
    premise = base_context.get("premise_seed") or "Untitled story"
    try:
        total_words = int(base_context.get("target_word_count") or 0)
    except (TypeError, ValueError):
        total_words = 0
    planning = getattr(config, "planning", None)
    act_count = max(1, int(getattr(planning, "baseline_act_count", 3) or 3))
    weights = getattr(planning, "baseline_word_weights", None)
    specs = _baseline_act_specs(act_count)
    allocations = _baseline_word_allocations(total_words, weights, len(specs))

    def _baseline(level: str, target_node: Any, constraints: dict) -> dict:
        return {
            "premise": premise,
            "central_conflict": f"The escalating struggle set in motion by: {premise}",
            "ending_target": "Resolve the central conflict and pay off the story's promises.",
            "arcs": [
                {
                    "arc_id": f"arc_{i + 1}",
                    "title": title,
                    "function": function,
                    "word_allocation": allocations[i],
                }
                for i, (title, function) in enumerate(specs)
            ],
            "promises": [
                {
                    "id": "promise_1",
                    "promise": f"The reader is promised an answer to the question the premise raises: {premise}",
                    "payoff": "Delivered in the final act as the central conflict resolves toward the ending target.",
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
    consult: Any = None,
) -> dict[str, Any]:
    """Plan the global story structure and persist the validated global plan.

    Returns the (mutated) orchestrator state. ``decider`` and ``registry`` are injectable
    seams (a synthetic decider keeps the node testable with no model/network); when
    ``decider`` is None the production M04-backed decider is built via
    ``make_planner_decider`` and awaited by the loop (the loop awaits any awaitable the
    decider returns, so sync test deciders work too).
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
    # Design decision (Data_Structures §2.7): two contradictory HARD annotations
    # (`requires_user_resolution`) are NOT auto-reconciled — the snapshot cannot reach
    # `approved` while one is unresolved, so guessing a middle path would be unsafe. We
    # surface the conflict (block reason + the structured `hard_conflicts` in the trace) for
    # the user / Build-14 recovery to resolve, and run no planning on contradictory input.
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
        # existing_arcs supports the continuation pass (append new arcs after exhaustion).
        # A true "revise the existing arcs" mode (re-planning bad arcs in place) is a deferred
        # follow-up, scoped as its own increment.
        "existing_arcs": list(state.get("existing_arcs") or []),
    }
    continuity: dict[str, Any] = {}

    # 4. registry + decider (model contact only through the injected/bridged decider seam).
    registry = registry or PlanningToolRegistry(db_path)
    if decider is None:
        decider = make_planner_decider(_NODE_NAME, base_context, config)

    # 5. run the bounded deliberation loop with a deterministic baseline floor.
    outcome = await run_planner_loop(
        level=_LEVEL,
        snapshot_id=snapshot_id,
        target_node=global_target,
        constraints=constraints,
        continuity=continuity,
        registry=registry,
        config=config,
        planner_decider=decider,
        deterministic_baseline=_build_global_baseline(base_context, config),
    )
    if outcome.records:
        trace.extend(outcome.records)

    # 5b. optional craft-consultant creative second pass: tighten a validated plan, then
    # RE-VALIDATE and only adopt the revision if it still passes (never persist unvalidated).
    if (
        getattr(config.planning, "creative_second_pass_enabled", False)
        and outcome.outcome in _REVISABLE_OUTCOMES
        and outcome.plan is not None
    ):
        consult_fn = consult
        if consult_fn is None:
            consult_fn = lambda p: run_creative_consult(p, base_context, config)  # noqa: E731
        try:
            consult_result = consult_fn(outcome.plan)
            revised = (
                await consult_result if inspect.isawaitable(consult_result) else consult_result
            )
            revised_validation = run_validators(_LEVEL, revised, constraints, continuity, config)
            if revised_validation.passes:
                outcome = dataclasses.replace(
                    outcome, plan=revised, validation=revised_validation
                )
                trace.append({"level": _LEVEL, "phase": "creative_consult", "adopted": True})
            else:
                trace.append(
                    {
                        "level": _LEVEL,
                        "phase": "creative_consult",
                        "adopted": False,
                        "failed_checks": list(revised_validation.failed_checks),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - best-effort; keep the original validated plan
            trace.append(
                {"level": _LEVEL, "phase": "creative_consult", "adopted": False, "error": repr(exc)}
            )

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
        # Loop exhausted caps + fallback ladder with no valid plan: signal recovery. Expected
        # to be rare — the deterministic baseline is the validated floor, so escalate fires
        # mainly on a store failure or a baseline that cannot validate, not in normal runs.
        state["planning_block_reason"] = _ESCALATION_BLOCK_REASON
    return state
