"""Module: M05 (Hierarchical Planning Cascade)
The bounded, harness-owned planner deliberation loop (Context_Injection_Algorithms.md
Phase 1 preamble; Module_Ability_Specification §5; Open_Problems.md "Planner
deliberation has an unmeasured cost/quality tradeoff").

Every planner level (global → arc → chapter → scene → beat) runs this one controller.
The model proposes; the harness disposes: each turn the loop asks an injected
`planner_decider` for exactly one `PlannerAction`, then *the harness* decides what
happens. It dispatches the four action types — `call_tool`, `revise_plan`,
`finalize_plan`, `raise_conflict` — gates acceptance of a `finalize_plan` solely on the
deterministic validators (`run_validators`, ignoring the planner's advisory
`self_check`), and enforces the per-level caps from config.

The loop is **pure**: it returns a `LoopOutcome` and performs no persistence, no
revision writes, and no escalation side effects. Mapping an outcome to snapshot/revision
writes and FSM routing is the planner node's job (a later sub-batch). Injected
dependencies (`planner_decider`, `deterministic_baseline`) make it fully testable
without a live model — the loop itself never calls `call_llm` or the network.

**Fallback ladder (Phase 1: `best_valid → deterministic_baseline → escalate`).** When no
validated finalize is reached within `planner_max_deliberation_loops[level]`, the loop
degrades safely without ever returning an invalid plan as a success/baseline: it returns
the best validated candidate it saw; else a deterministic baseline (injected, then
validated); else `escalate` carrying the last `ValidationResult`. A `raise_conflict`
short-circuits to `needs_clarification` and the ladder never overrides it.

**Cost-hygiene (the deliberation cost center, Open_Problems.md).**
- *Bounded accumulated context*: only a most-recent window of tool results is threaded
  back to the decider each turn (size read from config), so late turns cannot grow the
  render context without limit.
- *Early-finalize short-circuit*: a no-op / no-improvement revision over an already-valid
  plan accepts that plan immediately instead of burning the rest of the budget.
- *Intra-loop tool de-duplication*: an identical `(tool_name, args)` call within one
  deliberation reuses the prior result (including a cached `unavailable` marker) instead
  of re-issuing it and consuming the per-loop tool budget.
- *Wasted-turn accounting*: a non-finalizing, non-progressing turn (decider error,
  refused over-cap/unpermitted call, no-op invalid revision) is counted and surfaced so
  spinning is observable; the for-range already guarantees termination.

Cap semantics (per the Phase 1 pseudocode, which increments the tool-call count on each
tool call with no per-iteration reset): the deliberation loop is bounded by
`config.planning.planner_max_deliberation_loops[level]` iterations (tracked as
`planner_loop_index`), and executed tool calls across the run are bounded by
`config.planning.planner_max_tool_calls_per_loop[level]` — a single running counter that
mirrors the accumulating `planner_tool_call_count` FSM state field.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from fsm.planning_actions import PlannerAction
from fsm.planning_validators import ValidationResult, run_validators

# The four outcome statuses a planner node routes on.
OUTCOME_FINALIZED = "finalized"
OUTCOME_NEEDS_CLARIFICATION = "needs_clarification"
OUTCOME_FALLBACK_BASELINE = "fallback_baseline"
OUTCOME_ESCALATE = "escalate"

# Provisional fallback bound for the accumulated tool-result window threaded back to the
# decider each turn. There is no dedicated config key for this yet and this increment may
# touch only `fsm/planning_loop.py`, so the loop reads the named key
# `config.planning.planner_max_accumulated_tool_results` when present and otherwise uses
# this clearly-labelled default. A real config key (mirroring the 07.0a `planner_max_*`
# pattern) should be added in a config increment; it is a render-cost guard, not a tunable
# narrative threshold.
_ACCUMULATED_CONTEXT_CONFIG_KEY = "planner_max_accumulated_tool_results"
_DEFAULT_MAX_ACCUMULATED_TOOL_RESULTS = 20


@dataclass(frozen=True)
class DeliberationState:
    """The running state handed to the injected decider on each turn.

    Carries everything a planner needs to choose its next action — the current plan, a
    bounded most-recent window of accumulated tool results, the compiled constraints /
    continuity context, and the last validation result — plus the most recent refusal (so
    a refused tool call is fed back). Immutable, and `tool_results` is a defensive copy of
    the bounded window, so the decider cannot mutate or unboundedly grow the loop's state.
    """

    level: str
    target_node: Any
    snapshot_id: Any
    loop_index: int
    current_plan: dict | None
    best_valid_plan: dict | None
    tool_results: list[dict]
    constraints: dict
    continuity: Any
    last_validation: ValidationResult | None
    last_refusal: dict | None


@dataclass(frozen=True)
class LoopOutcome:
    """The structured result of one deliberation loop run (returned, never persisted).

    `outcome` is the gate the planner node routes on (`finalized` /
    `needs_clarification` / `fallback_baseline` / `escalate`). `plan` is the resulting
    plan (or None on `escalate` / `needs_clarification`) and — invariant — is non-None
    only when it passed `run_validators`. `validation` is the final `ValidationResult`;
    `best_valid_plan` is the best validated candidate seen; `baseline_source` distinguishes
    a `fallback_baseline` that reused the best-valid candidate from one built by the
    injected deterministic baseline. `records` is the per-iteration deliberation trace the
    node persists and folds into `planner_deliberation_trace`. `loops_used` is the
    deliberation count; `wasted_turns` counts non-progressing turns.
    """

    outcome: str
    level: str
    plan: dict | None = None
    validation: ValidationResult | None = None
    best_valid_plan: dict | None = None
    baseline_source: str | None = None
    conflicts: list[dict] | None = None
    records: list[dict] = field(default_factory=list)
    loops_used: int = 0
    tool_call_count: int = 0
    wasted_turns: int = 0


# Injected seams. The decider returns one PlannerAction given the running state; the
# deterministic baseline produces a fallback plan (or None) for a level. Real nodes back
# these with M04 / a static template; tests inject synthetic callables.
PlannerDecider = Callable[[DeliberationState], PlannerAction]
DeterministicBaseline = Callable[[str, Any, dict], dict | None]


def _dedup_key(tool_name: str | None, args: dict) -> tuple[str | None, str]:
    """Stable key for intra-loop tool-call de-duplication ((tool_name, canonical args))."""

    try:
        encoded = json.dumps(args, sort_keys=True, default=str)
    except TypeError:
        encoded = repr(args)
    return (tool_name, encoded)


def run_planner_loop(
    *,
    level: str,
    snapshot_id: Any,
    target_node: Any,
    constraints: dict,
    continuity: Any,
    registry: Any,
    config: Any,
    planner_decider: PlannerDecider,
    deterministic_baseline: DeterministicBaseline | None = None,
) -> LoopOutcome:
    """Run the bounded, harness-owned deliberation loop for one planner `level`.

    Asks `planner_decider` for one `PlannerAction` per turn and dispatches it. A
    `finalize_plan` is accepted only when `run_validators` passes — never on the planner's
    `self_check`. A `raise_conflict` ends the run immediately as `needs_clarification`.
    On cap exhaustion the fallback ladder runs (`best_valid → deterministic_baseline →
    escalate`), never returning an invalid plan as a success/baseline. Cost-hygiene guards
    (bounded accumulated context, early-finalize, intra-loop tool de-dup, wasted-turn
    accounting) bound deliberation cost. The loop never calls the network/`call_llm` and
    never writes to a store; it returns a `LoopOutcome`.
    """

    planning = getattr(config, "planning", None)
    if planning is None:
        raise ValueError("config has no `planning` section")
    try:
        max_loops = planning.planner_max_deliberation_loops[level]
        max_tool_calls = planning.planner_max_tool_calls_per_loop[level]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"no planning caps configured for level {level!r}") from exc

    max_accum = getattr(
        planning, _ACCUMULATED_CONTEXT_CONFIG_KEY, _DEFAULT_MAX_ACCUMULATED_TOOL_RESULTS
    )

    current_plan: dict | None = None
    best_valid_plan: dict | None = None
    best_valid_validation: ValidationResult | None = None
    tool_results: list[dict] = []
    tool_cache: dict[tuple[str | None, str], dict] = {}
    last_validation: ValidationResult | None = None
    last_refusal: dict | None = None
    tool_call_count = 0
    wasted_turns = 0
    records: list[dict] = []
    loops_used = 0

    def _window() -> list[dict]:
        """Bounded most-recent window of accumulated tool results (cost-hygiene)."""

        if not isinstance(max_accum, int) or max_accum <= 0:
            return []
        return list(tool_results[-max_accum:])

    for planner_loop_index in range(max_loops):
        loops_used = planner_loop_index + 1

        state = DeliberationState(
            level=level,
            target_node=target_node,
            snapshot_id=snapshot_id,
            loop_index=planner_loop_index,
            current_plan=current_plan,
            best_valid_plan=best_valid_plan,
            tool_results=_window(),
            constraints=constraints,
            continuity=continuity,
            last_validation=last_validation,
            last_refusal=last_refusal,
        )

        # Decider error: a flaky/non-conforming decider is a consumed, non-progressing
        # turn — count it and carry on so a single bad turn cannot crash the loop.
        try:
            action = planner_decider(state)
        except Exception as exc:  # noqa: BLE001 - degrade safely on any decider fault
            wasted_turns += 1
            records.append(
                {
                    "loop_index": planner_loop_index,
                    "action_type": "decider_error",
                    "error": repr(exc),
                }
            )
            continue

        if action.action_type == "call_tool":
            key = _dedup_key(action.tool_name, action.tool_args or {})

            # Intra-loop de-dup: reuse a prior identical call (even a cached unavailable
            # marker) without re-issuing it or consuming the per-loop tool budget.
            if key in tool_cache:
                last_refusal = None
                cached = tool_cache[key]
                records.append(
                    {
                        "loop_index": planner_loop_index,
                        "action_type": "call_tool",
                        "reused": True,
                        "tool_name": action.tool_name,
                        "available": cached.get("available"),
                        "outcome": cached.get("outcome"),
                        "trace_id": cached.get("trace_id"),
                    }
                )
                continue

            is_permitted = registry.permitted(level, action.tool_name)
            cap_reached = tool_call_count >= max_tool_calls
            if not is_permitted or cap_reached:
                reason = (
                    f"tool {action.tool_name!r} not permitted for level {level!r}"
                    if not is_permitted
                    else f"per-loop tool-call cap reached ({max_tool_calls})"
                )
                last_refusal = {
                    "tool_name": action.tool_name,
                    "reason": reason,
                    "loop_index": planner_loop_index,
                }
                wasted_turns += 1
                records.append(
                    {
                        "loop_index": planner_loop_index,
                        "action_type": "call_tool",
                        "accepted": False,
                        "wasted": True,
                        "tool_name": action.tool_name,
                        "reason": reason,
                    }
                )
                continue

            result = registry.call(
                level,
                action.tool_name,
                action.tool_args or {},
                snapshot_id,
                planner_loop_index,
            )
            tool_results.append(result)
            tool_cache[key] = result
            tool_call_count += 1
            last_refusal = None
            records.append(
                {
                    "loop_index": planner_loop_index,
                    "action_type": "call_tool",
                    "accepted": True,
                    "tool_name": action.tool_name,
                    "available": result.get("available"),
                    "outcome": result.get("outcome"),
                    "trace_id": result.get("trace_id"),
                }
            )
            continue

        if action.action_type == "revise_plan":
            proposed = action.revised_plan
            no_change = proposed == current_plan
            current_plan = proposed
            last_validation = run_validators(
                level, current_plan, constraints, continuity, config
            )
            record = {
                "loop_index": planner_loop_index,
                "action_type": "revise_plan",
                "validation_passes": last_validation.passes,
                "failed_checks": list(last_validation.failed_checks),
                "no_change": no_change,
            }
            if last_validation.passes:
                best_valid_plan = current_plan
                best_valid_validation = last_validation
                # Early-finalize: a no-op / no-improvement revision over an already-valid
                # plan accepts it now instead of burning the remaining loop budget. This
                # counts as the validated finalize.
                if no_change:
                    record["early_finalize"] = True
                    records.append(record)
                    return LoopOutcome(
                        outcome=OUTCOME_FINALIZED,
                        level=level,
                        plan=current_plan,
                        validation=last_validation,
                        best_valid_plan=best_valid_plan,
                        records=records,
                        loops_used=loops_used,
                        tool_call_count=tool_call_count,
                        wasted_turns=wasted_turns,
                    )
            elif no_change:
                # No change and still invalid → a non-progressing, wasted turn.
                wasted_turns += 1
                record["wasted"] = True
            records.append(record)
            continue

        if action.action_type == "finalize_plan":
            # HARNESS decides acceptance, not the planner's self_check (which is ignored).
            last_validation = run_validators(
                level, action.final_plan, constraints, continuity, config
            )
            records.append(
                {
                    "loop_index": planner_loop_index,
                    "action_type": "finalize_plan",
                    "validation_passes": last_validation.passes,
                    "failed_checks": list(last_validation.failed_checks),
                    "accepted": last_validation.passes,
                }
            )
            if last_validation.passes:
                return LoopOutcome(
                    outcome=OUTCOME_FINALIZED,
                    level=level,
                    plan=action.final_plan,
                    validation=last_validation,
                    best_valid_plan=action.final_plan,
                    records=records,
                    loops_used=loops_used,
                    tool_call_count=tool_call_count,
                    wasted_turns=wasted_turns,
                )
            # Not accepted: keep it as the current plan and keep deliberating; never
            # accept an invalid finalize.
            current_plan = action.final_plan
            continue

        if action.action_type == "raise_conflict":
            conflicts = list(action.conflicts or [])
            records.append(
                {
                    "loop_index": planner_loop_index,
                    "action_type": "raise_conflict",
                    "conflicts": len(conflicts),
                }
            )
            return LoopOutcome(
                outcome=OUTCOME_NEEDS_CLARIFICATION,
                level=level,
                conflicts=conflicts,
                validation=last_validation,
                best_valid_plan=best_valid_plan,
                records=records,
                loops_used=loops_used,
                tool_call_count=tool_call_count,
                wasted_turns=wasted_turns,
            )

        # PlannerAction.action_type is a four-value Literal, so this is unreachable for a
        # conforming decider; treat a non-conforming action as a wasted turn (degrade
        # safely) rather than crashing the loop.
        wasted_turns += 1
        records.append(
            {
                "loop_index": planner_loop_index,
                "action_type": "unsupported",
                "value": repr(getattr(action, "action_type", action)),
            }
        )
        continue

    # --- fallback ladder (only when no validated finalize occurred within the cap) ------
    # Strict order from Context_Injection_Algorithms.md Phase 1; every non-escalate return
    # below carries a plan that PASSED run_validators (the never-persist-invalid invariant).

    # 1. best validated candidate seen.
    if best_valid_plan is not None and best_valid_validation is not None:
        return LoopOutcome(
            outcome=OUTCOME_FALLBACK_BASELINE,
            level=level,
            plan=best_valid_plan,
            validation=best_valid_validation,
            best_valid_plan=best_valid_plan,
            baseline_source="best_valid",
            records=records,
            loops_used=loops_used,
            tool_call_count=tool_call_count,
            wasted_turns=wasted_turns,
        )

    # 2. deterministic baseline (injected), validated.
    if deterministic_baseline is not None:
        try:
            baseline = deterministic_baseline(level, target_node, constraints)
        except Exception:  # noqa: BLE001 - a faulty baseline must not crash; escalate.
            baseline = None
        if baseline is not None:
            baseline_validation = run_validators(
                level, baseline, constraints, continuity, config
            )
            if baseline_validation.passes:
                return LoopOutcome(
                    outcome=OUTCOME_FALLBACK_BASELINE,
                    level=level,
                    plan=baseline,
                    validation=baseline_validation,
                    best_valid_plan=best_valid_plan,
                    baseline_source="deterministic_baseline",
                    records=records,
                    loops_used=loops_used,
                    tool_call_count=tool_call_count,
                    wasted_turns=wasted_turns,
                )

    # 3. escalate — no valid plan; carry the last ValidationResult for failure recovery.
    return LoopOutcome(
        outcome=OUTCOME_ESCALATE,
        level=level,
        plan=None,
        validation=last_validation,
        best_valid_plan=best_valid_plan,
        records=records,
        loops_used=loops_used,
        tool_call_count=tool_call_count,
        wasted_turns=wasted_turns,
    )
