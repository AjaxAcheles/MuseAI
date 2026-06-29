"""Module: M05 (Hierarchical Planning Cascade)

Build the per-level planner *decider* the bounded deliberation loop calls each turn.

The harness owns the loop (``fsm/planning_loop.py``); the decider is the single seam
through which the model proposes its next move. Each turn the loop hands the decider the
running :class:`~fsm.planning_loop.DeliberationState`; the decider renders the level's
prompt template (M04 ``PromptLoader``), calls the M04 structured-output boundary
(``call_llm_structured``) constrained to the :class:`~fsm.planning_actions.PlannerAction`
schema, and returns exactly one parsed ``PlannerAction``. The model *proposes*; the loop
*disposes* — this decider performs no validation, no tool execution, and no persistence.

Level-agnostic by construction: every per-level difference is carried by ``node_name``
(which template), ``base_context`` (the static planning context for that level), and the
resolved endpoint. The same factory builds the global, arc, chapter, scene, and beat
deciders.

On a structured-output failure that M04 cannot salvage (``LLMCallError``), the decider
raises :class:`PlannerDeciderError` rather than crashing the node. The loop already treats
a raising decider as a non-finalizing (wasted) turn, so its caps and fallback ladder
(best-valid candidate -> deterministic baseline -> escalate) take over from there.

Integration note: this decider is ``async`` (it awaits the async inference boundary),
whereas the loop's current ``PlannerDecider`` seam is typed/called synchronously
(``fsm/planning_loop.py`` calls ``planner_decider(state)``). Wiring the real async decider
into the loop — making the loop ``await`` the decider — is a later increment; this module
only builds the decider and does not modify the loop (kept in scope).

``persist_loop_outcome`` is the single place that maps a finished ``LoopOutcome`` to
persistence and state. It centralizes the **never-persist-invalid** invariant: only a
validator-passed plan is ever written; a revision always preserves history (a new
``PlanningRevision`` is inserted and ``active_revision_id`` advances — never a silent
overwrite); ``needs_clarification`` and ``escalate`` persist no plan at all. It writes
validated planning *structure* and the proposal surface only — never prose, and it never
promotes a proposal node into committed narrative. LangGraph routing and the macro-outline
approval-gate edges are **not** wired here (that is Build 14).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from fsm.planning_actions import PlannerAction
from fsm.planning_annotations import compute_revision_diff
from fsm.planning_loop import LoopOutcome
from llm.call_llm import LLMCallError, call_llm_structured
from memory.event_log import write_event
from memory.sqlite_db import (
    get_planning_nodes,
    get_planning_snapshot,
    insert_planning_revision,
    transition_snapshot_status,
    upsert_planning_node,
)
from prompts.prompt_loader import PromptLoader


class PlannerDeciderError(RuntimeError):
    """A recoverable decider-turn failure (e.g. an unsalvageable structured output).

    Raised so the loop treats the turn as non-finalizing and applies its caps and
    fallback ladder — never to crash the planner node.
    """


# An async decider takes the running deliberation state and returns one PlannerAction.
AsyncPlannerDecider = Callable[[Any], Awaitable[PlannerAction]]


def _state_field(loop_state: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from the loop state, tolerating a dataclass or a mapping."""
    if isinstance(loop_state, Mapping):
        return loop_state.get(name, default)
    return getattr(loop_state, name, default)


def make_planner_decider(
    node_name: str,
    base_context: Mapping[str, Any],
    config: Any,
    *,
    endpoint_role: str = "planner",
    loader: PromptLoader | None = None,
    call_structured: Callable[..., Awaitable[Any]] | None = None,
    schema_model: type[PlannerAction] = PlannerAction,
) -> AsyncPlannerDecider:
    """Return the async ``decider(loop_state)`` the deliberation loop calls each turn.

    Per turn the decider:

    1. Merges ``base_context`` (the level's static planning context) with the running
       ``loop_state`` (current plan, accumulated tool results, compiled constraints, last
       validation, and the surrounding loop fields) into one render context — dynamic
       loop fields take precedence over static base keys on a collision.
    2. Renders ``node_name``'s prompt via the M04 ``PromptLoader`` (``StrictUndefined``, so
       a missing template variable fails loudly rather than blanking).
    3. Calls ``call_llm_structured`` on the role-resolved endpoint, constrained to the
       ``PlannerAction`` schema, and returns the parsed action.

    The endpoint is resolved by role from ``config.endpoints`` (default ``planner``) and
    the validate-retry cap from ``config.runtime.model_validate_retry_cap`` — no provider,
    port, or model name is hardcoded. Both are resolved eagerly here so a misconfigured
    ``endpoint_role`` fails at factory-build time, not mid-loop.

    ``loader``, ``call_structured``, and ``schema_model`` are injectable seams for tests;
    they default to the real M04 components.
    """
    endpoint = getattr(config.endpoints, endpoint_role)
    validate_retry_cap = config.runtime.model_validate_retry_cap
    prompt_loader = loader if loader is not None else PromptLoader()
    structured = call_structured if call_structured is not None else call_llm_structured

    async def decider(loop_state: Any) -> PlannerAction:
        # Static per-level context first; the running loop state overrides on collision.
        render_context: dict[str, Any] = dict(base_context)
        render_context.update(
            {
                "level": _state_field(loop_state, "level"),
                "loop_index": _state_field(loop_state, "loop_index"),
                "current_plan": _state_field(loop_state, "current_plan"),
                "best_valid_plan": _state_field(loop_state, "best_valid_plan"),
                "tool_results": _state_field(loop_state, "tool_results", []),
                "compiled_constraints": _state_field(loop_state, "constraints", {}),
                "continuity": _state_field(loop_state, "continuity"),
                "last_validation": _state_field(loop_state, "last_validation"),
                "last_refusal": _state_field(loop_state, "last_refusal"),
            }
        )

        rendered = prompt_loader.render(node_name, render_context)
        messages = [{"role": "user", "content": rendered}]

        try:
            action = await structured(
                messages,
                endpoint,
                schema_model=schema_model,
                validate_retry_cap=validate_retry_cap,
            )
        except LLMCallError as exc:
            # M04 exhausted retries + salvage: surface a recoverable decider error so the
            # loop counts a non-finalizing turn and lets its fallback ladder apply.
            raise PlannerDeciderError(
                f"planner decider for {node_name!r} could not obtain a valid "
                f"PlannerAction from endpoint role {endpoint_role!r}: {exc}"
            ) from exc

        return action

    return decider


# ---------------------------------------------------------------------------
# Outcome -> persistence/state mapping
# ---------------------------------------------------------------------------

# LoopOutcome.outcome values that carry a (by invariant) validator-passed plan to persist.
# `finalized` accepted a model `finalize_plan`; `fallback_baseline` used the best validated
# candidate or the deterministic baseline (the loop validates the baseline before returning).
_PERSIST_OUTCOMES = frozenset({"finalized", "fallback_baseline"})

# A needs_clarification outcome has no dedicated snapshot status in the §2.7 vocabulary
# (that state lives per-annotation as PlanningAnnotation.status='needs_clarification', set by
# the compiler). The snapshot itself moves to 'revision_requested' — a clarification/revision
# is being requested from the user — and the FSM block reason is the documented value.
_CLARIFICATION_SNAPSHOT_STATUS = "revision_requested"
_CLARIFICATION_BLOCK_REASON = "unresolved_hard_conflict"


@dataclass(frozen=True)
class PersistResult:
    """The structured result of mapping a LoopOutcome to persistence + state.

    The planner node folds these into ``OrchestratorState`` (``planning_snapshot_id``,
    ``active_planning_revision_id``, ``planning_block_reason``, and the conflict /
    validation payloads). ``persisted`` is True only when a validated plan + revision +
    event were written; it is False for ``needs_clarification`` and ``escalate``.
    """

    outcome: str
    persisted: bool
    planning_snapshot_id: str
    active_planning_revision_id: str | None
    revision_id: str | None = None
    plan: dict | None = None
    validation: Any = None
    conflicts: list[dict] | None = None
    planning_block_reason: str | None = None
    event_written: bool = False


def _target_node_id(target_node: Any) -> Any:
    """Best-effort id for a target node passed as a dict or a node_id string."""
    if isinstance(target_node, Mapping):
        return target_node.get("node_id") or target_node.get("id")
    return target_node


def _derive_revision_id(snapshot_id: str, diff_json: str) -> str:
    """Deterministic revision id from the diff content, so a replay is idempotent.

    The same logical commit (same snapshot + same diff) yields the same id, so a re-run
    after a crash hits ``insert_planning_revision``'s ON CONFLICT DO NOTHING instead of
    appending a duplicate revision.
    """
    digest = hashlib.sha256(f"{snapshot_id}|{diff_json}".encode("utf-8")).hexdigest()
    return f"rev_{digest[:24]}"


def _active_revision_id(db_path: str | Path, snapshot_id: str) -> str | None:
    snap = get_planning_snapshot(db_path, snapshot_id)
    return snap.get("active_revision_id") if snap else None


def persist_loop_outcome(
    outcome: LoopOutcome,
    *,
    level: str,
    snapshot_id: str,
    target_node: Any,
    persist_plan_fn: Callable[[dict], Any],
    db_path: str | Path,
    event_log_path: str | Path,
    prior_nodes: list[dict] | None = None,
    annotation_outcomes: Any = None,
    revision_id: str | None = None,
) -> PersistResult:
    """Map a finished ``LoopOutcome`` to persistence and state — the only place that does.

    Branches on ``outcome.outcome``:

    * ``finalized`` / ``fallback_baseline`` — assert the outcome carries a validator-passed
      plan (``plan`` present **and** ``validation.passes``); if not, raise ``ValueError``
      rather than persist an unvalidated plan. Then, in dependency order: (1) call the
      injected ``persist_plan_fn(plan)`` for the level-specific narrative-table writes; (2)
      upsert the plan's ``PlanningNode`` row(s); (3) insert a ``PlanningRevision`` whose
      ``diff_json`` is ``compute_revision_diff(prior_nodes, new_nodes, annotation_outcomes)``,
      which also advances ``active_revision_id`` (history preserved — never an overwrite); (4)
      mirror the commit to the append-only event log via ``write_event``. Every store write
      is idempotent/replay-safe and the event-log mirror is the recoverable record, so a
      crash mid-sequence leaves no partial *divergent* canonical truth — a replay converges.

    * ``needs_clarification`` — persist **no** plan. The conflicting annotations are already
      marked ``needs_clarification`` by the compiler (not re-written here). Move the snapshot
      to ``revision_requested`` and return the conflict info plus the
      ``unresolved_hard_conflict`` block reason for the node to put in state.

    * ``escalate`` — persist **no** plan and write **no** revision; return an escalation
      signal carrying the last ``ValidationResult`` for the node to route to recovery.

    Never writes prose and never promotes a proposal node into committed narrative —
    validated planning structure + the proposal surface only. No LangGraph route or
    approval-gate edge is wired here (Build 14).
    """
    if outcome.outcome in _PERSIST_OUTCOMES:
        # ---- never-persist-invalid invariant (centralized) ----
        if outcome.plan is None or outcome.validation is None or not outcome.validation.passes:
            raise ValueError(
                f"refusing to persist {outcome.outcome!r} for level {level!r}: outcome does "
                "not carry a validator-passed plan (plan present and validation.passes)"
            )
        plan = outcome.plan

        # (1) level-specific narrative-table writes, supplied by the node.
        persist_plan_fn(plan)

        # (2) upsert the proposal-surface PlanningNode rows carried by the plan.
        for node in plan.get("planning_nodes", []):
            upsert_planning_node(
                db_path,
                node_id=node["node_id"],
                snapshot_id=node.get("snapshot_id", snapshot_id),
                level=node.get("level", level),
                status=node["status"],
                parent_id=node.get("parent_id"),
                ordering=node.get("ordering", 0),
                title=node.get("title"),
                summary=node.get("summary"),
                purpose=node.get("purpose"),
                locked_pinned=node.get("locked_pinned", False),
            )

        # (3) revision: diff the full post-write node set against the prior set, then insert
        #     (advancing active_revision_id). Idempotent by a content-derived revision id.
        new_nodes = get_planning_nodes(db_path, snapshot_id)
        diff = compute_revision_diff(prior_nodes, new_nodes, annotation_outcomes)
        diff_json = json.dumps(diff, sort_keys=True)
        rev_id = revision_id or _derive_revision_id(snapshot_id, diff_json)
        parent_revision_id = _active_revision_id(db_path, snapshot_id)
        insert_planning_revision(
            db_path,
            revision_id=rev_id,
            snapshot_id=snapshot_id,
            diff_json=diff_json,
            parent_revision_id=parent_revision_id,
            change_summary=(
                f"{outcome.outcome} {level} plan"
                + (f" (baseline_source={outcome.baseline_source})" if outcome.baseline_source else "")
            ),
        )

        # (4) mirror the commit to the append-only event log (recoverable record).
        write_event(
            event_log_path,
            {
                "event_type": "planning_commit",
                "level": level,
                "snapshot_id": snapshot_id,
                "target_node_id": _target_node_id(target_node),
                "revision_id": rev_id,
                "parent_revision_id": parent_revision_id,
                "outcome": outcome.outcome,
                "baseline_source": outcome.baseline_source,
                "validation_passes": True,
                "diff": diff,
            },
        )

        return PersistResult(
            outcome=outcome.outcome,
            persisted=True,
            planning_snapshot_id=snapshot_id,
            active_planning_revision_id=rev_id,
            revision_id=rev_id,
            plan=plan,
            validation=outcome.validation,
            event_written=True,
        )

    if outcome.outcome == "needs_clarification":
        # No plan is written. The compiler already marked the conflicting annotations
        # needs_clarification — do not double-write them. Move the snapshot to the
        # clarification/revision-requested state and signal the block to the node.
        transition_snapshot_status(
            db_path, snapshot_id, status=_CLARIFICATION_SNAPSHOT_STATUS
        )
        return PersistResult(
            outcome=outcome.outcome,
            persisted=False,
            planning_snapshot_id=snapshot_id,
            active_planning_revision_id=_active_revision_id(db_path, snapshot_id),
            plan=None,
            validation=outcome.validation,
            conflicts=outcome.conflicts,
            planning_block_reason=_CLARIFICATION_BLOCK_REASON,
        )

    if outcome.outcome == "escalate":
        # Persist nothing and write no revision; hand the node the last validation so it can
        # route to failure recovery (the route itself is Build 14, not here).
        return PersistResult(
            outcome=outcome.outcome,
            persisted=False,
            planning_snapshot_id=snapshot_id,
            active_planning_revision_id=_active_revision_id(db_path, snapshot_id),
            plan=None,
            validation=outcome.validation,
        )

    raise ValueError(f"unknown LoopOutcome.outcome: {outcome.outcome!r}")
