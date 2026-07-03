"""Module: M01 (Coordinator & State Machine)
Vertical-slice planning runner: drive the five real M05 planner nodes in order,
emitting live events after each node.

A provisional, sequential stand-in for the compiled LangGraph wiring (Build 14,
``fsm/graph.py``). It runs the real planning cascade — snapshot creation,
annotation compilation, bounded deliberation loops, validators, persistence —
through each node's injectable decider seam:

* deterministic mode (default): a synthetic decider returns
  ``continue_deliberation`` every turn, so the bounded loop exhausts its
  configured cap and the production fallback ladder persists each node's
  validator-passed deterministic baseline. No model, no network.
* live mode: ``decider=None`` lets each node build the production
  M04-backed decider (``make_planner_decider`` → ``call_llm_structured``)
  against the configured planner endpoint.

Execution modes follow the design: ``macro_outline_before_draft`` plans global
+ all arcs + all chapters as the macro outline, then one scene + one beat
just-in-time; ``rolling`` plans one global → arc → chapter → scene → beat path.
The run-start mode snapshot is read from state (set by the generation manager
from the validated start payload, with config as its default) — never from
config at node time.

The macro-outline approval gate is real: with a generation manager attached the
runner parks on ``manager.wait_for_approval()`` (cancellable by Stop) and, once
approved, stamps the PlanningSnapshot ``approved`` and resumes into scene/beat
planning. Approval blocking uses ``awaiting_planning_approval`` /
``planning_block_reason`` only — never ``pause_requested`` /
``hard_stop_asserted``. No drafting module is routed to.
"""

from __future__ import annotations

import datetime as _dt
from collections import Counter
from typing import Any, Awaitable, Callable

from fsm.nodes.node_plan_arc import node_plan_arc
from fsm.nodes.node_plan_beat import node_plan_beat
from fsm.nodes.node_plan_chapter import node_plan_chapter
from fsm.nodes.node_plan_global import node_plan_global
from fsm.nodes.node_plan_scene import node_plan_scene
from fsm.planning_actions import PlannerAction
from memory import sqlite_db

EmitFn = Callable[[str, dict[str, Any]], Awaitable[None]]

# Vertical-slice model modes. "deterministic" never touches the network.
LLM_MODES = ("deterministic", "live")

APPROVAL_BLOCK_REASON = "awaiting_macro_approval"

_DETERMINISTIC_RATIONALE = (
    "Deterministic vertical-slice run: no model attached; defer to the "
    "harness's validated baseline planner."
)


def make_deterministic_decider() -> Callable[[Any], PlannerAction]:
    """A synthetic decider that thinks every turn and never finalizes.

    The bounded loop counts each turn against the level's configured cap, then
    runs the production fallback ladder, which validates and persists the
    node's deterministic baseline plan. Nothing about a live model is faked —
    the deliberation trace records plain ``continue_deliberation`` turns.
    """

    def decider(loop_state: Any) -> PlannerAction:
        del loop_state
        return PlannerAction(
            action_type="continue_deliberation", rationale=_DETERMINISTIC_RATIONALE
        )

    return decider


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pointer_text(pointer_data: dict[str, Any]) -> str:
    parts = []
    for label, key in (
        ("arc", "arc_id"),
        ("ch", "chapter_id"),
        ("scene", "scene_id"),
        ("beat", "beat_id"),
    ):
        value = pointer_data.get(key)
        if value:
            parts.append(f"{label} {value}")
    return " · ".join(parts) if parts else "start of book"


def _snapshot_payload(state: dict[str, Any]) -> dict[str, Any] | None:
    """Read the current PlanningSnapshot surface for the UI, or None if absent."""
    snapshot_id = state.get("planning_snapshot_id")
    db_path = state.get("sqlite_db_path")
    if not snapshot_id or db_path is None:
        return None
    snapshot = sqlite_db.get_planning_snapshot(db_path, snapshot_id)
    if snapshot is None:
        return None
    nodes = sqlite_db.get_planning_nodes(db_path, snapshot_id)
    revisions = sqlite_db.get_revisions_for_snapshot(db_path, snapshot_id)
    counts = dict(Counter(node.get("level") for node in nodes))
    counts_text = ", ".join(f"{level}: {n}" for level, n in sorted(counts.items()))
    return {
        "snapshot_id": snapshot_id,
        "status": snapshot.get("status"),
        "mode": snapshot.get("mode"),
        "active_revision_id": snapshot.get("active_revision_id"),
        "node_counts": counts,
        "revision_count": len(revisions),
        "macro_outline_ready": bool(state.get("macro_outline_ready")),
        "message": f"Snapshot {snapshot.get('status')} — {counts_text or 'no nodes yet'}",
    }


async def _pause_gate(resources: Any) -> None:
    """Hold between nodes while the generation manager reports paused."""
    manager = getattr(resources, "generation_manager", None)
    if manager is not None:
        await manager.wait_if_paused()


async def run_planning_vertical_slice(
    state: dict[str, Any],
    resources: Any,
    emit: EmitFn,
) -> dict[str, Any]:
    """Run the planning cascade for one vertical-slice pass and return the state.

    The returned state is "blocked" when ``planning_block_reason`` is set and
    could not be resolved in-run (unresolved hard conflict, escalation, or an
    approval gate with no manager attached to resume through); the caller
    decides how to present that. Exceptions propagate to the caller.
    """
    config = state.get("app_config") or state.get("config")
    if config is None:
        raise ValueError("state['app_config'] must carry the typed AppConfig")

    def log(event: str, **fields: Any) -> None:
        log_fn = getattr(resources, "log", None)
        if callable(log_fn):
            log_fn(event, **fields)

    # Run-start mode snapshot: set by the manager from the validated start
    # payload (config is the payload's default). Fall back to config only when
    # a bare runner is driven without the manager having set them.
    execution_mode = state.get("planning_execution_mode") or config.planning.execution_mode
    approval_mode = state.get("approval_mode") or config.planning.approval_mode
    state["planning_execution_mode"] = execution_mode
    state["approval_mode"] = approval_mode

    llm_mode = state.get("llm_mode") or "deterministic"
    if llm_mode not in LLM_MODES:
        raise ValueError(f"unknown llm_mode {llm_mode!r}; expected one of {LLM_MODES}")
    # None routes each node to its production M04-backed decider (live mode).
    decider = None if llm_mode == "live" else make_deterministic_decider()

    await emit(
        "phase_change",
        {
            "phase": "planning",
            "planning_execution_mode": execution_mode,
            "approval_mode": approval_mode,
            "llm_mode": llm_mode,
            "message": f"Planning started ({execution_mode}, approval {approval_mode}).",
        },
    )

    last_snapshot_payload: dict[str, Any] | None = None

    async def run_node(
        name: str, invoke: Callable[[], Awaitable[dict]]
    ) -> str | None:
        """Invoke one planner node, emit its telemetry, return its block reason."""
        nonlocal state, last_snapshot_payload
        revision_before = state.get("active_planning_revision_id")
        trace_before = len(state.get("planner_deliberation_trace") or [])
        log("node_start", node=name)

        state = await invoke()

        pointer = state.get("fsm_pointer")
        pointer_data = pointer.model_dump() if pointer is not None else {}
        await emit(
            "pointer_update",
            {**pointer_data, "message": f"Pointer: {_pointer_text(pointer_data)}"},
        )
        log("pointer_update", pointer=pointer_data)

        trace = state.get("planner_deliberation_trace") or []
        new_records = trace[trace_before:]
        revision_id = state.get("active_planning_revision_id")
        revision_changed = revision_id != revision_before
        block_reason = state.get("planning_block_reason")
        if block_reason:
            node_message = f"{name} blocked: {block_reason}"
        elif revision_changed:
            node_message = (
                f"{name} persisted revision {revision_id} "
                f"({len(new_records)} deliberation turns)"
            )
        else:
            node_message = f"{name} made no plan change ({len(new_records)} turns)"
        await emit(
            "planning_node",
            {
                "node": name,
                "revision_id": revision_id,
                "revision_changed": revision_changed,
                "deliberation_turns": len(new_records),
                "actions": [
                    record.get("action_type") or record.get("phase")
                    for record in new_records[-8:]
                    if isinstance(record, dict)
                ],
                "block_reason": block_reason,
                "pointer": pointer_data,
                "message": node_message,
            },
        )
        log(
            "node_end",
            node=name,
            revision_id=revision_id,
            revision_changed=revision_changed,
            turns=len(new_records),
            block_reason=block_reason,
        )

        snapshot_payload = _snapshot_payload(state)
        if snapshot_payload is not None and snapshot_payload != last_snapshot_payload:
            last_snapshot_payload = snapshot_payload
            await emit("planning_snapshot", snapshot_payload)
            log(
                "snapshot_update",
                snapshot_id=snapshot_payload["snapshot_id"],
                status=snapshot_payload["status"],
                revision_count=snapshot_payload["revision_count"],
            )

        if block_reason and block_reason != APPROVAL_BLOCK_REASON:
            await emit(
                "planning_blocked",
                {
                    "reason": block_reason,
                    "node": name,
                    "awaiting_approval": False,
                    "message": f"Planning blocked at {name}: {block_reason}",
                },
            )
            log("planning_blocked", node=name, reason=block_reason)
        return block_reason

    # --- Level 1 + 2: global, then arcs (one invocation plans all arc slots). --
    if await run_node(
        "node_plan_global",
        lambda: node_plan_global(state, decider=decider),
    ):
        return state
    await _pause_gate(resources)
    if await run_node(
        "node_plan_arc",
        lambda: node_plan_arc(state, decider=decider),
    ):
        return state

    # --- Level 3: chapters. Macro mode sweeps every stub chapter (one per
    # invocation) until the node reports the macro outline ready; rolling mode
    # plans only the next needed chapter. -----------------------------------
    if execution_mode == "macro_outline_before_draft":
        db_path = state.get("sqlite_db_path")
        snapshot_id = state.get("planning_snapshot_id")
        chapter_stub_count = len(
            sqlite_db.get_planning_nodes(db_path, snapshot_id, level="chapter")
        )
        for _ in range(max(chapter_stub_count, 1) + 1):
            await _pause_gate(resources)
            revision_before = state.get("active_planning_revision_id")
            reason = await run_node(
                "node_plan_chapter",
                lambda: node_plan_chapter(state, decider=decider),
            )
            if reason == APPROVAL_BLOCK_REASON:
                break  # gate handling below
            if reason:
                return state
            if state.get("macro_outline_ready"):
                break
            if state.get("active_planning_revision_id") == revision_before:
                raise RuntimeError(
                    "chapter planning made no progress and the macro outline "
                    "is not ready; aborting instead of looping"
                )
        else:
            raise RuntimeError(
                "chapter sweep exceeded the planned chapter count without "
                "reaching macro-outline readiness"
            )

        # --- Macro-outline approval gate (safe-boundary pause, not a failure).
        if state.get("awaiting_planning_approval"):
            await emit(
                "phase_change",
                {
                    "phase": "awaiting_approval",
                    "message": "Macro outline ready — awaiting approval.",
                },
            )
            await emit(
                "planning_blocked",
                {
                    "reason": APPROVAL_BLOCK_REASON,
                    "node": "node_plan_chapter",
                    "awaiting_approval": True,
                    "message": "Macro outline ready; planning paused for approval.",
                },
            )
            await emit(
                "approval_state",
                {
                    "awaiting_approval": True,
                    "macro_outline_ready": bool(state.get("macro_outline_ready")),
                    "macro_outline_approved": False,
                    "message": "Review the macro outline and approve to continue.",
                },
            )
            manager = getattr(resources, "generation_manager", None)
            if manager is None or not hasattr(manager, "wait_for_approval"):
                # Bare runner (no manager to resume through): return blocked.
                return state

            await manager.wait_for_approval()  # cancellable; released by /plan/approve

            state["macro_outline_approved"] = True
            state["awaiting_planning_approval"] = False
            state["planning_block_reason"] = None
            sqlite_db.transition_snapshot_status(
                state["sqlite_db_path"],
                state["planning_snapshot_id"],
                status="approved",
                approved_at=_utc_now_iso(),
            )
            log("approval_applied", snapshot_id=state.get("planning_snapshot_id"))
            await emit(
                "approval_state",
                {
                    "awaiting_approval": False,
                    "macro_outline_ready": True,
                    "macro_outline_approved": True,
                    "message": "Macro outline approved; resuming into scene/beat planning.",
                },
            )
            await emit(
                "phase_change",
                {"phase": "planning", "message": "Resumed just-in-time scene/beat planning."},
            )
    else:
        await _pause_gate(resources)
        if await run_node(
            "node_plan_chapter",
            lambda: node_plan_chapter(state, decider=decider),
        ):
            return state

    # --- Levels 4 + 5: one scene + one beat, just-in-time in both modes. ----
    await _pause_gate(resources)
    if await run_node(
        "node_plan_scene",
        lambda: node_plan_scene(state, decider=decider),
    ):
        return state
    await _pause_gate(resources)
    if await run_node(
        "node_plan_beat",
        lambda: node_plan_beat(state, decider=decider, adapt_fn=None),
    ):
        return state

    await emit(
        "phase_change",
        {"phase": "completed", "message": "Planning vertical slice finished."},
    )
    await emit(
        "done",
        {
            "message": "Planning vertical slice complete.",
            "planning_execution_mode": execution_mode,
            "macro_outline_ready": bool(state.get("macro_outline_ready")),
            "scene_needs_more": state.get("scene_needs_more"),
            "revision_id": state.get("active_planning_revision_id"),
        },
    )
    return state
