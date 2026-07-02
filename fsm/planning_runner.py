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
Approval blocking uses ``awaiting_planning_approval``/``planning_block_reason``
only — never ``pause_requested``/``hard_stop_asserted``. No drafting module is
routed to.
"""

from __future__ import annotations

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
    return {
        "snapshot_id": snapshot_id,
        "status": snapshot.get("status"),
        "mode": snapshot.get("mode"),
        "active_revision_id": snapshot.get("active_revision_id"),
        "node_counts": dict(Counter(node.get("level") for node in nodes)),
        "revision_count": len(revisions),
        "macro_outline_ready": bool(state.get("macro_outline_ready")),
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

    The returned state is "blocked" when ``planning_block_reason`` is set
    (approval gate, unresolved hard conflict, or planning escalation); the
    caller decides how to present that. Exceptions propagate to the caller.
    """
    config = state.get("app_config") or state.get("config")
    if config is None:
        raise ValueError("state['app_config'] must carry the typed AppConfig")

    # Snapshot the run's modes from config once, at run start (§1.1 contract:
    # nodes read these from state, never from config at node time).
    execution_mode = config.planning.execution_mode
    approval_mode = config.planning.approval_mode
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
        },
    )

    last_snapshot_payload: dict[str, Any] | None = None

    async def run_node(name: str, invoke: Callable[[], Awaitable[dict]]) -> bool:
        """Invoke one planner node, emit its telemetry, return True if blocked."""
        nonlocal state, last_snapshot_payload
        revision_before = state.get("active_planning_revision_id")
        trace_before = len(state.get("planner_deliberation_trace") or [])

        state = await invoke()

        pointer = state.get("fsm_pointer")
        pointer_data = pointer.model_dump() if pointer is not None else {}
        await emit("pointer_update", pointer_data)

        trace = state.get("planner_deliberation_trace") or []
        new_records = trace[trace_before:]
        revision_id = state.get("active_planning_revision_id")
        block_reason = state.get("planning_block_reason")
        await emit(
            "planning_node",
            {
                "node": name,
                "revision_id": revision_id,
                "revision_changed": revision_id != revision_before,
                "deliberation_turns": len(new_records),
                "actions": [
                    record.get("action_type") or record.get("phase")
                    for record in new_records[-8:]
                    if isinstance(record, dict)
                ],
                "block_reason": block_reason,
                "pointer": pointer_data,
            },
        )

        snapshot_payload = _snapshot_payload(state)
        if snapshot_payload is not None and snapshot_payload != last_snapshot_payload:
            last_snapshot_payload = snapshot_payload
            await emit("planning_snapshot", snapshot_payload)

        if block_reason:
            await emit(
                "planning_blocked",
                {
                    "reason": block_reason,
                    "node": name,
                    "awaiting_approval": bool(state.get("awaiting_planning_approval")),
                },
            )
            if state.get("awaiting_planning_approval"):
                await emit(
                    "approval_state",
                    {
                        "awaiting_approval": True,
                        "macro_outline_ready": bool(state.get("macro_outline_ready")),
                        "macro_outline_approved": bool(state.get("macro_outline_approved")),
                        "note": (
                            "Approval gate reached. Approval resume is not "
                            "implemented in this vertical slice."
                        ),
                    },
                )
            return True
        return False

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
            if await run_node(
                "node_plan_chapter",
                lambda: node_plan_chapter(state, decider=decider),
            ):
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
        if state.get("awaiting_planning_approval"):
            # Defensive: the chapter node emits its own block reason; this
            # branch only guards an approval pause without a block reason.
            return state
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
