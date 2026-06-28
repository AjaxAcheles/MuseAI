"""Module: M01 (Coordinator & State Machine)
Define typed orchestrator state, FSM pointer, and failure object schemas, including
the planning-subsystem fields the five-level planner cascade reads and mutates.
"""

import operator
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict


class FSM_Pointer(BaseModel):
    """Pointer to the current FSM location."""

    model_config = ConfigDict(extra="forbid", strict=True)

    arc_id: str
    chapter_id: str
    scene_id: str
    # `beat_index` is in-scene *position* (ordering), while `beat_id` below is the
    # current beat's *identity* — they are not the same field.
    beat_index: int
    # Identity of the current beat (matches Beats.id / a beat PlanningNode). Defaults to
    # "" so existing FSM_Pointer(...) construction sites keep working under
    # extra="forbid"/strict=True until the planner populates a real beat id.
    beat_id: str = ""


class FailureObject(BaseModel):
    """Structured failure details for recovery routing."""

    model_config = ConfigDict(extra="forbid", strict=True)

    error_code: str
    offending_text: str
    suggested_fix: str
    critic_source: str


def failure_object_json_schema() -> dict:
    """Return the JSON schema used to constrain critic failure output."""

    return FailureObject.model_json_schema()


def accumulate_or_reset(current: list, incoming: list) -> list:
    """Append non-empty contributions; treat an explicit empty list as reset."""

    if incoming == []:
        return []
    return [*current, *incoming]


class OrchestratorState(TypedDict):
    """Shared state passed between FSM nodes."""

    project_id: str
    fsm_pointer: FSM_Pointer
    active_context_package: dict[str, Any]
    current_draft_text: str
    streaming_buffer: str
    critic_failures: Annotated[list[FailureObject], accumulate_or_reset]
    stylometric_distance: float
    retry_count: int
    replan_count: int
    escalation_tier: int
    has_paradox: bool
    transient_dc_override: float | None
    pause_requested: bool
    hard_stop_asserted: bool
    failed_beat_cache: Annotated[list[dict[str, Any]], accumulate_or_reset]
    best_seen_draft: str | None

    # --- Planning subsystem (five-level cascade, deliberation loops, macro-outline +
    # approval). Approval-waiting is a safe boundary, never pause_requested /
    # hard_stop_asserted (Data_Structures.md §1.1; Module_Ability_Specification §5). ---
    planning_execution_mode: str
    approval_mode: str
    macro_outline_ready: bool
    macro_outline_approved: bool
    awaiting_planning_approval: bool
    planning_snapshot_id: str | None
    active_planning_revision_id: str | None
    planner_loop_index: int
    planner_tool_call_count: int
    # operator.add (append-only), NOT accumulate_or_reset: planner-action trace is
    # observability history that must never be reset by an empty contribution.
    planner_deliberation_trace: Annotated[list[dict[str, Any]], operator.add]
    planning_block_reason: str | None


def make_initial_state(
    project_id: str, fsm_pointer: FSM_Pointer, **overrides: Any
) -> OrchestratorState:
    """Create a fresh initial FSM state with optional explicit overrides."""

    state: OrchestratorState = {
        "project_id": project_id,
        "fsm_pointer": fsm_pointer,
        "active_context_package": {},
        "current_draft_text": "",
        "streaming_buffer": "",
        "critic_failures": [],
        "stylometric_distance": 0.0,
        "retry_count": 0,
        "replan_count": 0,
        "escalation_tier": 0,
        "has_paradox": False,
        "transient_dc_override": None,
        "pause_requested": False,
        "hard_stop_asserted": False,
        "failed_beat_cache": [],
        "best_seen_draft": None,
        # Planning subsystem — §1.x "Init on run" defaults. The mode fields default to
        # the system default and stay overridable via **overrides; Build 14 snapshots
        # the real config.planning.* values at graph entry (never read config here).
        "planning_execution_mode": "macro_outline_before_draft",
        "approval_mode": "off",
        "macro_outline_ready": False,
        "macro_outline_approved": False,
        "awaiting_planning_approval": False,
        "planning_snapshot_id": None,
        "active_planning_revision_id": None,
        "planner_loop_index": 0,
        "planner_tool_call_count": 0,
        "planner_deliberation_trace": [],
        "planning_block_reason": None,
    }
    unknown_keys = set(overrides) - set(state)
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unknown OrchestratorState override key(s): {unknown}")
    state.update(overrides)
    return state
