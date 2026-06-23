"""Module: M01 (Coordinator & State Machine)
Define typed orchestrator state, FSM pointer, and failure object schemas.
STUB:
"""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict


class FSM_Pointer(BaseModel):
    """Pointer to the current FSM location."""

    model_config = ConfigDict(extra="forbid", strict=True)

    arc_id: str
    chapter_id: str
    scene_id: str
    beat_index: int


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
    }
    unknown_keys = set(overrides) - set(state)
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unknown OrchestratorState override key(s): {unknown}")
    state.update(overrides)
    return state
