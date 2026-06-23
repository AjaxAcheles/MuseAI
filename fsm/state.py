"""Module: M01 (Coordinator & State Machine)
Define typed orchestrator state, FSM pointer, and failure object schemas.
STUB:
"""

from typing import TypedDict

from pydantic import BaseModel, ConfigDict


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


class OrchestratorState(TypedDict, total=False):
    """Shared state passed between FSM nodes."""
