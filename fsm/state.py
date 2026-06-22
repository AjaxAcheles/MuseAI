"""Module: M01 (Coordinator & State Machine)
Define typed orchestrator state, FSM pointer, and failure object schemas.
STUB:
"""

from typing import TypedDict


class FSM_Pointer(TypedDict, total=False):
    """Pointer to the current FSM location."""


class FailureObject(TypedDict, total=False):
    """Structured failure details for recovery routing."""


class OrchestratorState(TypedDict, total=False):
    """Shared state passed between FSM nodes."""
