"""Module: M05 (Hierarchical Planning Cascade)
Define the strict PlannerAction schema every planner level's LLM must return on each
turn of its bounded, harness-owned deliberation loop.

The deliberation loop is harness-owned: the model emits exactly one structured action
per turn; the harness validates it, executes any permitted tool, and decides whether a
`finalize_plan` is accepted. The model proposes; the harness disposes. The five
`action_type` values are exhaustive — `extra='forbid'` plus the `Literal` reject any
unknown action type or unexpected field, so a malformed action fails validation rather
than being silently coerced.

`self_check` is advisory metadata only. The programmatic harness validators
(`planner_required_checks` for the level, in M05's `fsm/planning_validators.py`) are the
authoritative acceptance gate; a model's self-reported `self_check` is never trusted for
accepting a `finalize_plan`.

This module defines the action schema only. The deliberation loop, the per-level tool
registry / permission matrix, the validators, and any node logic live in later M05
increments. Nothing here is wired to `call_llm`; M04 parses PlannerAction as a generic
structured output (see Module_Ability_Specification §4) when the planner nodes land.
"""

from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict


class PlannerAction(BaseModel):
    """One structured action a planner LLM returns per deliberation turn (§1.4).

    `action_type` selects which payload fields are meaningful; all payload fields are
    optional so a single model can carry any one action. Acceptance of `finalize_plan`
    is gated by the harness validators, never by `self_check`.
    """

    model_config = ConfigDict(extra="forbid")

    action_type: Literal[
        "call_tool", "revise_plan", "finalize_plan", "raise_conflict", "continue_deliberation"
    ]
    # call_tool:
    tool_name: str | None = None
    tool_args: Dict | None = None
    # revise_plan:
    revised_plan: Dict | None = None
    change_summary: str | None = None
    # finalize_plan:
    final_plan: Dict | None = None
    self_check: Dict | None = None  # {schema_valid, continuity_checked, depth_checked, user_annotations_addressed}
    # raise_conflict:
    conflicts: List[Dict] | None = None  # [{conflict_type, description, requires_user_resolution}]
    # continue_deliberation: a "think" turn — the planner records reasoning (in `rationale`)
    # and asks for another turn without finalizing. The harness threads the note forward and
    # does not count it as a wasted turn or charge the tool budget.
    rationale: str | None = None


def planner_action_json_schema() -> dict:
    """Return the JSON schema used to constrain planner deliberation output.

    Mirrors `fsm.state.failure_object_json_schema` so M04's structured-output path can
    later consume PlannerAction as a generic structured output.
    """

    return PlannerAction.model_json_schema()
