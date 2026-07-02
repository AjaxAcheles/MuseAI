"""Module: M05 (Hierarchical Planning Cascade)
Deterministic schema tests for the PlannerAction deliberation action contract.
"""

import json

import pytest
from pydantic import ValidationError

from fsm.planning_actions import PlannerAction, planner_action_json_schema


@pytest.mark.parametrize(
    "payload",
    [
        {
            "action_type": "call_tool",
            "tool_name": "read_arcs",
            "tool_args": {"arc_id": "arc-1"},
        },
        {
            "action_type": "revise_plan",
            "revised_plan": {"title": "candidate"},
            "change_summary": "tighten structure",
        },
        {
            "action_type": "finalize_plan",
            "final_plan": {"title": "final"},
            "self_check": {"schema_valid": False},
        },
        {
            "action_type": "raise_conflict",
            "conflicts": [
                {
                    "conflict_type": "annotation_conflict",
                    "description": "pin and remove target the same node",
                    "requires_user_resolution": True,
                }
            ],
        },
    ],
)
def test_four_core_planner_action_types_construct(payload):
    """The four design action types validate through the strict model."""

    action = PlannerAction(**payload)

    assert action.action_type == payload["action_type"]


def test_unknown_action_type_is_rejected():
    """A non-enumerated action type cannot enter the harness."""

    with pytest.raises(ValidationError):
        PlannerAction(action_type="draft_prose", final_plan={"title": "bad"})


def test_unexpected_extra_fields_are_rejected():
    """PlannerAction keeps Pydantic extra='forbid' at the LLM boundary."""

    with pytest.raises(ValidationError):
        PlannerAction(action_type="finalize_plan", final_plan={"title": "ok"}, prose="no")


def test_planner_action_json_schema_is_usable():
    """The helper returns a serializable schema with the strict action selector."""

    schema = planner_action_json_schema()

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert "action_type" in schema["required"]
    action_type = schema["properties"]["action_type"]
    assert {"call_tool", "revise_plan", "finalize_plan", "raise_conflict"}.issubset(
        set(action_type["enum"])
    )
    assert json.loads(json.dumps(schema)) == schema
