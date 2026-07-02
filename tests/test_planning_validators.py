"""Module: M05 (Hierarchical Planning Cascade)
Pure, deterministic tests for planner validators and the required-check runner.
"""

from types import SimpleNamespace

import pytest

from fsm.planning_validators import (
    REQUIRED_CHECK_REGISTRY,
    ValidationResult,
    run_validators,
    validate_continuity_warnings,
    validate_depth,
    validate_no_drafting,
    validate_ordering,
    validate_required_fields,
    validate_schema,
    validate_thread_coverage,
    validate_user_annotations,
)


def _config(checks_by_level):
    return SimpleNamespace(
        planning=SimpleNamespace(planner_required_checks=checks_by_level)
    )


def _well_formed_chapter_plan(**overrides):
    plan = {
        "title": "chapter-1 plan",
        "obligations": ["advance the open thread"],
        "thread_updates": [{"thread_id": "thread-1", "status": "progressing"}],
        "annotation_outcomes": {"ann-1": "applied"},
        "asserted_facts": [
            {"subject": "gate", "attribute": "state", "value": "locked"}
        ],
        "beats": [{"beat_index": 0}, {"beat_index": 1}],
    }
    plan.update(overrides)
    return plan


def test_validate_schema_passes_valid_plan_and_fails_matching_defects():
    assert validate_schema({"title": "plan"}, "chapter", required_keys=["title"]).passes

    missing = validate_schema({"summary": "plan"}, "chapter", required_keys=["title"])
    wrong_type = validate_schema(["not", "a", "dict"], "chapter")

    assert missing.failed_checks == ["validate_schema"]
    assert missing.details["validate_schema"]["reason"] == "missing_keys"
    assert wrong_type.failed_checks == ["validate_schema"]
    assert wrong_type.details["validate_schema"]["reason"] == "plan_not_dict"


def test_validate_required_fields_passes_valid_plan_and_fails_empty_fields():
    assert validate_required_fields(
        {"title": "plan", "purpose": "move thread"}, "chapter", required_fields=["title"]
    ).passes

    result = validate_required_fields(
        {"title": ""}, "chapter", required_fields=["title", "purpose"]
    )

    assert result.failed_checks == ["validate_required_fields"]
    assert result.details["validate_required_fields"]["missing_or_empty"] == [
        "title",
        "purpose",
    ]


def test_validate_ordering_passes_gapless_order_and_fails_gap_or_unorderable_items():
    assert validate_ordering({"scenes": [{"ordering": 0}, {"ordering": 1}]}).passes

    gap = validate_ordering({"scenes": [{"ordering": 0}, {"ordering": 2}]})
    unorderable = validate_ordering({"beats": [{"beat_index": 0}, {"id": "beat-2"}]})

    assert gap.failed_checks == ["validate_ordering"]
    assert gap.details["validate_ordering"]["offending"][0]["reason"] == (
        "not_monotonic_gapless"
    )
    assert unorderable.failed_checks == ["validate_ordering"]
    assert unorderable.details["validate_ordering"]["offending"][0]["reason"] == (
        "unorderable_items"
    )


def test_validate_no_drafting_passes_structure_and_fails_prose_anywhere():
    assert validate_no_drafting({"objective": "reach the gate"}).passes

    top_level = validate_no_drafting({"prose": "The gate creaked."})
    nested = validate_no_drafting({"beats": [{"draft_text": "No draft in plans."}]})

    assert top_level.failed_checks == ["validate_no_drafting"]
    assert top_level.details["validate_no_drafting"]["prose_fields"] == ["prose"]
    assert nested.failed_checks == ["validate_no_drafting"]
    assert nested.details["validate_no_drafting"]["prose_fields"] == [
        "beats[0].draft_text"
    ]


def test_validate_thread_coverage_passes_required_threads_and_fails_uncovered():
    plan = {"thread_updates": [{"thread_id": "thread-1"}]}

    assert validate_thread_coverage(plan, required_threads=["thread-1"]).passes

    result = validate_thread_coverage(plan, required_threads=["thread-1", "thread-2"])

    assert result.failed_checks == ["validate_thread_coverage"]
    assert result.details["validate_thread_coverage"]["uncovered_threads"] == [
        "thread-2"
    ]


def test_validate_user_annotations_passes_applied_hard_notes_and_fails_missing_outcomes():
    plan = {"annotation_outcomes": {"ann-1": "applied"}}
    hard = [{"annotation_id": "ann-1"}, {"annotation_id": "ann-2"}]

    assert validate_user_annotations(
        plan, hard_annotations=[{"annotation_id": "ann-1"}]
    ).passes

    result = validate_user_annotations(plan, hard_annotations=hard)

    assert result.failed_checks == ["validate_user_annotations"]
    assert result.details["validate_user_annotations"]["unsatisfied"] == [
        {"annotation_id": "ann-2", "outcome": None}
    ]


def test_validate_continuity_passes_matching_facts_and_fails_contradictions():
    continuity = [{"subject": "gate", "attribute": "state", "value": "locked"}]
    matching = {
        "asserted_facts": [
            {"subject": "gate", "attribute": "state", "value": "locked"}
        ]
    }
    conflicting = {
        "asserted_facts": [
            {"subject": "gate", "attribute": "state", "value": "open"}
        ]
    }

    assert validate_continuity_warnings(
        matching, continuity_facts=continuity
    ).passes

    result = validate_continuity_warnings(conflicting, continuity_facts=continuity)

    assert result.failed_checks == ["validate_continuity_warnings"]
    assert result.details["validate_continuity_warnings"]["contradictions"][0][
        "known_values"
    ] == ["locked"]


def test_validate_depth_passes_level_shape_and_fails_wrong_granularity():
    assert validate_depth({"obligations": ["scene constraints"]}, "chapter").passes

    result = validate_depth({"obligations": ["x"], "scenes": [{"id": "scene-1"}]}, "chapter")
    scene_result = validate_depth({"entry_state": "outside"}, "scene")

    assert result.failed_checks == ["validate_depth"]
    assert result.details["validate_depth"]["forbidden_granularity"] == ["scenes"]
    assert scene_result.failed_checks == ["validate_depth"]
    assert scene_result.details["validate_depth"]["missing_granularity"] == [
        "exit_state"
    ]


def test_run_validators_runs_exactly_configured_checks(monkeypatch):
    called = []

    def _configured_only(plan, level, constraints, continuity):
        called.append((plan, level, constraints, continuity))
        return ValidationResult(passes=True)

    monkeypatch.setitem(REQUIRED_CHECK_REGISTRY, "configured_only", _configured_only)
    config = _config({"chapter": ["configured_only"]})

    result = run_validators("chapter", {"anything": True}, {}, {}, config)

    assert result.passes is True
    assert result.failed_checks == []
    assert len(called) == 1
    assert called[0][1] == "chapter"


def test_run_validators_passes_valid_plan_and_reports_right_failed_checks():
    config = _config(
        {
            "chapter": [
                "schema",
                "required_fields",
                "thread_coverage",
                "annotation_satisfaction",
                "continuity",
                "depth",
                "ordering",
                "no_drafting",
            ]
        }
    )
    constraints = {
        "required_fields": ["title", "obligations"],
        "required_threads": ["thread-1"],
        "hard_annotations": [{"annotation_id": "ann-1"}],
    }
    continuity = {
        "facts": [{"subject": "gate", "attribute": "state", "value": "locked"}]
    }

    valid = run_validators(
        "chapter", _well_formed_chapter_plan(), constraints, continuity, config
    )
    invalid_plan = _well_formed_chapter_plan(
        title="",
        thread_updates=[],
        annotation_outcomes={"ann-1": "rejected"},
        asserted_facts=[{"subject": "gate", "attribute": "state", "value": "open"}],
        scenes=[{"ordering": 0}],
        beats=[{"beat_index": 0}, {"beat_index": 2}],
        prose="Narrative text belongs outside plans.",
    )
    invalid = run_validators("chapter", invalid_plan, constraints, continuity, config)

    assert valid.passes is True
    assert invalid.passes is False
    assert invalid.failed_checks == [
        "required_fields",
        "thread_coverage",
        "annotation_satisfaction",
        "continuity",
        "depth",
        "ordering",
        "no_drafting",
    ]


def test_run_validators_fails_closed_for_configured_missing_check():
    config = _config({"chapter": ["schema", "missing_required_check"]})

    with pytest.raises(ValueError, match="unknown required check"):
        run_validators("chapter", {"title": "plan"}, {}, {}, config)


def test_run_validators_ignores_planner_self_check():
    config = _config({"chapter": ["schema", "no_drafting"]})
    plan = {
        "title": "plan",
        "self_check": {
            "schema_valid": False,
            "continuity_checked": False,
            "depth_checked": False,
        },
    }

    result = run_validators("chapter", plan, {}, {}, config)

    assert result.passes is True
    assert result.failed_checks == []
