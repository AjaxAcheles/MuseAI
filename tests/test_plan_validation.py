"""Tests for deterministic chapter-obligation validation."""

from __future__ import annotations

import json
import re

import pytest

from museai.fsm.plan_validation import validate_concrete_obligation
from museai.llm.prompts import render_messages


def test_story_specific_obligation_is_accepted_and_normalized():
    text, problem = validate_concrete_obligation("  Mara   burns the false letter.  ")
    assert text == "Mara burns the false letter."
    assert problem is None


@pytest.mark.parametrize("value", ["An event that must occur", "  A RESOLUTION OR CONCLUSION TO THE STORY "])
def test_observed_output_schema_placeholders_are_rejected(value):
    text, problem = validate_concrete_obligation(value)
    assert text is None
    assert problem is not None
    assert "output-format placeholder" in problem


@pytest.mark.parametrize("value", ["", "   ", None, 3])
def test_empty_or_non_string_obligations_are_rejected(value):
    text, problem = validate_concrete_obligation(value)
    assert text is None
    assert problem is not None


def test_the_prompt_example_obligations_are_not_placeholders():
    """The chapter_planner template's own example obligations must pass
    validation. When they did not, a small model that echoed the template got
    its plan rejected by the very validator the template teaches it to satisfy."""
    messages = render_messages(
        "chapter_planner",
        project={"genre": "g", "premise": "p", "setting": "s"},
        arc={"description": "d"},
        threads=[],
        characters=[],
        research_mode=False,
    )
    content = messages[-1]["content"]
    match = re.search(r'"obligations":\s*(\[[^\]]*\])', content)
    assert match, "the output-format example must contain an obligations list"
    obligations = json.loads(match.group(1))
    assert obligations, "the example obligations list must be non-empty"
    for obligation in obligations:
        _, problem = validate_concrete_obligation(obligation)
        assert problem is None, f"example obligation is a placeholder: {obligation!r}"