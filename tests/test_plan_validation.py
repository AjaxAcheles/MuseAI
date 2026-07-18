"""Tests for deterministic chapter-obligation validation."""

from __future__ import annotations

import pytest

from museai.fsm.plan_validation import validate_concrete_obligation


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