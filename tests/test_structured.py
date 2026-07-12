"""Tests for museai.llm.structured — extraction, not prompting.

Several fixtures here are the literal shapes that broke the 2026-07-12
``the-last-postcard`` run: a planner reply truncated mid-JSON whose inner
array parsed as "the plan", a critic preamble containing a bracket, and prose
clean verdicts that burned every re-prompt.
"""

from __future__ import annotations

import pytest

from museai.fsm.state import FailureObject
from museai.llm.structured import (
    StructuredOutputError,
    TruncatedResponseError,
    is_clean_verdict,
    parse_failure_objects,
    parse_json_array,
)

VALID_FINDING = (
    '{"error_code": "CONTRADICTS_PRIOR_PROSE", '
    '"offending_text": "the red door", '
    '"suggested_fix": "make it blue again"}'
)


class TestElementKeysGuard:
    # A beat reply cut off at the token limit: the outer array never closes,
    # so the only balanced array is the inner thread_updates fragment.
    TRUNCATED_BEAT = (
        '[{"intent": "Mara opens the box", '
        '"thread_updates": [{"id": "t1", "status": "resolved"}], '
        '"exit_state": "Mara re'
    )

    def test_a_truncated_reply_must_not_yield_its_inner_array_as_the_plan(self):
        with pytest.raises(StructuredOutputError) as excinfo:
            parse_json_array(self.TRUNCATED_BEAT, what="beats", element_keys=("intent",))
        assert "expected keys" in str(excinfo.value)

    def test_without_the_guard_the_fragment_would_have_passed(self):
        # Documents why element_keys exists: acceptance, not extraction, is
        # what narrows. Extraction still finds the fragment.
        planned = parse_json_array(self.TRUNCATED_BEAT, what="beats")
        assert planned == [{"id": "t1", "status": "resolved"}]

    def test_a_real_plan_with_a_nested_array_still_parses(self):
        text = (
            'Here is the plan:\n```json\n'
            '[{"intent": "x", "thread_updates": [{"id": "t1", "status": "open"}]}]'
            "\n```"
        )
        planned = parse_json_array(text, what="beats", element_keys=("intent",))
        assert planned[0]["intent"] == "x"

    # The production shape from llm_io.log 14:47–14:50: a chapter plan cut at
    # ~130 tokens by the endpoint's default cap, mid-obligations.
    PRODUCTION_PREFIX = (
        '```json\n[\n  {\n    "ordering": 1,\n'
        '    "description": "Elena discovers the undelivered postcard while '
        'locking up the post office for the final time.",\n'
        '    "obligations": [\n'
        '      "Elena physically locks the post office door for the last time.",\n'
        '      "Elena finds the postcard hidden in a segregation bin."'
    )

    def test_a_reply_cut_mid_string_reports_no_array_not_a_wrong_one(self):
        # Cut inside a string: nothing balances, and the error must say so.
        text = self.PRODUCTION_PREFIX[:-1]  # drop the closing quote
        with pytest.raises(StructuredOutputError, match="could not extract"):
            parse_json_array(text, what="chapters", element_keys=("description",))

    def test_a_reply_cut_after_the_inner_array_closed_is_rejected(self):
        # The 14:50:22 failure shape: obligations closed, the outer array did
        # not. A string-element fragment was always rejected (as "element 0 is
        # not a JSON object"); the guard exists for dict-element fragments,
        # which used to be silently accepted. Both must raise.
        text = self.PRODUCTION_PREFIX + "\n    ]"
        with pytest.raises(StructuredOutputError):
            parse_json_array(text, what="chapters", element_keys=("description",))

    def test_any_one_of_the_keys_is_enough(self):
        text = '[{"description": "a chapter"}]'
        assert parse_json_array(
            text, what="chapters", element_keys=("description", "summary")
        )


class TestCriticSpanSearch:
    def test_a_bracketed_preamble_does_not_mask_the_array(self):
        text = f"I found 1 issue [see below]: [{VALID_FINDING}]"
        findings = parse_failure_objects(text)
        assert len(findings) == 1
        assert findings[0].error_code == "CONTRADICTS_PRIOR_PROSE"

    def test_a_bare_object_still_parses(self):
        findings = parse_failure_objects(f"Here you go: {VALID_FINDING}")
        assert len(findings) == 1

    def test_a_bracket_inside_a_finding_does_not_shadow_the_finding(self):
        """The array-first search scans raw text, so a `[...]` inside a string
        value is a candidate span. It must not stop the real object from being
        found once that fragment fails to validate."""
        text = (
            '{"error_code": "CONTRADICTS_CHARACTER", '
            '"offending_text": "Mara lied", '
            '"suggested_fix": "use [\\"silence\\"] instead"}'
        )
        findings = parse_failure_objects(text)
        assert len(findings) == 1
        assert findings[0].suggested_fix == 'use ["silence"] instead'

    @pytest.mark.parametrize(
        "text",
        ["[]", "```json\n[]\n```", "```[]```", "```json\n[]"],
    )
    def test_empty_array_variants_keep_parsing(self, text):
        assert parse_failure_objects(text) == []

    def test_unparseable_text_still_raises(self):
        with pytest.raises(StructuredOutputError):
            parse_failure_objects("I could not really decide about the draft")


class TestCleanVerdict:
    @pytest.mark.parametrize(
        "text",
        [
            "clear",
            "Clean.",
            "No continuity issues found.",
            "None found.",
            "The draft looks good.",
            "Nothing to report.",
        ],
    )
    def test_accepts_unambiguous_clean_prose(self, text):
        assert is_clean_verdict(text)
        assert parse_failure_objects(text) == []

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "The draft is",  # truncated mid-thought
            "Mostly clean, but the timeline might be off",  # hedging
            "It is unclear whether this contradicts chapter two",
            "No issues except the name change",
            "clean " * 60,  # over the length bound
            'clean, see [] above',  # contains a bracket
            "The offending_text is fine",  # schema vocabulary
        ],
    )
    def test_rejects_hedging_truncation_and_schema_talk(self, text):
        assert not is_clean_verdict(text)

    def test_a_hedged_verdict_still_raises_from_the_parser(self):
        with pytest.raises(StructuredOutputError):
            parse_failure_objects("Mostly clean, but the timeline might be off")

    def test_clean_prose_logs_a_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="museai.fsm"):
            assert parse_failure_objects("No continuity issues found.") == []
        assert any("critic_clean_prose" in r.getMessage() for r in caplog.records)


class TestTruncatedResponseError:
    def test_is_a_structured_output_error(self):
        # Callers that catch StructuredOutputError keep working unchanged.
        assert issubclass(TruncatedResponseError, StructuredOutputError)


class TestRegressionShapes:
    def test_a_valid_findings_array_round_trips(self):
        findings = parse_failure_objects(f"[{VALID_FINDING}]")
        assert isinstance(findings[0], FailureObject)

    def test_planner_array_in_a_fence_with_prose(self):
        text = 'Sure!\n```json\n[{"description": "ch1"}, {"description": "ch2"}]\n```\nDone.'
        assert len(parse_json_array(text, what="chapters", element_keys=("description",))) == 2
