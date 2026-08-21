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
    AmbiguousStructuredOutputError,
    StructuredOutputError,
    TruncatedResponseError,
    is_clean_verdict,
    parse_failure_objects,
    parse_json_array,
    response_truncation_remedy,
    truncation_remedy,
    validate_plain_text_response,
)

VALID_FINDING = (
    '{"error_code": "CONTRADICTS_PRIOR_PROSE", '
    '"offending_text": "the red door", '
    '"suggested_fix": "make it blue again", '
    '"critic_source": "continuity_critic"}'
)


class TestTruncationRemedy:
    """Which knob a cut-off reply actually points at.

    The 2026-07-24 failure: OLLAMA_CONTEXT_LENGTH was unset, so Ollama served a
    4096-token window against a declared 16384 while ignoring the num_ctx in
    extra_body. Every role died with generic "shorten the prompt" advice, which
    was the wrong fix — the prompt was sized correctly for the window the config
    described, and only the server was wrong.
    """

    def test_a_partial_reply_points_at_the_output_cap(self):
        remedy = truncation_remedy(empty=False)
        assert "max_output_tokens" in remedy

    def test_a_mismatched_window_is_named_with_both_numbers(self):
        remedy = truncation_remedy(
            empty=True,
            context_window=16384,
            served_prompt_tokens=4090,
            served_completion_tokens=6,
            mismatch_fraction=0.9,
        )
        assert "4096" in remedy and "16384" in remedy
        assert "OLLAMA_CONTEXT_LENGTH" in remedy
        # The distinguishing claim: the server, not the prompt, is the problem.
        assert "smaller window than the config believes" in remedy

    def test_a_genuinely_full_window_keeps_the_generic_advice(self):
        # Served tokens reach the declared window, so the config is honest and
        # the prompt really is too long. Naming a mismatch here would be a lie.
        remedy = truncation_remedy(
            empty=True,
            context_window=16384,
            served_prompt_tokens=16380,
            served_completion_tokens=0,
            mismatch_fraction=0.9,
        )
        assert "smaller window than the config believes" not in remedy
        assert "shorten the prompt" in remedy

    def test_without_served_counts_the_diagnosis_is_not_guessed(self):
        # No usage block means no evidence about the server's real window, and
        # an unfounded mismatch claim would send the reader after the wrong fix.
        remedy = truncation_remedy(
            empty=True, context_window=16384, served_prompt_tokens=None
        )
        assert "smaller window than the config believes" not in remedy

    def test_the_ollama_extra_body_trap_is_called_out(self):
        # extra_body's options.num_ctx is silently dropped by Ollama's /v1
        # endpoint, so advice that recommends it sends the reader in a circle.
        assert "silently" in truncation_remedy(empty=True)

    def test_response_helper_reads_a_response_and_endpoint(self):
        class _Response:
            text = ""
            served_prompt_tokens = 4090
            served_completion_tokens = 6

        class _Endpoint:
            context_window = 16384

        remedy = response_truncation_remedy(_Response(), _Endpoint())
        assert "4096" in remedy and "16384" in remedy

    def test_response_helper_degrades_on_an_unknown_shape(self):
        # A stub or an older response shape must not raise a second error on top
        # of the truncation it was called to explain.
        remedy = response_truncation_remedy(object(), None)
        assert "shorten the prompt" in remedy

    def test_a_reasoning_model_burning_its_cap_is_not_blamed_on_the_server(self):
        # 2026-07-24/25 live session: 28 of 30 critic truncations had
        # served_completion_tokens == max_output_tokens exactly, with a
        # declared context_window nowhere near full. The old code still
        # blamed OLLAMA_CONTEXT_LENGTH in all 28 cases.
        remedy = truncation_remedy(
            empty=True,
            context_window=16384,
            served_prompt_tokens=6362,
            served_completion_tokens=2048,
            max_output_tokens=2048,
            mismatch_fraction=0.9,
        )
        assert "OLLAMA_CONTEXT_LENGTH" not in remedy
        assert "max_output_tokens" in remedy
        assert "2048" in remedy

    def test_the_cap_diagnosis_names_reasoning_when_thinking_is_present(self):
        remedy = truncation_remedy(
            empty=True,
            served_completion_tokens=2048,
            max_output_tokens=2048,
            thinking="the door was red earlier so this scene contradicts...",
        )
        assert "reasoning_effort: none" in remedy
        assert "agents.<role>" in remedy
        assert "raise endpoint.max_output_tokens" not in remedy

    def test_the_cap_diagnosis_keeps_the_raise_cap_advice_without_thinking(self):
        remedy = truncation_remedy(
            empty=True,
            served_completion_tokens=2048,
            max_output_tokens=2048,
            thinking="",
        )
        assert "raise endpoint.max_output_tokens" in remedy
        assert "reasoning_effort" not in remedy

    def test_the_4096_vs_16384_case_still_produces_todays_message(self):
        # Run 2, 2026-07-24 18:11:36: a genuine served-window mismatch, no cap
        # involved. The new branch must not swallow this case.
        remedy = truncation_remedy(
            empty=True,
            context_window=16384,
            served_prompt_tokens=2792,
            served_completion_tokens=1304,
            max_output_tokens=None,
            mismatch_fraction=0.9,
        )
        assert "4096" in remedy and "16384" in remedy
        assert "OLLAMA_CONTEXT_LENGTH" in remedy

    def test_response_helper_threads_cap_and_thinking_through(self):
        class _Response:
            text = ""
            thinking = "reasoning about the contradiction at length"
            served_prompt_tokens = 6362
            served_completion_tokens = 2048

        class _Endpoint:
            context_window = 16384
            max_output_tokens = 2048

        remedy = response_truncation_remedy(_Response(), _Endpoint())
        assert "OLLAMA_CONTEXT_LENGTH" not in remedy
        assert "2048" in remedy


    def test_a_role_names_the_per_agent_cap_that_bound_the_call(self):
        class _Response:
            text = "partial verdict"

        remedy = response_truncation_remedy(_Response(), object(), role="critic")
        assert "agents.critic.max_output_tokens" in remedy
        assert "endpoint.max_output_tokens" not in remedy

    def test_an_unspecified_role_keeps_the_existing_wording_byte_for_byte(self):
        class _Response:
            text = "partial verdict"

        assert response_truncation_remedy(_Response(), object()) == (
            "raise endpoint.max_output_tokens (or leave it unset to omit the cap) "
            "or ask for a shorter answer"
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
        assert "could not extract a JSON array" in str(excinfo.value)

    def test_without_the_guard_the_truncated_outer_array_is_still_rejected(self):
        with pytest.raises(StructuredOutputError):
            parse_json_array(self.TRUNCATED_BEAT, what="beats")

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

    def test_a_bare_object_is_rejected(self):
        with pytest.raises(StructuredOutputError):
            parse_failure_objects(f"Here you go: {VALID_FINDING}")

    def test_a_bracket_inside_a_finding_does_not_shadow_the_finding(self):
        """The array-first search scans raw text, so a `[...]` inside a string
        value is a candidate span. It must not stop the real object from being
        found once that fragment fails to validate."""
        text = "[" + (
            '{"error_code": "CONTRADICTS_CHARACTER", '
            '"offending_text": "Mara lied", '
            '"suggested_fix": "use [\\"silence\\"] instead", '
            '"critic_source": "continuity_critic"}'
        ) + "]"
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
    VERBOSE_CLEAN_REPLY = (
        "The draft is clean. It fulfills all obligations in <beat_goal>: it shows Nell "
        "standing with the cracked ladder on her grass, establishes that nobody has "
        "seen it break (no neighbour or Ida noticed), and makes available the option of "
        "leaning it against Ida's shed without saying anything. The prose also "
        "correctly conveys that the garden forgets once items cross back, which is "
        "exactly the mechanism needed to justify Nell's silence. No contradictions with "
        "<story_world>, <characters>, <open_threads>..."
    )

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

    def test_accepts_the_observed_verbose_clean_reply(self):
        """The 2026-08-18 clean reply exceeded the old 240-character cap."""
        assert len(self.VERBOSE_CLEAN_REPLY) > 240
        assert is_clean_verdict(self.VERBOSE_CLEAN_REPLY)
        assert parse_failure_objects(self.VERBOSE_CLEAN_REPLY) == []

    def test_rejects_a_clean_phrase_that_arrives_after_discussion(self):
        text = (
            "The beat follows the recent prose and lands its required change. "
            "The draft is clean."
        )
        assert not is_clean_verdict(text)
        with pytest.raises(StructuredOutputError):
            parse_failure_objects(text)

    @pytest.mark.parametrize(
        "tail",
        [
            " It might still omit a detail.",
            " Its offending_text is irrelevant.",
            " The reviewer wrote [clean] in the margin.",
        ],
    )
    def test_long_clean_reply_keeps_whole_text_safety_guards(self, tail):
        text = "The draft is clean. " + ("Its obligations are all present. " * 12) + tail
        assert len(text) > 240
        assert not is_clean_verdict(text)
        with pytest.raises(StructuredOutputError):
            parse_failure_objects(text)

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "The draft is",  # truncated mid-thought
            "Mostly clean, but the timeline might be off",  # hedging
            "It is unclear whether this contradicts chapter two",
            "No issues except the name change",
            "clean " * 150,  # over the length bound
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

    def test_critic_source_is_optional_not_required(self):
        # state.py's FailureObject already defaults critic_source, and v1 has
        # exactly one critic — omitting the field is not a schema deviation.
        # A model that leaves it out must not be re-prompted for a value the
        # parser was always going to supply itself.
        three_key_finding = (
            '{"error_code": "CONTRADICTS_PRIOR_PROSE", '
            '"offending_text": "the red door", '
            '"suggested_fix": "make it blue again"}'
        )
        findings = parse_failure_objects(f"[{three_key_finding}]")
        assert len(findings) == 1
        assert findings[0].critic_source == "continuity_critic"

    def test_an_invalid_critic_source_is_still_rejected(self):
        # Optional, not unchecked: a model naming some other source is a real
        # schema deviation, distinct from simply leaving the key out.
        bad_source = (
            '{"error_code": "CONTRADICTS_PRIOR_PROSE", '
            '"offending_text": "the red door", '
            '"suggested_fix": "make it blue again", '
            '"critic_source": "some_other_critic"}'
        )
        with pytest.raises(StructuredOutputError, match="invalid critic_source"):
            parse_failure_objects(f"[{bad_source}]")

    def test_planner_array_in_a_fence_with_prose(self):
        text = 'Sure!\n```json\n[{"description": "ch1"}, {"description": "ch2"}]\n```\nDone.'
        assert len(parse_json_array(text, what="chapters", element_keys=("description",))) == 2


class TestAdversarialSerialization:
    @pytest.mark.parametrize(
        "payload",
        [
            '[{"description":"first"}][{"description":"second"}]',
            '[{"description":"first"}]\n[{"description":"second"}]',
        ],
    )
    def test_concatenated_or_repeated_arrays_are_ambiguous(self, payload):
        with pytest.raises(AmbiguousStructuredOutputError):
            parse_json_array(payload, what="chapters", element_keys=("description",))

    def test_duplicate_keys_are_rejected_instead_of_last_value_winning(self):
        payload = '[{"description":"safe","description":"overwritten"}]'
        with pytest.raises(StructuredOutputError, match="duplicate JSON key"):
            parse_json_array(payload, what="chapters")

    @pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_numbers_are_not_treated_as_json(self, constant):
        with pytest.raises(StructuredOutputError, match="non-finite JSON number"):
            parse_json_array(f'[{{"value":{constant}}}]', what="items")

    def test_a_single_object_is_not_coerced_to_an_array(self):
        with pytest.raises(StructuredOutputError, match="expected a JSON array"):
            parse_json_array('{"description":"only"}', what="chapters")


class TestPlainTextBoundary:
    @pytest.mark.parametrize(
        "payload",
        [
            "```markdown\nMara opened the door.\n```",
            '<think>reasoning</think>\nMara opened the door.',
            '{"name":"read_context","parameters":{"id":"x"}}',
            'I will use this call: {"name":"read_context","arguments":{"id":"x"}}',
            "Here is the revised prose: Mara opened the door.",
            "Mara opened the door.\nHope this helps!",
        ],
    )
    def test_formatting_artifacts_are_rejected(self, payload):
        with pytest.raises(StructuredOutputError):
            validate_plain_text_response(payload, what="draft")

    def test_unicode_and_literal_newlines_remain_valid_prose(self):
        payload = "Mara said, “Déjà vu.”\n\nSnow gathered at the café door."
        assert validate_plain_text_response(payload, what="draft") == payload
