"""Tests for museai.llm.tokenizer."""

from __future__ import annotations

import pytest

from museai.llm.tokenizer import (
    MESSAGE_OVERHEAD_TOKENS,
    REPLY_PRIMING_TOKENS,
    count_message_tokens,
    count_tokens,
)


class TestCharHeuristic:
    def test_empty_text_is_zero(self):
        assert count_tokens("", "char_heuristic") == 0

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("a", 1),  # ceil(1/4)
            ("abcd", 1),  # exactly 4 chars
            ("abcde", 2),  # ceil(5/4)
            ("hello world", 3),  # ceil(11/4)
            ("x" * 400, 100),
        ],
    )
    def test_ceil_of_length_over_four(self, text, expected):
        assert count_tokens(text, "char_heuristic") == expected

    def test_model_name_is_ignored(self):
        assert count_tokens("hello world", "char_heuristic", "any-model") == 3


class TestTiktoken:
    def test_returns_positive_int_for_unknown_model(self):
        count = count_tokens("hello world", "tiktoken", "some-unknown-model")
        assert isinstance(count, int)
        assert count > 0

    def test_returns_positive_int_without_model_name(self):
        assert count_tokens("hello world", "tiktoken") > 0

    def test_empty_text_is_zero(self):
        assert count_tokens("", "tiktoken") == 0

    def test_longer_text_costs_more(self):
        short = count_tokens("hello", "tiktoken")
        long = count_tokens("hello " * 50, "tiktoken")
        assert long > short

    def test_encoder_is_cached_across_calls(self):
        # Second call must not rebuild the BPE table; identity of the cached
        # encoder is what makes repeated counting cheap.
        from museai.llm.tokenizer import _get_encoder

        assert _get_encoder("some-unknown-model") is _get_encoder("some-unknown-model")


class TestUnknownFamily:
    def test_count_tokens_rejects_unknown_family(self):
        with pytest.raises(ValueError, match="unknown tokenizer_family"):
            count_tokens("hello", "hf_auto")

    def test_count_message_tokens_rejects_unknown_family(self):
        with pytest.raises(ValueError, match="unknown tokenizer_family"):
            count_message_tokens([{"role": "user", "content": "hi"}], "hf_auto")

    def test_unknown_family_rejected_even_for_empty_messages(self):
        with pytest.raises(ValueError, match="unknown tokenizer_family"):
            count_message_tokens([], "nonsense")


class TestMessageTokens:
    def test_empty_message_list_is_zero(self):
        assert count_message_tokens([], "char_heuristic") == 0

    def test_includes_per_message_overhead_and_reply_priming(self):
        messages = [{"role": "user", "content": "abcd"}]
        # role "user" -> ceil(4/4)=1, content "abcd" -> 1, plus overhead+priming
        expected = REPLY_PRIMING_TOKENS + MESSAGE_OVERHEAD_TOKENS + 1 + 1
        assert count_message_tokens(messages, "char_heuristic") == expected

    def test_overhead_scales_with_message_count(self):
        one = count_message_tokens([{"role": "user", "content": "abcd"}], "char_heuristic")
        two = count_message_tokens(
            [{"role": "user", "content": "abcd"}] * 2, "char_heuristic"
        )
        assert two - one == MESSAGE_OVERHEAD_TOKENS + 1 + 1

    def test_counts_more_than_content_alone(self):
        messages = [
            {"role": "system", "content": "You are a writer."},
            {"role": "user", "content": "Write a chapter."},
        ]
        content_only = sum(
            count_tokens(m["content"], "char_heuristic") for m in messages
        )
        assert count_message_tokens(messages, "char_heuristic") > content_only

    def test_non_string_fields_are_counted(self):
        """A tool-call message body still contributes tokens."""
        plain = count_message_tokens([{"role": "assistant", "content": ""}], "char_heuristic")
        with_tools = count_message_tokens(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "search", "arguments": "{}"}}
                    ],
                }
            ],
            "char_heuristic",
        )
        assert with_tools > plain

    def test_tiktoken_family_counts_messages(self):
        messages = [{"role": "user", "content": "hello world"}]
        assert count_message_tokens(messages, "tiktoken", "some-unknown-model") > 0
