"""Module: M04 (LLM Inference Boundary)
Synthetic tests for endpoint-routed token counting.
"""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import pytest

from llm import tokenizer
from llm.tokenizer import count_message_tokens, count_payload_tokens, count_tokens


def test_char_heuristic_counts_empty_and_short_strings() -> None:
    assert count_tokens("", "char_heuristic") == 0
    assert count_tokens("a", "char_heuristic") == 1
    assert count_tokens("abcd", "char_heuristic") == 1
    assert count_tokens("abcde", "char_heuristic") == 2


def test_tiktoken_known_or_unknown_model_counts_nonempty_text() -> None:
    known_count = count_tokens("hello world", "tiktoken", "gpt-4o")
    unknown_count = count_tokens(
        "hello world", "tiktoken", "definitely-unknown-tokenizer-model"
    )

    assert known_count > 0
    assert unknown_count > 0


def test_message_tokens_are_deterministic_and_include_structure() -> None:
    messages = [{"role": "user", "content": "hello world"}]

    first = count_message_tokens(messages, "char_heuristic")
    second = count_message_tokens(messages, "char_heuristic")

    assert first == second
    assert first > count_tokens("hello world", "char_heuristic")
    assert count_payload_tokens(messages, "char_heuristic") == first


def test_unknown_tokenizer_family_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown tokenizer_family"):
        count_tokens("hello", "provider_named_route")


def test_hf_auto_uses_optional_tokenizer_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTokenizer:
        def encode(self, text: str) -> list[str]:
            return text.split()

    class FakeAutoTokenizer:
        calls = 0

        @classmethod
        def from_pretrained(cls, model_name: str) -> FakeTokenizer:
            assert model_name == "local-synthetic-model"
            cls.calls += 1
            return FakeTokenizer()

    tokenizer._TOKENIZER_CACHE.clear()
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)
    )

    assert count_tokens("one two three", "hf_auto", "local-synthetic-model") == 3
    assert count_tokens("four five", "hf_auto", "local-synthetic-model") == 2
    assert FakeAutoTokenizer.calls == 1


def test_hf_auto_missing_optional_dependency_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("blocked optional dependency")
        return real_import(name, *args, **kwargs)

    tokenizer._TOKENIZER_CACHE.clear()
    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(RuntimeError, match="hf-tokenizer|transformers"):
        count_tokens("hello", "hf_auto", "local-synthetic-model")
