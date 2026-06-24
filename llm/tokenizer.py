"""Module: M04 (LLM Inference Boundary)
Backend-agnostic token counting routed by endpoint tokenizer family.

Counts are exact only for tokenizer implementations that expose exact encoders.
The character route and message structural overhead are deterministic budgeting
heuristics, not claims of backend-specific accounting precision.
"""

from __future__ import annotations

from math import ceil
from typing import Any

_TIKTOKEN_FAMILY = "tiktoken"
_HF_AUTO_FAMILY = "hf_auto"
_CHAR_HEURISTIC_FAMILY = "char_heuristic"
_TIKTOKEN_FALLBACK_ENCODING = "cl100k_base"
_CHAR_HEURISTIC_CHARS_PER_TOKEN = 4
# Deterministic scaffold estimate for role/content message wrappers. This keeps
# message payloads larger than raw content while avoiding backend-specific claims.
_MESSAGE_STRUCTURAL_OVERHEAD_TOKENS = 4

_TOKENIZER_CACHE: dict[tuple[str, str | None], Any] = {}


def count_tokens(
    text: str, tokenizer_family: str, model_name: str | None = None
) -> int:
    """Count tokens for text using the configured tokenizer family."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    family = _normalize_tokenizer_family(tokenizer_family)
    if family == _CHAR_HEURISTIC_FAMILY:
        if not text:
            return 0
        return ceil(len(text) / _CHAR_HEURISTIC_CHARS_PER_TOKEN)

    tokenizer = _get_tokenizer(family, model_name)
    return len(tokenizer.encode(text))


def count_message_tokens(
    messages: list[dict[str, str]],
    tokenizer_family: str,
    model_name: str | None = None,
) -> int:
    """Count role/content message payload tokens with deterministic overhead.

    The overhead is a stable per-message structural estimate for wrappers and
    separators. It is intentionally not presented as exact for every backend.
    """
    total = 0
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        total += _MESSAGE_STRUCTURAL_OVERHEAD_TOKENS
        total += count_tokens(role, tokenizer_family, model_name)
        total += count_tokens(content, tokenizer_family, model_name)
    return total


def count_payload_tokens(
    payload: str | list[dict[str, str]],
    tokenizer_family: str,
    model_name: str | None = None,
) -> int:
    """Count tokens for either raw text or role/content chat messages."""
    if isinstance(payload, str):
        return count_tokens(payload, tokenizer_family, model_name)
    if isinstance(payload, list):
        return count_message_tokens(payload, tokenizer_family, model_name)
    raise TypeError("payload must be a string or a list of message dictionaries")


def _normalize_tokenizer_family(tokenizer_family: str) -> str:
    family = tokenizer_family.strip().lower()
    if family not in {_TIKTOKEN_FAMILY, _HF_AUTO_FAMILY, _CHAR_HEURISTIC_FAMILY}:
        raise ValueError(f"unknown tokenizer_family: {tokenizer_family!r}")
    return family


def _get_tokenizer(tokenizer_family: str, model_name: str | None) -> Any:
    cache_key = (tokenizer_family, model_name)
    if cache_key not in _TOKENIZER_CACHE:
        if tokenizer_family == _TIKTOKEN_FAMILY:
            _TOKENIZER_CACHE[cache_key] = _load_tiktoken_encoding(model_name)
        elif tokenizer_family == _HF_AUTO_FAMILY:
            _TOKENIZER_CACHE[cache_key] = _load_hf_tokenizer(model_name)
        else:
            raise ValueError(f"unknown tokenizer_family: {tokenizer_family!r}")
    return _TOKENIZER_CACHE[cache_key]


def _load_tiktoken_encoding(model_name: str | None) -> Any:
    import tiktoken

    if model_name:
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            pass
    return tiktoken.get_encoding(_TIKTOKEN_FALLBACK_ENCODING)


def _load_hf_tokenizer(model_name: str | None) -> Any:
    if not model_name:
        raise ValueError("tokenizer_family='hf_auto' requires model_name")

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "tokenizer_family='hf_auto' requires the optional transformers "
            "dependency. Install it with `uv sync --extra hf-tokenizer`."
        ) from exc

    return AutoTokenizer.from_pretrained(model_name)
