"""Token counting for MuseAI v1.

Two tokenizer families are supported, selected per endpoint via
``EndpointConfig.tokenizer_family``:

* ``tiktoken``       — real BPE counts. Uses ``encoding_for_model(model_name)``
  when the model is known to tiktoken, otherwise falls back to the
  ``cl100k_base`` encoding. That fallback is a *tokenizer* fallback, not a
  statement about which model or provider is in use.
* ``char_heuristic`` — ``ceil(len(text) / 4)``, the usual four-characters-per-
  token approximation. Cheap, offline, and good enough for budgeting when no
  BPE table matches the endpoint's model.

Encoders are cached by ``(family, model_name)`` so repeated calls do not rebuild
or re-download a BPE table.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any, Mapping, Sequence

# Chat wire formats wrap each message in a few structural tokens (the role, and
# the delimiters framing the message). Four is the per-message constant OpenAI
# documents for cl100k-era chat models; it is close enough for the other
# families we count for, and it errs high rather than low.
MESSAGE_OVERHEAD_TOKENS = 4

# Every reply is primed with a short assistant header before generation starts.
REPLY_PRIMING_TOKENS = 2

# Used when the endpoint's model is not one tiktoken has a table for.
_FALLBACK_ENCODING = "cl100k_base"

_TIKTOKEN = "tiktoken"
_CHAR_HEURISTIC = "char_heuristic"


@lru_cache(maxsize=32)
def _get_encoder(model_name: str | None) -> Any:
    """Return a cached tiktoken encoder for ``model_name``.

    Falls back to ``cl100k_base`` when tiktoken has no table for the model.
    ``tiktoken`` is imported lazily so that merely importing this module — or
    anything that imports it — never touches the network or loads BPE tables.
    """
    import tiktoken

    if model_name:
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            pass
    return tiktoken.get_encoding(_FALLBACK_ENCODING)


def _as_text(value: Any) -> str:
    """Coerce a message field to text for counting purposes."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def count_tokens(
    text: str,
    tokenizer_family: str,
    model_name: str | None = None,
) -> int:
    """Count the tokens in ``text`` under the given tokenizer family.

    Raises ``ValueError`` for an unknown family.
    """
    if not text:
        return 0

    if tokenizer_family == _CHAR_HEURISTIC:
        return math.ceil(len(text) / 4)

    if tokenizer_family == _TIKTOKEN:
        return len(_get_encoder(model_name).encode(text))

    raise ValueError(f"unknown tokenizer_family: {tokenizer_family!r}")


def count_message_tokens(
    messages: Sequence[Mapping[str, Any]],
    tokenizer_family: str,
    model_name: str | None = None,
) -> int:
    """Count the tokens a chat-completions ``messages`` list will occupy.

    Each message costs ``MESSAGE_OVERHEAD_TOKENS`` on top of the tokens in its
    own fields, and the request as a whole costs ``REPLY_PRIMING_TOKENS`` for
    the assistant header that precedes the reply. An empty list costs nothing.

    Raises ``ValueError`` for an unknown family (even when ``messages`` is
    empty, so a misconfigured endpoint fails the same way every time).
    """
    if tokenizer_family not in (_TIKTOKEN, _CHAR_HEURISTIC):
        raise ValueError(f"unknown tokenizer_family: {tokenizer_family!r}")

    if not messages:
        return 0

    total = REPLY_PRIMING_TOKENS
    for message in messages:
        total += MESSAGE_OVERHEAD_TOKENS
        for value in message.values():
            total += count_tokens(_as_text(value), tokenizer_family, model_name)
    return total
