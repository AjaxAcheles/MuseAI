"""Dependency-free sentence helpers shared by the audit node and LLM seam.

Both layers need identical sentence boundaries, so the helpers live in ``core``
rather than making the endpoint-agnostic LLM seam depend inward on an FSM node.
"""

from __future__ import annotations

import re

# Sentence split on terminal punctuation followed by whitespace. Abbreviations
# ("Dr. Vance") over-split, which costs at most one extra sentence in the
# denominator — it never invents a passive, so it can only make the check more
# forgiving, which is the right way for a heuristic gate to be wrong.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[\"')\]”’]*\s+")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` offsets of the non-empty split sentences."""
    spans: list[tuple[int, int]] = []
    start = 0
    for boundary in _SENTENCE_SPLIT.finditer(text):
        segment = text[start : boundary.start()]
        left = len(segment) - len(segment.lstrip())
        right = len(segment.rstrip())
        if left < right:
            spans.append((start + left, start + right))
        start = boundary.end()

    segment = text[start:]
    left = len(segment) - len(segment.lstrip())
    right = len(segment.rstrip())
    if left < right:
        spans.append((start + left, start + right))
    return spans


def split_sentences(text: str) -> list[str]:
    """Split prose into non-empty sentences."""
    return [text[start:end] for start, end in sentence_spans(text)]
