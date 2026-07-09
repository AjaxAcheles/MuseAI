"""Structured-output validation for the continuity critic.

Models return JSON in prose. Sometimes they wrap it in a markdown fence, or
preface it with "Here are the issues I found:". This module turns whatever came
back into a validated ``list[FailureObject]``, or fails loudly.

Two bounded passes, never a loop:

1. **Strict** — ``json.loads`` on the response as given.
2. **Lenient** — strip markdown fences, then take the first balanced ``[...]``
   or ``{...}`` and parse that.

If both fail, :class:`StructuredOutputError` is raised for the caller to handle
by re-prompting, up to its own ``retry_cap``. An empty array is a valid, clean
result — it means the critic found nothing, which is the success case.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from museai.fsm.state import FailureObject

# Openers paired with their closers, for the balanced-span scan.
_PAIRS = {"[": "]", "{": "}"}


class StructuredOutputError(ValueError):
    """The model's response could not be read as a list of failure objects."""


def _strip_fences(text: str) -> str:
    """Remove markdown code fences, keeping the fenced body.

    Handles both ``` and ```json openers. When there is no fence, the text comes
    back unchanged.
    """
    fence = "```"
    if fence not in text:
        return text

    _, _, after_open = text.partition(fence)
    # A language hint ("json") runs to the end of the opening fence's line.
    if "\n" in after_open:
        first_line, _, remainder = after_open.partition("\n")
        if first_line.strip().isalpha():
            after_open = remainder

    body, _, _ = after_open.partition(fence)
    return body


def _first_balanced_span(text: str) -> str | None:
    """Return the first balanced ``[...]`` or ``{...}`` span, if any.

    Scans with string- and escape-awareness so a bracket inside a JSON string
    (``"suggested_fix": "drop the ]"``) does not close the span early.
    """
    start = next(
        (i for i, char in enumerate(text) if char in _PAIRS),
        None,
    )
    if start is None:
        return None

    opener = text[start]
    closer = _PAIRS[opener]
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def _validate(data: Any) -> list[FailureObject]:
    """Validate parsed JSON into failure objects.

    A bare object is accepted as a single finding; anything that is not an
    object or a list of objects is a hard failure.
    """
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise StructuredOutputError(
            f"expected a JSON array of failure objects, got {type(data).__name__}"
        )

    findings: list[FailureObject] = []
    for index, element in enumerate(data):
        try:
            findings.append(FailureObject.model_validate(element))
        except ValidationError as exc:
            raise StructuredOutputError(
                f"element {index} is not a valid failure object: {exc}"
            ) from exc
    return findings


def parse_failure_objects(raw_text: str, retry_cap: int) -> list[FailureObject]:
    """Parse a critic response into validated failure objects.

    ``retry_cap`` is the caller's re-prompt budget. This function never loops
    and never re-prompts; it makes one strict and one lenient extraction pass,
    then raises :class:`StructuredOutputError`. The cap is carried into the
    error message so the caller can report how many attempts remain.

    An empty array returns ``[]`` — a clean draft, not a failure.
    """
    if retry_cap < 1:
        raise ValueError(f"retry_cap must be at least 1, got {retry_cap}")

    if not raw_text or not raw_text.strip():
        raise StructuredOutputError(
            f"critic returned an empty response (retry_cap={retry_cap}); "
            "a clean result must be an explicit empty JSON array"
        )

    # Pass 1: strict.
    try:
        return _validate(json.loads(raw_text))
    except StructuredOutputError:
        raise  # parsed as JSON, but the shape is wrong — re-prompting won't fix the schema
    except json.JSONDecodeError:
        pass

    # Pass 2: lenient extraction.
    candidate = _first_balanced_span(_strip_fences(raw_text))
    if candidate is not None:
        try:
            return _validate(json.loads(candidate))
        except json.JSONDecodeError:
            pass

    raise StructuredOutputError(
        f"could not extract JSON failure objects from critic response "
        f"(retry_cap={retry_cap}): {raw_text[:200]!r}"
    )
