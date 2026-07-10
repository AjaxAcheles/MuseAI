"""Structured-output validation for JSON that arrives wrapped in prose.

Models return JSON in prose. Sometimes they wrap it in a markdown fence, or
preface it with "Here are the issues I found:". This module turns whatever came
back into validated Python, or fails loudly.

Two bounded *extraction* passes, never a loop:

1. **Direct** — ``json.loads`` on the response as given.
2. **Salvaged** — strip markdown fences, then take the first balanced ``[...]``
   or ``{...}`` and parse that.

(Not to be confused with :func:`parse_failure_objects`'s ``lenient`` flag, which
relaxes how each *element* is validated once extraction has already succeeded.)

Both entry points share those passes:

* :func:`parse_failure_objects` — the continuity critic's findings, validated
  into ``list[FailureObject]``. An empty array is a valid, clean result: it means
  the critic found nothing, which is the success case.
* :func:`parse_json_array` — the planners' chapter and beat arrays, returned as
  plain dicts. Here an empty array is *not* valid: a plan with no elements is a
  failed plan, not an empty one.

If both passes fail, :class:`StructuredOutputError` is raised for the caller to
handle.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ConfigDict, ValidationError

from museai.core.logging_setup import get_fsm_logger
from museai.fsm.state import FailureObject

# Openers paired with their closers, for the balanced-span scan.
_PAIRS = {"[": "]", "{": "}"}


class _LenientFailureObject(FailureObject):
    """``FailureObject`` that tolerates unknown keys, for degraded mode only.

    A weak model misspells a key (``offarming_text`` for ``offending_text``) as
    readily as it omits one. Ignoring the stray key still leaves ``offending_text``
    missing, so an element is *dropped* rather than defaulted: an empty
    ``offending_text`` makes ``revise.locate`` return ``None``, which silently
    escalates the beat to a whole-draft rewrite off a finding nobody wrote.
    """

    model_config = ConfigDict(extra="ignore")


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


def _validate(data: Any, *, lenient: bool = False) -> list[FailureObject]:
    """Validate parsed JSON into failure objects.

    A bare object is accepted as a single finding; anything that is not an
    object or a list of objects is a hard failure.

    Strictly, one bad element fails the whole response — the caller re-prompts.
    Leniently, a bad element is logged and skipped, so a run whose critic has
    given up on the schema still gets whatever findings were readable.
    """
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise StructuredOutputError(
            f"expected a JSON array of failure objects, got {type(data).__name__}"
        )

    model = _LenientFailureObject if lenient else FailureObject
    findings: list[FailureObject] = []
    for index, element in enumerate(data):
        try:
            findings.append(model.model_validate(element))
        except ValidationError as exc:
            if not lenient:
                raise StructuredOutputError(
                    f"element {index} is not a valid failure object: {exc}"
                ) from exc
            get_fsm_logger().warning(
                "critic_finding_skipped index=%d reason=unreadable_in_lenient_mode",
                index,
            )
    return findings


def parse_failure_objects(raw_text: str, *, lenient: bool = False) -> list[FailureObject]:
    """Parse a critic response into validated failure objects.

    Two bounded extraction passes — the response as given, then fence-stripped
    and reduced to its first balanced span — and never a loop. Re-prompting is
    the caller's job (see ``fsm/nodes/critics.py``), which is why no retry budget
    is threaded through here.

    ``lenient`` relaxes *element* validation only: unknown keys are ignored and
    an element that still will not validate is skipped rather than failing the
    response. Extraction itself stays strict, so unparseable text raises either
    way. Reserved for a run that has already degraded.

    The raised :class:`StructuredOutputError` carries the underlying validation
    detail, because that text is what gets fed back to the model on a re-prompt.

    An empty array returns ``[]`` — a clean draft, not a failure.
    """
    if not raw_text or not raw_text.strip():
        raise StructuredOutputError(
            "critic returned an empty response; "
            "a clean result must be an explicit empty JSON array"
        )

    # Pass 1: the response as given.
    try:
        return _validate(json.loads(raw_text), lenient=lenient)
    except json.JSONDecodeError:
        pass

    # Pass 2: strip fences and prose, then take the first balanced span.
    candidate = _first_balanced_span(_strip_fences(raw_text))
    if candidate is not None:
        try:
            return _validate(json.loads(candidate), lenient=lenient)
        except json.JSONDecodeError:
            pass

    raise StructuredOutputError(
        f"could not extract JSON failure objects from critic response: {raw_text[:200]!r}"
    )


def _as_object_array(data: Any, what: str) -> list[dict]:
    """Coerce parsed JSON to a non-empty list of objects."""
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise StructuredOutputError(
            f"expected a JSON array of {what}, got {type(data).__name__}"
        )
    if not data:
        raise StructuredOutputError(f"model returned an empty array of {what}")

    for index, element in enumerate(data):
        if not isinstance(element, dict):
            raise StructuredOutputError(
                f"{what} element {index} is not a JSON object, "
                f"got {type(element).__name__}"
            )
    return list(data)


def parse_json_array(raw_text: str, *, what: str) -> list[dict]:
    """Parse a planner response into a non-empty list of JSON objects.

    Same two bounded passes as :func:`parse_failure_objects`, but the elements
    stay plain dicts — a chapter and a beat have different shapes, and each
    planner validates its own fields.

    ``what`` names the thing being parsed ("chapters", "beats") so the raised
    error says which plan failed. Unlike the critic's parser, an empty array is
    a hard failure: a plan must contain at least one element, and returning
    ``[]`` here would let a node silently write nothing.
    """
    if not raw_text or not raw_text.strip():
        raise StructuredOutputError(f"model returned an empty response for {what}")

    # Pass 1: strict.
    try:
        return _as_object_array(json.loads(raw_text), what)
    except StructuredOutputError:
        raise  # parsed as JSON, but the shape is wrong — a re-prompt won't fix it
    except json.JSONDecodeError:
        pass

    # Pass 2: lenient extraction.
    candidate = _first_balanced_span(_strip_fences(raw_text))
    if candidate is not None:
        try:
            return _as_object_array(json.loads(candidate), what)
        except json.JSONDecodeError:
            pass

    raise StructuredOutputError(
        f"could not extract a JSON array of {what} from the model response: "
        f"{raw_text[:200]!r}"
    )
