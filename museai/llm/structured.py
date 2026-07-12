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
import re
from typing import Any, Sequence

from pydantic import ConfigDict, ValidationError

from museai.core.logging_setup import get_fsm_logger
from museai.fsm.state import FailureObject

# Openers paired with their closers, for the balanced-span scan.
_PAIRS = {"[": "]", "{": "}"}

# One `"key": "value"` pair occupying a whole line, captured as prefix / value /
# suffix. The value group is greedy, so it runs to the *last* quote on the line —
# which is the closing quote, whatever the model put between them.
_STRING_FIELD_LINE = re.compile(r'^(\s*"[\w-]+"\s*:\s*")(.*)("\s*,?\s*)$')

# A double quote not already escaped.
_BARE_QUOTE = re.compile(r'(?<!\\)"')


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


class FakeToolCallTextError(StructuredOutputError):
    """The planner wrote tool-call-shaped JSON as text instead of using tools."""


class TruncatedResponseError(StructuredOutputError):
    """The model's reply was cut off at the endpoint's token limit.

    Raised by callers that see ``finish_reason == "length"`` — the parsers here
    never see a finish reason. A truncated reply must not be parsed: a cleanly
    balanced inner fragment of a half-written plan can masquerade as the plan.
    """


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


def _balanced_span_from(text: str, start: int) -> str | None:
    """Return the balanced JSON span beginning at ``start``, if it closes."""
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
    return _balanced_span_from(text, start)


def _first_balanced_span_with_opener(text: str, opener: str) -> str | None:
    """Return the first balanced span that starts with ``opener``, if any."""
    for index, char in enumerate(text):
        if char != opener:
            continue
        span = _balanced_span_from(text, index)
        if span is not None:
            return span
    return None


def _balanced_spans_with_opener(text: str, opener: str) -> list[str]:
    """Return every balanced span that starts with ``opener``.

    Planner replies sometimes contain fake tool-call objects whose arguments
    themselves contain JSON-looking fragments. Callers can try every array span
    before accepting an earlier object, so a real plan later in the text is not
    masked by an incidental nested array.
    """
    spans: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != opener:
            index += 1
            continue
        span = _balanced_span_from(text, index)
        if span is None:
            index += 1
            continue
        spans.append(span)
        index += max(1, len(span))
    return spans


def _fake_tool_names(text: str) -> list[str]:
    """Names of tool-call-shaped objects the model wrote as plain text."""
    names: list[str] = []
    for match in re.finditer(r'"name"\s*:\s*"([A-Za-z_][\w-]*)"', text):
        tail = text[match.end() : match.end() + 160]
        if re.search(r'"parameters"\s*:', tail) and match.group(1) not in names:
            names.append(match.group(1))
    return names


def _fake_tool_call_error(what: str, names: list[str]) -> FakeToolCallTextError:
    listed = ", ".join(names[:6])
    if len(names) > 6:
        listed = f"{listed}, ..."
    return FakeToolCallTextError(
        f"{what} reply wrote tool calls as plain text instead of returning the "
        f"planner JSON array; tools were not executed: {listed or 'unknown'}"
    )


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


# --- Clean-verdict recognition (critic only) --------------------------------
# A critic sometimes phrases "no issues" as prose instead of `[]`. Rejecting
# that burns every re-prompt and pushes the run toward degraded mode over a
# draft the critic just said was fine. The reading below is deliberately
# narrow: any JSON-ish bracket, any schema vocabulary, any hedge, or anything
# beyond a short sentence still fails and re-prompts.
_CLEAN_VERDICT_MAX_CHARS = 240

_CLEAN_PHRASES = re.compile(
    r"\b(clean|clear|no (?:continuity )?(?:issues?|errors?|problems?)|"
    r"no continuity|none(?: found)?|nothing(?: to report| found)?|looks good)\b",
    re.IGNORECASE,
)

# Schema vocabulary: a reply naming a field or an error code is talking about
# findings, however it is phrased. Lower-cased comparison catches CONTRADICTS.
_SCHEMA_MARKERS = (
    "error_code",
    "offending_text",
    "suggested_fix",
    "critic_source",
    "contradicts",
    "unfulfilled",
    "false_real_world",
)

# Hedges: "clean, but…" is not a clean verdict, and neither is a reply that
# trails off mid-thought.
_HEDGE_MARKERS = re.compile(
    r"\b(but|however|although|though|except|unsure|not sure|maybe|might|"
    r"possibl[ey]|perhaps)\b",
    re.IGNORECASE,
)


def is_clean_verdict(text: str) -> bool:
    """Whether a prose critic reply unambiguously says the draft is clean.

    True only when the text is short, contains no ``[`` or ``{`` at all, none
    of the failure-object vocabulary, no hedging, and affirmatively matches a
    clean-ish phrase. Conservative by construction: a truncated or equivocating
    reply returns False and stays a parse failure.
    """
    stripped = text.strip()
    if not stripped or len(stripped) > _CLEAN_VERDICT_MAX_CHARS:
        return False
    if "[" in stripped or "{" in stripped:
        return False
    lowered = stripped.lower()
    if any(marker in lowered for marker in _SCHEMA_MARKERS):
        return False
    if _HEDGE_MARKERS.search(stripped):
        return False
    return bool(_CLEAN_PHRASES.search(stripped))


def parse_failure_objects(raw_text: str, *, lenient: bool = False) -> list[FailureObject]:
    """Parse a critic response into validated failure objects.

    Two bounded extraction passes — the response as given, then fence-stripped
    and scanned for balanced spans (every ``[...]`` candidate, then the first
    ``{...}``, matching the planners' search so a bracketed preamble cannot
    mask the payload) — and never a loop. Re-prompting is the caller's job
    (see ``fsm/nodes/critics.py``), which is why no retry budget is threaded
    through here.

    ``lenient`` relaxes *element* validation only: unknown keys are ignored and
    an element that still will not validate is skipped rather than failing the
    response. Extraction itself stays strict, so unparseable text raises either
    way. Reserved for a run that has already degraded.

    The raised :class:`StructuredOutputError` carries the underlying validation
    detail, because that text is what gets fed back to the model on a re-prompt.

    An empty array returns ``[]`` — a clean draft, not a failure. So does an
    unambiguous prose clean verdict (see :func:`is_clean_verdict`), logged at
    WARNING because the critic still broke its output contract.
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

    # Pass 2: strip fences and prose, then try every balanced array span — a
    # preamble like "I found 1 issue [see below]:" must not mask the real
    # array behind it — before falling back to the first object span.
    stripped = _strip_fences(raw_text)
    shape_error: StructuredOutputError | None = None
    for candidate in _balanced_spans_with_opener(stripped, "["):
        try:
            return _validate(json.loads(candidate), lenient=lenient)
        except StructuredOutputError as exc:
            shape_error = exc
        except json.JSONDecodeError:
            pass

    candidate = _first_balanced_span_with_opener(stripped, "{")
    if candidate is not None:
        try:
            return _validate(json.loads(candidate), lenient=lenient)
        except StructuredOutputError as exc:
            shape_error = exc
        except json.JSONDecodeError:
            pass

    # A reply that carried JSON-ish spans was talking findings, however badly;
    # is_clean_verdict rejects any text containing a bracket, so these two
    # rungs cannot both apply to one reply.
    if is_clean_verdict(raw_text):
        get_fsm_logger().warning(
            "critic_clean_prose verdict accepted as []: %r", raw_text.strip()[:120]
        )
        return []

    if shape_error is not None:
        raise shape_error

    raise StructuredOutputError(
        f"could not extract JSON failure objects from critic response: {raw_text[:200]!r}"
    )


def _as_object_array(
    data: Any, what: str, element_keys: Sequence[str] | None = None
) -> list[dict]:
    """Coerce parsed JSON to a non-empty list of objects.

    ``element_keys``, when given, requires every element to carry at least one
    of those keys. This is what stops a *truncated* reply from being accepted:
    a plan cut off mid-JSON often leaves some inner array — ``thread_updates``,
    ``obligations`` — as the only balanced span, and without a key check that
    fragment parses cleanly and becomes "the plan".
    """
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise StructuredOutputError(
            f"expected a JSON array of {what}, got {type(data).__name__}"
        )
    # A weak model sometimes wraps the array once more: `[[{...}, {...}]]`.
    # One level of unwrapping is unambiguous when every element is a list and
    # everything inside is an object; deeper nesting stays an error.
    if data and all(isinstance(element, list) for element in data):
        flattened = [item for element in data for item in element]
        if flattened and all(isinstance(item, dict) for item in flattened):
            data = flattened
    if not data:
        raise StructuredOutputError(f"model returned an empty array of {what}")

    for index, element in enumerate(data):
        if not isinstance(element, dict):
            raise StructuredOutputError(
                f"{what} element {index} is not a JSON object, "
                f"got {type(element).__name__}"
            )
        if "name" in element and isinstance(element.get("parameters"), dict):
            raise FakeToolCallTextError(
                f"{what} element {index} looks like a tool call, not a planner "
                f"object: {element.get('name')!r}"
            )
        if element_keys and not any(key in element for key in element_keys):
            raise StructuredOutputError(
                f"{what} element {index} has none of the expected keys "
                f"({', '.join(element_keys)}); it looks like a fragment of a "
                f"larger reply, not a {what} plan"
            )
    return list(data)


def repair_json_text(text: str) -> str:
    """Escape unescaped double quotes inside single-line JSON string values.

    A fiction planner quotes speech. Asked for ``exit_state`` it writes

        "exit_state": "Mara expresses vague regret ("It was stronger") while ...",

    which terminates the string at the first inner quote and makes the whole
    array unparseable. That killed a run on 2026-07-10 after seven committed
    beats.

    The repair is line-oriented because JSON forbids a raw newline inside a
    string: the closing quote of a value is therefore the *last* quote on its
    line, and everything between the opening and closing quotes is content. A
    character scanner deciding string boundaries by looking ahead for ``,`` or
    ``}`` would instead mangle ``"he said "yes", then left"``.

    Lines that are structural, or whose value is not a string, never match and
    pass through untouched. Valid JSON is returned byte-identical, because a
    valid string value contains no unescaped quote to rewrite.

    This is a heuristic, and it puts words in the model's mouth. Callers use it
    only after re-prompting has failed, and must say so out loud when it fires.
    """
    repaired: list[str] = []
    for line in text.splitlines():
        match = _STRING_FIELD_LINE.match(line)
        if match is None:
            repaired.append(line)
            continue
        prefix, value, suffix = match.groups()
        repaired.append(f"{prefix}{_BARE_QUOTE.sub(r'\\"', value)}{suffix}")
    return "\n".join(repaired)


def parse_json_array(
    raw_text: str,
    *,
    what: str,
    repair: bool = False,
    element_keys: Sequence[str] | None = None,
) -> list[dict]:
    """Parse a planner response into a non-empty list of JSON objects.

    Same two bounded passes as :func:`parse_failure_objects`, but the elements
    stay plain dicts — a chapter and a beat have different shapes, and each
    planner validates its own fields.

    ``what`` names the thing being parsed ("chapters", "beats") so the raised
    error says which plan failed. Unlike the critic's parser, an empty array is
    a hard failure: a plan must contain at least one element, and returning
    ``[]`` here would let a node silently write nothing.

    ``element_keys``, when given, rejects any candidate array whose elements
    carry none of those keys — the guard that keeps a truncated reply's inner
    ``thread_updates`` or ``obligations`` array from being accepted as the
    plan. Extraction still tries every balanced span, so a real plan later in
    the text is found; acceptance is what narrows.

    ``repair`` adds a third pass that runs :func:`repair_json_text` over the
    response. It is off by default and belongs to callers that have already
    spent their re-prompts — see ``museai/llm/planning.py``.
    """
    if not raw_text or not raw_text.strip():
        raise StructuredOutputError(f"model returned an empty response for {what}")

    # Pass 1: strict.
    try:
        return _as_object_array(json.loads(raw_text), what, element_keys)
    except StructuredOutputError:
        raise  # parsed as JSON, but the shape is wrong — a re-prompt won't fix it
    except json.JSONDecodeError:
        pass

    # Pass 2: lenient extraction.
    stripped = _strip_fences(raw_text)
    shape_error: StructuredOutputError | None = None
    for candidate in _balanced_spans_with_opener(stripped, "["):
        try:
            return _as_object_array(json.loads(candidate), what, element_keys)
        except FakeToolCallTextError:
            raise
        except StructuredOutputError as exc:
            shape_error = exc
        except json.JSONDecodeError:
            pass

    names = _fake_tool_names(stripped)
    if names:
        raise _fake_tool_call_error(what, names)

    candidate = _first_balanced_span(stripped)
    if candidate is not None:
        try:
            return _as_object_array(json.loads(candidate), what, element_keys)
        except FakeToolCallTextError:
            raise
        except StructuredOutputError as exc:
            shape_error = exc
        except json.JSONDecodeError:
            pass

    # Pass 3: repair the model's quoting, then extract again.
    if repair:
        repaired = repair_json_text(stripped)
        for candidate in _balanced_spans_with_opener(repaired, "["):
            try:
                return _as_object_array(json.loads(candidate), what, element_keys)
            except FakeToolCallTextError:
                raise
            except StructuredOutputError as exc:
                shape_error = exc
            except json.JSONDecodeError:
                pass
        names = _fake_tool_names(repaired)
        if names:
            raise _fake_tool_call_error(what, names)
        candidate = _first_balanced_span(repaired)
        if candidate is not None:
            try:
                return _as_object_array(json.loads(candidate), what, element_keys)
            except FakeToolCallTextError:
                raise
            except StructuredOutputError as exc:
                shape_error = exc
            except json.JSONDecodeError:
                pass

    if shape_error is not None:
        raise shape_error

    raise StructuredOutputError(
        f"could not extract a JSON array of {what} from the model response: "
        f"{raw_text[:200]!r}"
    )
