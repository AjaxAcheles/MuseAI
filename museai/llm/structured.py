"""Strict structured-output validation at the model boundary.

Ordinary markdown fences and surrounding prose are tolerated, but the payload
itself must be one unambiguous JSON array. Duplicate keys, non-finite numbers,
concatenated arrays, truncated outer arrays, and schema drift fail loudly so a
caller can use its bounded correction path instead of committing partial data.

Extraction first tries the complete response, then scans every balanced array
candidate. It accepts exactly one valid candidate; zero or multiple candidates
are errors.

(Not to be confused with :func:`parse_failure_objects`'s ``lenient`` flag, which
relaxes how each *element* is validated once extraction has already succeeded.)

The structured entry points share this policy:

* :func:`parse_failure_objects` — the continuity critic's findings, validated
  into ``list[FailureObject]``. An empty array is a valid, clean result: it means
  the critic found nothing, which is the success case.
* :func:`parse_json_array` — the planners' chapter and beat arrays, returned as
  plain dicts. Here an empty array is *not* valid: a plan with no elements is a
  failed plan, not an empty one.

If extraction or validation fails, :class:`StructuredOutputError` is raised for
the caller to handle.
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

CRITIC_ERROR_CODES = frozenset(
    {
        "CONTRADICTS_PRIOR_PROSE",
        "CONTRADICTS_CHARACTER",
        "CONTRADICTS_THREAD",
        "CONTRADICTS_PREMISE",
        "UNFULFILLED_OBLIGATION",
        "FALSE_REAL_WORLD_CLAIM",
        "INCOHERENT_BLOCKING",
    }
)

_META_PROSE_PREFIX = re.compile(
    r"^(?:here(?:'s| is)|sure[,!:]?|certainly[,!:]?)\s+"
    r"(?:(?:the|your|a)\s+)?(?:revised\s+)?(?:prose|draft|revision|story|text)\b",
    re.IGNORECASE,
)
_META_PROSE_SUFFIX = re.compile(
    r"(?:^|\n)\s*(?:hope this helps!?|let me know if you(?:'d| would) like[^\n]*)\s*$",
    re.IGNORECASE,
)


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


class UnlocatableFindingError(StructuredOutputError):
    """Every parsed critic finding quoted text absent from the current draft."""


class FakeToolCallTextError(StructuredOutputError):
    """The planner wrote tool-call-shaped JSON as text instead of using tools."""


class TruncatedResponseError(StructuredOutputError):
    """The model's reply was cut off at the endpoint's token limit.

    Raised by callers that see ``finish_reason == "length"`` — the parsers here
    never see a finish reason. A truncated reply must not be parsed: a cleanly
    balanced inner fragment of a half-written plan can masquerade as the plan.
    """


class AmbiguousStructuredOutputError(StructuredOutputError):
    """More than one complete payload could be read from one model reply."""


def truncation_remedy(
    *,
    empty: bool,
    context_window: int | None = None,
    served_prompt_tokens: int | None = None,
    served_completion_tokens: int | None = None,
    max_output_tokens: int | None = None,
    thinking: str | None = None,
    mismatch_fraction: float | None = None,
) -> str:
    """Remediation clause for a ``finish_reason == "length"`` failure.

    ``empty`` distinguishes the two ways a reply gets cut off, which need
    different fixes. Zero output tokens means the prompt filled the endpoint's
    context window and left no room to generate — widening the output cap does
    nothing; the window has to grow or the prompt has to shrink. A non-empty
    reply that stopped mid-array is a genuine output-length limit, where
    ``max_output_tokens`` is the right knob.

    A reasoning model breaks that dichotomy: it can spend its *entire* output
    grant on a ``<think>`` block and hand back ``text == ""`` with
    ``finish_reason == "length"``, which looks exactly like "the window left no
    room to generate" even though the window had room to spare — the model
    filled the *output* budget, not the *context* one. Checked first, before the
    served-window branch below: if ``served_completion_tokens`` reached
    ``max_output_tokens``, the cap is what bound, not the window. (Confirmed live
    2026-07-24/25: 28 of 30 critic truncations that session had
    ``served_completion_tokens == max_output_tokens`` exactly, and the served-window
    branch below blamed ``OLLAMA_CONTEXT_LENGTH`` for all 28 — a real server
    misconfiguration elsewhere in the same log, wrongly generalized.)

    The server's own counts, when it reported them, separate the *other* two
    further. A reply that died with zero output and did **not** hit its own cap
    has hit the server's wall, so the tokens it admits to processing approximate
    the window it is *actually* serving. When that total falls well short of the
    declared ``context_window``, the prompt is not too long in any absolute sense
    — the server is smaller than the config believes, and no amount of pruning to
    a budget derived from the wrong number will help. Saying so with both figures
    is the difference between a fix and an afternoon. (This is not hypothetical:
    on 2026-07-24 an unset ``OLLAMA_CONTEXT_LENGTH`` served 4096 against a
    declared 16384, and the generic advice below sent the reader after the prompt
    instead of the server.)

    ``mismatch_fraction`` defaults to
    ``generation.served_window_mismatch_fraction``; the argument exists so a
    caller (or a test) can pin it explicitly. The import is deferred so this
    module stays free of an ``fsm`` import at load time.
    """
    if not empty:
        return (
            "raise endpoint.max_output_tokens (or leave it unset to omit the cap) "
            "or ask for a shorter answer"
        )

    if (
        max_output_tokens is not None
        and served_completion_tokens is not None
        and served_completion_tokens >= max_output_tokens
    ):
        reasoning_note = (
            " The model spent the whole grant on internal reasoning (a non-empty "
            "<think> block was returned as `thinking`) and never got to an answer."
            if thinking and thinking.strip()
            else ""
        )
        return (
            f"the reply used its full output budget ({served_completion_tokens} "
            f"completion tokens against endpoint.max_output_tokens="
            f"{max_output_tokens}) before producing any answer text."
            f"{reasoning_note} This is a completion-length cap, not a context-"
            f"window problem: raise endpoint.max_output_tokens (and "
            f"output_reservation to match) or ask for a shorter answer"
        )

    if context_window is not None and served_prompt_tokens is not None:
        if mismatch_fraction is None:
            from museai.fsm.nodes.deps import get_node_config

            mismatch_fraction = (
                get_node_config().generation.served_window_mismatch_fraction
            )
        served_total = served_prompt_tokens + (served_completion_tokens or 0)
        if served_total < context_window * mismatch_fraction:
            return (
                f"the endpoint stopped after {served_total} tokens "
                f"({served_prompt_tokens} of them prompt), far short of the "
                f"declared endpoint.context_window of {context_window}. The "
                f"server is serving a smaller window than the config believes, "
                f"so every prompt budget derived from {context_window} is too "
                f"large. Fix the server rather than the prompt: for Ollama set "
                f"OLLAMA_CONTEXT_LENGTH and restart it, then confirm with "
                f"'ollama ps' that the CONTEXT column reads {context_window} "
                f"(note that 'options.num_ctx' in extra_body is silently ignored "
                f"by Ollama's /v1 endpoint). Otherwise lower "
                f"endpoint.context_window to what the server really serves"
            )

    return (
        "the prompt filled the endpoint's context window and left no room to "
        "generate (zero output tokens). Widen the model's context window "
        "(for Ollama: set OLLAMA_CONTEXT_LENGTH and restart the server — note "
        "that endpoint.extra_body={'options': {'num_ctx': N}} is silently "
        "ignored by Ollama's /v1 endpoint), shorten the prompt, or ask for a "
        "shorter answer"
    )


def response_truncation_remedy(response: Any, endpoint: Any = None) -> str:
    """:func:`truncation_remedy` filled in from an ``LLMResponse`` and its endpoint.

    Every caller that inspects ``finish_reason == "length"`` already holds both
    objects, and each was previously passing ``empty=`` by hand; this keeps the
    served-token diagnosis from having to be re-derived (or forgotten) at five
    call sites. Attributes are read defensively, matching how these call sites
    already reach for ``finish_reason``, so a caller holding a stub or an older
    response shape degrades to the generic advice instead of raising a second
    error on top of the truncation it was trying to report.

    The cap checked is ``response.effective_max_tokens`` — the cap the call
    actually ran under, whether it came from an explicit
    ``endpoint.max_output_tokens`` or was derived from ``context_window`` and the
    prompt size (see ``call_llm``) — falling back to
    ``endpoint.max_output_tokens`` for a response shape that predates that field.
    ``response.thinking`` is threaded through the same way. Without either, a
    truncation with an empty ``text`` looked identical to a context-window
    overrun regardless of which cap actually bound.
    """
    max_output_tokens = getattr(response, "effective_max_tokens", None)
    if max_output_tokens is None:
        max_output_tokens = getattr(endpoint, "max_output_tokens", None)
    return truncation_remedy(
        empty=not (getattr(response, "text", "") or "").strip(),
        context_window=getattr(endpoint, "context_window", None),
        served_prompt_tokens=getattr(response, "served_prompt_tokens", None),
        served_completion_tokens=getattr(response, "served_completion_tokens", None),
        max_output_tokens=max_output_tokens,
        thinking=getattr(response, "thinking", None),
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys instead of losing one."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StructuredOutputError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    """JSON has no NaN or infinity; accepting them silently corrupts numeric fields."""
    raise StructuredOutputError(f"non-finite JSON number {value!r} is not allowed")


def strict_json_loads(text: str) -> Any:
    """Decode standards-compliant JSON without duplicate keys or non-finite numbers."""
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_nonfinite_constant,
    )


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
            # Do not salvage a balanced child array from inside a truncated
            # outer array.  Doing so silently turns a partial plan into a
            # complete-looking response containing only its first nested item.
            break
        spans.append(span)
        index += max(1, len(span))
    return spans


def _fake_tool_names(text: str) -> list[str]:
    """Names of tool-call-shaped objects the model wrote as plain text."""
    names: list[str] = []
    for match in re.finditer(r'"name"\s*:\s*"([A-Za-z_][\w-]*)"', text):
        tail = text[match.end() : match.end() + 160]
        if (
            re.search(r'"(?:parameters|arguments)"\s*:', tail)
            and match.group(1) not in names
        ):
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

    Anything other than an array of objects is a hard failure.

    Strictly, one bad element fails the whole response — the caller re-prompts.
    Leniently, a bad element is logged and skipped, so a run whose critic has
    given up on the schema still gets whatever findings were readable.
    """
    if not isinstance(data, list):
        raise StructuredOutputError(
            f"expected a JSON array of failure objects, got {type(data).__name__}"
        )

    model = _LenientFailureObject if lenient else FailureObject
    findings: list[FailureObject] = []
    for index, element in enumerate(data):
        if not isinstance(element, dict):
            if not lenient:
                raise StructuredOutputError(
                    f"element {index} is not a JSON object, got {type(element).__name__}"
                )
            get_fsm_logger().warning(
                "critic_finding_skipped index=%d reason=not_an_object", index
            )
            continue
        if not lenient:
            # `critic_source` is optional, not required: the parser below
            # defaults it to "continuity_critic" itself (v1 has exactly one
            # critic), so a model that omits it has done nothing wrong. It is
            # still validated below when present — a model naming some other
            # source is a real schema deviation, not an omission.
            required = {"error_code", "offending_text", "suggested_fix"}
            allowed = required | {"critic_source"}
            actual = set(element)
            missing = required - actual
            unexpected = actual - allowed
            if missing or unexpected:
                deviations: list[str] = []
                if missing:
                    deviations.append(
                        f"missing required fields: {', '.join(sorted(missing))}"
                    )
                if unexpected:
                    deviations.append(
                        f"unexpected fields: {', '.join(sorted(unexpected))}"
                    )
                raise StructuredOutputError(
                    f"element {index} schema deviation: {'; '.join(deviations)}"
                )
        error_code = element.get("error_code")
        if error_code not in CRITIC_ERROR_CODES:
            if not lenient:
                raise StructuredOutputError(
                    f"element {index} has unknown error_code {error_code!r}"
                )
            get_fsm_logger().warning(
                "critic_finding_skipped index=%d reason=unknown_error_code", index
            )
            continue
        source = element.get("critic_source", "continuity_critic")
        if source != "continuity_critic":
            if not lenient:
                raise StructuredOutputError(
                    f"element {index} has invalid critic_source {source!r}"
                )
            get_fsm_logger().warning(
                "critic_finding_skipped index=%d reason=invalid_critic_source", index
            )
            continue
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
# beyond a short sentence still fails and re-prompts. The length bound is
# ``generation.critic_verdict_max_chars``.

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


def is_clean_verdict(text: str, max_chars: int | None = None) -> bool:
    """Whether a prose critic reply unambiguously says the draft is clean.

    True only when the text is short, contains no ``[`` or ``{`` at all, none
    of the failure-object vocabulary, no hedging, and affirmatively matches a
    clean-ish phrase. Conservative by construction: a truncated or equivocating
    reply returns False and stays a parse failure.

    ``max_chars`` defaults to ``generation.critic_verdict_max_chars``; the
    argument exists so a caller (or a test) can pin it explicitly. The import is
    deferred so this module stays free of an ``fsm`` import at load time.
    """
    if max_chars is None:
        from museai.fsm.nodes.deps import get_node_config

        max_chars = get_node_config().generation.critic_verdict_max_chars

    stripped = text.strip()
    if not stripped or len(stripped) > max_chars:
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
        return _validate(strict_json_loads(raw_text), lenient=lenient)
    except StructuredOutputError:
        raise
    except json.JSONDecodeError:
        pass

    # Pass 2: strip fences and prose, then try every balanced array span — a
    # preamble like "I found 1 issue [see below]:" must not mask the real
    # array behind it — before falling back to the first object span.
    shape_error: StructuredOutputError | None = None
    valid: list[list[FailureObject]] = []
    for candidate in _balanced_spans_with_opener(raw_text, "["):
        try:
            valid.append(_validate(strict_json_loads(candidate), lenient=lenient))
        except StructuredOutputError as exc:
            shape_error = exc
        except json.JSONDecodeError:
            pass
    if len(valid) > 1:
        raise AmbiguousStructuredOutputError(
            "critic response contained multiple complete JSON arrays"
        )
    if valid:
        return valid[0]

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
    if isinstance(data, dict) and "name" in data and isinstance(data.get("parameters"), dict):
        raise FakeToolCallTextError(
            f"{what} reply looks like a tool call, not a planner array: {data.get('name')!r}"
        )
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
        return _as_object_array(strict_json_loads(raw_text), what, element_keys)
    except StructuredOutputError:
        raise  # parsed as JSON, but the shape is wrong — a re-prompt won't fix it
    except json.JSONDecodeError:
        pass

    # Pass 2: lenient extraction.
    shape_error: StructuredOutputError | None = None
    valid: list[list[dict]] = []
    for candidate in _balanced_spans_with_opener(raw_text, "["):
        try:
            valid.append(_as_object_array(strict_json_loads(candidate), what, element_keys))
        except FakeToolCallTextError:
            raise
        except StructuredOutputError as exc:
            shape_error = exc
        except json.JSONDecodeError:
            pass
    if len(valid) > 1:
        raise AmbiguousStructuredOutputError(
            f"model response contained multiple complete JSON arrays of {what}"
        )
    if valid:
        return valid[0]

    names = _fake_tool_names(raw_text)
    if names:
        raise _fake_tool_call_error(what, names)

    # Pass 3: repair the model's quoting, then extract again.
    if repair:
        repaired = repair_json_text(raw_text)
        valid = []
        for candidate in _balanced_spans_with_opener(repaired, "["):
            try:
                valid.append(
                    _as_object_array(strict_json_loads(candidate), what, element_keys)
                )
            except FakeToolCallTextError:
                raise
            except StructuredOutputError as exc:
                shape_error = exc
            except json.JSONDecodeError:
                pass
        if len(valid) > 1:
            raise AmbiguousStructuredOutputError(
                f"model response contained multiple complete JSON arrays of {what}"
            )
        if valid:
            return valid[0]
        names = _fake_tool_names(repaired)
        if names:
            raise _fake_tool_call_error(what, names)

    if shape_error is not None:
        raise shape_error

    raise StructuredOutputError(
        f"could not extract a JSON array of {what} from the model response: "
        f"{raw_text[:200]!r}"
    )


def validate_plain_text_response(raw_text: str, *, what: str) -> str:
    """Return clean prose, rejecting wrappers, meta-text, and textual tool calls."""
    text = raw_text.strip()
    if not text:
        raise StructuredOutputError(f"model produced no prose for {what}")
    if "```" in text:
        raise StructuredOutputError(f"{what} was wrapped in a markdown code fence")
    if re.match(r"^<think\b", text, re.IGNORECASE):
        raise StructuredOutputError(f"{what} still contains a reasoning block")
    if _fake_tool_names(text):
        raise FakeToolCallTextError(
            f"{what} wrote a tool call as plain text instead of returning prose"
        )
    try:
        decoded = strict_json_loads(text)
    except (json.JSONDecodeError, StructuredOutputError):
        decoded = None
    if isinstance(decoded, (dict, list)):
        raise StructuredOutputError(f"{what} returned JSON instead of prose")
    if _META_PROSE_PREFIX.search(text) or _META_PROSE_SUFFIX.search(text):
        raise StructuredOutputError(f"{what} contains assistant meta-commentary")
    return text
