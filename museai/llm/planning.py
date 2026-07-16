"""Asking a planner for a JSON array, and insisting until it produces one.

Both planners — chapter and beat — ask the model for a JSON array and are dead
in the water without one. `parse_json_array` raises, `PlanningError` never gets a
chance, and the manager's catch-all kills a run that may already have committed
hours of prose.

The bounded correction path here mirrors ``fsm/nodes/critics.py``, which has
been carrying it since v1.12:

1. Ask. Parse strictly.
2. On a parse failure, show the model its own reply and the exact error, and ask
   again — up to ``generation.planner_parse_retries`` times. A model that
   forgot to escape a quote generally fixes it when told which line broke.
3. When the re-prompts are spent, try :func:`repair_json_text` on the last reply.

Repair is last, never first. A re-prompt returns the model's own words; the
repair returns *our reconstruction* of them, and the difference matters when the
text becomes a beat's ``intent``. Every repair is logged at WARNING and published
to the stream, because a plan the model did not literally write is not a thing to
pass off silently.

Failures are diagnosed before they are corrected. A reply cut off at the
endpoint's token limit (``finish_reason == "length"``) is *truncation*, not a
quoting mistake: it gets a truncation-specific correction asking for a shorter
plan, is logged as ``truncated`` rather than ``parse_failed``, and is never fed
to the quote repair — repairing half an array puts words in the model's mouth
twice over. An empty reply likewise gets its own correction, and neither case
appends the broken reply verbatim when doing so would teach the model nothing.

Retries continue from the agent loop's working conversation, so tool results
gathered on one attempt are not re-fetched on the next.

Unlike the critic, a planner cannot degrade. There is no honest empty plan: zero
chapters is a failed run, so exhausting every rung still raises.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from museai.core.config import EndpointConfig
from museai.core.logging_setup import log_node_event
from museai.core.stream_bus import bus
from museai.llm.client import call_llm
from museai.llm.structured import (
    FakeToolCallTextError,
    StructuredOutputError,
    TruncatedResponseError,
    parse_json_array,
)

# Carries the validation error verbatim. A model that wrote `("It was stronger")`
# inside a string needs to see which line broke, not a generic scolding — and it
# needs to be told the fix, because "invalid JSON" does not imply "escape it".
_CORRECTION_TEMPLATE = (
    "Your previous reply could not be parsed as JSON: {error}\n\n"
    "Return only the fenced JSON array described in the output format, and "
    "nothing else. Inside a string value, escape every double quote as \\\" or "
    "use single quotes for dialogue — an unescaped quote ends the string early "
    "and makes the whole array unreadable. Do not return the example values "
    "from the schema."
)

_FAKE_TOOL_CORRECTION_TEMPLATE = (
    "Your previous reply wrote tool calls as plain text, so no tool was executed: "
    "{error}\n\n"
    "Do not write {{\"name\": ..., \"parameters\": ...}} objects in your response. "
    "If you still need a tool, use the provided tool-call channel. If you have "
    "enough context, return only the fenced JSON array described in the output "
    "format. Do not include tool calls, notes, schema examples, or prose outside "
    "the array."
)

# Quote-escaping advice cannot fix a reply the endpoint cut off; asking for a
# shorter answer can.
_TRUNCATION_CORRECTION_TEMPLATE = (
    "Your previous reply was cut off at the endpoint's output token limit "
    "before the JSON array was complete.\n\n"
    "Return the complete fenced JSON array, but make it shorter: fewer "
    "elements if the plan allows it, and briefer field values — one or two "
    "sentences per description, no prose outside the array. Do not repeat the "
    "cut-off reply."
)

_EMPTY_CORRECTION_TEMPLATE = (
    "Your previous reply was empty.\n\n"
    "Return only the fenced JSON array described in the output format — at "
    "least one element, and nothing else. Do not reply with an empty message, "
    "reasoning only, or prose."
)

_INVALID_REPLY_MARKER = (
    "[Previous assistant reply omitted: it was invalid planner output and "
    "contained tool-call-shaped JSON as plain text rather than a JSON plan array.]"
)

_TRUNCATED_REPLY_MARKER = (
    "[Previous assistant reply omitted: it was cut off at the endpoint's "
    "output token limit before the JSON array was complete.]"
)

_EMPTY_REPLY_MARKER = "[Previous assistant reply omitted: it was empty.]"


async def call_llm_for_json_array(
    endpoint: EndpointConfig,
    messages: Sequence[Mapping[str, Any]],
    *,
    what: str,
    agent: str,
    node: str,
    retries: int,
    tools: Sequence[Mapping[str, Any]] | None = None,
    tool_impls: Mapping[str, Any] | None = None,
    max_tool_iterations: int = 1,
    on_tool_event: Any = None,
    tool_call_cap: int | None = None,
    tool_timeout: float | None = None,
    element_keys: Sequence[str] | None = None,
    item_validator: Callable[[dict, int], dict] | None = None,
) -> list[dict]:
    """Call ``agent`` until it yields a usable JSON array of ``what``, or give up.

    ``what`` ("chapters", "beats") names the plan in errors and log lines; ``node``
    names the FSM node for the structured log. ``element_keys`` is handed to
    :func:`parse_json_array` so a fragment of a truncated reply cannot pass as
    the plan. Raises :class:`StructuredOutputError` when every retry and the
    repair pass are spent, or :class:`TruncatedResponseError` when the model
    could not fit a plan inside the endpoint's output token limit.

    When ``tools`` and ``tool_impls`` are given, each attempt runs the bounded
    agent loop so the planner may gather context before answering; the loop
    always terminates in a plain reply, which is parsed exactly as before.
    ``on_tool_event`` is handed to that loop: one event per executed tool call.
    Corrections continue from the loop's working conversation, so a retry keeps
    the tool results the previous attempt already fetched.
    """
    if tools is not None and tool_impls is not None:
        # Imported here, not at module top: llm/ stays importable without fsm/,
        # and the loop itself only depends on this package's client.
        from museai.fsm.tools.loop import run_agent_loop

        async def _ask(conv: list, conv_out: list) -> Any:
            return await run_agent_loop(
                endpoint,
                conv,
                tools,
                tool_impls,
                max_tool_iterations,
                on_event=on_tool_event,
                agent=agent,
                tool_call_cap=tool_call_cap,
                tool_timeout=tool_timeout,
                conversation_out=conv_out,
            )
    else:

        async def _ask(conv: list, conv_out: list) -> Any:
            conv_out[:] = [dict(m) for m in conv]
            return await call_llm(
                endpoint, conv, agent=agent, stream=True, retry_on_empty=True
            )

    conversation = list(messages)
    last_text = ""
    last_error = ""
    last_truncated = False

    def _parse(text: str, *, repair: bool = False) -> list[dict]:
        planned = parse_json_array(
            text, what=what, repair=repair, element_keys=element_keys
        )
        if item_validator is None:
            return planned
        return [item_validator(item, index) for index, item in enumerate(planned, start=1)]

    for attempt in range(retries + 1):
        # The working conversation as the model saw it: the original messages
        # plus any tool calls and results this attempt made. Corrections build
        # on it so a retry does not re-run every tool.
        working: list = []
        response = await _ask(conversation, working)
        last_text = response.text
        last_truncated = getattr(response, "finish_reason", None) == "length"

        if last_truncated:
            # Do not parse a truncated reply: a balanced inner fragment of a
            # half-written array can pass for the plan (and once did).
            last_error = (
                f"the {what} reply was cut off at the endpoint's output token "
                f"limit (finish_reason='length')"
            )
            log_node_event(
                node,
                level=logging.WARNING,
                event="truncated",
                what=what,
                attempt=attempt + 1,
                retries=retries,
                error=last_error[:200],
            )
            if attempt == retries:
                break
            assistant_content = _TRUNCATED_REPLY_MARKER
            correction = _TRUNCATION_CORRECTION_TEMPLATE
        elif not last_text.strip():
            last_error = f"model returned an empty response for {what}"
            log_node_event(
                node,
                level=logging.WARNING,
                event="empty_reply",
                what=what,
                attempt=attempt + 1,
                retries=retries,
                error=last_error[:200],
            )
            if attempt == retries:
                break
            assistant_content = _EMPTY_REPLY_MARKER
            correction = _EMPTY_CORRECTION_TEMPLATE
        else:
            try:
                return _parse(last_text)
            except StructuredOutputError as exc:
                last_error = str(exc)
                fake_tool_text = isinstance(exc, FakeToolCallTextError)
                log_node_event(
                    node,
                    level=logging.WARNING,
                    event="fake_tool_call_text" if fake_tool_text else "parse_failed",
                    what=what,
                    attempt=attempt + 1,
                    retries=retries,
                    error=last_error[:200],
                )
                if attempt == retries:
                    break
                assistant_content = _INVALID_REPLY_MARKER if fake_tool_text else last_text
                correction = (
                    _FAKE_TOOL_CORRECTION_TEMPLATE
                    if fake_tool_text
                    else _CORRECTION_TEMPLATE
                ).format(error=last_error)

        conversation = [
            *(working or conversation),
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": correction},
        ]

    if last_truncated:
        # Repairing half a written array is nonsense; name the actual knob.
        raise TruncatedResponseError(
            f"every {what} reply was cut off at the endpoint's output token "
            f"limit; raise endpoint.max_output_tokens (or leave it unset to "
            f"omit the cap) or ask for a shorter plan"
        )

    # Last rung: the model will not fix its own quoting, so we do — and say so.
    planned = _parse(last_text, repair=True)
    log_node_event(
        node,
        level=logging.WARNING,
        event="json_repaired",
        what=what,
        count=len(planned),
        error=last_error[:200],
    )
    await bus.publish(
        "planner_repaired",
        {
            "node": node,
            "agent": agent,
            "what": what,
            "count": len(planned),
            "error": last_error[:300],
        },
    )
    return planned
