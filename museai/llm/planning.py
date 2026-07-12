"""Asking a planner for a JSON array, and insisting until it produces one.

Both planners — chapter and beat — ask the model for a JSON array and are dead
in the water without one. `parse_json_array` raises, `PlanningError` never gets a
chance, and the manager's catch-all kills a run that may already have committed
hours of prose.

The escalation ladder here mirrors ``fsm/nodes/critics.py``, which has been
carrying it since v1.12:

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

Unlike the critic, a planner cannot degrade. There is no honest empty plan: zero
chapters is a failed run, so exhausting every rung still raises.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from museai.core.config import EndpointConfig
from museai.core.logging_setup import log_node_event
from museai.core.stream_bus import bus
from museai.llm.client import call_llm
from museai.llm.structured import (
    FakeToolCallTextError,
    StructuredOutputError,
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

_INVALID_REPLY_MARKER = (
    "[Previous assistant reply omitted: it was invalid planner output and "
    "contained tool-call-shaped JSON as plain text rather than a JSON plan array.]"
)


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
) -> list[dict]:
    """Call ``agent`` until it yields a usable JSON array of ``what``, or give up.

    ``what`` ("chapters", "beats") names the plan in errors and log lines; ``node``
    names the FSM node for the structured log. Raises
    :class:`StructuredOutputError` when every retry and the repair pass are spent.

    When ``tools`` and ``tool_impls`` are given, each attempt runs the bounded
    agent loop so the planner may gather context before answering; the loop
    always terminates in a plain reply, which is parsed exactly as before.
    ``on_tool_event`` is handed to that loop: one event per executed tool call.
    """
    if tools is not None and tool_impls is not None:
        # Imported here, not at module top: llm/ stays importable without fsm/,
        # and the loop itself only depends on this package's client.
        from museai.fsm.tools.loop import run_agent_loop

        async def _ask(conv: list) -> Any:
            return await run_agent_loop(
                endpoint,
                conv,
                tools,
                tool_impls,
                max_tool_iterations,
                on_event=on_tool_event,
                agent=agent,
                tool_call_cap=tool_call_cap,
            )
    else:

        async def _ask(conv: list) -> Any:
            return await call_llm(endpoint, conv, agent=agent, stream=True)

    conversation = list(messages)
    last_text = ""
    last_error = ""

    for attempt in range(retries + 1):
        response = await _ask(conversation)
        last_text = response.text
        try:
            return parse_json_array(last_text, what=what)
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
                _FAKE_TOOL_CORRECTION_TEMPLATE if fake_tool_text else _CORRECTION_TEMPLATE
            ).format(error=last_error)
            conversation = [
                *conversation,
                {"role": "assistant", "content": assistant_content},
                {"role": "user", "content": correction},
            ]

    # Last rung: the model will not fix its own quoting, so we do — and say so.
    planned = parse_json_array(last_text, what=what, repair=True)
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
