"""A bounded, reusable tool-use loop.

This is the seam every agentic node in MuseAI runs on, and the seam future tools
plug into. It knows nothing about continuity, critics, or prose: it takes an
endpoint, a message list, a tool schema list, and a name -> callable registry,
and it drives the model until it stops asking for tools.

Two properties matter more than anything else here:

* **It is bounded.** A model that keeps calling tools cannot spin forever. Once
  ``max_iterations`` model turns have each come back asking for another tool,
  the loop makes one final call with ``tools`` omitted entirely, which leaves
  the endpoint no way to answer except in prose. The loop always terminates with
  a plain answer.
* **A tool never crashes the loop.** An unknown tool name, a malformed argument
  blob, or an exception inside a tool becomes a structured error *object*
  (``{"error": {"tool", "type", "message"}}``, serialised as that call's result)
  handed back to the model. Models recover from being told exactly what failed;
  they cannot recover from a traceback. Only ``call_llm`` itself may raise.

The loop also enforces a per-tool call cap (``tool_call_cap``): a model that
keeps re-running the same search gets a ``call_cap_exceeded`` error instead of
another result, so the bounded iterations are spent answering, not looping.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Awaitable, Callable, Mapping, Sequence

from museai.core.logging_setup import get_fsm_logger
from museai.llm.client import LLMResponse, call_llm

# Tool results echoed into an event are for a human watching a browser, not a
# transcript. The model still receives the full result.
_RESULT_PREVIEW_CHARS = 200

EventCallback = Callable[[dict[str, Any]], Any | Awaitable[Any]]


class AgentLoopError(ValueError):
    """The loop was configured impossibly — a non-positive iteration budget."""


def _error_payload(tool: str, error_type: str, message: str) -> str:
    """A tool failure as a structured object the model can read fields off."""
    return json.dumps(
        {"error": {"tool": tool, "type": error_type, "message": message}},
        ensure_ascii=False,
    )


async def _emit_event(on_event: EventCallback | None, event: dict[str, Any]) -> None:
    """Invoke an event callback that may be sync or async."""
    if on_event is None:
        return
    result = on_event(event)
    if inspect.isawaitable(result):
        await result


def _parse_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """Parse a tool call's ``arguments`` blob into kwargs.

    Returns ``(kwargs, error)``. Endpoints send arguments as a JSON *string*;
    some send a dict outright, and some send nothing at all for a zero-argument
    tool. All three are legitimate, so only unparseable text is an error.
    """
    if raw is None or raw == "":
        return {}, None
    if isinstance(raw, Mapping):
        return dict(raw), None
    if not isinstance(raw, str):
        return {}, f"arguments had unusable type {type(raw).__name__}"

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, f"arguments were not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return {}, f"arguments must decode to an object, got {type(parsed).__name__}"
    return parsed, None


def _stringify(result: Any) -> str:
    """Render a tool's return value as the string content of a tool message."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(result)


async def _invoke_tool(
    name: str,
    kwargs: dict[str, Any],
    tool_impls: Mapping[str, Callable[..., Any]],
) -> str:
    """Run one tool and return its result as a string, never raising.

    An unknown name and a raising tool both come back as an error string. The
    model reads it as that call's result and can try something else.
    """
    impl = tool_impls.get(name)
    if impl is None:
        known = ", ".join(sorted(tool_impls)) or "none"
        return _error_payload(
            name, "unknown_tool", f"unknown tool {name!r}; available tools: {known}"
        )

    try:
        if inspect.iscoroutinefunction(impl):
            result = await impl(**kwargs)
        else:
            # A sync tool (e.g. web_search's blocking HTTP) must not stall the
            # event loop — SSE streaming and the web UI share it.
            result = await asyncio.to_thread(impl, **kwargs)
            if inspect.isawaitable(result):
                result = await result
    except Exception as exc:  # noqa: BLE001 - a tool fault is data, not a crash
        return _error_payload(
            name, "tool_failure", f"{type(exc).__name__}: {exc}"
        )
    return _stringify(result)


async def _run_tool_calls(
    tool_calls: Sequence[Mapping[str, Any]],
    tool_impls: Mapping[str, Callable[..., Any]],
    on_event: EventCallback | None,
    call_counts: dict[str, int],
    call_cap: int | None,
) -> list[dict[str, Any]]:
    """Execute every tool call in one model turn, in order, into tool messages.

    ``call_counts`` persists across the whole loop; a tool at ``call_cap`` gets
    a ``call_cap_exceeded`` error instead of another execution.
    """
    messages: list[dict[str, Any]] = []
    logger = get_fsm_logger()

    for call in tool_calls:
        function = call.get("function") or {}
        name = function.get("name") or ""
        kwargs, error = _parse_arguments(function.get("arguments"))
        if error is not None:
            content = _error_payload(name, "bad_arguments", error)
        elif call_cap is not None and call_counts.get(name, 0) >= call_cap:
            content = _error_payload(
                name,
                "call_cap_exceeded",
                f"tool {name!r} was already called {call_cap} times in this "
                f"task; work with the results you have and answer now",
            )
        else:
            call_counts[name] = call_counts.get(name, 0) + 1
            content = await _invoke_tool(name, kwargs, tool_impls)

        logger.info(
            "agent_loop tool=%s args=%s result_chars=%d",
            name or "<unnamed>",
            json.dumps(kwargs, ensure_ascii=False, default=str),
            len(content),
        )
        await _emit_event(
            on_event,
            {
                "tool": name,
                "arguments": kwargs,
                "result_preview": content[:_RESULT_PREVIEW_CHARS],
            },
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "name": name,
                "content": content,
            }
        )
    return messages


async def run_agent_loop(
    endpoint: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    tool_impls: Mapping[str, Callable[..., Any]],
    max_iterations: int,
    on_event: EventCallback | None = None,
    agent: str = "system",
    on_token: Callable[[str], Awaitable[None]] | None = None,
    tool_call_cap: int | None = None,
) -> LLMResponse:
    """Drive the model through tool calls until it answers in prose.

    Runs at most ``max_iterations`` tool-enabled model turns. If the last of them
    still asks for a tool, one final tool-free call forces a plain answer, and
    that response is what comes back — so the return value never carries pending
    tool calls.

    ``agent`` names the node driving the loop for the chat transcript; each
    model turn inside the loop appears there as its own call.

    ``on_token`` streams every model turn's text tokens through to the caller
    (the drafter's live view). Turns that only request tools emit little or no
    text, so in practice the stream is the final answer.

    ``tool_call_cap`` bounds how many times any *one* tool may run across the
    whole loop; ``None`` leaves them uncapped.

    ``messages`` is not mutated; the loop works on its own copy.
    """
    if max_iterations < 1:
        raise AgentLoopError(f"max_iterations must be >= 1, got {max_iterations}")

    working: list[dict[str, Any]] = [dict(m) for m in messages]
    call_counts: dict[str, int] = {}

    for _ in range(max_iterations):
        response = await call_llm(
            endpoint, working, tools=tools, agent=agent, stream=True, on_token=on_token
        )
        if not response.tool_calls:
            return response

        working.append(
            {
                "role": "assistant",
                "content": response.text or "",
                "tool_calls": [dict(call) for call in response.tool_calls],
            }
        )
        working.extend(
            await _run_tool_calls(
                response.tool_calls, tool_impls, on_event, call_counts, tool_call_cap
            )
        )

    # The budget is spent and the model is still reaching for tools. Withholding
    # the schemas leaves it nothing to answer with but prose.
    get_fsm_logger().info(
        "agent_loop max_iterations=%d reached; forcing a tool-free answer",
        max_iterations,
    )
    return await call_llm(endpoint, working, agent=agent, stream=True, on_token=on_token)
