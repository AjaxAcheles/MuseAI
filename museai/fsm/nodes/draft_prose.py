"""Prose drafting node.

Renders the drafter prompt from ``active_context_package`` and streams one beat
of prose from the endpoint, republishing each token to the stream bus so the web
UI sees the draft as it is written.

The stream emits a single ``beat_start`` event before the first token, then one
``token`` event per token. The log does not follow suit: ``llm_io.log`` records
the call once, on the assembled text. Tokens are for the browser, not the disk.

No ``max_tokens`` is sent unless ``endpoint.max_output_tokens`` is set, per the
v1.02 wire-format finding: a reasoning-style endpoint bills hidden reasoning
against that budget, and a beat truncated mid-sentence is worse than a long
one. Either way a draft that comes back with ``finish_reason == "length"`` is
refused rather than committed half-written.
"""

from __future__ import annotations

from museai.core.logging_setup import log_node_event
from museai.core.stream_bus import bus
from museai.fsm.nodes.assemble_context import drafter_messages
from museai.fsm.nodes.deps import DraftingError, get_node_config
from museai.fsm.state import OrchestratorState
from museai.llm.structured import (
    StructuredOutputError,
    response_truncation_remedy,
    validate_plain_text_response,
)
from museai.fsm.tools.loop import run_agent_loop
from museai.fsm.tools.registry import tool_impls_for, tool_specs_for

PHASE = "Auditing"


async def draft_prose(state: OrchestratorState) -> dict:
    """Draft the active beat's prose, streaming it to the bus as it arrives.

    Returns the state delta: the finished prose in both ``current_draft_text``
    and ``streaming_buffer``.
    """
    config = get_node_config()
    package = state["active_context_package"]
    if not package:
        raise DraftingError(
            "draft_prose ran with no active_context_package; assemble_context "
            "must run first"
        )

    beat = package["beat"]
    beat_id = beat["id"]
    messages = drafter_messages(package)

    log_node_event(
        "draft_prose",
        event="start",
        beat_id=beat_id,
        context_tokens=package["budget"]["tokens"],
    )

    started = False

    async def on_token(token: str) -> None:
        nonlocal started
        if not started:
            started = True
            await bus.publish(
                "beat_start",
                {
                    "beat_id": beat_id,
                    "chapter_id": package["chapter"]["id"],
                    "ordering": beat["ordering"],
                },
            )
        await bus.publish("token", {"beat_id": beat_id, "text": token})

    async def on_tool_call(event: dict) -> None:
        log_node_event(
            "draft_prose",
            event="tool_call",
            beat_id=beat_id,
            tool=event["tool"],
            args=event["arguments"],
        )
        await bus.publish("drafter_tool", {"beat_id": beat_id, **event})

    response = await run_agent_loop(
        config.endpoint_for("drafter"),
        messages,
        tool_specs_for("drafter"),
        tool_impls_for("drafter"),
        config.generation.max_agent_iterations,
        on_event=on_tool_call,
        agent="drafter",
        on_token=on_token,
        tool_call_cap=config.generation.tool_call_cap,
        tool_timeout=config.generation.tool_timeout,
    )

    # The final turn's text is the draft. Tokens streamed to the browser may
    # include earlier tool-requesting turns; the committed prose never does.
    # Truncation first: a reply cut off at the token limit can be empty (a
    # reasoning model that spent the whole budget thinking) as easily as it can
    # stop mid-sentence, and "no prose" would name neither the cause nor the fix.
    if response.finish_reason == "length":
        raise DraftingError(
            f"the draft for beat {beat_id!r} was cut off (finish_reason='length'): "
            + response_truncation_remedy(
                response, config.endpoint_for("drafter"), role="drafter"
            )
        )
    try:
        draft = validate_plain_text_response(response.text, what=f"draft for beat {beat_id!r}")
    except StructuredOutputError as exc:
        raise DraftingError(str(exc)) from exc
    log_node_event(
        "draft_prose",
        event="drafted",
        beat_id=beat_id,
        words=len(draft.split()),
        tokens_out=response.tokens_out,
        finish_reason=response.finish_reason,
    )
    log_node_event("draft_prose", event="phase_change", phase=PHASE, beat_id=beat_id)
    await bus.publish(
        "phase_change", {"phase": PHASE, "node": "draft_prose", "beat_id": beat_id}
    )

    return {"current_draft_text": draft, "streaming_buffer": draft}
