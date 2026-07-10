"""Prose drafting node.

Renders the drafter prompt from ``active_context_package`` and streams one beat
of prose from the endpoint, republishing each token to the stream bus so the web
UI sees the draft as it is written.

The stream emits a single ``beat_start`` event before the first token, then one
``token`` event per token. The log does not follow suit: ``llm_io.log`` records
the call once, on the assembled text. Tokens are for the browser, not the disk.

No ``max_tokens`` is sent, per the v1.02 wire-format finding: a reasoning-style
endpoint bills hidden reasoning against that budget, and a beat truncated
mid-sentence is worse than a long one.
"""

from __future__ import annotations

from museai.core.logging_setup import log_node_event
from museai.core.stream_bus import bus
from museai.fsm.nodes.assemble_context import drafter_messages
from museai.fsm.nodes.deps import DraftingError, get_node_config
from museai.fsm.state import OrchestratorState
from museai.llm.client import call_llm

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
        word_target=beat["word_target"],
        context_tokens=package["budget"]["tokens"],
    )

    pieces: list[str] = []
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
                    "word_target": beat["word_target"],
                },
            )
        pieces.append(token)
        await bus.publish("token", {"beat_id": beat_id, "text": token})

    response = await call_llm(
        config.endpoint, messages, agent="drafter", stream=True, on_token=on_token
    )

    draft = "".join(pieces)
    if not draft.strip():
        raise DraftingError(
            f"the endpoint returned no prose for beat {beat_id!r} "
            f"(finish_reason={response.finish_reason!r})"
        )

    log_node_event(
        "draft_prose",
        event="drafted",
        beat_id=beat_id,
        words=len(draft.split()),
        word_target=beat["word_target"],
        tokens_out=response.tokens_out,
        finish_reason=response.finish_reason,
    )
    log_node_event("draft_prose", event="phase_change", phase=PHASE, beat_id=beat_id)
    await bus.publish(
        "phase_change", {"phase": PHASE, "node": "draft_prose", "beat_id": beat_id}
    )

    return {"current_draft_text": draft, "streaming_buffer": draft}
