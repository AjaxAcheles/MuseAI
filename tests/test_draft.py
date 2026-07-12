"""Tests for museai.fsm.nodes.draft_prose.

The agent loop's ``call_llm`` is replaced with a fake that drives the node's
``on_token`` callback exactly as the real streaming client does, so the bus
contract is exercised without a socket.
"""

from __future__ import annotations

import pytest

from museai.core.stream_bus import bus
from museai.fsm.tools import loop as loop_module
from museai.fsm.nodes.deps import DraftingError, set_node_config
from museai.fsm.nodes.draft_prose import draft_prose
from museai.fsm.state import FSM_Pointer, make_initial_state

BEAT_ID = "arc-1-c01-b01"
TOKENS = ["The post ", "came up ", "the path ", "in a canvas sack."]
PROSE = "".join(TOKENS)

PACKAGE = {
    "beat": {
        "id": BEAT_ID,
        "ordering": 1,
        "intent": "Mara finds the letter.",
        "entry_state": "A routine morning.",
        "exit_state": "Mara is holding her own handwriting.",
        "focal_character_id": "char-mara",
    },
    "pad_constraint": "Energy with nowhere to go. Checking exits.",
    "chapter": {
        "id": "arc-1-c01",
        "description": "Mara catalogs the letters.",
        "obligations": ["Mara dates the earliest letter."],
    },
    "threads": [{"id": "t1", "status": "open", "description": "Who writes them?",
                 "priority_score": 0.9}],
    "characters": [{"id": "char-mara", "name": "Mara", "description": "The keeper.",
                    "pad": {"pleasure": -0.2, "arousal": 0.1, "dominance": 0.3}}],
    "recent_prose": ["The lamp turned through the fog."],
    "budget": {"budget": 8000, "tokens_before": 400, "tokens": 400,
               "dropped_prose_passages": 0, "dropped_threads": 0,
               "over_budget": False},
}


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens_out = len(text.split())
        self.finish_reason = "stop"
        self.tool_calls = []


def _streaming_call_llm(tokens=TOKENS, seen_messages: list | None = None):
    """A fake ``call_llm`` that streams ``tokens`` through ``on_token``."""

    async def fake(endpoint, messages, *, stream=False, on_token=None, **kwargs):
        if seen_messages is not None:
            seen_messages.append(messages)
        assert stream is True, "draft_prose must stream"
        for token in tokens:
            await on_token(token)
        return _Response("".join(tokens))

    return fake


def _state(package=PACKAGE) -> dict:
    return make_initial_state(
        "test-project",
        FSM_Pointer(arc_id="arc-1", chapter_id="arc-1-c01", beat_index=0),
        active_context_package=package,
    )


@pytest.fixture
def configured(config_factory):
    config = config_factory()
    set_node_config(config)
    return config


async def test_tokens_are_published_and_the_draft_is_their_join(
    configured, monkeypatch
):
    monkeypatch.setattr(loop_module, "call_llm", _streaming_call_llm())

    queue = bus.subscribe()
    try:
        delta = await draft_prose(_state())
        events = [queue.get_nowait() for _ in range(queue.qsize())]
    finally:
        bus.unsubscribe(queue)

    assert delta["current_draft_text"] == PROSE
    assert delta["streaming_buffer"] == PROSE

    tokens = [e["data"]["text"] for e in events if e["type"] == "token"]
    assert tokens == TOKENS
    assert "".join(tokens) == delta["current_draft_text"]


async def test_beat_start_is_published_once_before_the_first_token(
    configured, monkeypatch
):
    monkeypatch.setattr(loop_module, "call_llm", _streaming_call_llm())

    queue = bus.subscribe()
    try:
        await draft_prose(_state())
        events = [queue.get_nowait() for _ in range(queue.qsize())]
    finally:
        bus.unsubscribe(queue)

    types = [event["type"] for event in events]
    assert types.count("beat_start") == 1
    assert types.index("beat_start") < types.index("token")

    beat_start = next(e["data"] for e in events if e["type"] == "beat_start")
    assert beat_start["beat_id"] == BEAT_ID
    assert "word_target" not in beat_start


async def test_the_run_ends_on_the_auditing_phase_change(configured, monkeypatch):
    monkeypatch.setattr(loop_module, "call_llm", _streaming_call_llm())

    queue = bus.subscribe()
    try:
        await draft_prose(_state())
        events = [queue.get_nowait() for _ in range(queue.qsize())]
    finally:
        bus.unsubscribe(queue)

    assert events[-1]["type"] == "phase_change"
    assert events[-1]["data"]["phase"] == "Auditing"


async def test_the_prompt_carries_the_beat_and_its_pad_constraint(
    configured, monkeypatch
):
    seen: list = []
    monkeypatch.setattr(
        loop_module, "call_llm", _streaming_call_llm(seen_messages=seen)
    )

    await draft_prose(_state())

    rendered = "\n".join(message["content"] for message in seen[0])
    assert PACKAGE["pad_constraint"] in rendered
    assert PACKAGE["beat"]["intent"] in rendered
    assert PACKAGE["chapter"]["obligations"][0] in rendered
    assert PACKAGE["recent_prose"][0] in rendered


async def test_drafting_without_an_assembled_context_is_a_hard_failure(
    configured, monkeypatch
):
    async def unreachable(*args, **kwargs):
        raise AssertionError("the endpoint must not be reached")

    monkeypatch.setattr(loop_module, "call_llm", unreachable)

    with pytest.raises(DraftingError, match="assemble_context"):
        await draft_prose(_state(package={}))


async def test_an_empty_stream_is_a_hard_failure(configured, monkeypatch):
    monkeypatch.setattr(loop_module, "call_llm", _streaming_call_llm(tokens=[]))

    with pytest.raises(DraftingError, match="no prose"):
        await draft_prose(_state())


async def test_the_loop_is_offered_the_drafter_roster(configured, monkeypatch):
    offered: list = []

    async def fake(endpoint, messages, *, stream=False, on_token=None, **kwargs):
        offered.append(kwargs.get("tools"))
        for token in TOKENS:
            await on_token(token)
        return _Response(PROSE)

    monkeypatch.setattr(loop_module, "call_llm", fake)

    await draft_prose(_state())

    assert [t["function"]["name"] for t in offered[0]] == [
        "get_current_pointer_context", "get_recent_commits", "search_manuscript",
        "get_character_sheet", "find_repetition",
    ]
