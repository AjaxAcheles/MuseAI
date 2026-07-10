"""The View Chat surface: thinking capture, chat events, transcript, routes.

The LLM-client tests here drive `call_llm` through `httpx.MockTransport`, like
`test_client.py`; the route tests boot the app through the shared `web_app`
fixture.
"""

from __future__ import annotations

import json

import httpx
import pytest

from museai.core import chat_log
from museai.core.config import EndpointConfig
from museai.core.stream_bus import bus
from museai.llm.client import (
    _LIVE_CALLS,
    LLMCallError,
    _ThinkTagSplitter,
    call_llm,
    live_chat_calls,
)

MESSAGES = [{"role": "user", "content": "Plan a chapter."}]


@pytest.fixture
def endpoint() -> EndpointConfig:
    return EndpointConfig(
        base_url="https://example.invalid/v1",
        api_key="secret-key-do-not-log",
        model_name="test-model",
        tokenizer_family="char_heuristic",
        request_timeout=5,
        temperature=0.3,
    )


@pytest.fixture
def chat_events(monkeypatch):
    """Capture every chat_* event published to the bus during a test."""
    captured: list[dict] = []
    original = bus.publish

    async def spy(event_type, data):
        if event_type.startswith("chat_"):
            captured.append({"type": event_type, "data": data})
        await original(event_type, data)

    monkeypatch.setattr(bus, "publish", spy)
    yield captured
    bus.last_snapshot.clear()


@pytest.fixture(autouse=True)
def unconfigured_transcript(monkeypatch):
    """Client-level tests must not inherit a transcript path from other tests."""
    monkeypatch.setattr(chat_log, "_path", None)


def sse(*chunks: dict, done: bool = True) -> bytes:
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def delta_chunk(**delta) -> dict:
    return {"choices": [{"delta": delta}]}


def transport_for(*chunks: dict) -> httpx.MockTransport:
    return httpx.MockTransport(lambda r: httpx.Response(200, content=sse(*chunks)))


# ------------------------------------------------------------ thinking capture


async def test_streamed_reasoning_content_becomes_thinking(endpoint):
    """DeepSeek/vLLM-style endpoints put chain-of-thought in reasoning_content."""
    seen: list[str] = []
    transport = transport_for(
        delta_chunk(reasoning_content="I should open"),
        delta_chunk(reasoning_content=" with the storm."),
        delta_chunk(content="The storm broke."),
    )
    result = await call_llm(endpoint, MESSAGES, stream=True, on_token=seen.append, transport=transport)

    assert result.thinking == "I should open with the storm."
    assert result.text == "The storm broke."
    assert seen == ["The storm broke."], "thinking must never reach on_token"


async def test_streamed_reasoning_field_variant_is_also_captured(endpoint):
    transport = transport_for(delta_chunk(reasoning="hmm"), delta_chunk(content="ok"))
    result = await call_llm(endpoint, MESSAGES, stream=True, transport=transport)
    assert result.thinking == "hmm"
    assert result.text == "ok"


async def test_streamed_think_tags_are_split_out(endpoint):
    """Ollama serving a reasoning model opens content with a <think> block."""
    seen: list[str] = []
    transport = transport_for(
        delta_chunk(content="<think>weigh the"),
        delta_chunk(content=" options</think>"),
        delta_chunk(content="The lamp held."),
    )
    result = await call_llm(endpoint, MESSAGES, stream=True, on_token=seen.append, transport=transport)

    assert result.thinking == "weigh the options"
    assert result.text == "The lamp held."
    assert "".join(seen) == "The lamp held."
    assert "<think>" not in result.text


async def test_think_tag_split_across_chunk_boundaries(endpoint):
    transport = transport_for(
        delta_chunk(content="<thi"),
        delta_chunk(content="nk>a thought</thi"),
        delta_chunk(content="nk>prose"),
    )
    result = await call_llm(endpoint, MESSAGES, stream=True, transport=transport)
    assert result.thinking == "a thought"
    assert result.text == "prose"


async def test_unclosed_think_block_is_all_thinking(endpoint):
    transport = transport_for(delta_chunk(content="<think>never stopped"))
    result = await call_llm(endpoint, MESSAGES, stream=True, transport=transport)
    assert result.thinking == "never stopped"
    assert result.text == ""


async def test_think_tag_mid_prose_is_prose(endpoint):
    """Only a block opening the reply is deliberation; mid-text tags are text."""
    transport = transport_for(delta_chunk(content="She wrote <think> on the board."))
    result = await call_llm(endpoint, MESSAGES, stream=True, transport=transport)
    assert result.thinking == ""
    assert result.text == "She wrote <think> on the board."


def test_splitter_holds_back_a_partial_close_tag():
    splitter = _ThinkTagSplitter()
    pieces = splitter.feed("<think>abc</th")
    pieces += splitter.feed("ink>done")
    pieces += splitter.flush()
    assert ("thinking", "abc") in [(k, "".join(t for k2, t in pieces if k2 == k)) for k in {"thinking"}]
    assert "".join(t for k, t in pieces if k == "response") == "done"
    assert "</th" not in "".join(t for k, t in pieces if k == "thinking")


async def test_nonstream_reasoning_content_field(endpoint):
    payload = {
        "choices": [
            {
                "message": {"content": "Chapter one.", "reasoning_content": "outline first"},
                "finish_reason": "stop",
            }
        ]
    }
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    result = await call_llm(endpoint, MESSAGES, transport=transport)
    assert result.thinking == "outline first"
    assert result.text == "Chapter one."


async def test_nonstream_think_block_is_split(endpoint):
    payload = {
        "choices": [
            {
                "message": {"content": "<think>quietly</think>\nChapter one."},
                "finish_reason": "stop",
            }
        ]
    }
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    result = await call_llm(endpoint, MESSAGES, transport=transport)
    assert result.thinking == "quietly"
    assert result.text == "Chapter one."


# --------------------------------------------------------------- chat events


async def test_call_publishes_start_tokens_and_end(endpoint, chat_events):
    transport = transport_for(
        delta_chunk(reasoning_content="mull"),
        delta_chunk(content="Answer."),
    )
    await call_llm(endpoint, MESSAGES, agent="beat_planner", stream=True, transport=transport)

    types = [event["type"] for event in chat_events]
    assert types[0] == "chat_start"
    assert types[-1] == "chat_end"
    assert "chat_token" in types

    start = chat_events[0]["data"]
    assert start["agent"] == "beat_planner"
    assert start["messages"] == MESSAGES
    assert start["model"] == "test-model"

    kinds = [e["data"]["kind"] for e in chat_events if e["type"] == "chat_token"]
    assert "thinking" in kinds and "response" in kinds

    end = chat_events[-1]["data"]
    assert end["ok"] is True
    assert end["text"] == "Answer."
    assert end["thinking"] == "mull"
    assert end["id"] == start["id"]


async def test_nonstreaming_call_still_gets_start_and_end(endpoint, chat_events):
    payload = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    await call_llm(endpoint, MESSAGES, agent="endpoint_test", transport=transport)

    types = [event["type"] for event in chat_events]
    assert types == ["chat_start", "chat_end"]
    assert chat_events[1]["data"]["text"] == "ok"


async def test_failed_call_ends_with_error(endpoint, chat_events):
    transport = httpx.MockTransport(lambda r: httpx.Response(400, content=b"nope"))
    with pytest.raises(LLMCallError):
        await call_llm(endpoint, MESSAGES, agent="drafter", transport=transport)

    end = chat_events[-1]["data"]
    assert chat_events[-1]["type"] == "chat_end"
    assert end["ok"] is False
    assert "400" in end["error"]


async def test_tokens_carry_a_monotonic_seq(endpoint, chat_events):
    """The seq lets a reloading page discard tokens its history partial covered."""
    transport = transport_for(
        delta_chunk(reasoning_content="mull"),
        delta_chunk(content="Ans"),
        delta_chunk(content="wer."),
    )
    await call_llm(endpoint, MESSAGES, stream=True, transport=transport)
    seqs = [e["data"]["seq"] for e in chat_events if e["type"] == "chat_token"]
    assert seqs == list(range(1, len(seqs) + 1))


async def test_in_flight_call_is_visible_with_its_partial_text(endpoint):
    """Mid-stream, `live_chat_calls` holds what has streamed so far; after the
    call ends it holds nothing — the transcript owns finished calls."""
    snapshots: list[list[dict]] = []

    async def on_token(token: str) -> None:
        snapshots.append(live_chat_calls())

    transport = transport_for(
        delta_chunk(reasoning_content="mull it over"),
        delta_chunk(content="Answer."),
    )
    await call_llm(
        endpoint, MESSAGES, agent="drafter", stream=True, on_token=on_token, transport=transport
    )

    assert snapshots, "on_token never fired"
    live = snapshots[-1]
    assert len(live) == 1
    assert live[0]["agent"] == "drafter"
    assert live[0]["messages"] == MESSAGES
    assert live[0]["thinking"] == "mull it over"
    assert live[0]["seq"] >= 1

    assert live_chat_calls() == []


async def test_a_failed_call_leaves_no_live_entry(endpoint):
    transport = httpx.MockTransport(lambda r: httpx.Response(400, content=b"nope"))
    with pytest.raises(LLMCallError):
        await call_llm(endpoint, MESSAGES, transport=transport)
    assert live_chat_calls() == []


async def test_chat_events_never_carry_the_api_key(endpoint, chat_events):
    payload = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    await call_llm(endpoint, MESSAGES, transport=transport)
    dumped = json.dumps([event["data"] for event in chat_events])
    assert "secret-key-do-not-log" not in dumped


# ----------------------------------------------------------------- transcript


async def test_transcript_records_start_and_end(endpoint, tmp_path, monkeypatch):
    path = tmp_path / "chat.jsonl"
    monkeypatch.setattr(chat_log, "_path", path)
    payload = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    await call_llm(endpoint, MESSAGES, agent="reviser", transport=transport)

    records = chat_log.replay(path, 10)
    assert [r["event"] for r in records] == ["chat_start", "chat_end"]
    assert records[0]["agent"] == "reviser"
    assert records[0]["messages"] == MESSAGES
    assert records[1]["text"] == "ok"


async def test_unconfigured_transcript_writes_nothing(endpoint, tmp_path):
    payload = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
    await call_llm(endpoint, MESSAGES, transport=transport)
    assert list(tmp_path.iterdir()) == []
    assert chat_log.transcript_path() is None


def test_replay_skips_torn_lines(tmp_path):
    path = tmp_path / "chat.jsonl"
    path.write_text(
        '{"event": "chat_start", "id": "a"}\n{"event": "chat_end", "id"\n', encoding="utf-8"
    )
    records = chat_log.replay(path, 10)
    assert len(records) == 1
    assert records[0]["id"] == "a"


def test_replay_returns_only_the_last_limit(tmp_path):
    path = tmp_path / "chat.jsonl"
    lines = [json.dumps({"event": "chat_start", "id": str(i)}) for i in range(10)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    records = chat_log.replay(path, 3)
    assert [r["id"] for r in records] == ["7", "8", "9"]


def test_replay_of_a_missing_file_is_empty(tmp_path):
    assert chat_log.replay(tmp_path / "nope.jsonl", 10) == []


# --------------------------------------------------------------------- routes


async def test_chat_page_renders_with_agent_filters(config_factory, web_app):
    app = await web_app(config_factory())
    response = await app.test_client().get("/chat")
    assert response.status_code == 200
    body = await response.get_data(as_text=True)
    assert "View Chat" in body
    assert "Drafter" in body
    assert "Beat Planner" in body
    assert "Critic" in body


async def test_history_replays_the_transcript(config_factory, web_app):
    config = config_factory()
    app = await web_app(config)

    path = chat_log.default_path(config.event_log_path)
    path.write_text(
        json.dumps({"event": "chat_start", "id": "c1", "agent": "drafter", "messages": []})
        + "\n"
        + json.dumps({"event": "chat_end", "id": "c1", "agent": "drafter", "ok": True, "text": "Prose."})
        + "\n",
        encoding="utf-8",
    )

    body = await (await app.test_client().get("/chat/history")).get_json()
    assert body["ok"] is True
    assert body["returned"] == 2
    assert body["records"][0]["agent"] == "drafter"
    assert body["records"][1]["text"] == "Prose."


async def test_history_is_empty_before_any_call(config_factory, web_app):
    app = await web_app(config_factory())
    body = await (await app.test_client().get("/chat/history")).get_json()
    assert body == {"ok": True, "returned": 0, "records": [], "partials": []}


async def test_history_carries_in_flight_partials(config_factory, web_app, monkeypatch):
    """A page loading mid-call gets the tokens that streamed before it arrived."""
    app = await web_app(config_factory())
    monkeypatch.setitem(
        _LIVE_CALLS,
        "live1",
        {
            "id": "live1",
            "agent": "drafter",
            "model": "test-model",
            "ts": 1.0,
            "stream": True,
            "messages": MESSAGES,
            "thinking": "half a thought",
            "text": "half a sen",
            "seq": 7,
        },
    )

    body = await (await app.test_client().get("/chat/history")).get_json()
    assert body["records"] == []
    assert len(body["partials"]) == 1
    partial = body["partials"][0]
    assert partial["id"] == "live1"
    assert partial["thinking"] == "half a thought"
    assert partial["text"] == "half a sen"
    assert partial["seq"] == 7


async def test_history_respects_the_limit(config_factory, web_app):
    config = config_factory()
    app = await web_app(config)
    path = chat_log.default_path(config.event_log_path)
    lines = [json.dumps({"event": "chat_start", "id": str(i)}) for i in range(20)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    body = await (await app.test_client().get("/chat/history?limit=5")).get_json()
    assert body["returned"] == 5
    assert body["records"][0]["id"] == "15"
