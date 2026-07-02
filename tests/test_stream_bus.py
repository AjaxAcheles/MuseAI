"""Module: M17 (Web UI & Real-time Observer Surface)
Synthetic, no-network tests for the SSE stream bus.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import uuid
from pathlib import Path

import pytest

from core.stream_bus import StreamBus, format_sse
from fsm.state import FSM_Pointer


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=10))


def test_subscriber_receives_published_event():
    async def scenario():
        bus = StreamBus()
        subscription = bus.subscribe()

        first = await anext(subscription)
        assert first["event"] == "status"
        assert first["data"]["status"] == "idle"

        await bus.publish("phase_change", {"phase": "planning"})
        event = await anext(subscription)
        assert event["event"] == "phase_change"
        assert event["data"] == {"phase": "planning"}
        assert event["timestamp"].endswith("Z")

        await subscription.aclose()

    _run(scenario())


def test_snapshot_updates_after_status_and_phase_events():
    async def scenario():
        bus = StreamBus()
        await bus.publish(
            "status", {"status": "running", "running": True, "message": "go"}
        )
        await bus.publish("phase_change", {"phase": "planning"})
        await bus.publish(
            "pointer_update", {"arc_id": "a1", "chapter_id": "c1", "scene_id": "", "beat_index": 0}
        )
        await bus.publish("planning_blocked", {"reason": "awaiting_macro_approval"})
        await bus.publish("error", {"message": "boom"})

        snapshot = bus.snapshot()
        assert snapshot["status"] == "running"
        assert snapshot["phase"] == "planning"
        assert snapshot["pointer"]["arc_id"] == "a1"
        assert snapshot["planning_block_reason"] == "awaiting_macro_approval"
        assert snapshot["last_error"] == "boom"
        assert snapshot["running"] is False  # error marks the run not-running

    _run(scenario())


def test_new_subscriber_gets_current_snapshot_immediately():
    async def scenario():
        bus = StreamBus()
        await bus.publish("phase_change", {"phase": "planning"})
        subscription = bus.subscribe()
        first = await anext(subscription)
        assert first["event"] == "status"
        assert first["data"]["phase"] == "planning"
        await subscription.aclose()

    _run(scenario())


def test_subscriber_cleanup_does_not_leak_after_generator_close():
    async def scenario():
        bus = StreamBus()
        first_sub = bus.subscribe()
        second_sub = bus.subscribe()
        await anext(first_sub)
        await anext(second_sub)
        assert bus.subscriber_count() == 2

        await first_sub.aclose()
        assert bus.subscriber_count() == 1

        # The surviving subscriber still receives events.
        await bus.publish("status", {"status": "running", "running": True})
        event = await anext(second_sub)
        assert event["data"]["status"] == "running"

        await second_sub.aclose()
        assert bus.subscriber_count() == 0

    _run(scenario())


def test_event_payloads_are_json_serializable():
    async def scenario():
        bus = StreamBus()
        subscription = bus.subscribe()
        await anext(subscription)

        pointer = FSM_Pointer(arc_id="a1", chapter_id="c1", scene_id="s1", beat_index=2)
        await bus.publish(
            "planning_node",
            {
                "pointer": pointer,
                "path": Path("data") / "fictionwriter.db",
                "uuid": uuid.uuid4(),
                "when": dt.datetime(2026, 7, 2, 12, 0, tzinfo=dt.timezone.utc),
                "error": ValueError("synthetic failure"),
                "nested": {"paths": [Path("a"), Path("b")]},
            },
        )
        event = await anext(subscription)
        serialized = json.dumps(event)  # must not raise
        assert "synthetic failure" in serialized
        assert event["data"]["pointer"]["arc_id"] == "a1"
        assert isinstance(event["data"]["path"], str)

        await subscription.aclose()

    _run(scenario())


def test_slow_subscriber_never_blocks_publishing():
    async def scenario():
        bus = StreamBus(max_queue_size=3)
        slow = bus.subscribe()
        await anext(slow)  # attach, then never read again

        for index in range(10):
            await bus.publish("status", {"status": "running", "message": str(index)})

        # Publisher never blocked; the slow queue holds only the newest events.
        newest = None
        for _ in range(3):
            newest = await asyncio.wait_for(anext(slow), timeout=1)
        assert newest["data"]["message"] == "9"

        await slow.aclose()

    _run(scenario())


def test_publish_rejects_unknown_event_type():
    async def scenario():
        bus = StreamBus()
        with pytest.raises(ValueError):
            await bus.publish("word_count", {})

    _run(scenario())


def test_sse_formatting():
    event = {
        "event": "phase_change",
        "data": {"phase": "planning"},
        "timestamp": "2026-07-02T12:00:00Z",
    }
    frame = format_sse(event)
    assert frame.startswith("event: phase_change\ndata: ")
    assert frame.endswith("\n\n")
    payload = json.loads(frame.split("data: ", 1)[1])
    assert payload["data"]["phase"] == "planning"
