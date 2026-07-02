"""Module: M17 (Web UI & Real-time Observer Surface)
Synthetic tests for the SSE fan-out bus (`core/stream_bus.py`): multi-subscriber
fan-out, per-run isolation, the end-of-stream sentinel, and unsubscribe. No network
or browser — the bus is exercised directly against asyncio queues.
"""

import asyncio

from core.stream_bus import StreamBus


def test_publish_fans_out_to_all_subscribers_of_a_run():
    async def scenario():
        bus = StreamBus()
        q1 = bus.subscribe("run1")
        q2 = bus.subscribe("run1")
        bus.publish("run1", {"type": "token", "text": "hi"})
        assert (await q1.get())["text"] == "hi"
        assert (await q2.get())["text"] == "hi"

    asyncio.run(scenario())


def test_runs_are_isolated():
    async def scenario():
        bus = StreamBus()
        q_a = bus.subscribe("A")
        q_b = bus.subscribe("B")
        bus.publish("A", {"n": 1})
        assert (await q_a.get())["n"] == 1
        assert q_b.empty()  # B never sees A's events

    asyncio.run(scenario())


def test_close_delivers_done_sentinel_and_forgets_run():
    async def scenario():
        bus = StreamBus()
        q = bus.subscribe("run1")
        bus.close("run1")
        assert StreamBus.is_done(await q.get())
        # After close the run is forgotten: a later publish reaches no one and is a no-op.
        bus.publish("run1", {"late": True})
        assert q.empty()

    asyncio.run(scenario())


def test_unsubscribe_stops_delivery():
    async def scenario():
        bus = StreamBus()
        q = bus.subscribe("run1")
        bus.unsubscribe("run1", q)
        bus.publish("run1", {"x": 1})
        assert q.empty()

    asyncio.run(scenario())
