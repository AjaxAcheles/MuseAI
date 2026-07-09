"""In-process async pub/sub for Server-Sent Events.

The FSM nodes publish events here; the web layer subscribes and relays them to
browsers over SSE. Deliberately free of any Quart/web import so nodes can
publish without pulling in a web dependency.

``last_snapshot`` accumulates the most recent value of each event type so a
client that reconnects can hydrate current state immediately instead of waiting
for the next live event.
"""

from __future__ import annotations

import asyncio
from typing import Any


# Per-subscriber backlog cap. A subscriber that stops draining (a stalled SSE
# connection) must not grow its queue for the whole run; beyond the cap the
# oldest events are dropped — a reconnecting client re-hydrates from
# ``last_snapshot`` anyway.
_MAX_QUEUE_EVENTS = 1024


class StreamBus:
    """A simple fan-out bus: one publisher, many subscriber queues."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self.last_snapshot: dict[str, dict[str, Any]] = {}

    async def publish(self, event_type: str, data: dict[str, Any]) -> None:
        """Publish an event to all subscribers and update the snapshot."""
        event = {"type": event_type, "data": data}
        self.last_snapshot[event_type] = data
        for queue in list(self._subscribers):
            while True:
                try:
                    queue.put_nowait(event)
                    break
                except asyncio.QueueFull:
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Register a new subscriber and return its queue."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_MAX_QUEUE_EVENTS)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove a subscriber's queue; safe to call more than once."""
        self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


# Module-level singleton shared across the process.
bus = StreamBus()
