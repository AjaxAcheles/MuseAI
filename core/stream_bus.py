"""Module: M17 (Web UI & Real-time Observer Surface)
Fan out server-sent events between the generation driver and web routes.

An in-process async pub/sub keyed by ``run_id``. FSM nodes and the generation
driver ``publish`` small JSON-able event dicts (streamed prose tokens, node
lifecycle, run status); each connected dashboard SSE response ``subscribe``s and
drains its own queue. Publish is non-blocking (``put_nowait``) so a slow or
disconnected browser can never stall generation — a subscriber that overflows is
dropped rather than back-pressuring the driver.
"""

from __future__ import annotations

import asyncio
from typing import Any

# Bounded per-subscriber buffer. A browser that falls this far behind is dropped
# rather than allowed to back-pressure the generation loop.
_SUBSCRIBER_MAX_QUEUE = 2048

# Sentinel pushed to a subscriber's queue when its run finishes, so the SSE
# generator can close cleanly instead of hanging on the next ``get()``.
_DONE = object()


class StreamBus:
    """In-process fan-out of run events to any number of SSE subscribers."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue]] = {}

    def subscribe(self, run_id: str) -> asyncio.Queue:
        """Register a new subscriber queue for ``run_id`` and return it."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=_SUBSCRIBER_MAX_QUEUE)
        self._subscribers.setdefault(run_id, set()).add(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        """Drop a subscriber queue (e.g. when its browser disconnects)."""
        subscribers = self._subscribers.get(run_id)
        if subscribers is not None:
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(run_id, None)

    def publish(self, run_id: str, event: dict[str, Any]) -> None:
        """Fan ``event`` out to every current subscriber of ``run_id``.

        Non-blocking: a subscriber whose bounded queue is full is silently
        skipped for this event (it has fallen too far behind to keep up), so a
        stalled browser never blocks the driver.
        """
        for queue in tuple(self._subscribers.get(run_id, ())):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                continue

    def close(self, run_id: str) -> None:
        """Signal end-of-stream to every subscriber of ``run_id``.

        Each subscriber receives the ``_DONE`` sentinel so its SSE generator can
        terminate; the run is then forgotten.
        """
        for queue in tuple(self._subscribers.get(run_id, ())):
            try:
                queue.put_nowait(_DONE)
            except asyncio.QueueFull:
                # Full queue: force the sentinel by dropping the oldest item.
                try:
                    queue.get_nowait()
                    queue.put_nowait(_DONE)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass
        self._subscribers.pop(run_id, None)

    @staticmethod
    def is_done(item: Any) -> bool:
        """True if ``item`` is the end-of-stream sentinel from :meth:`close`."""
        return item is _DONE
