"""Module: M17 (Web UI & Real-time Observer Surface)
Fan out server-sent events between the generation runtime and web routes.

An in-memory async pub/sub bus. Publishers (the generation manager and the
planning runner) push typed events; each SSE client subscribes and receives its
own bounded queue so one slow browser tab can never block publishing or other
subscribers. The bus also maintains a "latest snapshot" of run state so a page
load / reconnect can render current status immediately without replaying
history (there is deliberately no global event history list).

Event object shape (every payload is reduced to JSON-safe primitives first;
``run_id`` identifies the generation run the event belongs to, None outside a
run):

    {"event": "phase_change", "data": {...},
     "timestamp": "2026-07-02T12:00:00Z", "run_id": "a1b2c3d4e5f6"}
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as _dt
import json
import uuid
from collections.abc import AsyncIterator, Mapping
from pathlib import Path, PurePath
from typing import Any

# Event types the vertical slice publishes. Publishing an unknown type is a
# programming error and fails loudly rather than silently streaming junk.
EVENT_TYPES = frozenset(
    {
        "status",
        "phase_change",
        "pointer_update",
        "planning_snapshot",
        "planning_node",
        "planning_blocked",
        "approval_state",
        "done",
        "error",
    }
)

# Per-subscriber queue bound. A structural backpressure limit for the SSE
# plumbing (oldest events are dropped for a stalled client), not a narrative
# tunable; if it ever needs calibration it should graduate to a config key.
DEFAULT_MAX_SUBSCRIBER_QUEUE = 256


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with a trailing Z."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def jsonable(value: Any) -> Any:
    """Reduce ``value`` to JSON-serializable primitives.

    Handles dataclasses, Pydantic models, Mappings, sequences, Paths, UUIDs,
    datetimes, and exceptions. Anything else unknown degrades to ``str(value)``
    so a publish can never raise on a payload repr.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):  # Pydantic v2 models (FSM_Pointer, PlannerAction, ...)
        try:
            return jsonable(model_dump(mode="json"))
        except Exception:  # noqa: BLE001 - fall through to the generic reductions
            pass
    if isinstance(value, (Path, PurePath)):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    return str(value)


def format_sse(event: Mapping[str, Any]) -> str:
    """Format one bus event as an SSE frame: ``event: <type>``, ``data: <json>``."""
    payload = json.dumps(
        {
            "event": event.get("event"),
            "data": event.get("data"),
            "timestamp": event.get("timestamp"),
            "run_id": event.get("run_id"),
        },
        ensure_ascii=False,
    )
    return f"event: {event.get('event')}\ndata: {payload}\n\n"


class StreamBus:
    """SSE fan-out bus with per-subscriber bounded queues and a latest snapshot."""

    def __init__(self, max_queue_size: int = DEFAULT_MAX_SUBSCRIBER_QUEUE) -> None:
        self._max_queue_size = max_queue_size
        self._run_id: str | None = None
        self._subscribers: set[asyncio.Queue] = set()
        self._snapshot: dict[str, Any] = {
            "running": False,
            "status": "idle",
            "phase": "idle",
            "message": "",
            "pointer": None,
            "planning_execution_mode": None,
            "approval_mode": None,
            "planning_block_reason": None,
            "awaiting_planning_approval": False,
            "last_error": None,
            "planning_snapshot": None,
            "planning_node": None,
        }

    # -- snapshot ----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """A JSON-safe copy of the latest run-state snapshot."""
        return json.loads(json.dumps(self._snapshot))

    def seed_snapshot(self, **fields: Any) -> None:
        """Set initial snapshot fields (e.g. config modes) before any event flows."""
        for key, value in fields.items():
            if key not in self._snapshot:
                raise KeyError(f"unknown snapshot field {key!r}")
            self._snapshot[key] = jsonable(value)

    def subscriber_count(self) -> int:
        """Number of currently attached subscriber queues (for tests/telemetry)."""
        return len(self._subscribers)

    def set_run_id(self, run_id: str | None) -> None:
        """Stamp subsequent event envelopes with the active generation run id."""
        self._run_id = run_id

    def _apply_to_snapshot(self, event_type: str, data: Any) -> None:
        mapping = data if isinstance(data, Mapping) else {}
        if event_type == "status":
            for key in ("running", "status", "message"):
                if key in mapping:
                    self._snapshot[key] = mapping[key]
        elif event_type == "phase_change":
            if "phase" in mapping:
                self._snapshot["phase"] = mapping["phase"]
        elif event_type == "pointer_update":
            self._snapshot["pointer"] = data
        elif event_type == "planning_snapshot":
            self._snapshot["planning_snapshot"] = data
        elif event_type == "planning_node":
            self._snapshot["planning_node"] = data
        elif event_type == "planning_blocked":
            self._snapshot["planning_block_reason"] = mapping.get("reason")
            self._snapshot["awaiting_planning_approval"] = bool(
                mapping.get("awaiting_approval", False)
            )
        elif event_type == "approval_state":
            awaiting = bool(mapping.get("awaiting_approval", False))
            self._snapshot["awaiting_planning_approval"] = awaiting
            if not awaiting and self._snapshot.get("planning_block_reason") == (
                "awaiting_macro_approval"
            ):
                # Approval resolved: the gate's block reason is no longer true.
                self._snapshot["planning_block_reason"] = None
        elif event_type == "done":
            self._snapshot["running"] = False
            if "message" in mapping:
                self._snapshot["message"] = mapping["message"]
        elif event_type == "error":
            self._snapshot["running"] = False
            self._snapshot["last_error"] = mapping.get("message") or jsonable(data)
        # Mode fields ride along on any event that carries them (run start).
        for key in ("planning_execution_mode", "approval_mode"):
            if key in mapping:
                self._snapshot[key] = mapping[key]

    # -- pub/sub -----------------------------------------------------------

    async def publish(self, event_type: str, data: dict[str, Any] | Any) -> None:
        """Publish one event to all subscribers without ever blocking.

        A subscriber whose bounded queue is full loses its oldest queued event
        (never the publisher's time). Payloads are reduced to JSON-safe
        primitives before enqueueing so every subscriber sees serializable data.
        """
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown stream event type {event_type!r}")
        event = {
            "event": event_type,
            "data": jsonable(data),
            "timestamp": _utc_now_iso(),
            "run_id": self._run_id,
        }
        self._apply_to_snapshot(event_type, event["data"])
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop this slow subscriber's oldest event to make room; the
                # snapshot event it receives on reconnect restores currency.
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass
        # Cooperative yield so long publish bursts let SSE consumers drain.
        await asyncio.sleep(0)

    async def subscribe(self) -> AsyncIterator[dict[str, Any]]:
        """Yield events for one client; the current snapshot is delivered first.

        The subscriber queue is registered for the generator's lifetime and
        removed when the client disconnects (generator close/cancellation), so
        disconnected clients never leak.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue_size)
        queue.put_nowait(
            {
                "event": "status",
                "data": self.snapshot(),
                "timestamp": _utc_now_iso(),
                "run_id": self._run_id,
            }
        )
        self._subscribers.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)
