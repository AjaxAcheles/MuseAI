"""Module: M17 (Web UI & Real-time Observer Surface)

Serve the dashboard page and the per-run server-sent-event stream.

``GET /`` renders the minimal control/observer page; ``GET /events/<run_id>``
opens an SSE stream that drains that run's :class:`~core.stream_bus.StreamBus`
subscription and forwards each event to the browser as ``text/event-stream``. The
subscription is always released when the client disconnects.
"""

from __future__ import annotations

import json

from quart import Blueprint, Response, current_app, render_template

from core.stream_bus import StreamBus


def create_blueprint() -> Blueprint:
    """Create the dashboard blueprint (index page + SSE event surface)."""
    bp = Blueprint("dashboard", __name__)

    @bp.get("/")
    async def index() -> str:
        return await render_template("dashboard.html")

    @bp.get("/events/<run_id>")
    async def events(run_id: str) -> Response:
        bus: StreamBus = current_app.generation_manager.stream_bus
        queue = bus.subscribe(run_id)

        async def event_stream():
            try:
                while True:
                    item = await queue.get()
                    if StreamBus.is_done(item):
                        yield "event: done\ndata: {}\n\n"
                        break
                    yield f"data: {json.dumps(item)}\n\n"
            finally:
                bus.unsubscribe(run_id, queue)

        return Response(
            event_stream(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return bp
