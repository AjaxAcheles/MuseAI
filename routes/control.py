"""Module: M17 (Web UI & Real-time Observer Surface)
Expose the SSE event stream, run status, and start/pause/resume/stop controls.

Route handlers stay thin: resources come from ``core.runtime.get_resources()``
(never re-initialized here), long work stays in the generation manager's
background task, and every control response uses the consistent
``{"ok", "status", "message"}`` JSON shape. No handler returns a raw exception
to the browser.
"""

from __future__ import annotations

import logging
from typing import Any

from quart import Blueprint, Response, jsonify, request

import core.runtime as runtime
from core.runtime import get_resources
from core.stream_bus import format_sse
from memory.event_log import tail_events

logger = logging.getLogger(__name__)

# /api/logs/recent limit envelope: request-size hygiene, not a tunable threshold.
_LOGS_DEFAULT_LIMIT = 200
_LOGS_MAX_LIMIT = 1000


def create_blueprint() -> Blueprint:
    """Create the control blueprint (/events, /api/status, /control/*)."""
    bp = Blueprint("control", __name__)

    def _manager() -> Any | None:
        return get_resources().generation_manager

    def _error_response(message: str, http_status: int) -> tuple[Response, int]:
        return jsonify({"ok": False, "status": "error", "message": message}), http_status

    async def _control(action: str) -> Any:
        manager = _manager()
        if manager is None:
            return _error_response("Generation manager is not initialized.", 503)
        try:
            if action == "start":
                payload = await request.get_json(silent=True)
                result = await manager.start(payload or {})
            elif action == "pause":
                result = await manager.pause()
            elif action == "resume":
                result = await manager.resume()
            else:
                result = await manager.stop()
        except Exception:  # noqa: BLE001 - never leak a traceback to the browser
            logger.exception("control action %r failed", action)
            return _error_response(f"Control action {action!r} failed; see server logs.", 500)
        return jsonify(result)

    @bp.get("/api/status")
    async def api_status() -> Any:
        resources = get_resources()
        manager = resources.generation_manager
        return jsonify(
            {
                "ok": True,
                "manager": manager.status()
                if manager is not None
                else {"status": "idle", "running": False},
                "snapshot": resources.event_bus.snapshot(),
                "stores": {
                    name: {"kind": handle.kind, "note": handle.note}
                    for name, handle in resources.stores.items()
                },
            }
        )

    @bp.get("/healthz")
    async def healthz() -> Any:
        if runtime._ACTIVE_RESOURCES is None:  # noqa: SLF001 - health probe of init state
            return (
                jsonify(
                    {
                        "ok": False,
                        "runtime_initialized": False,
                        "message": "Runtime resources are not initialized.",
                    }
                ),
                503,
            )
        resources = get_resources()
        manager = resources.generation_manager
        return jsonify(
            {
                "ok": True,
                "runtime_initialized": True,
                "run": manager.status()
                if manager is not None
                else {"status": "idle", "running": False},
                "stores": {
                    name: handle.kind for name, handle in resources.stores.items()
                },
            }
        )

    @bp.get("/api/logs/recent")
    async def api_logs_recent() -> Any:
        resources = get_resources()
        if resources.logs_dir is None:
            return jsonify({"ok": True, "entries": [], "message": "App log is not wired."})
        try:
            limit = int(request.args.get("limit", _LOGS_DEFAULT_LIMIT))
        except ValueError:
            limit = _LOGS_DEFAULT_LIMIT
        limit = max(1, min(limit, _LOGS_MAX_LIMIT))
        log_path = resources.logs_dir / "app.jsonl"
        try:
            entries = tail_events(log_path, limit)
        except ValueError:
            # A malformed line must not take the debug surface down with it.
            logger.exception("app log tail failed")
            return jsonify(
                {"ok": False, "entries": [], "message": "App log contains a malformed line."}
            )
        return jsonify({"ok": True, "entries": entries, "limit": limit})

    @bp.get("/events")
    async def events() -> Response:
        resources = get_resources()
        bus = resources.event_bus

        async def stream():
            # subscribe() delivers the latest snapshot first, so a reconnect
            # renders current state without replaying history.
            resources.log("sse_subscribe", subscribers=bus.subscriber_count() + 1)
            try:
                async for event in bus.subscribe():
                    yield format_sse(event)
            finally:
                resources.log("sse_disconnect", subscribers=bus.subscriber_count())

        response = Response(stream(), mimetype="text/event-stream")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Accel-Buffering"] = "no"
        # Long-lived stream: disable Quart's response body timeout.
        response.timeout = None
        return response

    @bp.post("/control/start")
    async def control_start() -> Any:
        return await _control("start")

    @bp.post("/control/pause")
    async def control_pause() -> Any:
        return await _control("pause")

    @bp.post("/control/resume")
    async def control_resume() -> Any:
        return await _control("resume")

    @bp.post("/control/stop")
    async def control_stop() -> Any:
        return await _control("stop")

    return bp
