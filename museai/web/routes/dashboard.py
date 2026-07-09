"""Dashboard, generation, status, and SSE routes."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from quart import Blueprint, Response, jsonify, render_template

from museai.core.stream_bus import bus
from museai.fsm.export import committed_word_count
from museai.fsm.manager import GenerationManagerError
from museai.memory.db import connect_db, get_project
from museai.web.app import get_config, get_manager

bp = Blueprint("dashboard", __name__)


def _sse(event: dict[str, Any]) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"


def _seeded_project_id() -> str | None:
    cfg = get_config()
    conn = connect_db(cfg.db_path)
    try:
        configured = get_project(conn, cfg.project_id)
        if configured is not None:
            return cfg.project_id
        row = conn.execute("SELECT id FROM Projects ORDER BY id ASC LIMIT 1").fetchone()
        return str(row["id"]) if row is not None else None
    finally:
        conn.close()


def _pointer_dict() -> dict[str, Any] | None:
    manager = get_manager()
    if manager.state is None:
        return None
    pointer = manager.state.get("fsm_pointer")
    if pointer is None:
        return None
    return pointer.model_dump() if hasattr(pointer, "model_dump") else dict(pointer)


@bp.get("/")
@bp.get("/dashboard")
async def dashboard():
    return await render_template("dashboard.html")


@bp.get("/stream")
async def stream():
    queue = bus.subscribe()

    async def events() -> AsyncIterator[str]:
        try:
            yield _sse({"type": "hydration", "data": dict(bus.last_snapshot)})
            while True:
                yield _sse(await queue.get())
        finally:
            bus.unsubscribe(queue)

    return Response(events(), content_type="text/event-stream")


@bp.post("/generate")
async def generate():
    project_id = _seeded_project_id()
    if project_id is None:
        return jsonify({"ok": False, "error": "No project is seeded. Load a seed before starting generation."}), 400

    manager = get_manager()
    if manager.status not in {"idle", "done", "stopped"}:
        return jsonify({"ok": True, "status": manager.status, "message": "Generation is already active."})
    try:
        await manager.start(project_id)
    except GenerationManagerError as exc:
        return jsonify({"ok": False, "error": str(exc), "status": manager.status}), 409
    return jsonify({"ok": True, "status": manager.status, "project_id": project_id})


@bp.get("/status")
async def status():
    cfg = get_config()
    manager = get_manager()
    return jsonify(
        {
            "ok": True,
            "status": manager.status,
            "pointer": _pointer_dict(),
            "project_word_total": committed_word_count(cfg),
        }
    )
