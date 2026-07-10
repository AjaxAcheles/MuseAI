"""Dashboard, generation, status, committed-story, outline, and SSE routes."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, AsyncIterator

from quart import Blueprint, Response, jsonify, render_template

from museai.core.stream_bus import bus
from museai.fsm.export import committed_word_count
from museai.fsm.manager import GenerationManagerError
from museai.memory.db import (
    connect_db,
    get_arcs,
    get_beats_for_chapter,
    get_chapters_for_arc,
    get_committed_beats,
    get_project,
)
from museai.web.app import get_config, get_manager

bp = Blueprint("dashboard", __name__)

# Manager states from which a fresh run may be started. Mirrors the guard in `generate`.
_STARTABLE = {"idle", "done", "stopped", "error"}


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


def _active_project_id() -> str | None:
    """The project the manager is running, else whatever is seeded on disk."""
    manager = get_manager()
    if manager.state is not None and manager.state.get("project_id"):
        return str(manager.state["project_id"])
    return _seeded_project_id()


@bp.get("/")
@bp.get("/dashboard")
async def dashboard():
    # The Load Seed drawer prefills the same starter seed the Seed & Plan page offers.
    from museai.web.routes.seed import example_seed_text

    return await render_template("dashboard.html", example_seed=example_seed_text())


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

    return Response(
        events(),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@bp.post("/generate")
async def generate():
    project_id = _seeded_project_id()
    if project_id is None:
        return jsonify({"ok": False, "error": "No project is seeded. Load a seed before starting generation."}), 400

    manager = get_manager()
    if manager.status not in _STARTABLE:
        return jsonify({"ok": True, "status": manager.status, "message": "Generation is already active."})
    try:
        await manager.start(project_id)
    except GenerationManagerError as exc:
        return jsonify({"ok": False, "error": str(exc), "status": manager.status}), 409
    return jsonify({"ok": True, "status": manager.status, "project_id": project_id})


def _word_target(project_id: str | None) -> int | None:
    """Return the seeded project's word target, or ``None`` before seed load."""
    if project_id is None:
        return None
    cfg = get_config()
    conn = connect_db(cfg.db_path)
    try:
        project = get_project(conn, project_id)
        if project is not None and project["word_count_target"]:
            return int(project["word_count_target"])
    finally:
        conn.close()
    return cfg.generation.word_count_target


def _project_summary(conn: sqlite3.Connection, project_id: str) -> dict[str, Any] | None:
    row = get_project(conn, project_id)
    if row is None:
        return None
    return {
        "id": row["id"],
        "genre": row["genre"],
        "premise": row["premise"],
        "word_count_target": row["word_count_target"],
    }


def _seed_counts(conn: sqlite3.Connection, project_id: str) -> dict[str, int]:
    def count(table: str) -> int:
        row = conn.execute(
            f"SELECT COUNT(*) AS n FROM {table} WHERE project_id=?", (project_id,)
        ).fetchone()
        return int(row["n"] or 0)

    return {"arcs": count("Arcs"), "threads": count("Threads"), "characters": count("Characters")}


def _last_commit(conn: sqlite3.Connection, project_id: str) -> dict[str, Any] | None:
    """The most recent committed beat in narrative order, or None before the first commit."""
    row = conn.execute(
        """
        SELECT Beats.id AS beat_id, Beats.word_count AS word_count,
               Chapters.id AS chapter_id, Arcs.id AS arc_id
        FROM Beats
        JOIN Chapters ON Beats.chapter_id = Chapters.id
        JOIN Arcs ON Chapters.arc_id = Arcs.id
        WHERE Arcs.project_id = ? AND Beats.status='completed' AND Beats.prose IS NOT NULL
        ORDER BY Arcs.ordering DESC, Chapters.ordering DESC, Beats.ordering DESC
        LIMIT 1
        """,
        (project_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "beat_id": row["beat_id"],
        "chapter_id": row["chapter_id"],
        "arc_id": row["arc_id"],
        "word_count": int(row["word_count"] or 0),
    }


@bp.get("/status")
async def status():
    """Authoritative run state. The browser trusts this over anything it saw on the stream."""
    cfg = get_config()
    manager = get_manager()
    project_id = _active_project_id()

    project: dict[str, Any] | None = None
    counts = {"arcs": 0, "threads": 0, "characters": 0}
    last_commit: dict[str, Any] | None = None
    if project_id is not None:
        conn = connect_db(cfg.db_path)
        try:
            project = _project_summary(conn, project_id)
            if project is not None:
                counts = _seed_counts(conn, project_id)
                last_commit = _last_commit(conn, project_id)
        finally:
            conn.close()

    seed_loaded = project is not None
    return jsonify(
        {
            "ok": True,
            "status": manager.status,
            "pointer": _pointer_dict(),
            "project_word_total": committed_word_count(cfg),
            "word_target": _word_target(project_id),
            "seed_loaded": seed_loaded,
            "can_generate": seed_loaded and manager.status in _STARTABLE,
            "project": project,
            "counts": counts,
            "last_commit": last_commit,
            "endpoint": {"base_url": cfg.endpoint.base_url, "model_name": cfg.endpoint.model_name},
        }
    )


@bp.get("/committed")
async def committed():
    """Committed prose only — never a draft, a revision, or a best-seen review candidate."""
    cfg = get_config()
    project_id = _active_project_id()
    if project_id is None:
        return jsonify({"ok": True, "project_id": None, "project_word_total": 0, "beats": []})

    conn = connect_db(cfg.db_path)
    try:
        rows = get_committed_beats(conn, project_id)
    finally:
        conn.close()

    beats = [
        {
            "beat_id": row["beat_id"],
            "ordering": row["beat_ordering"],
            "arc_id": row["arc_id"],
            "arc_ordering": row["arc_ordering"],
            "chapter_id": row["chapter_id"],
            "chapter_ordering": row["chapter_ordering"],
            "chapter_description": row["chapter_description"],
            "prose": row["prose"],
            "word_count": int(row["word_count"] or 0),
        }
        for row in rows
    ]
    return jsonify(
        {
            "ok": True,
            "project_id": project_id,
            "project_word_total": committed_word_count(cfg),
            "beats": beats,
        }
    )


@bp.get("/outline")
async def outline():
    """Narrative structure for the progress rail: arcs, and whatever the planners have built.

    Beat prose is deliberately excluded — the rail shows position, not text. Before the
    planners run, a seeded project has arcs and no chapters, and this returns exactly that.
    """
    cfg = get_config()
    project_id = _active_project_id()
    if project_id is None:
        return jsonify({"ok": True, "project_id": None, "pointer": None, "arcs": []})

    conn = connect_db(cfg.db_path)
    try:
        arcs = []
        for arc in get_arcs(conn, project_id):
            chapters = []
            for chapter in get_chapters_for_arc(conn, arc["id"]):
                beats = [
                    {
                        "id": beat["id"],
                        "ordering": beat["ordering"],
                        "status": beat["status"],
                        "word_count": int(beat["word_count"] or 0),
                    }
                    for beat in get_beats_for_chapter(conn, chapter["id"])
                ]
                chapters.append(
                    {
                        "id": chapter["id"],
                        "ordering": chapter["ordering"],
                        "description": chapter["description"],
                        "status": chapter["status"],
                        "beats": beats,
                    }
                )
            arcs.append(
                {
                    "id": arc["id"],
                    "ordering": arc["ordering"],
                    "description": arc["description"],
                    "status": arc["status"],
                    "chapters": chapters,
                }
            )
    finally:
        conn.close()

    return jsonify({"ok": True, "project_id": project_id, "pointer": _pointer_dict(), "arcs": arcs})
