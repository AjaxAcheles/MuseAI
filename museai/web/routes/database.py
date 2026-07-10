"""Read-only inspection of the project database and the append-only event log.

This blueprint never writes. It exists so a curious operator can see exactly what the engine
persisted, without a SQLite client. Only the eight tables that actually exist are exposed; an
unknown record type is rejected rather than guessed at.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from quart import Blueprint, jsonify, render_template, request

from museai.core.events import replay_events
from museai.memory.db import connect_db
from museai.web.app import get_config

bp = Blueprint("database", __name__)

# The real schema, in the order a reader would want to walk it. Doubles as the SQL
# injection guard: a table name reaches a query only by matching this tuple exactly.
RECORD_TYPES: tuple[str, ...] = (
    "Projects",
    "Arcs",
    "Chapters",
    "Beats",
    "Threads",
    "Characters",
    "CharacterEmotions",
    "CommitIntent",
)

# Ordering that reflects narrative or insertion order rather than SQLite's rowid accident.
_ORDER_BY: dict[str, str] = {
    "Projects": "id ASC",
    "Arcs": "ordering ASC",
    "Chapters": "arc_id ASC, ordering ASC",
    "Beats": "chapter_id ASC, ordering ASC",
    "Threads": "priority_score DESC, id ASC",
    "Characters": "name ASC",
    "CharacterEmotions": "character_id ASC",
    "CommitIntent": "id DESC",
}

_DEFAULT_LIMIT = 200
_MAX_LIMIT = 1000


def _clamp_limit(raw: str | None, default: int = _DEFAULT_LIMIT) -> int:
    try:
        value = int(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default
    return max(1, min(value, _MAX_LIMIT))


def _matches(row: dict[str, Any], needle: str) -> bool:
    """Case-insensitive substring match across every column of a row."""
    return any(needle in str(value).lower() for value in row.values() if value is not None)


@bp.get("/database")
async def database():
    return await render_template("database.html", record_types=RECORD_TYPES)


@bp.get("/database/records")
async def records():
    record_type = request.args.get("type", "")
    if record_type not in RECORD_TYPES:
        known = ", ".join(RECORD_TYPES)
        return jsonify({"ok": False, "error": f"Unknown record type {record_type!r}. Known types: {known}."}), 400

    query = request.args.get("q", "").strip().lower()
    limit = _clamp_limit(request.args.get("limit"))

    conn = connect_db(get_config().db_path)
    try:
        # `record_type` is constrained to RECORD_TYPES above, so this interpolation is safe.
        rows = conn.execute(f"SELECT * FROM {record_type} ORDER BY {_ORDER_BY[record_type]}").fetchall()
    except sqlite3.OperationalError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    finally:
        conn.close()

    items = [dict(row) for row in rows]
    if query:
        items = [item for item in items if _matches(item, query)]
    total = len(items)
    return jsonify({"ok": True, "type": record_type, "total": total, "returned": min(total, limit), "records": items[:limit]})


@bp.get("/database/event-log")
async def event_log():
    """Tail of the append-only JSONL log. A missing log is an empty log, not an error."""
    limit = _clamp_limit(request.args.get("limit"), default=50)
    cfg = get_config()
    events = list(replay_events(cfg.event_log_path))
    tail = events[-limit:]
    return jsonify({"ok": True, "total": len(events), "returned": len(tail), "events": tail})
