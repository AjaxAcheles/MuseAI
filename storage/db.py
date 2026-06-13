"""Local persistence for MuseAI (aiosqlite).

Two tables:
  projects  — one row per creative brief + the pipeline outputs as they complete.
  stories   — the finished, assembled story for a project (what export reads).

Stage outputs are stored as JSON text blobs so a page refresh can resume the
reader view, and so the pipeline can persist incrementally.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional

import aiosqlite

from config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id           TEXT PRIMARY KEY,
    created_at   REAL NOT NULL,
    status       TEXT NOT NULL DEFAULT 'draft',
    brief_json   TEXT,
    premise_json TEXT,
    bible_json   TEXT,
    outline_json TEXT
);
CREATE TABLE IF NOT EXISTS stories (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES projects(id),
    title       TEXT,
    content_md  TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_stories_project ON stories(project_id);
"""


async def init_db() -> None:
    async with aiosqlite.connect(settings.db_path) as db:
        await db.executescript(_SCHEMA)
        await db.commit()


def _new_id() -> str:
    return uuid.uuid4().hex


async def create_project(brief: dict[str, Any]) -> str:
    pid = _new_id()
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            "INSERT INTO projects (id, created_at, status, brief_json) VALUES (?, ?, ?, ?)",
            (pid, time.time(), "draft", json.dumps(brief)),
        )
        await db.commit()
    return pid


async def get_project(pid: str) -> Optional[dict[str, Any]]:
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM projects WHERE id = ?", (pid,)) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        project = {
            "id": row["id"],
            "created_at": row["created_at"],
            "status": row["status"],
            "brief": _loads(row["brief_json"]),
            "premise": _loads(row["premise_json"]),
            "bible": _loads(row["bible_json"]),
            "outline": _loads(row["outline_json"]),
        }
        async with db.execute(
            "SELECT * FROM stories WHERE project_id = ? ORDER BY created_at DESC LIMIT 1",
            (pid,),
        ) as cur:
            srow = await cur.fetchone()
        project["story"] = (
            {
                "id": srow["id"],
                "title": srow["title"],
                "content_md": srow["content_md"],
                "created_at": srow["created_at"],
            }
            if srow
            else None
        )
        return project


async def update_brief(pid: str, brief: dict[str, Any]) -> None:
    await _set_column(pid, "brief_json", json.dumps(brief))


async def save_stage(pid: str, stage: str, payload: Any) -> None:
    """Persist one structured stage output (premise|bible|outline)."""
    column = {"premise": "premise_json", "bible": "bible_json", "outline": "outline_json"}[stage]
    await _set_column(pid, column, json.dumps(payload))


async def set_status(pid: str, status: str) -> None:
    await _set_column(pid, "status", status)


async def save_story(pid: str, title: str, content_md: str) -> str:
    sid = _new_id()
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(
            "INSERT INTO stories (id, project_id, title, content_md, created_at) VALUES (?, ?, ?, ?, ?)",
            (sid, pid, title, content_md, time.time()),
        )
        await db.commit()
    return sid


async def _set_column(pid: str, column: str, value: str) -> None:
    async with aiosqlite.connect(settings.db_path) as db:
        await db.execute(f"UPDATE projects SET {column} = ? WHERE id = ?", (value, pid))
        await db.commit()


def _loads(text: Optional[str]) -> Any:
    return json.loads(text) if text else None
