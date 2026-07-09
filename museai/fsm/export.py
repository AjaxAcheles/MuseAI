"""Manuscript export for MuseAI v1.

The export surface is deliberately singular: one Markdown manuscript assembled
from committed beat prose in narrative order. It does not include uncommitted
draft text, best-seen review candidates, continuity notes, timelines, bibles, or
any other non-v1 artifact.
"""

from __future__ import annotations

import re
from pathlib import Path

from museai.core.config import AppConfig
from museai.core.logging_setup import get_fsm_logger
from museai.memory.db import connect_db

_WORD = re.compile(r"\S+")


def committed_word_count(config: AppConfig) -> int:
    """Total words from completed beats only."""
    conn = connect_db(config.db_path)
    try:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(Beats.word_count), 0) AS total
            FROM Beats
            JOIN Chapters ON Beats.chapter_id = Chapters.id
            JOIN Arcs ON Chapters.arc_id = Arcs.id
            WHERE Arcs.project_id=? AND Beats.status='completed'
            """,
            (config.project_id,),
        ).fetchone()
        return int(row["total"] or 0)
    finally:
        conn.close()


def _chapter_slug(description: str, fallback: str) -> str:
    title = " ".join(description.split()).strip()
    return title or fallback


def export_manuscript(config: AppConfig) -> Path:
    """Write the committed manuscript to ``data/output/<project_id>.md``.

    Completed beats are read in narrative order: arc ordering, then chapter
    ordering, then beat ordering. Chapters with no committed prose are omitted.
    The returned path points to the written Markdown file.
    """
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{config.project_id}.md"

    conn = connect_db(config.db_path)
    try:
        rows = conn.execute(
            """
            SELECT
                Chapters.id AS chapter_id,
                Chapters.description AS chapter_description,
                Beats.prose AS prose
            FROM Beats
            JOIN Chapters ON Beats.chapter_id = Chapters.id
            JOIN Arcs ON Chapters.arc_id = Arcs.id
            WHERE Arcs.project_id=?
              AND Beats.status='completed'
              AND Beats.prose IS NOT NULL
            ORDER BY Arcs.ordering ASC, Chapters.ordering ASC, Beats.ordering ASC
            """,
            (config.project_id,),
        ).fetchall()
    finally:
        conn.close()

    sections: list[str] = []
    current_chapter: str | None = None
    chapter_index = 0
    manuscript_words = 0

    for row in rows:
        if row["chapter_id"] != current_chapter:
            current_chapter = row["chapter_id"]
            chapter_index += 1
            title = _chapter_slug(row["chapter_description"] or "", f"Chapter {chapter_index}")
            sections.append(f"## Chapter {chapter_index}: {title}")

        prose = "\n\n".join(part.strip() for part in (row["prose"] or "").splitlines() if part.strip())
        if prose:
            manuscript_words += len(_WORD.findall(prose))
            sections.append(prose)

    content = "\n\n".join(sections).strip()
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")

    get_fsm_logger().info(
        "manuscript_exported project_id=%s path=%s chapters=%d words=%d",
        config.project_id,
        path,
        chapter_index,
        manuscript_words,
    )
    return path