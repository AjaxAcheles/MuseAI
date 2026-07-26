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


def committed_word_count(config: AppConfig, project_id: str | None = None) -> int:
    """Total words from completed beats only.

    ``project_id`` defaults to the configured project; callers that know which
    project a run actually generated (the manager, the web routes) pass it
    explicitly so a stale config cannot count the wrong project.
    """
    project = project_id or config.project_id
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
            (project,),
        ).fetchone()
        return int(row["total"] or 0)
    finally:
        conn.close()


def _chapter_epigraph(description: str) -> str:
    return " ".join(description.split()).strip()


def export_manuscript(config: AppConfig, project_id: str | None = None) -> Path:
    """Write the committed manuscript to ``config.output_dir/<project_id>.md``.

    Completed beats are read in narrative order: arc ordering, then chapter
    ordering, then beat ordering. Chapters with no committed prose are omitted.
    Chapter numbering runs continuously across arcs; each new arc opens with its
    own heading and the chapter's planned description follows its heading as an
    italic epigraph. The returned path points to the written Markdown file.

    ``project_id`` defaults to the configured project — see
    :func:`committed_word_count` for why callers should pass the run's own.
    """
    project = project_id or config.project_id
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{project}.md"

    conn = connect_db(config.db_path)
    try:
        rows = conn.execute(
            """
            SELECT
                Arcs.id AS arc_id,
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
            (project,),
        ).fetchall()
    finally:
        conn.close()

    sections: list[str] = []
    current_arc: str | None = None
    current_chapter: str | None = None
    arc_index = 0
    chapter_index = 0
    manuscript_words = 0

    for row in rows:
        if row["arc_id"] != current_arc:
            current_arc = row["arc_id"]
            arc_index += 1
            sections.append(f"# Arc {arc_index}")
        if row["chapter_id"] != current_chapter:
            current_chapter = row["chapter_id"]
            chapter_index += 1
            sections.append(f"## Chapter {chapter_index}")
            epigraph = _chapter_epigraph(row["chapter_description"] or "")
            if epigraph:
                sections.append(f"*{epigraph}*")

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
        project,
        path,
        chapter_index,
        manuscript_words,
    )
    return path
