"""SQLite persistence for MuseAI v1.

Raw ``sqlite3`` — no ORM. Every write is an idempotent upsert keyed by ``id`` so
replaying the event log is always safe. Reads are precise and ordered.

Callers own transactions: wrap writes in ``with conn:`` to commit. There is no
``Scenes`` table and no non-v1 store — v1 plans at Chapter + Beat granularity.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect_db(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection with row access by name and foreign keys enforced."""
    path = Path(db_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


_SCHEMA = """
CREATE TABLE IF NOT EXISTS Projects (
    id TEXT PRIMARY KEY,
    genre TEXT,
    premise TEXT,
    word_count_target INTEGER
);

CREATE TABLE IF NOT EXISTS Arcs (
    id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES Projects(id),
    ordering INTEGER,
    description TEXT NOT NULL,
    status TEXT CHECK(status IN ('planned','active','completed'))
);

CREATE TABLE IF NOT EXISTS Chapters (
    id TEXT PRIMARY KEY,
    arc_id TEXT REFERENCES Arcs(id),
    ordering INTEGER,
    description TEXT NOT NULL,
    obligations TEXT,
    status TEXT CHECK(status IN ('planned','active','completed'))
);

CREATE TABLE IF NOT EXISTS Beats (
    id TEXT PRIMARY KEY,
    chapter_id TEXT REFERENCES Chapters(id),
    ordering INTEGER,
    beat_spec TEXT,
    pad_constraint TEXT,
    word_target INTEGER,
    prose TEXT,
    word_count INTEGER DEFAULT 0,
    status TEXT CHECK(status IN ('planned','active','completed'))
);

CREATE TABLE IF NOT EXISTS Threads (
    id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES Projects(id),
    description TEXT NOT NULL,
    status TEXT CHECK(status IN ('open','progressing','closed')),
    priority_score REAL
);

CREATE TABLE IF NOT EXISTS Characters (
    id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES Projects(id),
    name TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS CharacterEmotions (
    character_id TEXT PRIMARY KEY REFERENCES Characters(id),
    pleasure REAL CHECK(pleasure BETWEEN -1.0 AND 1.0),
    arousal REAL CHECK(arousal BETWEEN -1.0 AND 1.0),
    dominance REAL CHECK(dominance BETWEEN -1.0 AND 1.0),
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS CommitIntent (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    beat_id TEXT,
    arc_id TEXT,
    chapter_id TEXT,
    beat_index INTEGER,
    status TEXT CHECK(status IN ('pending','committed')),
    initiated_at TEXT,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_arcs_project_ordering ON Arcs(project_id, ordering);
CREATE INDEX IF NOT EXISTS idx_chapters_arc_ordering ON Chapters(arc_id, ordering);
CREATE INDEX IF NOT EXISTS idx_beats_chapter_ordering ON Beats(chapter_id, ordering);
CREATE INDEX IF NOT EXISTS idx_threads_project_status ON Threads(project_id, status);
CREATE INDEX IF NOT EXISTS idx_commitintent_status ON CommitIntent(status);
"""


def init_db(db_path: str | Path) -> None:
    """Create the schema if it does not exist. Idempotent."""
    conn = connect_db(db_path)
    try:
        with conn:
            conn.executescript(_SCHEMA)
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Upserts — idempotent, keyed by id.                                          #
# --------------------------------------------------------------------------- #

def upsert_project(
    conn: sqlite3.Connection,
    *,
    id: str,
    genre: str | None = None,
    premise: str | None = None,
    word_count_target: int | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO Projects (id, genre, premise, word_count_target)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            genre = excluded.genre,
            premise = excluded.premise,
            word_count_target = excluded.word_count_target
        """,
        (id, genre, premise, word_count_target),
    )


def upsert_arc(
    conn: sqlite3.Connection,
    *,
    id: str,
    project_id: str,
    ordering: int,
    description: str,
    status: str,
) -> None:
    conn.execute(
        """
        INSERT INTO Arcs (id, project_id, ordering, description, status)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            project_id = excluded.project_id,
            ordering = excluded.ordering,
            description = excluded.description,
            status = excluded.status
        """,
        (id, project_id, ordering, description, status),
    )


def upsert_chapter(
    conn: sqlite3.Connection,
    *,
    id: str,
    arc_id: str,
    ordering: int,
    description: str,
    obligations: str | None = None,
    status: str = "planned",
) -> None:
    conn.execute(
        """
        INSERT INTO Chapters (id, arc_id, ordering, description, obligations, status)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            arc_id = excluded.arc_id,
            ordering = excluded.ordering,
            description = excluded.description,
            obligations = excluded.obligations,
            -- A finished chapter is never un-finished by a later write. Re-planning
            -- an arc would otherwise reset completed chapters to 'planned' and the
            -- graph would redraft prose it had already committed.
            status = CASE
                WHEN Chapters.status = 'completed' THEN 'completed'
                ELSE excluded.status
            END
        """,
        (id, arc_id, ordering, description, obligations, status),
    )


def upsert_beat(
    conn: sqlite3.Connection,
    *,
    id: str,
    chapter_id: str,
    ordering: int,
    beat_spec: str | None = None,
    pad_constraint: str | None = None,
    word_target: int | None = None,
    prose: str | None = None,
    word_count: int = 0,
    status: str = "planned",
) -> None:
    conn.execute(
        """
        INSERT INTO Beats (id, chapter_id, ordering, beat_spec, pad_constraint,
                           word_target, prose, word_count, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            chapter_id = excluded.chapter_id,
            ordering = excluded.ordering,
            beat_spec = excluded.beat_spec,
            pad_constraint = excluded.pad_constraint,
            word_target = excluded.word_target,
            -- Prose is written once, by `commit`, and is the manuscript. A caller
            -- that passes no prose (the planners) is describing a beat, not
            -- unwriting it, so an absent value must never blank the column — nor
            -- orphan `word_count` from the text it counts.
            prose = COALESCE(excluded.prose, Beats.prose),
            word_count = CASE
                WHEN excluded.prose IS NULL THEN Beats.word_count
                ELSE excluded.word_count
            END,
            status = CASE
                WHEN Beats.status = 'completed' THEN 'completed'
                ELSE excluded.status
            END
        """,
        (id, chapter_id, ordering, beat_spec, pad_constraint,
         word_target, prose, word_count, status),
    )


def upsert_thread(
    conn: sqlite3.Connection,
    *,
    id: str,
    project_id: str,
    description: str,
    status: str = "open",
    priority_score: float | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO Threads (id, project_id, description, status, priority_score)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            project_id = excluded.project_id,
            description = excluded.description,
            status = excluded.status,
            priority_score = excluded.priority_score
        """,
        (id, project_id, description, status, priority_score),
    )


def upsert_character(
    conn: sqlite3.Connection,
    *,
    id: str,
    project_id: str,
    name: str,
    description: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO Characters (id, project_id, name, description)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            project_id = excluded.project_id,
            name = excluded.name,
            description = excluded.description
        """,
        (id, project_id, name, description),
    )


def upsert_character_emotions(
    conn: sqlite3.Connection,
    *,
    character_id: str,
    pleasure: float,
    arousal: float,
    dominance: float,
    updated_at: str | None = None,
) -> None:
    """Upsert the single current PAD row for a character, in place."""
    conn.execute(
        """
        INSERT INTO CharacterEmotions
            (character_id, pleasure, arousal, dominance, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(character_id) DO UPDATE SET
            pleasure = excluded.pleasure,
            arousal = excluded.arousal,
            dominance = excluded.dominance,
            updated_at = excluded.updated_at
        """,
        (character_id, pleasure, arousal, dominance, updated_at or _utc_now()),
    )


# --------------------------------------------------------------------------- #
# CommitIntent — the write-ahead marker for a beat commit.                    #
# --------------------------------------------------------------------------- #

def create_commit_intent(
    conn: sqlite3.Connection,
    *,
    beat_id: str,
    arc_id: str,
    chapter_id: str,
    beat_index: int,
    initiated_at: str | None = None,
) -> int:
    """Record a pending commit intent and return its row id."""
    cur = conn.execute(
        """
        INSERT INTO CommitIntent
            (beat_id, arc_id, chapter_id, beat_index, status, initiated_at)
        VALUES (?, ?, ?, ?, 'pending', ?)
        """,
        (beat_id, arc_id, chapter_id, beat_index, initiated_at or _utc_now()),
    )
    return int(cur.lastrowid)


def mark_commit_committed(
    conn: sqlite3.Connection, intent_id: int, completed_at: str | None = None
) -> None:
    conn.execute(
        "UPDATE CommitIntent SET status='committed', completed_at=? WHERE id=?",
        (completed_at or _utc_now(), intent_id),
    )


def delete_commit_intent(conn: sqlite3.Connection, intent_id: int) -> None:
    conn.execute("DELETE FROM CommitIntent WHERE id=?", (intent_id,))


def get_pending_intents(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM CommitIntent WHERE status='pending' ORDER BY id ASC"
    ).fetchall()


def set_beat_status(conn: sqlite3.Connection, beat_id: str, status: str) -> None:
    """Targeted status update — used by recovery to reset an uncommitted beat."""
    conn.execute("UPDATE Beats SET status=? WHERE id=?", (status, beat_id))


# --------------------------------------------------------------------------- #
# Reads — precise and ordered.                                               #
# --------------------------------------------------------------------------- #

def get_project(conn: sqlite3.Connection, project_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM Projects WHERE id=?", (project_id,)
    ).fetchone()


def get_arcs(conn: sqlite3.Connection, project_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM Arcs WHERE project_id=? ORDER BY ordering ASC",
        (project_id,),
    ).fetchall()


def get_chapters_for_arc(conn: sqlite3.Connection, arc_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM Chapters WHERE arc_id=? ORDER BY ordering ASC",
        (arc_id,),
    ).fetchall()


def get_beats_for_chapter(
    conn: sqlite3.Connection, chapter_id: str
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM Beats WHERE chapter_id=? ORDER BY ordering ASC",
        (chapter_id,),
    ).fetchall()


def get_open_threads(conn: sqlite3.Connection, project_id: str) -> list[sqlite3.Row]:
    """Open threads for a project, highest priority first."""
    return conn.execute(
        """
        SELECT * FROM Threads
        WHERE project_id=? AND status='open'
        ORDER BY priority_score DESC
        """,
        (project_id,),
    ).fetchall()


def get_threads_for_project(
    conn: sqlite3.Connection, project_id: str
) -> list[sqlite3.Row]:
    """Every thread for a project, unresolved first, then closed.

    The beat planner needs to see *closed* threads too — a resolved thread it
    can no longer re-dramatize — not just the open ones. Ordered open →
    progressing → closed, each band highest-priority first.
    """
    return conn.execute(
        """
        SELECT * FROM Threads
        WHERE project_id=?
        ORDER BY
            CASE status WHEN 'open' THEN 0 WHEN 'progressing' THEN 1 ELSE 2 END,
            priority_score DESC
        """,
        (project_id,),
    ).fetchall()


def get_characters(conn: sqlite3.Connection, project_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM Characters WHERE project_id=? ORDER BY name ASC",
        (project_id,),
    ).fetchall()


def get_character_emotions(
    conn: sqlite3.Connection, character_id: str
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM CharacterEmotions WHERE character_id=?",
        (character_id,),
    ).fetchone()


def reset_active_beats(conn: sqlite3.Connection, project_id: str) -> int:
    """Return in-flight beats to ``planned``, so a restart redrafts them cleanly.

    A crash mid-beat leaves ``status='active'`` with no prose — a half-state the
    planners skip and the drafter never revisits. Only prose-less beats are
    touched: a beat that has prose is either committed or awaiting review, and
    neither is ours to undo. Idempotent; returns the number of rows reset.
    """
    cursor = conn.execute(
        """
        UPDATE Beats SET status='planned'
        WHERE status='active'
          AND prose IS NULL
          AND chapter_id IN (
              SELECT Chapters.id FROM Chapters
              JOIN Arcs ON Chapters.arc_id = Arcs.id
              WHERE Arcs.project_id = ?
          )
        """,
        (project_id,),
    )
    return cursor.rowcount


def get_committed_beats(conn: sqlite3.Connection, project_id: str) -> list[sqlite3.Row]:
    """Every committed beat in narrative order, with its arc and chapter context.

    A beat counts as committed only when the commit node has both persisted its prose
    and flipped its status to ``completed``. Planned and active beats are never returned,
    so a caller cannot accidentally present a discarded draft as manuscript.
    """
    return conn.execute(
        """
        SELECT
            Beats.id            AS beat_id,
            Beats.ordering      AS beat_ordering,
            Beats.prose         AS prose,
            Beats.word_count    AS word_count,
            Chapters.id         AS chapter_id,
            Chapters.ordering   AS chapter_ordering,
            Chapters.description AS chapter_description,
            Arcs.id             AS arc_id,
            Arcs.ordering       AS arc_ordering
        FROM Beats
        JOIN Chapters ON Beats.chapter_id = Chapters.id
        JOIN Arcs ON Chapters.arc_id = Arcs.id
        WHERE Arcs.project_id = ?
          AND Beats.status = 'completed'
          AND Beats.prose IS NOT NULL
        ORDER BY Arcs.ordering ASC, Chapters.ordering ASC, Beats.ordering ASC
        """,
        (project_id,),
    ).fetchall()


def get_recent_committed_beats(
    conn: sqlite3.Connection, project_id: str, limit: int
) -> list[sqlite3.Row]:
    """Most-recent completed beats (with prose) across chapters, newest-last.

    Narrative order runs arc -> chapter -> beat. We take the last ``limit`` in
    that order and return them oldest-first so the newest sits at the end,
    ready to inject as chronological recent context.
    """
    rows = conn.execute(
        """
        SELECT Beats.*
        FROM Beats
        JOIN Chapters ON Beats.chapter_id = Chapters.id
        JOIN Arcs ON Chapters.arc_id = Arcs.id
        WHERE Arcs.project_id = ?
          AND Beats.status = 'completed'
          AND Beats.prose IS NOT NULL
        ORDER BY Arcs.ordering DESC, Chapters.ordering DESC, Beats.ordering DESC
        LIMIT ?
        """,
        (project_id, limit),
    ).fetchall()
    return list(reversed(rows))
