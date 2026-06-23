"""Module: M02 (Persistent Memory Stores & Interfaces)
Manage the ACID relational hub and story-planning schema.

This paste covers connection lifecycle only: opening a deterministic, local-file
sqlite3 connection with the row factory and foreign-key enforcement every caller
relies on. Schema creation, reads, writes, and CommitIntent scanning land in later
increments.
"""

import sqlite3
from pathlib import Path


def _resolve_db_path(db_path: str | Path) -> Path:
    """Coerce a string/Path target into a Path and ensure its parent exists.

    Local-file only: the parent directory is created if missing so the database
    artifact can be opened on first run without a separate provisioning step.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def connect_db(db_path: str | Path) -> sqlite3.Connection:
    """Open a sqlite3 connection to the local relational hub.

    Accepts a ``str`` or ``Path``, creating the parent directory if needed. Every
    connection sets ``row_factory = sqlite3.Row`` for name-addressable rows and
    enables ``PRAGMA foreign_keys = ON``, since SQLite leaves foreign-key
    enforcement off per-connection by default. Connection setup is deterministic
    and local-file only — no ORM, no external server, no Docker behavior.
    """
    path = _resolve_db_path(db_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# Outline and planning schema for the relational hub. Hard CHECK constraints pin
# the documented status vocabularies at the store level so the generator can never
# persist an out-of-vocabulary state. Parent tables are declared before their
# children to keep the dependency order obvious; SQLite enforces foreign keys only
# on row writes, not at CREATE TABLE time.
_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS Arcs (
        id TEXT PRIMARY KEY,
        description TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('planned', 'active', 'completed'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS Chapters (
        id TEXT PRIMARY KEY,
        arc_id TEXT NOT NULL REFERENCES Arcs(id),
        description TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('planned', 'active', 'completed'))
    )
    """,
    # `ordering` is an explicit sort key: read paths sort scenes by `ordering ASC`
    # rather than creation time, so scenes generated simultaneously stay in their
    # planned narrative order.
    """
    CREATE TABLE IF NOT EXISTS Scenes (
        id TEXT PRIMARY KEY,
        chapter_id TEXT NOT NULL REFERENCES Chapters(id),
        description TEXT NOT NULL,
        word_budget INTEGER NOT NULL DEFAULT 0,
        ordering INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL CHECK(status IN ('planned', 'active', 'completed'))
    )
    """,
    # The design lists Beats among the relational tables but gives no explicit
    # column schema (§2.1 defines Arcs/Chapters/Scenes/Threads/CommitIntent/
    # RaptorNodes only). Kept minimal to the documented planner/commit/commit-router
    # needs: a stable `id` (beat writes upsert keyed by beat_id), the parent
    # `scene_id`, the `beat_index` carried by FSM_Pointer/CommitIntent for in-scene
    # ordering, and a status under the same planning vocabulary. Prose, PAD targets,
    # and other beat payload columns are deferred to the write-helper increment that
    # documents them rather than invented here.
    """
    CREATE TABLE IF NOT EXISTS Beats (
        id TEXT PRIMARY KEY,
        scene_id TEXT NOT NULL REFERENCES Scenes(id),
        beat_index INTEGER NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('planned', 'active', 'completed'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS Threads (
        id TEXT PRIMARY KEY,
        description TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('open', 'progressing', 'closed')),
        priority_score REAL NOT NULL
    )
    """,
    # The design names Characters as a normalized table but gives no explicit
    # column schema (§2.1 has no Characters CREATE TABLE). Kept minimal to an
    # addressable identity and a display name; richer attributes are deferred to
    # the increment that documents them rather than invented here.
    """
    CREATE TABLE IF NOT EXISTS Characters (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL
    )
    """,
    # PAD bounds enforced at the store level, not by caller validation: each axis is
    # hard-clamped to the documented inclusive [-1.00, 1.00] range so the generator
    # can never persist an out-of-range score (e.g. a hallucinated 999.0). One
    # current-state row per character (PK = character_id); the per-beat emotional
    # history lives in the append-only event ledger, not duplicated here.
    """
    CREATE TABLE IF NOT EXISTS CharacterEmotions (
        character_id TEXT PRIMARY KEY REFERENCES Characters(id),
        pleasure REAL NOT NULL CHECK(pleasure BETWEEN -1.00 AND 1.00),
        arousal REAL NOT NULL CHECK(arousal BETWEEN -1.00 AND 1.00),
        dominance REAL NOT NULL CHECK(dominance BETWEEN -1.00 AND 1.00)
    )
    """,
    # CommitIntent schema per §2.1 verbatim. The plain TEXT pointer columns are
    # intentionally not declared as foreign keys (they mirror an in-flight FSM
    # pointer that may precede the committed parent rows). Commit/crash orchestration
    # belongs to M10; this increment only creates the table and its status index.
    """
    CREATE TABLE IF NOT EXISTS CommitIntent (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        beat_id TEXT NOT NULL,
        arc_id TEXT NOT NULL,
        chapter_id TEXT NOT NULL,
        scene_id TEXT NOT NULL,
        beat_index INTEGER NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('pending', 'committed')),
        initiated_at TEXT NOT NULL,
        completed_at TEXT
    )
    """,
    # RaptorNodes schema per §2.1 verbatim: a self-referential summary tree
    # persisted across restarts, written by node_compress_memory at chapter
    # boundaries.
    """
    CREATE TABLE IF NOT EXISTS RaptorNodes (
        id TEXT PRIMARY KEY,
        parent_id TEXT REFERENCES RaptorNodes(id),
        level TEXT NOT NULL CHECK(level IN ('beat', 'scene', 'chapter', 'arc', 'global')),
        summary TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
)

# Indexes for the deterministic exact-read paths the design relies on. Each is
# `IF NOT EXISTS` so a second init_db() call is a no-op. Composite indexes fuse a
# foreign-key lookup with its ordered traversal (scenes by `ordering`, beats by
# `beat_index`) so the planner's child-in-order reads need no separate sort.
_INDEX_STATEMENTS: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_chapters_arc_id ON Chapters(arc_id)",
    "CREATE INDEX IF NOT EXISTS idx_scenes_chapter_ordering ON Scenes(chapter_id, ordering)",
    "CREATE INDEX IF NOT EXISTS idx_beats_scene_index ON Beats(scene_id, beat_index)",
    "CREATE INDEX IF NOT EXISTS idx_threads_status ON Threads(status)",
    "CREATE INDEX IF NOT EXISTS idx_commitintent_status ON CommitIntent(status)",
    "CREATE INDEX IF NOT EXISTS idx_raptornodes_parent ON RaptorNodes(parent_id)",
    "CREATE INDEX IF NOT EXISTS idx_raptornodes_level ON RaptorNodes(level)",
)


def init_db(db_path: str | Path) -> None:
    """Create the full relational schema on the local hub.

    Opens its own connection via ``connect_db``, issues every ``CREATE TABLE IF NOT
    EXISTS`` and ``CREATE INDEX IF NOT EXISTS`` statement, commits, and closes the
    connection it opened. Idempotent: running it twice against the same file neither
    drops data nor raises. Creates the documented relational tables (Arcs, Chapters,
    Scenes, Beats, Threads, Characters, CharacterEmotions, CommitIntent, RaptorNodes)
    with their hard CHECK constraints, plus the indexes for the deterministic
    exact-read paths. Data writes, read helpers, runtime wiring, and CommitIntent
    crash-recovery orchestration (M10) are out of scope here.
    """
    conn = connect_db(db_path)
    try:
        for statement in _SCHEMA_STATEMENTS:
            conn.execute(statement)
        for statement in _INDEX_STATEMENTS:
            conn.execute(statement)
        conn.commit()
    finally:
        conn.close()
