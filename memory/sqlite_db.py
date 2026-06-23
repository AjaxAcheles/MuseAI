"""Module: M02 (Persistent Memory Stores & Interfaces)
Manage the ACID relational hub and story-planning schema.

Provides the local-file connection lifecycle (``connect_db``), idempotent schema
creation (``init_db``), and the first relational write primitive: an idempotent
beat-commit upsert keyed by ``beat_id`` (``upsert_beat_commit``). PAD/thread sub-
writes, read helpers, non-SQLite stores, and CommitIntent crash-recovery
orchestration (M10) land in later increments.
"""

import sqlite3
from datetime import datetime, timezone
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
    # The design gives no explicit Beats CREATE TABLE (§2.1 defines Arcs/Chapters/
    # Scenes/Threads/CommitIntent/RaptorNodes only); columns are scoped to the
    # documented planner/commit/commit-router needs. Identity/structure: a stable
    # `id` (beat writes upsert keyed by beat_id), parent `scene_id`, the `beat_index`
    # carried by FSM_Pointer/CommitIntent for in-scene ordering, and a planning
    # `status`. Committed state (node_commit_transaction step 2): `prose` holds the
    # validated committed prose delta (nullable so a planner-created beat can exist
    # before it is drafted); `word_count` backs `_commit_router`'s SUM(word_count)
    # continuation check; `committed_at` is the commit timestamp. PAD/thread sub-
    # writes are owned by separate tables and the next paste, not stored here.
    """
    CREATE TABLE IF NOT EXISTS Beats (
        id TEXT PRIMARY KEY,
        scene_id TEXT NOT NULL REFERENCES Scenes(id),
        beat_index INTEGER NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('planned', 'active', 'completed')),
        prose TEXT,
        word_count INTEGER NOT NULL DEFAULT 0,
        committed_at TEXT
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
    # can never persist an out-of-range score (e.g. a hallucinated 999.0). Keyed by
    # (beat_id, character_id) because node_commit_transaction bundles the per-beat PAD
    # snapshot into the beat commit and writes it "upsert keyed by beat_id"; the
    # composite primary key makes replaying a beat update the same rows in place
    # rather than appending duplicates. This supersedes the minimal current-state
    # shape guessed when the table was first created.
    """
    CREATE TABLE IF NOT EXISTS CharacterEmotions (
        beat_id TEXT NOT NULL REFERENCES Beats(id),
        character_id TEXT NOT NULL REFERENCES Characters(id),
        pleasure REAL NOT NULL CHECK(pleasure BETWEEN -1.00 AND 1.00),
        arousal REAL NOT NULL CHECK(arousal BETWEEN -1.00 AND 1.00),
        dominance REAL NOT NULL CHECK(dominance BETWEEN -1.00 AND 1.00),
        PRIMARY KEY (beat_id, character_id)
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
    # CharacterEmotions PK is (beat_id, character_id); a per-character FK lookup
    # would otherwise be uncovered, so index character_id to keep the documented
    # foreign-key-lookup paths indexed.
    "CREATE INDEX IF NOT EXISTS idx_characteremotions_character ON CharacterEmotions(character_id)",
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


# Upsert the committed-beat row, keyed by the beat's primary key so a crash-recovery
# replay of the same beat updates in place and never duplicates the row. `excluded`
# refers to the values proposed by the INSERT; every committed-state column is
# overwritten from it on conflict.
_BEAT_COMMIT_UPSERT = """
    INSERT INTO Beats (id, scene_id, beat_index, status, prose, word_count, committed_at)
    VALUES (:beat_id, :scene_id, :beat_index, :status, :prose, :word_count, :committed_at)
    ON CONFLICT(id) DO UPDATE SET
        scene_id = excluded.scene_id,
        beat_index = excluded.beat_index,
        status = excluded.status,
        prose = excluded.prose,
        word_count = excluded.word_count,
        committed_at = excluded.committed_at
"""

# Per-character PAD snapshot for a beat, upserted on the (beat_id, character_id)
# primary key so replaying the same beat updates the same rows in place.
_PAD_STATE_UPSERT = """
    INSERT INTO CharacterEmotions (beat_id, character_id, pleasure, arousal, dominance)
    VALUES (:beat_id, :character_id, :pleasure, :arousal, :dominance)
    ON CONFLICT(beat_id, character_id) DO UPDATE SET
        pleasure = excluded.pleasure,
        arousal = excluded.arousal,
        dominance = excluded.dominance
"""

# Thread status transition applied by thread id. An UPDATE is idempotent by
# construction (re-applying the same status is a no-op) and the schema's hard status
# CHECK rejects any out-of-vocabulary value. Thread creation (description/priority)
# belongs to the planner, not this commit helper.
_THREAD_STATUS_UPDATE = "UPDATE Threads SET status = :status WHERE id = :thread_id"


def _upsert_pad_states(
    conn: sqlite3.Connection,
    beat_id: str,
    pad_states: dict[str, dict[str, float]],
) -> None:
    """Upsert each character's PAD snapshot for ``beat_id`` on the open connection.

    ``pad_states`` maps ``character_id`` -> ``{"pleasure", "arousal", "dominance"}``,
    matching the ``pad_states`` block of the ``beat_commit`` payload. Relational-only
    and replay-safe; the caller owns the surrounding transaction.
    """
    rows = [
        {
            "beat_id": beat_id,
            "character_id": character_id,
            "pleasure": pad["pleasure"],
            "arousal": pad["arousal"],
            "dominance": pad["dominance"],
        }
        for character_id, pad in pad_states.items()
    ]
    if rows:
        conn.executemany(_PAD_STATE_UPSERT, rows)


def _apply_thread_updates(
    conn: sqlite3.Connection,
    thread_updates: dict[str, str],
) -> None:
    """Apply thread status transitions by thread id on the open connection.

    ``thread_updates`` maps ``thread_id`` -> new ``status``. Idempotent by thread id;
    the schema's status CHECK enforces the allowed vocabulary. The caller owns the
    surrounding transaction.
    """
    rows = [
        {"thread_id": thread_id, "status": status}
        for thread_id, status in thread_updates.items()
    ]
    if rows:
        conn.executemany(_THREAD_STATUS_UPDATE, rows)


def upsert_beat_commit(
    db_path: str | Path,
    *,
    beat_id: str,
    scene_id: str,
    beat_index: int,
    prose: str,
    word_count: int | None = None,
    committed_at: str | None = None,
    status: str = "completed",
    pad_states: dict[str, dict[str, float]] | None = None,
    thread_updates: dict[str, str] | None = None,
) -> None:
    """Idempotently persist a validated beat commit to the relational hub.

    This is the SQLite half of ``node_commit_transaction`` step 2: it upserts the
    ``Beats`` row keyed by ``beat_id`` and, bundled into the same atomic commit, the
    per-character PAD snapshot and any Thread status transitions — matching the design
    requirement that PAD state travels inside the beat commit rather than as a
    standalone event. The signature is explicit; callers pass already-validated,
    committed input only — never in-flight draft text.

    ``pad_states`` maps ``character_id`` -> ``{"pleasure", "arousal", "dominance"}``
    and is upserted on the ``(beat_id, character_id)`` key, so replaying the same beat
    updates the same PAD rows in place. ``thread_updates`` maps ``thread_id`` -> new
    ``status`` and is applied idempotently by id under the schema's status CHECK.

    ``word_count`` defaults to a whitespace token count of ``prose`` (the value
    ``_commit_router`` sums for its continuation check); ``committed_at`` defaults to
    the current UTC ISO-8601 timestamp. All writes run inside one explicit
    transaction: on any error the transaction is rolled back and the exception
    re-raised, so a beat commit is never partially applied.

    Event-log append, Graphiti edges, crash replay, and M10 commit orchestration are
    out of scope here; this helper is relational-only.
    """
    if word_count is None:
        word_count = len(prose.split())
    if committed_at is None:
        committed_at = datetime.now(timezone.utc).isoformat()

    params = {
        "beat_id": beat_id,
        "scene_id": scene_id,
        "beat_index": beat_index,
        "status": status,
        "prose": prose,
        "word_count": word_count,
        "committed_at": committed_at,
    }

    init_db(db_path)
    conn = connect_db(db_path)
    try:
        conn.execute("BEGIN")
        # Beat row first so the (beat_id) FK on the PAD snapshot is satisfied within
        # the same transaction before its rows are written.
        conn.execute(_BEAT_COMMIT_UPSERT, params)
        if pad_states:
            _upsert_pad_states(conn, beat_id, pad_states)
        if thread_updates:
            _apply_thread_updates(conn, thread_updates)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# --- Deterministic read helpers -------------------------------------------------
# Exact relational reads for the planning and commit-router paths. Each returns plain
# dicts (never raw sqlite3.Row) so callers stay decoupled from the cursor. "Remaining"
# means any row not yet 'completed'. Ordered reads honour the design's sort keys:
# scenes by `ordering ASC` (not creation time), beats by `beat_index ASC`.
_REMAINING_STATUSES = ("planned", "active")  # everything that is not 'completed'


def _read_one(db_path: str | Path, query: str, params: tuple) -> dict | None:
    """Run a single-row exact read, returning a plain dict or None."""
    init_db(db_path)
    conn = connect_db(db_path)
    try:
        row = conn.execute(query, params).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def _read_all(db_path: str | Path, query: str, params: tuple) -> list[dict]:
    """Run a multi-row exact read, returning a list of plain dicts."""
    init_db(db_path)
    conn = connect_db(db_path)
    try:
        return [dict(row) for row in conn.execute(query, params).fetchall()]
    finally:
        conn.close()


def get_arc(db_path: str | Path, arc_id: str) -> dict | None:
    """Return the Arc row for ``arc_id`` as a dict, or None if absent."""
    return _read_one(db_path, "SELECT * FROM Arcs WHERE id = ?", (arc_id,))


def get_chapters_for_arc(db_path: str | Path, arc_id: str) -> list[dict]:
    """Return all chapters under ``arc_id`` (foreign-key exact lookup)."""
    return _read_all(db_path, "SELECT * FROM Chapters WHERE arc_id = ?", (arc_id,))


def get_scenes_for_chapter_ordered(db_path: str | Path, chapter_id: str) -> list[dict]:
    """Return scenes under ``chapter_id`` sorted by ``ordering ASC``.

    Sorts by the explicit `ordering` key, not creation time, so scenes generated
    simultaneously keep their planned narrative order.
    """
    return _read_all(
        db_path,
        "SELECT * FROM Scenes WHERE chapter_id = ? ORDER BY ordering ASC",
        (chapter_id,),
    )


def get_beats_for_scene_ordered(db_path: str | Path, scene_id: str) -> list[dict]:
    """Return beats under ``scene_id`` sorted by ``beat_index ASC``."""
    return _read_all(
        db_path,
        "SELECT * FROM Beats WHERE scene_id = ? ORDER BY beat_index ASC",
        (scene_id,),
    )


def get_remaining_beats_for_scene(db_path: str | Path, scene_id: str) -> list[dict]:
    """Return not-yet-completed beats under ``scene_id``, ordered by ``beat_index``."""
    return _read_all(
        db_path,
        "SELECT * FROM Beats WHERE scene_id = ? AND status IN (?, ?) "
        "ORDER BY beat_index ASC",
        (scene_id, *_REMAINING_STATUSES),
    )


def get_remaining_scenes_for_chapter(db_path: str | Path, chapter_id: str) -> list[dict]:
    """Return not-yet-completed scenes under ``chapter_id``, ordered by ``ordering``."""
    return _read_all(
        db_path,
        "SELECT * FROM Scenes WHERE chapter_id = ? AND status IN (?, ?) "
        "ORDER BY ordering ASC",
        (chapter_id, *_REMAINING_STATUSES),
    )


def get_remaining_chapters_for_arc(db_path: str | Path, arc_id: str) -> list[dict]:
    """Return not-yet-completed chapters under ``arc_id``."""
    return _read_all(
        db_path,
        "SELECT * FROM Chapters WHERE arc_id = ? AND status IN (?, ?)",
        (arc_id, *_REMAINING_STATUSES),
    )


def get_open_threads(db_path: str | Path) -> list[dict]:
    """Return open Threads — canonical relational truth for planner/critic consumers."""
    return _read_all(
        db_path,
        "SELECT * FROM Threads WHERE status = ? ORDER BY priority_score DESC",
        ("open",),
    )


def get_beat(db_path: str | Path, beat_id: str, *, strict: bool = False) -> dict | None:
    """Return the committed Beat row for ``beat_id`` as a dict.

    Returns None when the beat is absent. Pass ``strict=True`` to raise ``KeyError``
    instead — for callers that treat a missing committed beat as a hard error.
    """
    beat = _read_one(db_path, "SELECT * FROM Beats WHERE id = ?", (beat_id,))
    if beat is None and strict:
        raise KeyError(f"No Beats row for beat_id={beat_id!r}")
    return beat


def get_latest_pad_for_character(db_path: str | Path, character_id: str) -> dict | None:
    """Return the most recent committed PAD snapshot for ``character_id``, or None.

    "Most recent" is the character's PAD from the latest committed beat, ordered by
    the beat's ``committed_at`` then ``beat_index`` (both descending). Returns None
    when the character has no committed PAD rows yet.
    """
    return _read_one(
        db_path,
        "SELECT ce.* FROM CharacterEmotions ce "
        "JOIN Beats b ON b.id = ce.beat_id "
        "WHERE ce.character_id = ? "
        "ORDER BY b.committed_at DESC, b.beat_index DESC LIMIT 1",
        (character_id,),
    )


def get_latest_pad_for_scene(db_path: str | Path, scene_id: str) -> list[dict]:
    """Return the latest PAD snapshot per character within ``scene_id``.

    For each character appearing in the scene's beats, returns the PAD row from the
    highest ``beat_index`` in that scene (the scene's most recent emotional state for
    that character). Ordered by ``character_id`` for deterministic output; ``[]`` when
    the scene has no committed PAD rows.
    """
    return _read_all(
        db_path,
        "SELECT ce.* FROM CharacterEmotions ce "
        "JOIN Beats b ON b.id = ce.beat_id "
        "WHERE b.scene_id = ? AND b.beat_index = ("
        "    SELECT MAX(b2.beat_index) FROM CharacterEmotions ce2 "
        "    JOIN Beats b2 ON b2.id = ce2.beat_id "
        "    WHERE b2.scene_id = b.scene_id AND ce2.character_id = ce.character_id"
        ") ORDER BY ce.character_id ASC",
        (scene_id,),
    )


def get_pending_commit_intents(db_path: str | Path) -> list[dict]:
    """Return pending CommitIntent rows (observational only), oldest first.

    A read-only window onto in-flight commits. Replay, human-review, and recovery
    decisions belong to the later commit/crash increments — never made here.
    """
    return _read_all(
        db_path,
        "SELECT * FROM CommitIntent WHERE status = ? ORDER BY id ASC",
        ("pending",),
    )


def get_raptor_nodes_by_level(db_path: str | Path, level: str) -> list[dict]:
    """Return RaptorNodes at ``level``, ordered by id for deterministic output."""
    return _read_all(
        db_path,
        "SELECT * FROM RaptorNodes WHERE level = ? ORDER BY id ASC",
        (level,),
    )


def get_raptor_nodes_by_parent(db_path: str | Path, parent_id: str | None) -> list[dict]:
    """Return RaptorNodes whose parent is ``parent_id``, ordered by id.

    ``parent_id=None`` selects root nodes (``parent_id IS NULL``); ``IS ?`` binds NULL
    correctly so the same helper serves both root and child lookups.
    """
    return _read_all(
        db_path,
        "SELECT * FROM RaptorNodes WHERE parent_id IS ? ORDER BY id ASC",
        (parent_id,),
    )
