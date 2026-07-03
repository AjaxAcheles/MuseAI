"""Module: M02 (Persistent Memory Stores & Interfaces)
Manage the ACID relational hub and story-planning schema.

Provides the local-file connection lifecycle (``connect_db``), idempotent schema
creation (``init_db``), and the first relational write primitive: an idempotent
beat-commit upsert keyed by ``beat_id`` (``upsert_beat_commit``). PAD/thread sub-
writes, read helpers, non-SQLite stores, and CommitIntent crash-recovery
orchestration (M10) land in later increments.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


_INITIALIZED_DB_PATHS: set[Path] = set()


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


def _ensure_db_initialized(db_path: str | Path) -> None:
    """Initialize ``db_path`` once per process, and again if the file is removed."""
    path = _resolve_db_path(db_path)
    cache_key = path.resolve()
    if cache_key not in _INITIALIZED_DB_PATHS or not path.exists():
        init_db(path)
        _INITIALIZED_DB_PATHS.add(cache_key)


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
    # --- Planning-persistence tables (§2.7), verbatim from the design DDL --------
    # These live in the same DB file as the narrative tables above but are a separate
    # *proposal surface*: a PlanningSnapshot is a versioned outline the user can
    # approve/reject/annotate before any of it becomes committed prose. They never
    # alter the nine narrative tables. Hard CHECK constraints pin the documented enum
    # vocabularies at the store level. Declared in §2.7 order; `PlanningSnapshot` and
    # `PlanningRevision` form a nullable circular reference (a snapshot's
    # `active_revision_id` stays NULL until its first revision exists), which is safe
    # because SQLite resolves foreign-key parents on row writes, not at CREATE time.
    """
    CREATE TABLE IF NOT EXISTS PlanningSnapshot (
        snapshot_id        TEXT PRIMARY KEY,
        project_id         TEXT NOT NULL,
        mode               TEXT NOT NULL,
        status             TEXT NOT NULL CHECK(status IN
                             ('draft','user_annotated','revision_requested','revised','approved','rejected','superseded')),
        created_at         TEXT NOT NULL,
        approved_at        TEXT,
        active_revision_id TEXT REFERENCES PlanningRevision(revision_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS PlanningNode (
        node_id       TEXT PRIMARY KEY,
        snapshot_id   TEXT NOT NULL REFERENCES PlanningSnapshot(snapshot_id),
        level         TEXT NOT NULL CHECK(level IN ('global','arc','chapter','scene','beat')),
        parent_id     TEXT REFERENCES PlanningNode(node_id),
        ordering      INTEGER NOT NULL DEFAULT 0,
        title         TEXT,
        summary       TEXT,
        purpose       TEXT,
        status        TEXT NOT NULL,
        locked_pinned INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS PlanningAnnotation (
        annotation_id  TEXT PRIMARY KEY,
        snapshot_id    TEXT NOT NULL REFERENCES PlanningSnapshot(snapshot_id),
        target_node_id TEXT NOT NULL REFERENCES PlanningNode(node_id),
        target_level   TEXT NOT NULL CHECK(target_level IN ('global','arc','chapter','scene','beat')),
        note_type      TEXT NOT NULL CHECK(note_type IN
                         ('constraint','preference','concern','question','regenerate_request','pin','remove','move','tone','continuity')),
        scope          TEXT NOT NULL CHECK(scope IN ('this_node','children','subtree','sibling_sequence','global')),
        priority       TEXT NOT NULL CHECK(priority IN ('low','normal','high','hard')),
        text           TEXT NOT NULL,
        status         TEXT NOT NULL CHECK(status IN
                         ('pending','applied','partially_applied','rejected','needs_clarification','superseded')),
        planner_response TEXT,
        created_at     TEXT NOT NULL,
        resolved_at    TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS PlanningRevision (
        revision_id        TEXT PRIMARY KEY,
        snapshot_id        TEXT NOT NULL REFERENCES PlanningSnapshot(snapshot_id),
        parent_revision_id TEXT REFERENCES PlanningRevision(revision_id),
        change_summary     TEXT,
        diff_json          TEXT NOT NULL,
        created_at         TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS PlannerToolCallTrace (
        trace_id        TEXT PRIMARY KEY,
        snapshot_id     TEXT REFERENCES PlanningSnapshot(snapshot_id),
        planner_level   TEXT NOT NULL CHECK(planner_level IN ('global','arc','chapter','scene','beat')),
        planner_node_id TEXT,
        loop_index      INTEGER NOT NULL,
        tool_name       TEXT NOT NULL,
        tool_args_json  TEXT,
        result_summary  TEXT,
        success         INTEGER NOT NULL,
        error           TEXT,
        created_at      TEXT NOT NULL
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
    # Planning proposal-surface lookups (§2.7): nodes by snapshot+level and by parent
    # for tree traversal, annotations by snapshot and by target node, revisions by
    # snapshot, and tool-call traces by snapshot+planner level.
    "CREATE INDEX IF NOT EXISTS idx_planningnode_snapshot_level ON PlanningNode(snapshot_id, level)",
    "CREATE INDEX IF NOT EXISTS idx_planningnode_parent ON PlanningNode(parent_id)",
    "CREATE INDEX IF NOT EXISTS idx_planningannotation_snapshot ON PlanningAnnotation(snapshot_id)",
    "CREATE INDEX IF NOT EXISTS idx_planningannotation_target ON PlanningAnnotation(target_node_id)",
    "CREATE INDEX IF NOT EXISTS idx_planningrevision_snapshot ON PlanningRevision(snapshot_id)",
    "CREATE INDEX IF NOT EXISTS idx_plannertoolcalltrace_snapshot_level ON PlannerToolCallTrace(snapshot_id, planner_level)",
)


def init_db(db_path: str | Path) -> None:
    """Create the full relational schema on the local hub.

    Opens its own connection via ``connect_db``, issues every ``CREATE TABLE IF NOT
    EXISTS`` and ``CREATE INDEX IF NOT EXISTS`` statement, commits, and closes the
    connection it opened. Idempotent: running it twice against the same file neither
    drops data nor raises. Creates the documented narrative tables (Arcs, Chapters,
    Scenes, Beats, Threads, Characters, CharacterEmotions, CommitIntent, RaptorNodes)
    and the §2.7 planning proposal-surface tables (PlanningSnapshot, PlanningNode,
    PlanningAnnotation, PlanningRevision, PlannerToolCallTrace) with their hard CHECK
    constraints, plus the indexes for the deterministic exact-read paths. Planning
    read/write helpers, data writes, runtime wiring, and CommitIntent crash-recovery
    orchestration (M10) are out of scope here.
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

    _ensure_db_initialized(db_path)
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
    _ensure_db_initialized(db_path)
    conn = connect_db(db_path)
    try:
        row = conn.execute(query, params).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def _read_all(db_path: str | Path, query: str, params: tuple) -> list[dict]:
    """Run a multi-row exact read, returning a list of plain dicts."""
    _ensure_db_initialized(db_path)
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


# --- Planning proposal-surface helpers (§2.7) -----------------------------------
# Persistence primitives over the five planning tables: the macro-outline approval
# flow and the annotation-driven revision loop. These are a *proposal surface* — they
# never promote nodes into the canonical narrative tables or into committed prose.
# Writes are idempotent/replay-safe per logical operation: snapshot/node writes upsert
# by primary key; annotation/revision/trace inserts use ON CONFLICT DO NOTHING so a
# replay of the same logical write never duplicates or silently overwrites a prior row
# (a revision must never overwrite — §2.7 lifecycle). The annotation compiler, conflict
# detection, planner loop, validators, and tool registry are the M05 library, not here.


def _now_iso() -> str:
    """Current UTC timestamp as an ISO-8601 string (matches the relational writers)."""
    return datetime.now(timezone.utc).isoformat()


def _planning_write(db_path: str | Path, fn):
    """Run ``fn(conn)`` inside one explicit transaction on the planning DB.

    Initializes the schema if needed, opens a connection (FKs ON), runs ``fn`` under a
    single ``BEGIN``/commit, rolls back and re-raises on any error, and always closes
    the connection. ``fn`` returns the helper's result (typically a re-read row dict).
    """
    _ensure_db_initialized(db_path)
    conn = connect_db(db_path)
    try:
        conn.execute("BEGIN")
        result = fn(conn)
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _fetch_one(conn: sqlite3.Connection, query: str, params: dict) -> dict | None:
    """Single-row read on an open connection, returning a plain dict or None."""
    row = conn.execute(query, params).fetchone()
    return dict(row) if row is not None else None


# --- PlanningSnapshot ------------------------------------------------------------
# Upsert keyed by snapshot_id. A create replay may refresh stable identity fields, but
# lifecycle state belongs to the dedicated transition/revision helpers and is never
# rewound by this helper after the snapshot has advanced.
_PLANNING_SNAPSHOT_UPSERT = """
    INSERT INTO PlanningSnapshot
        (snapshot_id, project_id, mode, status, created_at, approved_at, active_revision_id)
    VALUES
        (:snapshot_id, :project_id, :mode, :status, :created_at, :approved_at, :active_revision_id)
    ON CONFLICT(snapshot_id) DO UPDATE SET
        project_id = excluded.project_id,
        mode = excluded.mode
"""


def create_planning_snapshot(
    db_path: str | Path,
    *,
    snapshot_id: str,
    project_id: str,
    mode: str,
    status: str = "draft",
    created_at: str | None = None,
    approved_at: str | None = None,
    active_revision_id: str | None = None,
) -> dict:
    """Create (or replay-upsert) a PlanningSnapshot, returning the stored row.

    Idempotent keyed by ``snapshot_id``: re-running with the same arguments leaves a
    single row and preserves the original ``created_at``. On conflict, this helper
    refreshes only stable identity fields (``project_id`` and ``mode``); lifecycle
    fields (``status``, ``approved_at``, ``active_revision_id``) are preserved so a
    create replay cannot rewind approval or revision state. ``status`` is checked at
    the store level against the planning vocabulary; this helper does not duplicate
    that enum in Python.
    """
    params = {
        "snapshot_id": snapshot_id,
        "project_id": project_id,
        "mode": mode,
        "status": status,
        "created_at": created_at if created_at is not None else _now_iso(),
        "approved_at": approved_at,
        "active_revision_id": active_revision_id,
    }

    def _do(conn: sqlite3.Connection) -> dict:
        conn.execute(_PLANNING_SNAPSHOT_UPSERT, params)
        return _fetch_one(
            conn,
            "SELECT * FROM PlanningSnapshot WHERE snapshot_id = :snapshot_id",
            {"snapshot_id": snapshot_id},
        )

    return _planning_write(db_path, _do)


def transition_snapshot_status(
    db_path: str | Path,
    snapshot_id: str,
    *,
    status: str,
    approved_at: str | None = None,
    has_unresolved_conflict: bool = False,
) -> dict | None:
    """Transition a snapshot's ``status``, returning the updated row (None if absent).

    The status vocabulary is enforced by the table ``CHECK`` (an invalid value raises
    ``sqlite3.IntegrityError``). When ``status == 'approved'`` the ``approved_at``
    stamp is set once (defaulting to now) and preserved on replay. Moving out of
    ``approved`` clears the approval stamp. This helper persists state only — the real
    approval gate lives in M05/M01; if it is handed ``has_unresolved_conflict=True``
    while asked to set ``approved``, it raises ``ValueError`` rather than persisting an
    approval that a ``needs_clarification`` annotation should still be blocking.
    Idempotent: re-applying the same transition does not restamp approval time.
    """
    if status == "approved" and has_unresolved_conflict:
        raise ValueError(
            "cannot set status='approved' while an unresolved (needs_clarification) "
            "conflict is flagged"
        )

    if status == "approved":
        params = {
            "snapshot_id": snapshot_id,
            "status": status,
            "approved_at": approved_at if approved_at is not None else _now_iso(),
        }
        sql = (
            "UPDATE PlanningSnapshot SET status = :status, "
            "approved_at = COALESCE(approved_at, :approved_at) "
            "WHERE snapshot_id = :snapshot_id"
        )
    else:
        params = {"snapshot_id": snapshot_id, "status": status}
        sql = (
            "UPDATE PlanningSnapshot SET status = :status, approved_at = NULL "
            "WHERE snapshot_id = :snapshot_id"
        )

    def _do(conn: sqlite3.Connection) -> dict | None:
        conn.execute(sql, params)
        return _fetch_one(
            conn,
            "SELECT * FROM PlanningSnapshot WHERE snapshot_id = :snapshot_id",
            {"snapshot_id": snapshot_id},
        )

    return _planning_write(db_path, _do)


def set_snapshot_active_revision(
    db_path: str | Path,
    snapshot_id: str,
    revision_id: str | None,
) -> dict | None:
    """Point a snapshot's ``active_revision_id`` at ``revision_id`` (or NULL to clear).

    Idempotent UPDATE; the FK requires a non-NULL ``revision_id`` to already exist in
    ``PlanningRevision``. Returns the updated row, or None if the snapshot is absent.
    """

    def _do(conn: sqlite3.Connection) -> dict | None:
        conn.execute(
            "UPDATE PlanningSnapshot SET active_revision_id = :revision_id "
            "WHERE snapshot_id = :snapshot_id",
            {"snapshot_id": snapshot_id, "revision_id": revision_id},
        )
        return _fetch_one(
            conn,
            "SELECT * FROM PlanningSnapshot WHERE snapshot_id = :snapshot_id",
            {"snapshot_id": snapshot_id},
        )

    return _planning_write(db_path, _do)


def get_planning_snapshot(db_path: str | Path, snapshot_id: str) -> dict | None:
    """Return the PlanningSnapshot row for ``snapshot_id`` as a dict, or None."""
    return _read_one(
        db_path,
        "SELECT * FROM PlanningSnapshot WHERE snapshot_id = ?",
        (snapshot_id,),
    )


def get_planning_snapshots(db_path: str | Path) -> list[dict]:
    """Return all PlanningSnapshot rows, newest-first by ``created_at``.

    ``snapshot_id`` is the deterministic secondary sort so equal timestamps
    still read back in a stable order. Lets an observer surface find the
    latest persisted snapshot without tracking ids in memory.
    """
    return _read_all(
        db_path,
        "SELECT * FROM PlanningSnapshot ORDER BY created_at DESC, snapshot_id DESC",
        (),
    )


# --- PlanningNode ----------------------------------------------------------------
# Upsert keyed by node_id so a replanned node updates in place rather than duplicating.
_PLANNING_NODE_UPSERT = """
    INSERT INTO PlanningNode
        (node_id, snapshot_id, level, parent_id, ordering, title, summary, purpose,
         status, locked_pinned)
    VALUES
        (:node_id, :snapshot_id, :level, :parent_id, :ordering, :title, :summary,
         :purpose, :status, :locked_pinned)
    ON CONFLICT(node_id) DO UPDATE SET
        snapshot_id = excluded.snapshot_id,
        level = excluded.level,
        parent_id = excluded.parent_id,
        ordering = excluded.ordering,
        title = excluded.title,
        summary = excluded.summary,
        purpose = excluded.purpose,
        status = excluded.status,
        locked_pinned = excluded.locked_pinned
"""


def upsert_planning_node(
    db_path: str | Path,
    *,
    node_id: str,
    snapshot_id: str,
    level: str,
    status: str,
    parent_id: str | None = None,
    ordering: int = 0,
    title: str | None = None,
    summary: str | None = None,
    purpose: str | None = None,
    locked_pinned: bool = False,
) -> dict:
    """Idempotently upsert a PlanningNode keyed by ``node_id``, returning the row.

    ``level`` is enforced by the table ``CHECK``. ``locked_pinned`` is stored as 0/1;
    its conflict-on-removal semantics are M05's concern, this layer only persists the
    flag. A replay of the same logical node leaves exactly one row.
    """
    params = {
        "node_id": node_id,
        "snapshot_id": snapshot_id,
        "level": level,
        "parent_id": parent_id,
        "ordering": ordering,
        "title": title,
        "summary": summary,
        "purpose": purpose,
        "status": status,
        "locked_pinned": int(bool(locked_pinned)),
    }

    def _do(conn: sqlite3.Connection) -> dict:
        conn.execute(_PLANNING_NODE_UPSERT, params)
        return _fetch_one(
            conn,
            "SELECT * FROM PlanningNode WHERE node_id = :node_id",
            {"node_id": node_id},
        )

    return _planning_write(db_path, _do)


def update_planning_node_status(
    db_path: str | Path,
    node_id: str,
    status: str,
) -> dict | None:
    """Update a PlanningNode's ``status`` (idempotent), returning the row or None."""

    def _do(conn: sqlite3.Connection) -> dict | None:
        conn.execute(
            "UPDATE PlanningNode SET status = :status WHERE node_id = :node_id",
            {"node_id": node_id, "status": status},
        )
        return _fetch_one(
            conn,
            "SELECT * FROM PlanningNode WHERE node_id = :node_id",
            {"node_id": node_id},
        )

    return _planning_write(db_path, _do)


def set_planning_node_locked(
    db_path: str | Path,
    node_id: str,
    locked: bool,
) -> dict | None:
    """Set or clear a PlanningNode's ``locked_pinned`` flag, returning the row or None.

    Persists the user-pin flag only; emitting a ``raise_conflict`` when a planner
    revision would remove a pinned node is M05 behavior, not enforced here.
    """

    def _do(conn: sqlite3.Connection) -> dict | None:
        conn.execute(
            "UPDATE PlanningNode SET locked_pinned = :locked WHERE node_id = :node_id",
            {"node_id": node_id, "locked": int(bool(locked))},
        )
        return _fetch_one(
            conn,
            "SELECT * FROM PlanningNode WHERE node_id = :node_id",
            {"node_id": node_id},
        )

    return _planning_write(db_path, _do)


def get_planning_nodes(
    db_path: str | Path,
    snapshot_id: str,
    *,
    level: str | None = None,
) -> list[dict]:
    """Return a snapshot's PlanningNodes ordered by ``ordering``, optionally one level.

    ``level=None`` returns every node under ``snapshot_id``; otherwise only that level.
    Ordered by ``ordering ASC`` then ``node_id`` for deterministic output.
    """
    if level is None:
        return _read_all(
            db_path,
            "SELECT * FROM PlanningNode WHERE snapshot_id = ? "
            "ORDER BY ordering ASC, node_id ASC",
            (snapshot_id,),
        )
    return _read_all(
        db_path,
        "SELECT * FROM PlanningNode WHERE snapshot_id = ? AND level = ? "
        "ORDER BY ordering ASC, node_id ASC",
        (snapshot_id, level),
    )


def get_planning_nodes_by_parent(
    db_path: str | Path,
    snapshot_id: str,
    parent_id: str | None,
) -> list[dict]:
    """Return snapshot-scoped PlanningNodes whose parent is ``parent_id``.

    ``parent_id=None`` selects root nodes (``parent_id IS NULL``); ``IS ?`` binds NULL
    correctly so the same helper serves both root and child traversal.
    """
    return _read_all(
        db_path,
        "SELECT * FROM PlanningNode WHERE snapshot_id = ? AND parent_id IS ? "
        "ORDER BY ordering ASC, node_id ASC",
        (snapshot_id, parent_id),
    )


# --- PlanningAnnotation ----------------------------------------------------------
# Insert is append-only and history-preserving: ON CONFLICT(annotation_id) DO NOTHING
# so a replay never duplicates and never overwrites an existing annotation. Annotations
# are never deleted — status transitions record outcomes for the diff/UI.
_PLANNING_ANNOTATION_INSERT = """
    INSERT INTO PlanningAnnotation
        (annotation_id, snapshot_id, target_node_id, target_level, note_type, scope,
         priority, text, status, planner_response, created_at, resolved_at)
    VALUES
        (:annotation_id, :snapshot_id, :target_node_id, :target_level, :note_type,
         :scope, :priority, :text, :status, :planner_response, :created_at,
         :resolved_at)
    ON CONFLICT(annotation_id) DO NOTHING
"""


def insert_planning_annotation(
    db_path: str | Path,
    *,
    annotation_id: str,
    snapshot_id: str,
    target_node_id: str,
    target_level: str,
    note_type: str,
    scope: str,
    priority: str,
    text: str,
    status: str = "pending",
    planner_response: str | None = None,
    created_at: str | None = None,
    resolved_at: str | None = None,
) -> dict:
    """Insert a PlanningAnnotation (append-only), returning the stored row.

    ``note_type``/``scope``/``priority``/``status``/``target_level`` are all enforced
    by table ``CHECK`` sets. Idempotent keyed by ``annotation_id`` via
    ``ON CONFLICT DO NOTHING`` so a replay neither duplicates nor overwrites; updates go
    through ``update_annotation_status``. Annotations are never deleted (history is kept
    for the revision diff/UI).
    """
    params = {
        "annotation_id": annotation_id,
        "snapshot_id": snapshot_id,
        "target_node_id": target_node_id,
        "target_level": target_level,
        "note_type": note_type,
        "scope": scope,
        "priority": priority,
        "text": text,
        "status": status,
        "planner_response": planner_response,
        "created_at": created_at if created_at is not None else _now_iso(),
        "resolved_at": resolved_at,
    }

    def _do(conn: sqlite3.Connection) -> dict:
        conn.execute(_PLANNING_ANNOTATION_INSERT, params)
        return _fetch_one(
            conn,
            "SELECT * FROM PlanningAnnotation WHERE annotation_id = :annotation_id",
            {"annotation_id": annotation_id},
        )

    return _planning_write(db_path, _do)


def update_annotation_status(
    db_path: str | Path,
    annotation_id: str,
    *,
    status: str,
    planner_response: str | None = None,
    resolved_at: str | None = None,
) -> dict | None:
    """Update a PlanningAnnotation's outcome fields, returning the row or None.

    Sets ``status`` (enforced by the table ``CHECK``) plus the ``planner_response`` and
    ``resolved_at`` outcome fields in place. Idempotent; never deletes the annotation.
    """

    def _do(conn: sqlite3.Connection) -> dict | None:
        conn.execute(
            "UPDATE PlanningAnnotation SET status = :status, "
            "planner_response = :planner_response, resolved_at = :resolved_at "
            "WHERE annotation_id = :annotation_id",
            {
                "annotation_id": annotation_id,
                "status": status,
                "planner_response": planner_response,
                "resolved_at": resolved_at,
            },
        )
        return _fetch_one(
            conn,
            "SELECT * FROM PlanningAnnotation WHERE annotation_id = :annotation_id",
            {"annotation_id": annotation_id},
        )

    return _planning_write(db_path, _do)


def get_annotations_for_node(db_path: str | Path, target_node_id: str) -> list[dict]:
    """Return annotations targeting ``target_node_id``, oldest-first by ``created_at``."""
    return _read_all(
        db_path,
        "SELECT * FROM PlanningAnnotation WHERE target_node_id = ? "
        "ORDER BY created_at ASC, annotation_id ASC",
        (target_node_id,),
    )


def get_annotations_for_snapshot(db_path: str | Path, snapshot_id: str) -> list[dict]:
    """Return all annotations under ``snapshot_id``, oldest-first by ``created_at``."""
    return _read_all(
        db_path,
        "SELECT * FROM PlanningAnnotation WHERE snapshot_id = ? "
        "ORDER BY created_at ASC, annotation_id ASC",
        (snapshot_id,),
    )


# --- PlanningRevision ------------------------------------------------------------
# A revision is never a silent overwrite (§2.7): insert is ON CONFLICT DO NOTHING and
# the prior revision row is preserved. The owning snapshot's active_revision_id advances
# only when this call inserted a new revision, so replaying an older revision is a no-op.
_PLANNING_REVISION_INSERT = """
    INSERT INTO PlanningRevision
        (revision_id, snapshot_id, parent_revision_id, change_summary, diff_json,
         created_at)
    VALUES
        (:revision_id, :snapshot_id, :parent_revision_id, :change_summary, :diff_json,
         :created_at)
    ON CONFLICT(revision_id) DO NOTHING
"""


def insert_planning_revision(
    db_path: str | Path,
    *,
    revision_id: str,
    snapshot_id: str,
    diff_json: str,
    parent_revision_id: str | None = None,
    change_summary: str | None = None,
    created_at: str | None = None,
) -> dict:
    """Insert a PlanningRevision and advance the snapshot's ``active_revision_id``.

    Both the revision insert and the ``active_revision_id`` advance happen in one
    transaction, so a newly inserted revision becomes active while every prior revision
    row is preserved (a revision never silently overwrites another). Idempotent keyed by
    ``revision_id``: a replay inserts nothing new and leaves the active pointer where it
    already is. Returns the stored revision row.
    """
    params = {
        "revision_id": revision_id,
        "snapshot_id": snapshot_id,
        "parent_revision_id": parent_revision_id,
        "change_summary": change_summary,
        "diff_json": diff_json,
        "created_at": created_at if created_at is not None else _now_iso(),
    }

    def _do(conn: sqlite3.Connection) -> dict:
        cursor = conn.execute(_PLANNING_REVISION_INSERT, params)
        if cursor.rowcount == 1:
            conn.execute(
                "UPDATE PlanningSnapshot SET active_revision_id = :revision_id "
                "WHERE snapshot_id = :snapshot_id",
                {"revision_id": revision_id, "snapshot_id": snapshot_id},
            )
        return _fetch_one(
            conn,
            "SELECT * FROM PlanningRevision WHERE revision_id = :revision_id",
            {"revision_id": revision_id},
        )

    return _planning_write(db_path, _do)


def get_revisions_for_snapshot(
    db_path: str | Path,
    snapshot_id: str,
    *,
    newest_first: bool = True,
) -> list[dict]:
    """Return a snapshot's revisions ordered by ``created_at`` (newest-first default).

    ``newest_first=False`` returns them oldest-first. ``revision_id`` is the secondary
    sort key for deterministic ordering when timestamps collide.
    """
    direction = "DESC" if newest_first else "ASC"
    return _read_all(
        db_path,
        f"SELECT * FROM PlanningRevision WHERE snapshot_id = ? "
        f"ORDER BY created_at {direction}, revision_id {direction}",
        (snapshot_id,),
    )


# --- PlannerToolCallTrace --------------------------------------------------------
# Append-only observability rows for the planner loop. Idempotent keyed by trace_id.
_PLANNER_TRACE_INSERT = """
    INSERT INTO PlannerToolCallTrace
        (trace_id, snapshot_id, planner_level, planner_node_id, loop_index, tool_name,
         tool_args_json, result_summary, success, error, created_at)
    VALUES
        (:trace_id, :snapshot_id, :planner_level, :planner_node_id, :loop_index,
         :tool_name, :tool_args_json, :result_summary, :success, :error, :created_at)
    ON CONFLICT(trace_id) DO NOTHING
"""


def insert_planner_tool_call_trace(
    db_path: str | Path,
    *,
    trace_id: str,
    planner_level: str,
    loop_index: int,
    tool_name: str,
    success: bool,
    snapshot_id: str | None = None,
    planner_node_id: str | None = None,
    tool_args_json: str | None = None,
    result_summary: str | None = None,
    error: str | None = None,
    created_at: str | None = None,
) -> dict:
    """Insert a PlannerToolCallTrace observability row, returning the stored row.

    ``planner_level`` is enforced by the table ``CHECK``; ``success`` is stored as 0/1.
    Idempotent keyed by ``trace_id`` via ``ON CONFLICT DO NOTHING`` so a replay of the
    same logical trace never duplicates.
    """
    params = {
        "trace_id": trace_id,
        "snapshot_id": snapshot_id,
        "planner_level": planner_level,
        "planner_node_id": planner_node_id,
        "loop_index": loop_index,
        "tool_name": tool_name,
        "tool_args_json": tool_args_json,
        "result_summary": result_summary,
        "success": int(bool(success)),
        "error": error,
        "created_at": created_at if created_at is not None else _now_iso(),
    }

    def _do(conn: sqlite3.Connection) -> dict:
        conn.execute(_PLANNER_TRACE_INSERT, params)
        return _fetch_one(
            conn,
            "SELECT * FROM PlannerToolCallTrace WHERE trace_id = :trace_id",
            {"trace_id": trace_id},
        )

    return _planning_write(db_path, _do)


def get_traces_for_snapshot(
    db_path: str | Path,
    snapshot_id: str,
    *,
    planner_level: str | None = None,
    loop_index: int | None = None,
) -> list[dict]:
    """Return tool-call traces for ``snapshot_id``, optionally filtered.

    Optional ``planner_level`` and ``loop_index`` narrow the result. Ordered by
    ``loop_index`` then ``created_at`` then ``trace_id`` for deterministic output.
    """
    clauses = ["snapshot_id = ?"]
    params: list = [snapshot_id]
    if planner_level is not None:
        clauses.append("planner_level = ?")
        params.append(planner_level)
    if loop_index is not None:
        clauses.append("loop_index = ?")
        params.append(loop_index)
    where = " AND ".join(clauses)
    return _read_all(
        db_path,
        f"SELECT * FROM PlannerToolCallTrace WHERE {where} "
        f"ORDER BY loop_index ASC, created_at ASC, trace_id ASC",
        tuple(params),
    )


# --- Plan-time outline writers (§2.1 narrative rows, structural scaffolding) ------
# These persist *validated planner structure* into the canonical narrative tables:
# the minimal structural identity of an arc/chapter/scene/beat. They are distinct from
# the commit-time path (`upsert_beat_commit` / `node_commit_transaction`, M10): they
# never write draft or committed prose. A plan-time Beats row carries only structural
# columns (id/scene_id/beat_index/status); `prose`/`word_count`/`committed_at` are left
# untouched so a later commit fills them without the planner ever overwriting prose.
#
# The narrative tables hold only minimal structural columns, so the richer planner
# detail the design routes to the proposal surface — chapter obligations (dramatic
# function / expected emotional shift / required thread progress / scene-planning
# constraints) and the tailored beat PAD behavioral-constraint string — is written into
# a paired ``PlanningNode`` row (``level='chapter'``/``'beat'``) in the *same DB*, per
# LangGraph_Nodes.md (node_plan_chapter writes "Chapter rows … and PlanningNode rows at
# level='chapter'"; node_plan_beat writes "Beat Nodes … including the tailored PAD
# behavioral constraint string"). All writers upsert by primary key, so a replay of the
# same logical plan write never duplicates a row.
_DEFAULT_PLAN_STATUS = "planned"  # narrative status vocabulary: planned/active/completed

_ARC_PLAN_UPSERT = """
    INSERT INTO Arcs (id, description, status)
    VALUES (:id, :description, :status)
    ON CONFLICT(id) DO UPDATE SET
        description = excluded.description,
        status = excluded.status
"""

_CHAPTER_PLAN_UPSERT = """
    INSERT INTO Chapters (id, arc_id, description, status)
    VALUES (:id, :arc_id, :description, :status)
    ON CONFLICT(id) DO UPDATE SET
        arc_id = excluded.arc_id,
        description = excluded.description,
        status = excluded.status
"""

_SCENE_PLAN_UPSERT = """
    INSERT INTO Scenes (id, chapter_id, description, word_budget, ordering, status)
    VALUES (:id, :chapter_id, :description, :word_budget, :ordering, :status)
    ON CONFLICT(id) DO UPDATE SET
        chapter_id = excluded.chapter_id,
        description = excluded.description,
        word_budget = excluded.word_budget,
        ordering = excluded.ordering,
        status = excluded.status
"""

# Structural beat columns only — deliberately omits prose/word_count/committed_at so a
# plan-time write never clobbers committed prose and a later commit upsert is unaffected.
_BEAT_PLAN_UPSERT = """
    INSERT INTO Beats (id, scene_id, beat_index, status)
    VALUES (:id, :scene_id, :beat_index, :status)
    ON CONFLICT(id) DO UPDATE SET
        scene_id = excluded.scene_id,
        beat_index = excluded.beat_index,
        status = excluded.status
"""


def upsert_arc_plan(
    db_path: str | Path,
    *,
    arc_id: str,
    description: str,
    status: str = _DEFAULT_PLAN_STATUS,
) -> dict:
    """Idempotently persist a validated arc-plan row, returning the stored Arc dict.

    Writes the minimal `Arcs` structural row keyed by ``arc_id``. ``status`` is enforced
    by the table ``CHECK``. Structural scaffolding only — no prose, no commit fields.
    """
    params = {"id": arc_id, "description": description, "status": status}

    def _do(conn: sqlite3.Connection) -> dict:
        conn.execute(_ARC_PLAN_UPSERT, params)
        return _fetch_one(conn, "SELECT * FROM Arcs WHERE id = :id", {"id": arc_id})

    return _planning_write(db_path, _do)


def upsert_chapter_plan(
    db_path: str | Path,
    *,
    chapter_id: str,
    arc_id: str,
    description: str,
    status: str = _DEFAULT_PLAN_STATUS,
    snapshot_id: str | None = None,
    node_id: str | None = None,
    parent_node_id: str | None = None,
    ordering: int = 0,
    title: str | None = None,
    dramatic_function: str | None = None,
    expected_emotional_shift: str | None = None,
    required_thread_progress: str | None = None,
    scene_planning_constraints: str | None = None,
    locked_pinned: bool = False,
) -> dict:
    """Persist a validated chapter-plan row plus its obligations, returning the dict.

    Writes the minimal `Chapters` structural row keyed by ``chapter_id``. The chapter's
    obligations — dramatic function, expected emotional shift, required thread progress,
    and scene-planning constraints — are persisted to a paired ``PlanningNode``
    (``level='chapter'``) in the same DB, serialized as JSON in the node's ``purpose``
    (the design routes this detail to the proposal surface, not the narrative tables).

    Pass ``snapshot_id`` and ``node_id`` to write that PlanningNode. If any obligation
    field is supplied without both, a ``ValueError`` is raised rather than silently
    dropping the obligations. The returned dict is the Chapters row, with the stored
    PlanningNode row under a ``"planning_node"`` key when one was written. Idempotent:
    both rows upsert by primary key, so a replay never duplicates. Structural/proposal
    scaffolding only — never narrative prose.
    """
    obligations = {
        "dramatic_function": dramatic_function,
        "expected_emotional_shift": expected_emotional_shift,
        "required_thread_progress": required_thread_progress,
        "scene_planning_constraints": scene_planning_constraints,
    }
    has_obligations = any(v is not None for v in obligations.values())
    write_node = snapshot_id is not None and node_id is not None
    if has_obligations and not write_node:
        raise ValueError(
            "chapter obligations require snapshot_id and node_id (they are persisted to "
            "the PlanningNode proposal surface, not the narrative Chapters row)"
        )

    chapter_params = {
        "id": chapter_id,
        "arc_id": arc_id,
        "description": description,
        "status": status,
    }
    node_params = None
    if write_node:
        node_params = {
            "node_id": node_id,
            "snapshot_id": snapshot_id,
            "level": "chapter",
            "parent_id": parent_node_id,
            "ordering": ordering,
            "title": title if title is not None else description,
            "summary": dramatic_function,
            "purpose": json.dumps(obligations),
            "status": status,
            "locked_pinned": int(bool(locked_pinned)),
        }

    def _do(conn: sqlite3.Connection) -> dict:
        conn.execute(_CHAPTER_PLAN_UPSERT, chapter_params)
        result = _fetch_one(
            conn, "SELECT * FROM Chapters WHERE id = :id", {"id": chapter_id}
        )
        if node_params is not None:
            conn.execute(_PLANNING_NODE_UPSERT, node_params)
            result["planning_node"] = _fetch_one(
                conn,
                "SELECT * FROM PlanningNode WHERE node_id = :node_id",
                {"node_id": node_id},
            )
        return result

    return _planning_write(db_path, _do)


def upsert_scene_plan(
    db_path: str | Path,
    *,
    scene_id: str,
    chapter_id: str,
    description: str,
    ordering: int,
    word_budget: int = 0,
    status: str = _DEFAULT_PLAN_STATUS,
) -> dict:
    """Idempotently persist a validated scene-plan row, returning the stored Scene dict.

    Writes the `Scenes` structural row keyed by ``scene_id`` and sets the explicit
    ``ordering`` sort key (read paths sort scenes by ``ordering ASC``, not creation
    time). Structural scaffolding only — no prose, no commit fields.
    """
    params = {
        "id": scene_id,
        "chapter_id": chapter_id,
        "description": description,
        "word_budget": word_budget,
        "ordering": ordering,
        "status": status,
    }

    def _do(conn: sqlite3.Connection) -> dict:
        conn.execute(_SCENE_PLAN_UPSERT, params)
        return _fetch_one(conn, "SELECT * FROM Scenes WHERE id = :id", {"id": scene_id})

    return _planning_write(db_path, _do)


def upsert_beat_plan(
    db_path: str | Path,
    *,
    beat_id: str,
    scene_id: str,
    beat_index: int,
    status: str = _DEFAULT_PLAN_STATUS,
    snapshot_id: str | None = None,
    node_id: str | None = None,
    parent_node_id: str | None = None,
    ordering: int = 0,
    title: str | None = None,
    pad_constraint: str | None = None,
    immediate_objective: str | None = None,
    physical_constraints: str | None = None,
    locked_pinned: bool = False,
) -> dict:
    """Persist a validated beat-plan row plus its PAD constraint, returning the dict.

    Writes the minimal `Beats` structural row keyed by ``beat_id`` (id/scene_id/
    beat_index/status only — ``prose``/``word_count``/``committed_at`` are left for the
    commit path and never touched here). The tailored PAD behavioral-constraint string
    (with the beat's immediate objective and physical constraints) is persisted to a
    paired ``PlanningNode`` (``level='beat'``) in the same DB, serialized as JSON in the
    node's ``purpose``.

    Pass ``snapshot_id`` and ``node_id`` to write that PlanningNode. If ``pad_constraint``
    (or the other beat detail fields) is supplied without both, a ``ValueError`` is
    raised rather than silently dropping it. The returned dict is the Beats row, with the
    stored PlanningNode under a ``"planning_node"`` key when one was written. Idempotent:
    both rows upsert by primary key. Structural/proposal scaffolding only — never prose.
    """
    beat_detail = {
        "immediate_objective": immediate_objective,
        "physical_constraints": physical_constraints,
        "pad_constraint": pad_constraint,
    }
    has_detail = any(v is not None for v in beat_detail.values())
    write_node = snapshot_id is not None and node_id is not None
    if has_detail and not write_node:
        raise ValueError(
            "beat PAD/objective detail requires snapshot_id and node_id (it is persisted "
            "to the PlanningNode proposal surface, not the narrative Beats row)"
        )

    beat_params = {
        "id": beat_id,
        "scene_id": scene_id,
        "beat_index": beat_index,
        "status": status,
    }
    node_params = None
    if write_node:
        node_params = {
            "node_id": node_id,
            "snapshot_id": snapshot_id,
            "level": "beat",
            "parent_id": parent_node_id,
            "ordering": ordering,
            "title": title,
            "summary": immediate_objective,
            "purpose": json.dumps(beat_detail),
            "status": status,
            "locked_pinned": int(bool(locked_pinned)),
        }

    def _do(conn: sqlite3.Connection) -> dict:
        conn.execute(_BEAT_PLAN_UPSERT, beat_params)
        result = _fetch_one(conn, "SELECT * FROM Beats WHERE id = :id", {"id": beat_id})
        if node_params is not None:
            conn.execute(_PLANNING_NODE_UPSERT, node_params)
            result["planning_node"] = _fetch_one(
                conn,
                "SELECT * FROM PlanningNode WHERE node_id = :node_id",
                {"node_id": node_id},
            )
        return result

    return _planning_write(db_path, _do)
