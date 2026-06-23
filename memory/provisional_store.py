"""Module: M02 (Persistent Memory Stores & Interfaces)
Hold unconfirmed ingestion/coreference claims apart from canonical truth.

The historical ingestion loop (M12) imputes ambiguous coreference links below the
genre certainty floor and saves them as *provisional* beliefs. This store keeps
those claims physically separate from the canonical relational hub
(`memory/sqlite_db.py` → `data/fictionwriter.db`) so an unconfirmed assumption can
never contaminate ground truth. Context assembly (M03) later injects them by
confidence tier, and node_commit_transaction / the optional Alignment Dashboard
resolve them at chapter boundaries — none of that lives here.

Format: a dedicated **SQLite file** (default `data/provisional_claims.db`),
distinct from the canonical DB. A single `ProvisionalClaims` table is keyed by a
stable `claim_id`; `upsert_claim` is idempotent so re-ingesting the same span
updates the existing row in place rather than duplicating it.

Confidence is stored exactly as supplied. This module does NOT interpret the
high/mid/low confidence thresholds — confidence-band reads take their boundaries
from the caller, and the floor itself is a config-driven decision owned by
context-assembly callers, not the storage boundary.

Review state transitions (reviewed/confirmed/rejected) update a claim in place and
never delete it, so history is preserved for the optional Alignment UI and later
reconciliation.

Scope: storage and read/review-state only. Alignment UI routes, the ingestion
pipeline, Graphiti/SQLite belief promotion (M10, at chapter boundaries), and
chapter-boundary belief resolution are out of scope and live in their own modules.
"""

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# Default file is intentionally a separate artifact from the canonical
# `data/fictionwriter.db`; callers may override the path per project/test.
DEFAULT_PROVISIONAL_PATH = "data/provisional_claims.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ProvisionalClaims (
    claim_id TEXT PRIMARY KEY,
    source_ref TEXT,
    subject_id TEXT,
    entity_id TEXT,
    claim_text TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    reviewer_note TEXT,
    reviewed_at TEXT
)
"""

# Status vocabulary. `provisional` is the unreviewed default (a "pending" claim);
# the review terminals record a human/automated decision without deleting the row.
# These are state labels, not numeric thresholds.
PENDING_STATUS = "provisional"
REVIEW_STATUSES = ("reviewed", "confirmed", "rejected")


def _connect(path: str | Path) -> sqlite3.Connection:
    """Open a sqlite3 connection to the separate provisional-claims file.

    Creates the parent directory if missing and sets ``row_factory`` for
    name-addressable rows. Local-file only — no shared handle with the canonical
    relational hub.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the table if absent and add any missing columns in place.

    Idempotent and tolerant of an older table (e.g. one created before the
    ``reviewed_at`` column existed): the column is added without rewriting rows,
    so history is preserved.
    """
    conn.execute(_SCHEMA)
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(ProvisionalClaims)")}
    if "reviewed_at" not in existing:
        conn.execute("ALTER TABLE ProvisionalClaims ADD COLUMN reviewed_at TEXT")


def init_provisional_store(path: str | Path = DEFAULT_PROVISIONAL_PATH) -> None:
    """Create the provisional-claims table if it does not exist.

    Idempotent: safe to call on every startup. Establishes the store as a file
    separate from the canonical relational hub.
    """
    conn = _connect(path)
    try:
        _ensure_schema(conn)
        conn.commit()
    finally:
        conn.close()


def _derive_claim_id(
    source_ref: str | None,
    subject_id: str | None,
    entity_id: str | None,
    claim_text: str,
) -> str:
    """Derive a deterministic content key when no explicit claim_id is supplied.

    Hashes the identifying content so re-ingesting the same imputed link yields
    the same key, keeping the upsert idempotent without a caller-managed ID.
    """
    canonical = "\x1f".join(
        "" if part is None else str(part)
        for part in (source_ref, subject_id, entity_id, claim_text)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def upsert_claim(
    path: str | Path,
    *,
    claim_text: str,
    confidence: float,
    claim_id: str | None = None,
    source_ref: str | None = None,
    subject_id: str | None = None,
    entity_id: str | None = None,
    status: str = "provisional",
    reviewer_note: str | None = None,
) -> str:
    """Insert or update a single provisional claim, idempotently.

    Keyed by ``claim_id``; when omitted, a deterministic content key is derived
    from the source reference, subject/entity IDs, and claim text so replaying the
    same imputed link never duplicates a row. On conflict the mutable fields are
    updated in place while ``created_at`` is preserved from first write.

    ``confidence`` is stored verbatim — this boundary does not compare it to any
    high/mid/low threshold. Returns the resolved ``claim_id``.
    """
    resolved_id = claim_id or _derive_claim_id(
        source_ref, subject_id, entity_id, claim_text
    )
    created_at = datetime.now(timezone.utc).isoformat()

    conn = _connect(path)
    try:
        _ensure_schema(conn)
        conn.execute(
            """
            INSERT INTO ProvisionalClaims (
                claim_id, source_ref, subject_id, entity_id, claim_text,
                confidence, status, created_at, reviewer_note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(claim_id) DO UPDATE SET
                source_ref = excluded.source_ref,
                subject_id = excluded.subject_id,
                entity_id = excluded.entity_id,
                claim_text = excluded.claim_text,
                confidence = excluded.confidence,
                status = excluded.status,
                reviewer_note = excluded.reviewer_note
            """,
            (
                resolved_id,
                source_ref,
                subject_id,
                entity_id,
                claim_text,
                confidence,
                status,
                created_at,
                reviewer_note,
            ),
        )
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        raise
    finally:
        conn.close()
    return resolved_id


def _rows_to_dicts(rows) -> list[dict]:
    """Return query rows as plain dicts (never raw ``sqlite3.Row``)."""
    return [dict(row) for row in rows]


def list_pending_claims(path: str | Path) -> list[dict]:
    """Return all unreviewed (`provisional`) claims as plain dicts.

    A missing store initializes cleanly and yields ``[]``. Ordered by confidence
    DESC then ``claim_id`` ASC for deterministic, replay-safe output.
    """
    conn = _connect(path)
    try:
        _ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT * FROM ProvisionalClaims
            WHERE status = ?
            ORDER BY confidence DESC, claim_id ASC
            """,
            (PENDING_STATUS,),
        ).fetchall()
    finally:
        conn.close()
    return _rows_to_dicts(rows)


def list_claims_by_confidence(
    path: str | Path,
    *,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
    status: str | None = None,
) -> list[dict]:
    """Return claims whose confidence falls in a caller-supplied band.

    Band boundaries are inclusive and entirely caller-driven — this store applies
    no high/mid/low interpretation. Either bound may be omitted to leave that side
    open. An optional ``status`` filter (e.g. ``PENDING_STATUS``) narrows the set.
    A missing store initializes cleanly and yields ``[]``. Ordered by confidence
    DESC then ``claim_id`` ASC for deterministic output.
    """
    clauses: list[str] = []
    params: list = []
    if min_confidence is not None:
        clauses.append("confidence >= ?")
        params.append(min_confidence)
    if max_confidence is not None:
        clauses.append("confidence <= ?")
        params.append(max_confidence)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    conn = _connect(path)
    try:
        _ensure_schema(conn)
        rows = conn.execute(
            f"SELECT * FROM ProvisionalClaims {where} "
            "ORDER BY confidence DESC, claim_id ASC",
            params,
        ).fetchall()
    finally:
        conn.close()
    return _rows_to_dicts(rows)


def get_claim(path: str | Path, claim_id: str) -> dict | None:
    """Return a single claim as a plain dict, or ``None`` if absent.

    A missing store initializes cleanly and yields ``None``.
    """
    conn = _connect(path)
    try:
        _ensure_schema(conn)
        row = conn.execute(
            "SELECT * FROM ProvisionalClaims WHERE claim_id = ?", (claim_id,)
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None


def mark_claim_reviewed(
    path: str | Path,
    claim_id: str,
    *,
    status: str,
    reviewer_note: str | None = None,
) -> dict:
    """Record a review decision on a claim in place — never deletes the row.

    ``status`` must be one of :data:`REVIEW_STATUSES` (``reviewed``/``confirmed``/
    ``rejected``); the row is updated, stamping ``reviewed_at`` and, when supplied,
    ``reviewer_note`` (an omitted note leaves the existing one untouched). History
    is preserved for the Alignment UI and later reconciliation. Re-applying the
    same decision is idempotent aside from the refreshed ``reviewed_at`` stamp.
    Returns the updated claim as a plain dict.

    Raises:
        ValueError: if ``status`` is not a recognized review status.
        KeyError: if no claim with ``claim_id`` exists.
    """
    if status not in REVIEW_STATUSES:
        raise ValueError(
            f"status must be one of {REVIEW_STATUSES}, got {status!r}"
        )
    reviewed_at = datetime.now(timezone.utc).isoformat()

    conn = _connect(path)
    try:
        _ensure_schema(conn)
        cursor = conn.execute(
            """
            UPDATE ProvisionalClaims
            SET status = ?,
                reviewer_note = COALESCE(?, reviewer_note),
                reviewed_at = ?
            WHERE claim_id = ?
            """,
            (status, reviewer_note, reviewed_at, claim_id),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"no provisional claim with claim_id {claim_id!r}")
        conn.commit()
        row = conn.execute(
            "SELECT * FROM ProvisionalClaims WHERE claim_id = ?", (claim_id,)
        ).fetchone()
    except sqlite3.Error:
        conn.rollback()
        raise
    finally:
        conn.close()
    return dict(row)


class ProvisionalStore:
    """Provisional fact store.

    STUB: legacy class placeholder, superseded by the module-level
    ``init_provisional_store`` / ``upsert_claim`` functions. Retained as an honest
    stub; not yet wired.
    """

    def add(self, *args, **kwargs):
        """Add a provisional fact."""
        raise NotImplementedError("STUB: use upsert_claim")
