"""Module: M02 (Persistent Memory Stores & Interfaces)
Synthetic tests for the provisional-claim store (`memory/provisional_store.py`,
exercising 04.05): idempotent upsert, pending listing, caller-supplied
confidence-band queries, and non-destructive review-state transitions.

All fixtures are file-local under ``tmp_path`` — these tests never write to the
real ``data/`` directory or the canonical ``fictionwriter.db``. Alignment UI,
ingestion, Graphiti promotion, and M10 belief resolution are out of scope here.
Band thresholds are always supplied by the caller; none are hardcoded.
"""

from core import runtime
from memory.provisional_store import (
    PENDING_STATUS,
    get_claim,
    init_provisional_store,
    list_claims_by_confidence,
    list_pending_claims,
    mark_claim_reviewed,
    upsert_claim,
)


def _store_path(tmp_path):
    """A nested temp provisional-store path, separate from any canonical DB."""
    return tmp_path / "data" / "provisional_claims.db"


def _all_claims(path):
    """Every stored claim (no band/status filter) — used for counting."""
    return list_claims_by_confidence(path)


def test_upsert_same_claim_twice_leaves_one_stored_claim(tmp_path):
    store = _store_path(tmp_path)
    # No explicit claim_id -> deterministic content key, so a replay collapses.
    fields = dict(
        claim_text="She is Mara",
        confidence=0.62,
        source_ref="ch3:span[1040:1052]",
        subject_id="pron_7",
        entity_id="char_mara",
    )
    id1 = upsert_claim(store, **fields)
    id2 = upsert_claim(store, **fields)

    assert id1 == id2
    assert len(_all_claims(store)) == 1


def test_list_pending_claims_returns_only_unreviewed(tmp_path):
    store = _store_path(tmp_path)
    upsert_claim(store, claim_id="p1", claim_text="maybe Tom", confidence=0.30)
    upsert_claim(store, claim_id="p2", claim_text="clearly Mara", confidence=0.90)
    # A claim that has been reviewed should drop out of the pending list.
    upsert_claim(store, claim_id="done1", claim_text="was Tom", confidence=0.70)
    mark_claim_reviewed(store, "done1", status="confirmed")

    pending = list_pending_claims(store)
    assert {c["claim_id"] for c in pending} == {"p1", "p2"}
    assert all(c["status"] == PENDING_STATUS for c in pending)


def test_confidence_band_query_uses_caller_supplied_thresholds(tmp_path):
    store = _store_path(tmp_path)
    upsert_claim(store, claim_id="low", claim_text="low", confidence=0.10)
    upsert_claim(store, claim_id="mid", claim_text="mid", confidence=0.62)
    upsert_claim(store, claim_id="high", claim_text="high", confidence=0.95)

    # Caller picks the band; the store applies no high/mid/low interpretation.
    mid_band = list_claims_by_confidence(store, min_confidence=0.50, max_confidence=0.80)
    assert {c["claim_id"] for c in mid_band} == {"mid"}

    # An open-ended upper bound, plus a status filter, both caller-driven.
    at_least_half = list_claims_by_confidence(
        store, min_confidence=0.50, status=PENDING_STATUS
    )
    assert {c["claim_id"] for c in at_least_half} == {"mid", "high"}

    # Inclusive boundaries: a band whose edges equal stored confidences includes them.
    inclusive = list_claims_by_confidence(store, min_confidence=0.10, max_confidence=0.95)
    assert {c["claim_id"] for c in inclusive} == {"low", "mid", "high"}


def test_mark_reviewed_preserves_record_and_stores_note(tmp_path):
    store = _store_path(tmp_path)

    for claim_id, decision in (
        ("c_confirm", "confirmed"),
        ("c_reject", "rejected"),
        ("c_review", "reviewed"),
    ):
        upsert_claim(store, claim_id=claim_id, claim_text="t", confidence=0.5)
        note = f"reviewer says {decision}"
        updated = mark_claim_reviewed(store, claim_id, status=decision, reviewer_note=note)

        # The record is updated in place, never deleted.
        assert updated["status"] == decision
        assert updated["reviewer_note"] == note
        assert updated["reviewed_at"]  # a review timestamp was stamped

        stored = get_claim(store, claim_id)
        assert stored is not None
        assert stored["status"] == decision
        assert stored["reviewer_note"] == note

    # All three reviewed claims are still present (history preserved).
    assert len(_all_claims(store)) == 3


def test_provisional_path_is_separate_from_canonical_db(tmp_path):
    store = _store_path(tmp_path)
    init_provisional_store(store)

    assert store.name == "provisional_claims.db"
    assert "fictionwriter.db" not in str(store)
    # Distinct artifact from the canonical relational DB.
    assert store.name != runtime.SQLITE_DB_PATH.name
    # Nothing was written to a canonical fictionwriter.db in the fixture tree.
    assert not (tmp_path / "data" / "fictionwriter.db").exists()
