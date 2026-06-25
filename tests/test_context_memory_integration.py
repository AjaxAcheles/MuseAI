"""Module: M03 (Context Assembly & Budgeting) × M02 (Persistent Memory Stores)

Integration coverage for ``node_assemble_context`` reading the *real* relational
and provisional stores (synthetic-seeded through the public 04.0x helpers), sizing
the package through the config-selected drafting endpoint's ``tokenizer_family``,
applying drop-priority pruning and three-tier coreference injection, and degrading
gracefully where the M02 temporal (Graphiti) and flavour (Chroma) stores are still
deferred stubs.

The node is model-free: this test makes no inference/LLM HTTP call (``httpx`` and
the inference boundary are guarded to raise if touched). It is provider-neutral —
the tokenizer family is resolved from the real config loader, never hardcoded.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from core.config_loader import load_config
from fsm.nodes import node_assemble_context as context_node
from fsm.state import FSM_Pointer
from memory import provisional_store, sqlite_db

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
ENDPOINT_NAMES = ("planner", "drafter", "critic", "pad_translator", "craft_consultant")


# --------------------------------------------------------------------------- #
# Synthetic seed helpers — public store APIs where they exist; raw SQL only for
# the planner-owned tables (Arcs/Chapters/Scenes/Characters/Threads/RaptorNodes)
# that have no write helper yet. Beat commits and provisional claims go through
# their real helpers.
# --------------------------------------------------------------------------- #


def seed_relational_store(db_path: str | Path) -> None:
    """Seed a small ordered project into a real SQLite relational hub."""

    sqlite_db.init_db(db_path)

    # Planner-owned parent rows: no public write helper owns these yet, so seed
    # them directly (FK-ordered) rather than inventing one.
    conn = sqlite_db.connect_db(db_path)
    try:
        conn.execute(
            "INSERT INTO Arcs (id, description, status) VALUES (?, ?, ?)",
            ("arc-1", "rescue arc", "active"),
        )
        conn.execute(
            "INSERT INTO Chapters (id, arc_id, description, status) VALUES (?, ?, ?, ?)",
            ("chapter-1", "arc-1", "the descent", "active"),
        )
        conn.executemany(
            "INSERT INTO Scenes "
            "(id, chapter_id, description, word_budget, ordering, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("scene-1", "chapter-1", "the gate", 600, 0, "active"),
                ("scene-2", "chapter-1", "the vault", 600, 1, "planned"),
            ],
        )
        conn.execute(
            "INSERT INTO Characters (id, name) VALUES (?, ?)",
            ("char-ada", "Ada"),
        )
        conn.executemany(
            "INSERT INTO Threads (id, description, status, priority_score) "
            "VALUES (?, ?, ?, ?)",
            [
                ("thread-open", "who holds the key", "open", 0.80),
                ("thread-prog", "the missing sister", "open", 0.60),
            ],
        )
        conn.executemany(
            "INSERT INTO RaptorNodes (id, parent_id, level, summary, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                ("raptor-arc", None, "arc", "arc-level summary", "2026-01-01T00:00:00Z"),
                (
                    "raptor-scene",
                    "raptor-arc",
                    "scene",
                    "scene-level summary",
                    "2026-01-01T00:00:00Z",
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    # Committed beats go through the real 04.02 write helper, including the bundled
    # PAD snapshot and a thread status transition (open -> progressing).
    sqlite_db.upsert_beat_commit(
        db_path,
        beat_id="beat-0",
        scene_id="scene-1",
        beat_index=0,
        prose="Ada set her hand against the cold gate.",
        status="completed",
        pad_states={"char-ada": {"pleasure": 0.1, "arousal": 0.4, "dominance": -0.2}},
        thread_updates={"thread-prog": "progressing"},
    )
    sqlite_db.upsert_beat_commit(
        db_path,
        beat_id="beat-1",
        scene_id="scene-1",
        beat_index=1,
        prose="The lock answered with a single dry click.",
        status="completed",
    )


def seed_provisional_store(path: str | Path) -> None:
    """Seed provisional coreference claims spanning the high/mid/low bands."""

    provisional_store.init_provisional_store(path)
    provisional_store.upsert_claim(
        path,
        claim_id="claim-high",
        claim_text="HIGH_MARKER: she -> Ada",
        confidence=0.95,
        status="provisional",
    )
    provisional_store.upsert_claim(
        path,
        claim_id="claim-mid",
        claim_text="MID_MARKER: her -> the sister",
        confidence=0.60,
        status="provisional",
    )
    provisional_store.upsert_claim(
        path,
        claim_id="claim-low",
        claim_text="LOW_MARKER: it -> the vault",
        confidence=0.20,
        status="provisional",
    )


def load_real_config(monkeypatch: pytest.MonkeyPatch):
    """Load the real config.yaml, injecting placeholder endpoint secrets.

    Secrets are required by the loader but never live in config.yaml; inject a
    placeholder per endpoint so validation passes without a real backend. This is
    the genuine config loader and the real role->endpoint routing — nothing about
    sizing is hardcoded.
    """

    for name in ENDPOINT_NAMES:
        monkeypatch.setenv(f"{name.upper()}_API_KEY", "placeholder-not-used")
    return load_config(CONFIG_PATH)


def make_state(config: Any, *, db_path: Path, provisional_path: Path) -> dict[str, Any]:
    return {
        "fsm_pointer": FSM_Pointer(
            arc_id="arc-1", chapter_id="chapter-1", scene_id="scene-1", beat_index=0
        ),
        "app_config": config,
        "sqlite_db_path": str(db_path),
        "provisional_store_path": str(provisional_path),
    }


def guard_no_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any inference HTTP or call_llm invocation a hard failure."""

    def _explode(*args: Any, **kwargs: Any):
        raise AssertionError("node_assemble_context must not make an inference call")

    # The inference boundary builds an httpx.AsyncClient; the node never should.
    monkeypatch.setattr(httpx, "AsyncClient", _explode)
    # Belt and braces: trip if the inference boundary itself is invoked.
    import llm.call_llm as call_llm_module

    monkeypatch.setattr(call_llm_module, "call_llm", _explode, raising=False)
    monkeypatch.setattr(call_llm_module, "call_llm_structured", _explode, raising=False)


def run_node(state: dict[str, Any]) -> context_node.ContextPackage:
    """Invoke the real async node and return the assembled package."""

    result_state = asyncio.run(context_node.node_assemble_context(state))
    return result_state["active_context_package"]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_assembles_over_real_stores_with_config_routed_sizing_and_degradation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "fictionwriter.db"
    provisional_path = tmp_path / "provisional_claims.db"
    seed_relational_store(db_path)
    seed_provisional_store(provisional_path)

    guard_no_inference(monkeypatch)
    config = load_real_config(monkeypatch)
    drafter = config.endpoints.drafter

    # Spy that records the routing used by sizing while delegating to the real
    # tokenizer, so we prove the configured family/model were threaded through.
    real_count_tokens = context_node.count_tokens
    routed_calls: list[tuple[str, str | None]] = []

    def spy_count_tokens(text: str, tokenizer_family: str, model_name: str | None = None):
        routed_calls.append((tokenizer_family, model_name))
        return real_count_tokens(text, tokenizer_family, model_name)

    monkeypatch.setattr(context_node, "count_tokens", spy_count_tokens)

    package = run_node(make_state(config, db_path=db_path, provisional_path=provisional_path))

    # --- relational layer reflects the seeded project -----------------------
    assert package["meta"]["relational"]["available"] is True
    assert package["relational"]["arc"]["id"] == "arc-1"
    scene_ids = [s["id"] for s in package["relational"]["scenes_for_chapter"]]
    assert scene_ids == ["scene-1", "scene-2"]  # ordered by `ordering ASC`
    committed_ids = {b["id"] for b in package["relational"]["committed_beats"]}
    assert {"beat-0", "beat-1"} <= committed_ids
    open_thread_ids = {t["id"] for t in package["relational"]["open_threads"]}
    assert "thread-open" in open_thread_ids
    assert "thread-prog" not in open_thread_ids  # transitioned to 'progressing'

    # --- summary layer reflects the seeded RaptorNodes ----------------------
    assert package["meta"]["summaries"]["available"] is True
    summary_ids = {
        node["id"]
        for level in package["summaries"]["by_level"].values()
        for node in level
    }
    assert {"raptor-arc", "raptor-scene"} <= summary_ids

    # --- sizing routed through the config-resolved tokenizer_family ----------
    sizing = package["meta"]["token_sizing"]
    assert sizing["tokenizer_family"] == drafter.tokenizer_family
    assert sizing["model_name"] == drafter.model_name
    assert sizing["context_token_budget"] == config.context.token_budget
    assert sizing["total_tokens"] == sum(sizing["layers"].values())
    assert routed_calls, "count_tokens must be exercised during sizing"
    assert all(
        call == (drafter.tokenizer_family, drafter.model_name) for call in routed_calls
    )

    # --- three-tier coreference injection (bands from real config) ----------
    resolution = package["meta"]["coreference_resolution"]
    assert resolution["high_confidence_threshold"] == config.context.coreference_high_confidence
    assert resolution["mid_confidence_threshold"] == config.context.coreference_mid_confidence
    assert resolution["confirmed_fact_count"] == 1
    assert resolution["epistemic_belief_count"] == 1
    assert resolution["excluded_low_confidence_count"] == 1

    fact_claim_ids = {f["claim"]["claim_id"] for f in package["relational"]["coreference_facts"]}
    assert fact_claim_ids == {"claim-high"}
    beliefs = package["coreference_candidates"]
    assert len(beliefs) == 1
    assert beliefs[0]["claim"]["claim_id"] == "claim-mid"
    assert beliefs[0]["injection_type"] == "epistemic_belief"
    assert beliefs[0]["is_confirmed"] is False

    # low-confidence claim is excluded from every retained layer
    serialized = json.dumps(package, sort_keys=True, default=str)
    assert "LOW_MARKER" not in serialized
    assert "claim-low" not in serialized

    # --- graceful degradation of the stubbed temporal/flavour stores --------
    assert package["temporal"] == {"records": [], "coreference_candidates": []}
    assert package["flavour"] == []
    assert package["meta"]["temporal"]["available"] is False
    assert "not yet implemented" in package["meta"]["temporal"]["reason"]
    assert package["meta"]["flavour"]["available"] is False
    assert "not yet implemented" in package["meta"]["flavour"]["reason"]

    # under the real (large) budget nothing is pruned
    assert package["meta"]["pruned_layers"] == []
    assert package["meta"]["over_budget"] is False


def test_over_budget_prunes_summaries_and_preserves_relational_truth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "fictionwriter.db"
    provisional_path = tmp_path / "provisional_claims.db"
    seed_relational_store(db_path)
    seed_provisional_store(provisional_path)

    guard_no_inference(monkeypatch)
    real_config = load_real_config(monkeypatch)
    drafter = real_config.endpoints.drafter

    # Sizing config only: reuse the real config-resolved tokenizer family/model
    # (not hardcoded) and the real coreference bands, but force a tiny budget so
    # pruning must run. flavour is already empty (Chroma stubbed), so drop-order
    # proceeds to the summary tiers; relational truth must survive.
    tight_config = SimpleNamespace(
        endpoints=SimpleNamespace(
            drafter=SimpleNamespace(
                tokenizer_family=drafter.tokenizer_family,
                model_name=drafter.model_name,
            )
        ),
        context=SimpleNamespace(
            token_budget=20,
            coreference_high_confidence=real_config.context.coreference_high_confidence,
            coreference_mid_confidence=real_config.context.coreference_mid_confidence,
        ),
    )

    package = run_node(
        make_state(tight_config, db_path=db_path, provisional_path=provisional_path)
    )

    # Relational canonical truth is never dropped, even while over budget.
    assert package["relational"]["arc"]["id"] == "arc-1"
    assert [s["id"] for s in package["relational"]["scenes_for_chapter"]] == [
        "scene-1",
        "scene-2",
    ]

    # The summary tiers were pruned in least-critical-first order; the budgeting
    # decision is recorded in meta.
    pruned = package["meta"]["pruned_layers"]
    assert any(layer.startswith("summaries.by_level.") for layer in pruned)
    assert all(package["summaries"]["by_level"][lvl] == [] for lvl in ("beat", "scene"))
    assert "final_token_total" in package["meta"]
    assert "initial_token_total" in package["meta"]
    # relational alone exceeds the deliberately tiny budget, so it stays over budget
    assert package["meta"]["over_budget"] is True
