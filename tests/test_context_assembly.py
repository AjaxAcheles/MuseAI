"""Module: M03 (Context Assembly & Budgeting)
Synthetic tests for backend-agnostic context sizing and pruning.
"""

from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from math import ceil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fsm.nodes import node_assemble_context as context_node
from memory import provisional_store, sqlite_db


def make_config(
    *,
    token_budget: int,
    tokenizer_family: str = "char_heuristic",
    high_confidence: float = 0.91,
    mid_confidence: float = 0.37,
) -> SimpleNamespace:
    return SimpleNamespace(
        endpoints=SimpleNamespace(
            drafter=SimpleNamespace(
                tokenizer_family=tokenizer_family,
                model_name="synthetic-drafter",
            )
        ),
        context=SimpleNamespace(
            token_budget=token_budget,
            coreference_high_confidence=high_confidence,
            coreference_mid_confidence=mid_confidence,
        ),
    )


def make_state(config: SimpleNamespace) -> dict[str, Any]:
    return {
        "fsm_pointer": SimpleNamespace(
            arc_id="arc-1",
            chapter_id="chapter-1",
            scene_id="scene-1",
            beat_index=0,
        ),
        "app_config": config,
        "sqlite_db_path": "unused-synthetic.sqlite",
        "provisional_store_path": "unused-provisional.sqlite",
    }


def install_context_layers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    relational: dict[str, Any] | None = None,
    summaries: dict[str, Any] | None = None,
    flavour: list[dict[str, Any]] | None = None,
    temporal: dict[str, Any] | None = None,
    coreference_candidates: list[dict[str, Any]] | None = None,
) -> None:
    relational_layer = relational if relational is not None else {"fact": "relational"}
    summary_layer = summaries if summaries is not None else make_summaries()
    flavour_layer = flavour if flavour is not None else [{"text": "flavour"}]
    temporal_layer = (
        temporal
        if temporal is not None
        else {"records": [], "coreference_candidates": []}
    )
    provisional_layer = (
        coreference_candidates if coreference_candidates is not None else []
    )

    monkeypatch.setattr(
        context_node,
        "_read_relational_layer",
        lambda db_path, pointer: deepcopy(relational_layer),
    )
    monkeypatch.setattr(
        context_node,
        "_read_summary_layer",
        lambda db_path: deepcopy(summary_layer),
    )
    monkeypatch.setattr(
        context_node,
        "_read_flavour_layer",
        lambda pointer: deepcopy(flavour_layer),
    )
    monkeypatch.setattr(
        context_node,
        "_read_temporal_layer",
        lambda pointer: deepcopy(temporal_layer),
    )
    monkeypatch.setattr(
        context_node.provisional_store,
        "list_pending_claims",
        lambda path: deepcopy(provisional_layer),
    )


def make_summaries(
    *,
    beat_text: str = "beat summary",
    scene_text: str = "scene summary",
    chapter_text: str = "chapter summary",
    arc_text: str = "arc summary",
) -> dict[str, Any]:
    return {
        "roots": [],
        "by_level": {
            "global": [],
            "arc": [{"id": "sum-arc", "summary": arc_text}],
            "chapter": [{"id": "sum-chapter", "summary": chapter_text}],
            "scene": [{"id": "sum-scene", "summary": scene_text}],
            "beat": [{"id": "sum-beat", "summary": beat_text}],
        },
    }


def char_heuristic_count(layer: Any) -> int:
    serialized = json.dumps(layer, sort_keys=True, separators=(",", ":"), default=str)
    if not serialized:
        return 0
    return ceil(len(serialized) / 4)


def assemble_package(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: SimpleNamespace,
    relational: dict[str, Any] | None = None,
    summaries: dict[str, Any] | None = None,
    flavour: list[dict[str, Any]] | None = None,
    temporal: dict[str, Any] | None = None,
    coreference_candidates: list[dict[str, Any]] | None = None,
) -> context_node.ContextPackage:
    install_context_layers(
        monkeypatch,
        relational=relational,
        summaries=summaries,
        flavour=flavour,
        temporal=temporal,
        coreference_candidates=coreference_candidates,
    )
    return context_node.build_context_package(make_state(config))


def test_char_heuristic_layer_sizing_uses_configured_drafter_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    known_relational_layer = {"known": "abcdefghijklmnop"}
    package = assemble_package(
        monkeypatch,
        config=make_config(token_budget=10_000, tokenizer_family="char_heuristic"),
        relational=known_relational_layer,
        summaries={"roots": [], "by_level": {}},
        flavour=[],
        temporal={"records": [], "coreference_candidates": []},
    )

    sizing = package["meta"]["token_sizing"]

    assert sizing["tokenizer_family"] == "char_heuristic"
    assert sizing["layers"] == {
        layer_name: char_heuristic_count(package[layer_name])
        for layer_name in context_node.CONTEXT_LAYER_NAMES
    }
    assert sizing["layers"]["relational"] == ceil(
        len('{"coreference_facts":[],"known":"abcdefghijklmnop"}') / 4
    )


def test_layer_sizing_routes_every_layer_by_configured_tokenizer_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str | None]] = []

    def fake_count_tokens(
        text: str, tokenizer_family: str, model_name: str | None = None
    ) -> int:
        calls.append((tokenizer_family, model_name))
        return max(1, len(text) % 11)

    monkeypatch.setattr(context_node, "count_tokens", fake_count_tokens)

    package = assemble_package(
        monkeypatch,
        config=make_config(token_budget=10_000, tokenizer_family="hf_auto"),
        relational={"known": "route-by-config"},
        summaries={"roots": [], "by_level": {}},
        flavour=[],
        temporal={"records": [], "coreference_candidates": []},
    )

    assert package["meta"]["token_sizing"]["tokenizer_family"] == "hf_auto"
    assert len(calls) >= len(context_node.CONTEXT_LAYER_NAMES)
    assert all(
        call == ("hf_auto", "synthetic-drafter")
        for call in calls
    )


def test_token_total_equals_sum_of_retained_layer_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = assemble_package(monkeypatch, config=make_config(token_budget=10_000))
    sizing = package["meta"]["token_sizing"]

    assert sizing["total_tokens"] == sum(sizing["layers"].values())
    assert package["meta"]["final_token_total"] == sizing["total_tokens"]


def test_under_budget_package_is_unpruned(monkeypatch: pytest.MonkeyPatch) -> None:
    flavour = [{"text": "retained flavour"}]
    summaries = make_summaries()

    package = assemble_package(
        monkeypatch,
        config=make_config(token_budget=10_000),
        summaries=summaries,
        flavour=flavour,
    )

    assert package["meta"]["pruned_layers"] == []
    assert package["meta"]["over_budget"] is False
    assert package["flavour"] == flavour
    assert package["summaries"]["by_level"]["beat"] == summaries["by_level"]["beat"]


def test_over_budget_prunes_flavour_then_summary_tiers_with_recompute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    large_text = "x" * 400
    summaries = make_summaries(
        beat_text=large_text,
        scene_text=large_text,
        chapter_text=large_text,
        arc_text=large_text,
    )

    package = assemble_package(
        monkeypatch,
        config=make_config(token_budget=230),
        relational={"fact": "small"},
        summaries=summaries,
        flavour=[{"text": large_text}],
        temporal={"records": [], "coreference_candidates": []},
    )

    assert package["meta"]["pruned_layers"] == [
        "flavour",
        "summaries.by_level.beat",
        "summaries.by_level.scene",
        "summaries.by_level.chapter",
    ]
    assert package["flavour"] == []
    assert package["summaries"]["by_level"]["beat"] == []
    assert package["summaries"]["by_level"]["scene"] == []
    assert package["summaries"]["by_level"]["chapter"] == []
    assert package["summaries"]["by_level"]["arc"]
    assert package["meta"]["final_token_total"] <= 230


def test_relational_truth_is_never_dropped_when_still_over_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_fact = "r" * 800
    large_text = "x" * 400

    package = assemble_package(
        monkeypatch,
        config=make_config(token_budget=20),
        relational={"canonical_fact": canonical_fact},
        summaries=make_summaries(
            beat_text=large_text,
            scene_text=large_text,
            chapter_text=large_text,
            arc_text=large_text,
        ),
        flavour=[{"text": large_text}],
        temporal={"records": [], "coreference_candidates": []},
    )

    assert package["relational"]["canonical_fact"] == canonical_fact
    assert package["flavour"] == []
    assert package["summaries"]["by_level"]["beat"] == []
    assert package["summaries"]["by_level"]["scene"] == []
    assert package["summaries"]["by_level"]["chapter"] == []
    assert package["summaries"]["by_level"]["arc"] == []
    assert package["meta"]["pruned_layers"] == [
        "flavour",
        "summaries.by_level.beat",
        "summaries.by_level.scene",
        "summaries.by_level.chapter",
        "summaries.by_level.arc",
    ]
    assert package["meta"]["over_budget"] is True
    assert package["meta"]["final_token_total"] > 20


def test_budget_and_coreference_bands_are_read_from_injected_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = assemble_package(
        monkeypatch,
        config=make_config(
            token_budget=12_345,
            high_confidence=0.77,
            mid_confidence=0.33,
        ),
        coreference_candidates=[],
    )

    assert package["meta"]["token_sizing"]["context_token_budget"] == 12_345
    assert package["meta"]["coreference_resolution"]["high_confidence_threshold"] == 0.77
    assert package["meta"]["coreference_resolution"]["mid_confidence_threshold"] == 0.33


# --------------------------------------------------------------------------- #
# 06.T2 — graceful degradation against the stubbed M02 stores
# --------------------------------------------------------------------------- #
#
# These tests deliberately exercise the *real* (not monkeypatched) Graphiti and
# Chroma reads. Both clients are still honest stubs whose `.query()` raises
# `NotImplementedError("STUB")`, so the `_safe_store_read` seam must degrade the
# temporal and flavour layers to empty/unavailable while the genuinely-built
# SQLite relational hub, the persisted RaptorNodes summaries, and the separate
# provisional-claim store still populate from real, seeded fixtures.


def seed_relational_store(db_path: str | Path) -> None:
    """Seed a real SQLite hub the relational + summary reads can resolve."""

    sqlite_db.init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO Arcs (id, description, status) VALUES (?, ?, ?)",
            ("arc-1", "seeded arc", "active"),
        )
        conn.execute(
            "INSERT INTO Chapters (id, arc_id, description, status) "
            "VALUES (?, ?, ?, ?)",
            ("chapter-1", "arc-1", "seeded chapter", "active"),
        )
        conn.execute(
            "INSERT INTO Scenes "
            "(id, chapter_id, description, word_budget, ordering, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("scene-1", "chapter-1", "seeded scene", 500, 0, "active"),
        )
        conn.execute(
            "INSERT INTO Beats "
            "(id, scene_id, beat_index, status, prose, word_count, committed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("beat-1", "scene-1", 0, "planned", None, 0, None),
        )
        conn.execute(
            "INSERT INTO RaptorNodes (id, parent_id, level, summary, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("raptor-scene-1", None, "scene", "seeded scene summary", "2026-01-01T00:00:00Z"),
        )
        conn.commit()
    finally:
        conn.close()


def make_real_state(
    config: SimpleNamespace,
    *,
    db_path: str | Path,
    provisional_path: str | Path,
) -> dict[str, Any]:
    state = make_state(config)
    state["sqlite_db_path"] = str(db_path)
    state["provisional_store_path"] = str(provisional_path)
    return state


def test_stubbed_graphiti_chroma_degrade_to_empty_unavailable_layers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "fictionwriter.db"
    provisional_path = tmp_path / "provisional_claims.db"
    seed_relational_store(db_path)

    state = make_real_state(
        make_config(token_budget=10_000),
        db_path=db_path,
        provisional_path=provisional_path,
    )

    # No monkeypatching of the layer reads: the real GraphitiClient/ChromaClient
    # stubs raise NotImplementedError, and the seam must absorb only that.
    package = context_node.build_context_package(state)

    # A valid package is still produced.
    assert set(context_node.CONTEXT_LAYER_NAMES) <= set(package.keys())
    assert "meta" in package

    # Temporal (Graphiti) and flavour (Chroma) degrade to empty layers...
    assert package["flavour"] == []
    assert package["temporal"] == {"records": [], "coreference_candidates": []}

    # ...and meta records them unavailable with a not-yet-built reason.
    assert package["meta"]["temporal"]["available"] is False
    assert "not yet implemented" in package["meta"]["temporal"]["reason"]
    assert package["meta"]["flavour"]["available"] is False
    assert "not yet implemented" in package["meta"]["flavour"]["reason"]


def test_real_relational_and_summary_layers_populate_under_degradation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "fictionwriter.db"
    provisional_path = tmp_path / "provisional_claims.db"
    seed_relational_store(db_path)

    # Seed a real provisional claim in the mid band so it survives resolution as
    # an explicitly-unconfirmed belief, proving the separate store was read.
    provisional_store.upsert_claim(
        provisional_path,
        claim_id="claim-mid",
        claim_text="she == the captain",
        confidence=0.60,
        status="provisional",
    )

    state = make_real_state(
        make_config(token_budget=10_000, high_confidence=0.85, mid_confidence=0.50),
        db_path=db_path,
        provisional_path=provisional_path,
    )

    package = context_node.build_context_package(state)

    # Relational truth read from the real SQLite hub.
    assert package["meta"]["relational"]["available"] is True
    assert package["relational"]["arc"]["id"] == "arc-1"
    assert any(
        scene["id"] == "scene-1" for scene in package["relational"]["scenes_for_chapter"]
    )

    # RaptorNodes summaries read from the real hub.
    assert package["meta"]["summaries"]["available"] is True
    scene_summaries = package["summaries"]["by_level"]["scene"]
    assert any(node["id"] == "raptor-scene-1" for node in scene_summaries)

    # The provisional candidate was read from the real store and tiered to a
    # mid-band unconfirmed belief.
    assert package["meta"]["coreference_candidates"]["available"] is True
    beliefs = package["coreference_candidates"]
    assert len(beliefs) == 1
    assert beliefs[0]["claim"]["claim_id"] == "claim-mid"
    assert beliefs[0]["injection_type"] == "epistemic_belief"


def test_unexpected_store_error_is_not_swallowed_by_the_seam(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "fictionwriter.db"
    provisional_path = tmp_path / "provisional_claims.db"
    seed_relational_store(db_path)

    def boom(db_path: Any, pointer: Any) -> dict[str, Any]:
        raise RuntimeError("unexpected store failure")

    # An unexpected, non-degrade error (not the documented NotImplementedError)
    # must propagate — the seam only absorbs the not-yet-built signal.
    monkeypatch.setattr(context_node, "_read_relational_layer", boom)

    state = make_real_state(
        make_config(token_budget=10_000),
        db_path=db_path,
        provisional_path=provisional_path,
    )

    with pytest.raises(RuntimeError, match="unexpected store failure"):
        context_node.build_context_package(state)


# --------------------------------------------------------------------------- #
# 06.T2 — three-tier coreference-confidence classification
# --------------------------------------------------------------------------- #


def make_tiered_candidates() -> list[dict[str, Any]]:
    """High / mid / low synthetic provisional coreference candidates."""

    return [
        {"claim_id": "c-high", "claim_text": "HIGH_MARKER", "confidence": 0.95},
        {"claim_id": "c-mid", "claim_text": "MID_MARKER", "confidence": 0.60},
        {"claim_id": "c-low", "claim_text": "LOW_MARKER", "confidence": 0.20},
    ]


def test_three_tier_classification_uses_injected_bands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = assemble_package(
        monkeypatch,
        config=make_config(
            token_budget=10_000, high_confidence=0.85, mid_confidence=0.50
        ),
        coreference_candidates=make_tiered_candidates(),
    )

    resolution = package["meta"]["coreference_resolution"]
    assert resolution["confirmed_fact_count"] == 1
    assert resolution["epistemic_belief_count"] == 1
    assert resolution["excluded_low_confidence_count"] == 1

    # High-confidence link resolves to a confirmed relational fact.
    facts = package["relational"]["coreference_facts"]
    assert len(facts) == 1
    assert facts[0]["claim"]["claim_id"] == "c-high"
    assert facts[0]["injection_type"] == "confirmed_fact"
    assert facts[0]["is_confirmed"] is True


def test_mid_confidence_is_unconfirmed_belief_never_fact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = assemble_package(
        monkeypatch,
        config=make_config(
            token_budget=10_000, high_confidence=0.85, mid_confidence=0.50
        ),
        coreference_candidates=make_tiered_candidates(),
    )

    beliefs = package["coreference_candidates"]
    assert len(beliefs) == 1
    belief = beliefs[0]
    assert belief["claim"]["claim_id"] == "c-mid"
    assert belief["injection_type"] == "epistemic_belief"
    assert belief["confirmation_status"] == "unconfirmed"
    assert belief["is_confirmed"] is False

    # The mid-confidence link must never be promoted to a confirmed fact.
    fact_ids = {fact["claim"]["claim_id"] for fact in package["relational"]["coreference_facts"]}
    assert "c-mid" not in fact_ids


def test_low_confidence_link_never_appears_in_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = assemble_package(
        monkeypatch,
        config=make_config(
            token_budget=10_000, high_confidence=0.85, mid_confidence=0.50
        ),
        coreference_candidates=make_tiered_candidates(),
    )

    serialized = json.dumps(package, sort_keys=True, default=str)
    assert "LOW_MARKER" not in serialized
    assert "c-low" not in serialized


def test_band_thresholds_drive_reclassification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Lowering the high band below the mid candidate promotes it to a fact,
    # proving classification is driven by the injected config, not hardcoded.
    package = assemble_package(
        monkeypatch,
        config=make_config(
            token_budget=10_000, high_confidence=0.50, mid_confidence=0.30
        ),
        coreference_candidates=make_tiered_candidates(),
    )

    resolution = package["meta"]["coreference_resolution"]
    assert resolution["confirmed_fact_count"] == 2
    assert resolution["epistemic_belief_count"] == 0
    assert resolution["excluded_low_confidence_count"] == 1

    fact_ids = {fact["claim"]["claim_id"] for fact in package["relational"]["coreference_facts"]}
    assert fact_ids == {"c-high", "c-mid"}
