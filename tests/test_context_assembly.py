"""Module: M03 (Context Assembly & Budgeting)
Synthetic tests for backend-agnostic context sizing and pruning.
"""

from __future__ import annotations

import json
from copy import deepcopy
from math import ceil
from types import SimpleNamespace
from typing import Any

import pytest

from fsm.nodes import node_assemble_context as context_node


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
