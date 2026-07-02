"""Module: M05 (Hierarchical Planning Cascade)
Deterministic tests for PAD EWMA smoothing and grounded translation rungs.
"""

import asyncio
from types import SimpleNamespace

import pytest

from fsm.pad_translation import (
    PADTarget,
    RUNG_FALLBACK,
    RUNG_STATIC,
    compose_baseline_string,
    load_pad_regions,
    region_key_for_pad,
    smooth_pad_target,
    translate_pad_to_behaviour,
)


def _config(alpha):
    return SimpleNamespace(
        thresholds=SimpleNamespace(pad_ewma_alpha=alpha)
    )


def test_smooth_pad_target_is_deterministic_and_respects_alpha_extremes():
    prior = PADTarget(-0.4, 0.2, -0.8)
    proposed = PADTarget(0.8, -0.6, 0.4)

    prior_only = smooth_pad_target(
        prior, proposed, _config(alpha=0.0).thresholds.pad_ewma_alpha
    )
    proposed_only = smooth_pad_target(
        prior, proposed, _config(alpha=1.0).thresholds.pad_ewma_alpha
    )
    mixed_alpha = _config(alpha=0.25).thresholds.pad_ewma_alpha
    mixed_once = smooth_pad_target(prior, proposed, mixed_alpha)
    mixed_twice = smooth_pad_target(prior, proposed, mixed_alpha)

    assert prior_only == prior
    assert proposed_only == proposed
    assert mixed_once == mixed_twice
    assert tuple(mixed_once) == pytest.approx((-0.1, 0.0, -0.5))


def test_pad_coordinates_resolve_to_octants_and_neutral_band():
    band = load_pad_regions()["neutral_band"]

    assert region_key_for_pad(PADTarget(band / 2, -band / 2, band / 2)) == "neutral"
    assert region_key_for_pad(PADTarget(band * 2, -band * 2, band * 2)) == "P+A-D+"
    assert region_key_for_pad({"pleasure": -0.7, "arousal": 0.4, "dominance": -0.2}) == (
        "P-A+D-"
    )


def test_translate_static_rung_is_llm_free_and_returns_static_baseline():
    target = PADTarget(0.7, 0.4, 0.5)
    result = asyncio.run(
        translate_pad_to_behaviour(target, {"scene_intent": "synthetic"}, adapt_fn=None)
    )

    assert result.rung == RUNG_STATIC
    assert result.region_key == "P+A+D+"
    assert result.text == compose_baseline_string("P+A+D+")


def test_translate_fallback_rung_recovers_from_raising_adapter():
    target = PADTarget(-0.8, 0.6, -0.7)
    calls = []

    async def _raising_adapter(baseline, region_key, pad_target, context):
        calls.append(
            {
                "baseline": baseline,
                "region_key": region_key,
                "pad_target": pad_target,
                "context": dict(context),
            }
        )
        raise RuntimeError("adapter unavailable")

    result = asyncio.run(
        translate_pad_to_behaviour(
            target, {"scene_intent": "synthetic"}, adapt_fn=_raising_adapter
        )
    )

    assert result.rung == RUNG_FALLBACK
    assert result.region_key == "P-A+D-"
    assert result.text == compose_baseline_string("P-A+D-")
    assert len(calls) == 1
    assert calls[0]["baseline"] == result.text
    assert calls[0]["pad_target"] == target


def test_pad_regions_json_loads_and_covers_all_used_regions():
    data = load_pad_regions()
    expected_regions = {
        "P+A+D+",
        "P+A+D-",
        "P+A-D+",
        "P+A-D-",
        "P-A+D+",
        "P-A+D-",
        "P-A-D+",
        "P-A-D-",
        "neutral",
    }

    assert expected_regions <= set(data["regions"])
    for region_key in expected_regions:
        entry = data["regions"][region_key]
        assert entry["behaviour"]
        assert entry["physical_cues"]
        assert entry["register"]
        assert compose_baseline_string(region_key)
