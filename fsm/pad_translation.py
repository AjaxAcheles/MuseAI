"""Module: M05 (Hierarchical Planning Cascade)

Deterministic PAD-target smoothing and grounded behavioural translation.

This module owns two of the steps of ``node_plan_beat``'s PAD Grounded
Translation Pipeline (``_design/implementation/LangGraph_Nodes.md`` Phase B):

* ``smooth_pad_target`` — a pure EWMA over the three affect axes (pleasure,
  arousal, dominance). The smoothing weight is **read by the caller** from the
  existing ``config.thresholds.pad_ewma_alpha`` key and passed in; this module
  never hardcodes it and never introduces a planning-scoped alpha.
* ``translate_pad_to_behaviour`` — the grounded ladder that converts a PAD
  target into a compact behavioural-constraint string for a beat plan:

      static baseline  ->  optional small-tier LLM adaptation  ->  static fallback

  The **static** and **fallback** rungs are the floor: they are pure
  ``prompts/pad_regions.json`` lookups and run with no LLM, network, or store.
  The **adaptation** rung is an injected seam (``adapt_fn``); when it is absent
  or raises, the ladder returns the static baseline unchanged. ``node_plan_beat``
  is responsible for choosing whether adaptation is permitted, for logging the
  fallback at WARNING, and for persisting the result — none of that lives here.

Region resolution reads the neutral-band half-width from the data file, so the
band can be tuned without a code change; this module hardcodes no affect
threshold, no endpoint/model name, and no provider branch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, NamedTuple, Sequence

# ---------------------------------------------------------------------------
# PAD coordinate representation
# ---------------------------------------------------------------------------

_PAD_REGIONS_PATH = Path(__file__).resolve().parent.parent / "prompts" / "pad_regions.json"

# Affect axes are normalized to this inclusive interval (mirrors the SQLite
# CharacterEmotions PAD CHECK in memory/sqlite_db.py). EWMA of two in-range
# values stays in range; the clamp is a defensive boundary only.
_AXIS_MIN = -1.0
_AXIS_MAX = 1.0


class PADTarget(NamedTuple):
    """An (pleasure, arousal, dominance) affect coordinate, each in [-1, 1]."""

    pleasure: float
    arousal: float
    dominance: float


def _clamp_axis(value: float) -> float:
    return max(_AXIS_MIN, min(_AXIS_MAX, value))


def _coerce_pad(pad: Any) -> PADTarget:
    """Accept a PADTarget, a 3-sequence, or a mapping and return a PADTarget.

    Mappings may key the axes as ``pleasure``/``arousal``/``dominance``,
    ``p``/``a``/``d``, or their uppercase forms. Coercion does not clamp — it
    only normalizes shape — so callers see their values verbatim.
    """
    if isinstance(pad, PADTarget):
        return pad
    if isinstance(pad, Mapping):
        def _axis(*names: str) -> float:
            for name in names:
                if name in pad:
                    return float(pad[name])
            raise KeyError(f"PAD mapping is missing any of {names!r}")

        return PADTarget(
            _axis("pleasure", "p", "P"),
            _axis("arousal", "a", "A"),
            _axis("dominance", "d", "D"),
        )
    if isinstance(pad, Sequence) and not isinstance(pad, (str, bytes)):
        values = list(pad)
        if len(values) != 3:
            raise ValueError(f"PAD sequence must have exactly 3 axes, got {len(values)}")
        return PADTarget(float(values[0]), float(values[1]), float(values[2]))
    raise TypeError(f"Unsupported PAD coordinate type: {type(pad)!r}")


# ---------------------------------------------------------------------------
# EWMA smoothing
# ---------------------------------------------------------------------------

def smooth_pad_target(prior_pad: Any, proposed_pad: Any, alpha: float) -> PADTarget:
    """Exponentially smooth a proposed PAD target against the prior target.

    Per-axis: ``smoothed = alpha * proposed + (1 - alpha) * prior`` — the
    standard EWMA where ``alpha`` weights the new observation. ``alpha`` is
    supplied by the caller from ``config.thresholds.pad_ewma_alpha``; it is never
    read or defaulted here. Pure and deterministic; the result is clamped to the
    [-1, 1] affect range. ``prior_pad``/``proposed_pad`` accept any shape that
    :func:`_coerce_pad` understands.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"pad_ewma_alpha must be in [0, 1], got {alpha!r}")
    prior = _coerce_pad(prior_pad)
    proposed = _coerce_pad(proposed_pad)
    return PADTarget(
        _clamp_axis(alpha * proposed.pleasure + (1.0 - alpha) * prior.pleasure),
        _clamp_axis(alpha * proposed.arousal + (1.0 - alpha) * prior.arousal),
        _clamp_axis(alpha * proposed.dominance + (1.0 - alpha) * prior.dominance),
    )


# ---------------------------------------------------------------------------
# Region lookup (static behavioural floor)
# ---------------------------------------------------------------------------

_REGIONS_DATA: dict[str, Any] | None = None


def load_pad_regions() -> dict[str, Any]:
    """Load and cache the ``prompts/pad_regions.json`` lookup table.

    Returns the parsed top-level object (``schema_version``, ``neutral_band``,
    ``regions``). Cached after the first read; pure data, no LLM/network/store.
    """
    global _REGIONS_DATA
    if _REGIONS_DATA is None:
        _REGIONS_DATA = json.loads(_PAD_REGIONS_PATH.read_text(encoding="utf-8"))
    return _REGIONS_DATA


def _neutral_band() -> float:
    """Read the neutral dead-zone half-width from the data file (never hardcoded)."""
    data = load_pad_regions()
    if "neutral_band" not in data:
        raise KeyError("pad_regions.json is missing the required 'neutral_band' constant")
    return float(data["neutral_band"])


def region_key_for_pad(pad_target: Any) -> str:
    """Resolve a PAD coordinate to its region key by axis sign.

    A coordinate whose every axis lies within +/- ``neutral_band`` of zero maps
    to ``"neutral"``; otherwise each axis contributes ``+`` (>= 0) or ``-`` to a
    ``"P{..}A{..}D{..}"`` octant key. The band width comes from the data file.
    """
    pad = _coerce_pad(pad_target)
    band = _neutral_band()
    if all(abs(axis) <= band for axis in pad):
        return "neutral"

    def _sign(value: float) -> str:
        return "+" if value >= 0 else "-"

    return f"P{_sign(pad.pleasure)}A{_sign(pad.arousal)}D{_sign(pad.dominance)}"


def _region_entry(region_key: str) -> dict[str, Any]:
    regions = load_pad_regions().get("regions", {})
    if region_key not in regions:
        raise KeyError(f"pad_regions.json has no region {region_key!r}")
    return regions[region_key]


def compose_baseline_string(region_key: str) -> str:
    """Compose a compact, writing-useful behavioural-constraint string for a region.

    Folds the region's ``behaviour`` descriptor, ``physical_cues``, and
    ``register`` into one line suitable for a beat plan. Pure data lookup.
    """
    entry = _region_entry(region_key)
    behaviour = entry.get("behaviour", "").strip()
    cues = entry.get("physical_cues", [])
    register = entry.get("register", "").strip()
    parts = [behaviour]
    if cues:
        parts.append("Physical cues: " + "; ".join(cues) + ".")
    if register:
        parts.append(f"Register: {register}.")
    return " ".join(part for part in parts if part)


# ---------------------------------------------------------------------------
# Grounded translation ladder
# ---------------------------------------------------------------------------

# Rung markers recorded on the returned translation.
RUNG_STATIC = "static"      # no adapter supplied; static baseline is the floor
RUNG_ADAPTED = "adapted"    # small-tier adaptation succeeded
RUNG_FALLBACK = "fallback"  # adaptation was attempted but failed -> static floor


@dataclass(frozen=True)
class BehaviourTranslation:
    """The behavioural-constraint string plus provenance for a PAD target."""

    text: str
    rung: str
    region_key: str


# An adapter takes the static baseline plus the resolved region, the PAD target,
# and scene/character context, and returns a tailored constraint string.
AdaptFn = Callable[[str, str, PADTarget, Mapping[str, Any]], Awaitable[str]]


async def translate_pad_to_behaviour(
    pad_target: Any,
    context: Mapping[str, Any] | None = None,
    *,
    adapt_fn: AdaptFn | None = None,
) -> BehaviourTranslation:
    """Translate a PAD target into a behavioural-constraint string via the ladder.

    Rungs, in order:

    1. **static** — look up the region's baseline behavioural descriptor from
       ``pad_regions.json`` and compose the floor string. Always succeeds.
    2. **adaptation (optional)** — if ``adapt_fn`` is supplied, ``await`` it to
       tailor the baseline to ``context``. On *any* error (including a missing
       prompt template or a failed inference call) the ladder falls through.
    3. **fallback** — return the static baseline unchanged.

    The static and fallback rungs run with no LLM, network, or store; only the
    injected ``adapt_fn`` may reach a model. The returned ``rung`` marks which
    rung produced the string (``static`` when no adapter was supplied,
    ``adapted`` on success, ``fallback`` when an adapter was tried but failed).
    """
    ctx: Mapping[str, Any] = context or {}
    region_key = region_key_for_pad(pad_target)
    baseline = compose_baseline_string(region_key)

    if adapt_fn is None:
        return BehaviourTranslation(text=baseline, rung=RUNG_STATIC, region_key=region_key)

    pad = _coerce_pad(pad_target)
    try:
        adapted = await adapt_fn(baseline, region_key, pad, ctx)
    except Exception:
        # Best-effort rung: any failure degrades to the static floor so planning
        # is never blocked here. node_plan_beat logs this fallback at WARNING.
        return BehaviourTranslation(text=baseline, rung=RUNG_FALLBACK, region_key=region_key)

    if not isinstance(adapted, str) or not adapted.strip():
        return BehaviourTranslation(text=baseline, rung=RUNG_FALLBACK, region_key=region_key)
    return BehaviourTranslation(text=adapted.strip(), rung=RUNG_ADAPTED, region_key=region_key)


# ---------------------------------------------------------------------------
# Default small-tier adapter (optional seam — the pipeline never depends on it)
# ---------------------------------------------------------------------------

def build_pad_adapt_fn(
    config: Any,
    *,
    endpoint_role: str = "pad_translator",
    template_node: str = "node_plan_beat_pad",
    call: Any = None,
    loader: Any = None,
) -> AdaptFn:
    """Build a default ``adapt_fn`` that wraps ``call_llm`` on the small-tier endpoint.

    The endpoint is resolved by role from ``config.endpoints`` (default
    ``pad_translator``); no provider/model name is hardcoded. The prompt is
    rendered through the M04 ``PromptLoader`` (``template_node``), so no prompt
    prose lives in this module — the constraint string and context are passed as
    template variables. Until that template is authored the render raises and
    :func:`translate_pad_to_behaviour` degrades to the static floor, which is the
    intended fault-tolerant behaviour.

    ``call`` and ``loader`` are injectable seams for tests. Imports are lazy so
    the static-only path never pulls the HTTP/inference stack.
    """
    endpoint = getattr(config.endpoints, endpoint_role)

    async def _adapt(
        baseline: str,
        region_key: str,
        pad_target: PADTarget,
        context: Mapping[str, Any],
    ) -> str:
        _call = call
        _loader = loader
        if _call is None:
            from llm.call_llm import call_llm as _call  # lazy: avoid HTTP stack on static path
        if _loader is None:
            from prompts.prompt_loader import PromptLoader

            _loader = PromptLoader()

        rendered = _loader.render(
            template_node,
            {
                "baseline_behaviour": baseline,
                "pad_region": region_key,
                "pleasure": pad_target.pleasure,
                "arousal": pad_target.arousal,
                "dominance": pad_target.dominance,
                **dict(context),
            },
        )
        response = await _call([{"role": "user", "content": rendered}], endpoint)
        return response.text

    return _adapt
