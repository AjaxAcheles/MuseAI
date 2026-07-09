"""Deterministic PAD → behavioural-constraint lookup.

A target emotional state reaches the drafter as a behavioural constraint, never
as raw coordinates: ``pleasure=-0.6, arousal=0.7, dominance=-0.5`` means nothing
to a model in isolation, and asking it to interpret the numbers produces a
different reading every time. So each axis is quantized into one of three bands
and the resulting region is looked up in ``pad_baselines.json``, a static table
of 27 authored strings — the full 3×3×3 grid. The string is injected verbatim.
No model is involved, and the same coordinate always yields the same constraint.

The band for an axis value ``v``::

    "neg"  if v < -PAD_BAND_THRESHOLD
    "pos"  if v >  PAD_BAND_THRESHOLD
    "neu"  otherwise

A region key is the three bands joined in axis order, pleasure first:
``"pos_neu_neg"`` is pleasure positive, arousal neutral, dominance negative.
"""

from __future__ import annotations

import json
from functools import lru_cache
from itertools import product
from pathlib import Path

PAD_BASELINES_PATH = Path(__file__).resolve().parent / "pad_baselines.json"

# Axis order of a region key, and of every PAD triple in this codebase.
PAD_AXES = ("pleasure", "arousal", "dominance")

# The half-width of the neutral band. An axis reading between -0.33 and 0.33
# carries no directional signal, so it quantizes to "neu" rather than to
# whichever sign it happens to hold. This splits each axis into three roughly
# equal thirds of its [-1.0, 1.0] range.
PAD_BAND_THRESHOLD = 0.33

PAD_BANDS = ("neg", "neu", "pos")


def quantize_axis(value: float) -> str:
    """Quantize one PAD axis reading to its band."""
    value = float(value)
    if value < -PAD_BAND_THRESHOLD:
        return "neg"
    if value > PAD_BAND_THRESHOLD:
        return "pos"
    return "neu"


def pad_key(pleasure: float, arousal: float, dominance: float) -> str:
    """The ``"P_A_D"`` region key of a PAD coordinate."""
    return "_".join(
        quantize_axis(value) for value in (pleasure, arousal, dominance)
    )


def pad_keys() -> tuple[str, ...]:
    """Every region key the table must define: the full 3×3×3 grid."""
    return tuple("_".join(bands) for bands in product(PAD_BANDS, repeat=3))


@lru_cache(maxsize=1)
def load_pad_baselines() -> dict[str, str]:
    """Load the static region → behavioural-constraint table.

    The table must be complete: a missing region would leave a beat with no
    constraint, so an incomplete table is a hard failure at first use.
    """
    with open(PAD_BASELINES_PATH, encoding="utf-8") as fh:
        baselines = json.load(fh)

    expected = pad_keys()
    missing = [key for key in expected if key not in baselines]
    if missing:
        raise KeyError(
            f"{PAD_BASELINES_PATH.name} is incomplete: missing {len(missing)} of "
            f"{len(expected)} PAD regions: {missing}"
        )
    return baselines


def resolve_pad_constraint(
    pleasure: float, arousal: float, dominance: float
) -> str:
    """Return the behavioural constraint for a target PAD coordinate.

    Total over the whole cube: every coordinate quantizes into one of the 27
    regions, and every region has an authored string.
    """
    key = pad_key(pleasure, arousal, dominance)
    baselines = load_pad_baselines()

    try:
        constraint = baselines[key]
    except KeyError as exc:
        raise KeyError(
            f"{PAD_BASELINES_PATH.name} has no entry for PAD region {key!r} "
            f"(pleasure={pleasure}, arousal={arousal}, dominance={dominance}); "
            f"the table must define all {len(pad_keys())} regions"
        ) from exc

    if not constraint.strip():
        raise ValueError(
            f"{PAD_BASELINES_PATH.name} defines PAD region {key!r} as an empty "
            f"string; every region needs an authored behavioural constraint"
        )
    return constraint
