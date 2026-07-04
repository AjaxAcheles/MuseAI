"""Module: M07 (Quality Gauntlet)

Stage-1 (model-free) audit node. Runs deterministic Python checks against
``current_draft_text`` — no LLM call, no network, no store write — and reports findings
through the same ``FailureObject``/``critic_failures`` contract the LLM critics use:

  1. **Structural checks** — cheap layout heuristics (empty draft, an odd count of
     ``"`` quote marks, an odd count of a markdown span delimiter) each raise their own
     ``FailureObject`` with a verbatim ``offending_text`` span.
  2. **Passive-voice check, density-based** — a deterministic "be-verb + participle"
     heuristic classifies each sentence; a single aggregate ``FailureObject`` fires only
     when the passive fraction exceeds ``config.thresholds.passive_voice_density`` (an
     occasional passive sentence in an otherwise active beat must not fire).
  3. **``best_seen_draft`` retention** — the current draft's failure count is compared
     against a fresh re-run of the same checks against the stored ``best_seen_draft``
     (no separate failure-count field exists in ``OrchestratorState``, so the comparison
     is recomputed rather than cached); the fewer-failures draft wins ties go to the
     existing ``best_seen_draft``. In-memory only, never persisted.
  4. **``stylometric_distance``** — computed through an injectable ``distance_provider``
     seam. The default provider returns the stubbed ``0.0`` because ``memory/style_store.py``
     is still a deferred M02 stub (per the design's own Sprint-3 note that STEL $D_c$
     gating runs against a stubbed value while routing is fully real); the stub condition
     is logged rather than silently assumed.

``critic_failures`` is a reducer-backed field (``accumulate_or_reset``): this node
contributes only the delta it found this call (never merges in prior history and never
mutates the existing list object in place), consistent with the reducer contract the
compiled graph will apply.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from core.logger import get_logger
from fsm.state import FailureObject

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
_NODE_NAME = "node_programmatic_audit"

DistanceProvider = Callable[[str], float]

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_BE_VERB_RE = re.compile(r"\b(?:am|is|are|was|were|be|been|being)\s+(\w+)\b", re.IGNORECASE)
_MARKDOWN_DELIMITERS = ("**", "`")

# Deterministic heuristic, not a linguistic parser: a "be" verb immediately followed by
# a regular ("-ed") or listed irregular past participle. Intentionally over-inclusive on
# rare irregulars and under-inclusive on participles it doesn't list — documented here as
# a heuristic rather than presented as grammatically exact.
_IRREGULAR_PAST_PARTICIPLES = frozenset(
    {
        "written", "seen", "done", "made", "known", "given", "taken", "shown", "left",
        "found", "held", "built", "sent", "brought", "thought", "kept", "told",
        "understood", "chosen", "broken", "spoken", "driven", "eaten", "forgotten",
        "hidden", "ridden", "risen", "sung", "swum", "worn", "torn", "born", "drawn",
        "grown", "thrown", "flown", "blown", "sewn", "mown", "hewn", "begun", "frozen",
        "stolen", "woken", "bitten", "bound", "caught", "dealt", "felt", "fought",
        "heard", "led", "lost", "meant", "met", "paid", "said", "sold", "sought",
        "taught", "won",
    }
)


def _resolve_config(state: dict[str, Any]) -> Any:
    """Return the active typed config (mirrors the built nodes' resolution)."""
    config = state.get("app_config") or state.get("config")
    if config is not None:
        return config
    from core.config_loader import load_config

    return load_config(state.get("config_path", CONFIG_PATH))


def _is_passive_sentence(sentence: str) -> bool:
    """True if a "be" verb in ``sentence`` is followed by a past-participle-shaped word."""
    for match in _BE_VERB_RE.finditer(sentence):
        word = match.group(1).lower()
        if word.endswith("ed") or word in _IRREGULAR_PAST_PARTICIPLES:
            return True
    return False


def _passive_failure(draft_text: str, threshold: float) -> FailureObject | None:
    """One aggregate density-based passive-voice failure, or None below threshold."""
    sentences = [s for s in _SENTENCE_BOUNDARY_RE.split(draft_text) if s.strip()]
    if not sentences:
        return None
    passive_sentences = [s for s in sentences if _is_passive_sentence(s)]
    if not passive_sentences:
        return None
    density = len(passive_sentences) / len(sentences)
    if density <= threshold:
        return None
    worst = max(passive_sentences, key=len).strip()
    return FailureObject(
        error_code="PASSIVE_VOICE_DENSITY",
        offending_text=worst,
        suggested_fix=(
            f"Rewrite flagged sentences in active voice: passive density "
            f"{density:.2f} exceeds the {threshold:.2f} threshold."
        ),
        critic_source="programmatic_audit",
    )


def _structural_failures(draft_text: str) -> list[FailureObject]:
    """Cheap layout checks: empty draft, unclosed quote, unclosed markdown span."""
    if not draft_text.strip():
        return [
            FailureObject(
                error_code="EMPTY_DRAFT",
                offending_text="",
                suggested_fix="Generate non-empty prose for this beat.",
                critic_source="programmatic_audit",
            )
        ]

    failures: list[FailureObject] = []

    if draft_text.count('"') % 2 == 1:
        tail_start = draft_text.rfind('"')
        failures.append(
            FailureObject(
                error_code="UNCLOSED_QUOTE",
                offending_text=draft_text[tail_start:],
                suggested_fix="Close the open double-quoted dialogue or quotation.",
                critic_source="programmatic_audit",
            )
        )

    for delim in _MARKDOWN_DELIMITERS:
        if draft_text.count(delim) % 2 == 1:
            tail_start = draft_text.rfind(delim)
            failures.append(
                FailureObject(
                    error_code="UNCLOSED_MARKDOWN",
                    offending_text=draft_text[tail_start:],
                    suggested_fix=f"Close the open '{delim}' markdown span.",
                    critic_source="programmatic_audit",
                )
            )

    return failures


def _run_checks(draft_text: str, threshold: float) -> list[FailureObject]:
    """All Stage-1 checks against a single draft text (pure function of text + config)."""
    failures = _structural_failures(draft_text)
    passive = _passive_failure(draft_text, threshold)
    if passive is not None:
        failures.append(passive)
    return failures


def _default_distance_provider(_draft_text: str) -> float:
    """Stubbed stylometric distance pending the M02 style store."""
    return 0.0


async def node_programmatic_audit(
    state: dict[str, Any],
    *,
    distance_provider: DistanceProvider | None = None,
) -> dict[str, Any]:
    """Run the model-free Stage-1 audit against the active draft.

    ``distance_provider`` is an injectable seam for ``stylometric_distance``; the
    default (``None``) uses the stubbed provider and logs that condition rather than
    reading the still-stubbed style store.
    """
    config = _resolve_config(state)
    threshold = config.thresholds.passive_voice_density
    draft_text = state.get("current_draft_text") or ""

    new_failures = _run_checks(draft_text, threshold)

    stubbed = distance_provider is None
    provider = distance_provider if distance_provider is not None else _default_distance_provider
    distance = provider(draft_text)
    if stubbed:
        logger = get_logger(_NODE_NAME)
        logger.info(
            json.dumps(
                {
                    "node_name": _NODE_NAME,
                    "note": "stylometric_distance stubbed at 0.0 pending memory/style_store.py",
                },
                separators=(",", ":"),
            )
        )

    best_seen = state.get("best_seen_draft")
    if best_seen is None:
        should_update_best = True
    else:
        best_failure_count = len(_run_checks(best_seen, threshold))
        should_update_best = len(new_failures) < best_failure_count

    state["stylometric_distance"] = distance
    # Delta only — the reducer contract merges this with prior history; a shared list
    # reference is never mutated in place here.
    state["critic_failures"] = new_failures
    if should_update_best:
        state["best_seen_draft"] = draft_text

    return state
