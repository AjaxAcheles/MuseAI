"""Run the deterministic prose audit on a draft, before it is submitted.

This is the exact same set of model-free checks the ``audit`` node will run —
passive-voice density, verbatim prose overlap with recent committed prose,
and named-emotion density — offered as a tool so the drafter and reviser can
self-correct instead of paying a full critic-revise round trip for a problem a
regex can name.
"""

from __future__ import annotations

from typing import Any

from museai.fsm.nodes.audit import (
    EMOTION_ERROR_CODE,
    ERROR_CODE,
    OVERLAP_ERROR_CODE,
    _emotion_pattern,
    emotion_word_density,
    paragraph_overlaps,
    phrase_echoes,
    passive_voice_density,
)
from museai.fsm.nodes.deps import get_node_config
from museai.fsm.tools.project_db import project_connection
from museai.memory.db import get_recent_committed_beats

CHECK_DRAFT_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "check_draft",
        "description": (
            "Run the deterministic prose checks that will audit your draft: "
            "passive-voice density, prose that duplicates recently "
            "committed prose, and sentences that name an emotion outright. "
            "Returns pass/fail with the offending sentences, so you can fix "
            "them before answering."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The draft prose to check.",
                },
            },
            "required": ["text"],
        },
    },
}


def _quotes(sentences: list[str]) -> list[str]:
    """Enough of each offending sentence to find it again in the draft."""
    tools = get_node_config().tools
    return [
        s[: tools.check_draft_quote_chars]
        for s in sentences[: tools.check_draft_max_quotes]
    ]


def check_draft(text: str) -> dict:
    """The audit's verdict on ``text``: ``passes`` plus one finding per breach."""
    draft = (text or "").strip()
    if not draft:
        return {"error": "no text was given to check"}

    config = get_node_config()
    generation = config.generation
    findings: list[dict] = []

    density, passives = passive_voice_density(draft)
    if density > generation.passive_voice_threshold:
        findings.append(
            {
                "error_code": ERROR_CODE,
                "detail": (
                    f"{density:.0%} of sentences are in the passive voice, over "
                    f"the {generation.passive_voice_threshold:.0%} limit. Rewrite "
                    f"the passive clauses so the actor performs the verb."
                ),
                "offending": _quotes(passives),
            }
        )

    with project_connection() as (conn, project_id):
        recent = get_recent_committed_beats(
            conn, project_id, generation.recent_prose_beats
        )
    overlaps = paragraph_overlaps(
        draft,
        [row["prose"] for row in recent],
        threshold=generation.repetition_threshold,
        min_sentences=generation.repetition_min_run,
        allowlist=list(generation.repetition_allowlist),
    )
    if overlaps:
        findings.append(
            {
                "error_code": OVERLAP_ERROR_CODE,
                "detail": (
                    "These paragraphs duplicate prose already committed. Write "
                    "fresh prose that moves the beat forward instead."
                ),
                "offending": _quotes(overlaps),
            }
        )

    echoes = [
        paragraph
        for paragraph in phrase_echoes(
            draft,
            [row["prose"] for row in recent],
            min_words=generation.repetition_min_phrase_words,
            allowlist=list(generation.repetition_allowlist),
        )
        if paragraph not in overlaps
    ]
    if echoes:
        findings.append(
            {
                "error_code": OVERLAP_ERROR_CODE,
                "detail": (
                    "These paragraphs reuse short phrases from prose already "
                    "committed. Rewrite the repeated words in fresh prose that "
                    "moves the beat forward instead."
                ),
                "offending": _quotes(echoes),
            }
        )

    emotion_density, tells = emotion_word_density(
        draft, _emotion_pattern(generation.emotion_words)
    )
    if emotion_density > generation.emotion_word_threshold:
        findings.append(
            {
                "error_code": EMOTION_ERROR_CODE,
                "detail": (
                    f"{emotion_density:.0%} of sentences name an emotion "
                    f"outright, over the "
                    f"{generation.emotion_word_threshold:.0%} limit. Show the "
                    f"feeling through action and perception instead."
                ),
                "offending": _quotes(tells),
            }
        )

    return {"passes": not findings, "findings": findings}
