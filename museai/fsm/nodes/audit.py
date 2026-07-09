"""Programmatic audit of the current draft.

Cheap, deterministic, model-free checks that run before the continuity critic
spends a token. Anything a regex can decide should not cost an inference call.

v1 audits exactly one thing: **passive-voice density**. There is no drift
metric, no stylometric score, no style store — those are not in v1, and a
number nobody computes is worse than no number at all.

The check is a proportion, never an absolute count. One passive sentence in a
long beat is prose; half the beat in passive voice is a draft that reads limp.
Only the proportion crossing ``generation.passive_voice_threshold`` raises a
finding, and it raises exactly one: a `PASSIVE_VOICE_DENSITY` `FailureObject`
against the beat, not one per offending sentence.

The detector is a regex, not a parser, and every ambiguity in it is resolved
toward *not* flagging. A missed passive costs one limp sentence; a false one
sends the reviser to rewrite prose that was already fine.
"""

from __future__ import annotations

import re

from museai.core.logging_setup import log_node_event
from museai.core.stream_bus import bus
from museai.fsm.nodes.deps import get_node_config
from museai.fsm.state import FailureObject, OrchestratorState

CRITIC_SOURCE = "programmatic_audit"
ERROR_CODE = "PASSIVE_VOICE_DENSITY"

# The offending sentence quoted back to the reviser. Enough to locate it.
_QUOTE_CHARS = 240

# Sentence split on terminal punctuation followed by whitespace. Abbreviations
# ("Dr. Vance") over-split, which costs at most one extra sentence in the
# denominator — it never invents a passive, so it can only make the check more
# forgiving, which is the right way for a heuristic gate to be wrong.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+")

# A passive clause in English is an inflection of "to be" (or the colloquial
# "get" passive) followed by a past participle, optionally with an adverb
# wedged between: "was opened", "is quietly taken", "had been written".
#
# Regular participles end in -ed and are matched by shape. Irregular ones do not
# and cannot be, so they are enumerated: an -en/-n ending alone would swallow
# "was often" and "is green". The two together cover ordinary prose without a
# parser, and the residue (a missed irregular) errs toward *not* flagging.
_IRREGULAR_PARTICIPLES = frozenset(
    """
    been begun bent bitten blown born borne bought bound broken brought built
    burnt caught chosen clung cut dealt done drawn driven drunk eaten fallen fed
    felt fought found flown forgiven forgotten frozen given gone ground grown
    heard held hidden hit hung hurt kept knelt known laid led left lent let lit
    lost made meant met paid put quit read ridden risen run said seen sent set
    sewn shaken shed shone shot shown shut slain slept slid sold sought sown
    spent spilt split spoken spread sprung stolen struck strung stung sunk
    swept swollen sworn swum taken taught thrown told torn understood upset
    withdrawn woken won worn wound written
    """.split()
)

# "She was tired" has the shape of a passive but is a copula plus a predicate
# adjective, and no regex can tell it from "She was seized" structurally. The
# frequent offenders are enumerated and excused. A missed passive costs nothing;
# faulting a writer for "she was worried" would train the reviser to write worse.
_ADJECTIVAL_PARTICIPLES = frozenset(
    """
    amused annoyed ashamed bored concerned confused crowded delighted
    determined disappointed embarrassed excited exhausted frightened interested
    involved pleased prepared relieved satisfied scared surprised tired troubled
    used worried
    """.split()
)

_BE_FORMS = r"(?:am|is|are|was|were|be|been|being|get|gets|got|gotten)"

# Optionally one adverb ("was quietly opened") or a negation ("was not opened").
_INTERVENING = r"(?:\s+(?:not|never|already|just|also|\w+ly))?"

_PASSIVE = re.compile(
    rf"\b{_BE_FORMS}\b{_INTERVENING}\s+(\w+)\b",
    re.IGNORECASE,
)


def _is_participle(word: str) -> bool:
    """True when ``word`` can be the past participle of a passive clause."""
    lowered = word.lower()
    if lowered in _ADJECTIVAL_PARTICIPLES:
        return False
    if lowered in _IRREGULAR_PARTICIPLES:
        return True
    # "need", "seed", "speed" are nouns, not participles; -ed on a stem of at
    # least four characters is the reliable shape.
    return lowered.endswith("ed") and len(lowered) >= 4


def split_sentences(text: str) -> list[str]:
    """Split prose into non-empty sentences."""
    return [s for s in (part.strip() for part in _SENTENCE_SPLIT.split(text)) if s]


def is_passive(sentence: str) -> bool:
    """True when the sentence contains at least one passive-voice clause."""
    return any(_is_participle(match.group(1)) for match in _PASSIVE.finditer(sentence))


def passive_voice_density(text: str) -> tuple[float, list[str]]:
    """Return the passive proportion of ``text`` and the offending sentences.

    An empty draft has a density of ``0.0`` — nothing to fault.
    """
    sentences = split_sentences(text)
    if not sentences:
        return 0.0, []

    passives = [sentence for sentence in sentences if is_passive(sentence)]
    return len(passives) / len(sentences), passives


async def audit(state: OrchestratorState) -> dict:
    """Run the programmatic checks over ``current_draft_text``.

    Returns the state delta ``{"critic_failures": [...]}``. An empty list resets
    the reducer, which is what a fresh audit of a new draft should do: the
    previous cycle's findings describe prose that no longer exists.
    """
    config = get_node_config()
    threshold = config.generation.passive_voice_threshold
    draft = state["current_draft_text"]
    beat_index = state["fsm_pointer"].beat_index

    density, passives = passive_voice_density(draft)
    breached = density > threshold

    failures: list[FailureObject] = []
    if breached:
        failures.append(
            FailureObject(
                error_code=ERROR_CODE,
                offending_text=passives[0][:_QUOTE_CHARS],
                suggested_fix=(
                    f"{density:.0%} of sentences in this beat are in the passive "
                    f"voice, over the {threshold:.0%} limit. Rewrite the passive "
                    f"clauses so the actor performs the verb."
                ),
                critic_source=CRITIC_SOURCE,
            )
        )

    log_node_event(
        "audit",
        event="audited",
        beat_index=beat_index,
        passive_density=f"{density:.3f}",
        threshold=threshold,
        passive_sentences=len(passives),
        failures=len(failures),
    )
    await bus.publish(
        "audit",
        {
            "beat_index": beat_index,
            "passive_density": round(density, 3),
            "threshold": threshold,
            "failures": [f.model_dump() for f in failures],
        },
    )

    return {"critic_failures": failures}
