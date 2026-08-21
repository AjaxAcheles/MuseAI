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
from difflib import SequenceMatcher

from museai.core.logging_setup import log_node_event
from museai.core.stream_bus import bus
from museai.fsm.nodes.deps import get_node_config
from museai.fsm.state import FailureObject, OrchestratorState

CRITIC_SOURCE = "programmatic_audit"
ERROR_CODE = "PASSIVE_VOICE_DENSITY"
OVERLAP_ERROR_CODE = "PARAGRAPH_OVERLAP"
EMOTION_ERROR_CODE = "EMOTION_TELL"
TIC_ERROR_CODE = "STYLE_TIC"
POV_ERROR_CODE = "POINT_OF_VIEW_INTRUSION"

# Sentence split on terminal punctuation followed by whitespace. Abbreviations
# ("Dr. Vance") over-split, which costs at most one extra sentence in the
# denominator — it never invents a passive, so it can only make the check more
# forgiving, which is the right way for a heuristic gate to be wrong.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[\"')\]”’]*\s+")

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


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` offsets of the non-empty split sentences."""
    spans: list[tuple[int, int]] = []
    start = 0
    for boundary in _SENTENCE_SPLIT.finditer(text):
        segment = text[start : boundary.start()]
        left = len(segment) - len(segment.lstrip())
        right = len(segment.rstrip())
        if left < right:
            spans.append((start + left, start + right))
        start = boundary.end()

    segment = text[start:]
    left = len(segment) - len(segment.lstrip())
    right = len(segment.rstrip())
    if left < right:
        spans.append((start + left, start + right))
    return spans


def split_sentences(text: str) -> list[str]:
    """Split prose into non-empty sentences."""
    return [text[start:end] for start, end in sentence_spans(text)]


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


# ------------------------------------------------------------ repetition guard

# Paragraphs are separated by a blank line. A drafted beat that reproduces a
# committed one copies whole paragraphs, so paragraph is the right unit: a lone
# repeated *sentence* sitting inside an otherwise-new paragraph never trips this,
# which is what lets a short deliberate refrain through.
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")

# For comparison only: collapse whitespace and drop punctuation/case so that
# "He said, 'stop.'" and "He said 'Stop'" read as the same prose. The offending
# text quoted to the reviser is always the original, never this normalized form.
_NON_COMPARE = re.compile(r"[^\w\s]")


def split_paragraphs(text: str) -> list[str]:
    """Split prose into non-empty paragraphs, preserving the original text."""
    return [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]


def _normalize_for_compare(text: str) -> str:
    return " ".join(_NON_COMPARE.sub(" ", text).lower().split())


def _is_allowlisted(paragraph_norm: str, allowlist_norm: list[str]) -> bool:
    """A paragraph is exempt if it carries a declared refrain (as a substring)."""
    return any(phrase and phrase in paragraph_norm for phrase in allowlist_norm)


def _normalized_verbatim_phrase_matches(
    phrase_norm: str, passages: list[tuple[str, str]], *, max_words: int
) -> list[str]:
    """Return original passages whose supplied normalized text contains a phrase."""
    if not phrase_norm or len(phrase_norm.split()) > max_words:
        return []
    return [passage for passage, norm in passages if phrase_norm in norm]


def verbatim_phrase_matches(
    phrase: str, passages: list[str], *, max_words: int
) -> list[str]:
    """Return passages containing a normalized short phrase verbatim.

    This is shared by the audit's short-echo guard and ``find_repetition`` so
    they use one bounded phrase-matching rule.
    """
    phrase_norm = _normalize_for_compare(phrase)
    normalized_passages = [
        (passage, _normalize_for_compare(passage)) for passage in passages
    ]
    return _normalized_verbatim_phrase_matches(
        phrase_norm, normalized_passages, max_words=max_words
    )


def longest_shared_verbatim_word_run(text: str, passages: list[str]) -> int:
    """Return the longest normalized consecutive word run shared with a passage."""
    words = _normalize_for_compare(text).split()
    if not words:
        return 0

    longest = 0
    for passage in passages:
        passage_words = _normalize_for_compare(passage).split()
        previous = [0] * (len(passage_words) + 1)
        for word in words:
            current = [0] * (len(passage_words) + 1)
            for index, passage_word in enumerate(passage_words, start=1):
                if word == passage_word:
                    current[index] = previous[index - 1] + 1
                    longest = max(longest, current[index])
            previous = current
    return longest


def phrase_echoes(
    draft: str,
    committed: list[str],
    *,
    min_words: int,
    allowlist: list[str],
) -> list[str]:
    """Return paragraphs with a non-allowlisted verbatim run of ``min_words`` or more; there is no maximum."""
    allowlist_norm = [_normalize_for_compare(p) for p in allowlist]
    corpus_ngrams = {
        tuple(words[index : index + min_words])
        for passage in committed
        for paragraph in split_paragraphs(passage)
        for words in [_normalize_for_compare(paragraph).split()]
        for index in range(len(words) - min_words + 1)
    }

    offenders: list[str] = []
    for paragraph in split_paragraphs(draft):
        words = _normalize_for_compare(paragraph).split()
        ngrams = [
            tuple(words[index : index + min_words])
            for index in range(len(words) - min_words + 1)
        ]
        norm = " ".join(words)
        if not norm or _is_allowlisted(norm, allowlist_norm):
            corpus_ngrams.update(ngrams)
            continue

        if any(ngram in corpus_ngrams for ngram in ngrams):
            offenders.append(paragraph)
        corpus_ngrams.update(ngrams)
    return offenders


def paragraph_overlaps(
    draft: str,
    committed: list[str],
    *,
    threshold: float,
    min_sentences: int,
    allowlist: list[str],
) -> list[str]:
    """Return drafted paragraphs that duplicate committed (or earlier draft) prose.

    A paragraph is faulted when its normalized similarity to any committed
    paragraph — or any *earlier* paragraph in the same draft — reaches
    ``threshold`` AND it is at least ``min_sentences`` sentences long (the size
    gate that lets short deliberate refrains pass) AND it is not allowlisted.
    """
    allowlist_norm = [_normalize_for_compare(p) for p in allowlist]
    # Each committed entry is a whole beat's prose; compare paragraph-to-paragraph,
    # so split every committed passage into its paragraphs first.
    committed_norm = [
        _normalize_for_compare(para)
        for passage in committed
        for para in split_paragraphs(passage)
    ]

    draft_paragraphs = split_paragraphs(draft)
    offenders: list[str] = []
    seen_norm: list[str] = []
    for paragraph in draft_paragraphs:
        norm = _normalize_for_compare(paragraph)
        # Compare against committed prose and paragraphs already seen in this
        # draft, so a beat that repeats itself is caught too.
        corpus = committed_norm + seen_norm
        seen_norm.append(norm)
        if not norm or len(split_sentences(paragraph)) < min_sentences:
            continue
        if _is_allowlisted(norm, allowlist_norm):
            continue
        best = max(
            (SequenceMatcher(None, norm, other).ratio() for other in corpus if other),
            default=0.0,
        )
        if best >= threshold:
            offenders.append(paragraph)
    return offenders


# ---------------------------------------------------------- emotion-tell guard


def _emotion_pattern(words: list[str]) -> re.Pattern[str] | None:
    """A whole-word alternation of the configured emotion vocabulary."""
    cleaned = [re.escape(w.strip()) for w in words if w.strip()]
    if not cleaned:
        return None
    return re.compile(rf"\b(?:{'|'.join(cleaned)})\b", re.IGNORECASE)


def emotion_word_density(text: str, pattern: re.Pattern[str] | None) -> tuple[float, list[str]]:
    """Proportion of sentences that name an emotion outright, and those sentences.

    A named emotion is the classic *tell*: "she felt pure panic" states the score
    the prose was supposed to dramatize. An empty draft, or no vocabulary, is 0.0.
    """
    if pattern is None:
        return 0.0, []
    sentences = split_sentences(text)
    if not sentences:
        return 0.0, []
    offenders = [s for s in sentences if pattern.search(s)]
    return len(offenders) / len(sentences), offenders


# First-person pronouns are evidence of a third-person intrusion only outside
# dialogue. ``I`` followed by a period is skipped: that inexpensive exception
# avoids treating an initial or acronym component as narration.
_FIRST_PERSON_PRONOUN = re.compile(
    r"\b(?:me|my|mine|myself|we|us|our|ours|ourselves)\b|\b(?-i:I)\b(?!\.)",
    re.IGNORECASE,
)


def _is_single_quote_opener(text: str, index: int) -> bool:
    """True only for a leading single-quote dialogue delimiter, never ``Nell's``."""
    previous = text[index - 1] if index else ""
    following = text[index + 1] if index + 1 < len(text) else ""
    return not previous.isalnum() and following.isalpha()


def _strip_dialogue(text: str) -> str:
    """Mask paired quoted dialogue while retaining sentence boundaries.

    Straight and typographic double quotes are ordinary delimiters. A straight
    single quote opens dialogue only at a word boundary, so possessive and
    contraction apostrophes cannot swallow later narration. Unpaired marks are
    left alone: malformed punctuation must not make the rest of a beat invisible
    to an audit.
    """
    masked = list(text)
    quote_end: str | None = None
    start: int | None = None

    for index, character in enumerate(text):
        if quote_end is None:
            if character == '"':
                quote_end, start = '"', index
            elif character == "“":
                quote_end, start = "”", index
            elif character == "‘":
                quote_end, start = "’", index
            elif character == "'" and _is_single_quote_opener(text, index):
                quote_end, start = "'", index
            continue

        is_apostrophe = (
            quote_end == "'"
            and character == "'"
            and index > 0
            and index + 1 < len(text)
            and text[index - 1].isalnum()
            and text[index + 1].isalnum()
        )
        if character != quote_end or is_apostrophe:
            continue

        assert start is not None
        for masked_index in range(start, index + 1):
            if masked[masked_index] not in "\r\n.!?\"'”’":
                masked[masked_index] = " "
        quote_end, start = None, None

    return "".join(masked)


def first_person_narration_sentences(text: str) -> list[str]:
    """Return original sentences with first-person narration outside dialogue."""
    masked = _strip_dialogue(text)
    # The masked text is character-for-character with the original, but it must
    # never be re-split: masking ``"It is 9 a.m., Nell,"`` leaves the period
    # followed by a space, creating a boundary absent from the original.
    return [
        text[start:end]
        for start, end in sentence_spans(text)
        if _FIRST_PERSON_PRONOUN.search(masked[start:end])
    ]


def _list_offenders(sentences: list[str], budget: int) -> str:
    """Render every offending sentence for the revision prompt, not the draft.

    A density finding is about the whole beat, not one sentence — pointing the
    reviser at only the first offender lets it fix that single span while the
    proportion barely moves. This goes into ``suggested_fix``, which is
    instruction prose the reviser reads, never text matched against the draft
    — unlike ``offending_text``, which must stay a single sentence so
    ``revise.locate`` can find it as one contiguous span and keep this failure
    in span mode instead of dragging every co-occurring failure into a
    full-beat rewrite.

    Whole sentences are dropped rather than the joined string sliced: a quote
    cut mid-word is not locatable prose, and a reviser told "all offending
    sentences" and then shown two-and-a-half of them fixes what it can see and
    leaves the proportion where it was. What does not fit is counted out loud.
    """
    kept: list[str] = []
    used = 0
    for index, sentence in enumerate(sentences):
        quoted = f'"{sentence}"'
        # "; " between entries, and room for the "(+N more)" tail if any
        # sentence after this one will have to be dropped.
        separator = 2 if kept else 0
        remaining = len(sentences) - index - 1
        tail = len(f" (+{remaining} more)") if remaining else 0
        if used + separator + len(quoted) + tail > budget:
            break
        kept.append(quoted)
        used += separator + len(quoted)

    if not kept:
        # A single sentence longer than the whole budget: one truncated quote
        # still beats saying nothing about a breach the reviser has to fix.
        return f'"{sentences[0][:budget]}"' if sentences else ""

    listed = "; ".join(kept)
    dropped = len(sentences) - len(kept)
    return f"{listed} (+{dropped} more)" if dropped else listed


async def audit(state: OrchestratorState) -> dict:
    """Run the programmatic checks over ``current_draft_text``.

    Returns the state delta ``{"critic_failures": [...]}``. An empty list resets
    the reducer, which is what a fresh audit of a new draft should do: the
    previous cycle's findings describe prose that no longer exists.

    Six model-free checks run here: passive-voice density, verbatim paragraph
    and short-phrase overlap against committed prose, named-emotion density,
    stock-phrase (style-tic) density, and declared-third-person POV intrusion.
    All append ``FailureObject``s to the same list,
    which flows into the existing draft→audit→revise loop.
    """
    config = get_node_config()
    generation = config.generation
    threshold = generation.passive_voice_threshold
    # The offending prose quoted back to the reviser. Enough to locate it.
    quote_chars = generation.audit_quote_chars
    # Wider, and for prose the reviser only reads: the full offender list a
    # density failure appends to its suggested_fix. See `_list_offenders`.
    offender_list_chars = generation.audit_offender_list_chars
    draft = state["current_draft_text"]
    beat_index = state["fsm_pointer"].beat_index

    min_sentences = generation.passive_min_sentences
    all_sentences = split_sentences(draft)
    density, passives = passive_voice_density(draft)
    breached = len(all_sentences) >= min_sentences and density > threshold

    failures: list[FailureObject] = []
    if breached:
        failures.append(
            FailureObject(
                error_code=ERROR_CODE,
                offending_text=passives[0][:quote_chars],
                suggested_fix=(
                    f"{density:.0%} of sentences in this beat are in the passive "
                    f"voice, over the {threshold:.0%} limit. Rewrite the passive "
                    f"clauses so the actor performs the verb. All offending "
                    f"sentences: {_list_offenders(passives, offender_list_chars)}"
                ),
                critic_source=CRITIC_SOURCE,
                whole_draft=True,
            )
        )

    # --- repetition guard ---------------------------------------------------
    package = state.get("active_context_package") or {}
    # Compare against the whole committed manuscript when the package carries
    # it: the recent-prose window is token-budgeted for the drafter and shrinks
    # exactly when the manuscript grows, which is when distant copies appear.
    # ``committed_prose`` is never rendered into a prompt, so it costs nothing.
    corpus = package.get("committed_prose") or package.get("recent_prose") or []
    # Effective allowlist: author-declared config phrases + this beat's declared
    # refrain. The drafter can never write to either, so it cannot exempt its
    # own copies.
    beat = package.get("beat") or {}
    allowlist = list(generation.repetition_allowlist) + list(
        beat.get("intended_refrain") or []
    )
    overlaps = paragraph_overlaps(
        draft,
        corpus,
        threshold=generation.repetition_threshold,
        min_sentences=generation.repetition_min_run,
        allowlist=allowlist,
    )
    echoes = [
        paragraph
        for paragraph in phrase_echoes(
            draft,
            corpus,
            min_words=generation.repetition_min_phrase_words,
            allowlist=allowlist,
        )
        if paragraph not in overlaps
    ]
    for paragraph in overlaps:
        failures.append(
            FailureObject(
                error_code=OVERLAP_ERROR_CODE,
                offending_text=paragraph[:quote_chars],
                suggested_fix=(
                    "This paragraph duplicates prose already committed earlier in "
                    "the manuscript. Do not restate it — write fresh prose that "
                    "moves the beat forward from where the story now stands."
                ),
                critic_source=CRITIC_SOURCE,
            )
        )
    for paragraph in echoes:
        failures.append(
            FailureObject(
                error_code=OVERLAP_ERROR_CODE,
                offending_text=paragraph[:quote_chars],
                suggested_fix=(
                    "This paragraph reuses a short phrase from prose already "
                    "committed earlier in the manuscript. Rewrite the repeated "
                    "words in fresh prose that moves the beat forward."
                ),
                critic_source=CRITIC_SOURCE,
            )
        )

    # --- emotion-tell guard -------------------------------------------------
    emotion_pattern = _emotion_pattern(generation.emotion_words)
    emotion_density, emotion_sentences = emotion_word_density(draft, emotion_pattern)
    emotion_breached = (
        len(all_sentences) >= min_sentences
        and emotion_density > generation.emotion_word_threshold
    )
    if emotion_breached:
        failures.append(
            FailureObject(
                error_code=EMOTION_ERROR_CODE,
                offending_text=emotion_sentences[0][:quote_chars],
                suggested_fix=(
                    f"{emotion_density:.0%} of sentences name an emotion outright, "
                    f"over the {generation.emotion_word_threshold:.0%} limit. Cut "
                    f"or understate the named emotions — let the beat's events and "
                    f"the character's choices imply the feeling. Do not add new "
                    f"emotional description to compensate. All offending "
                    f"sentences: {_list_offenders(emotion_sentences, offender_list_chars)}"
                ),
                critic_source=CRITIC_SOURCE,
                whole_draft=True,
            )
        )

    # --- style-tic guard -----------------------------------------------------
    # The same sentence-density machinery as the emotion gate, over a separate
    # vocabulary: stock gestures ("trembling", "deep breath") and abstract
    # emotional shorthand ("the weight of", "closure") that generated drafts
    # reach for instead of a specific action or image.
    tic_pattern = _emotion_pattern(generation.tic_phrases)
    tic_density, tic_sentences = emotion_word_density(draft, tic_pattern)
    tic_breached = (
        len(all_sentences) >= min_sentences
        and tic_density > generation.tic_phrase_threshold
    )
    if tic_breached:
        failures.append(
            FailureObject(
                error_code=TIC_ERROR_CODE,
                offending_text=tic_sentences[0][:quote_chars],
                suggested_fix=(
                    f"{tic_density:.0%} of sentences lean on a stock gesture or "
                    f"abstract emotional shorthand, over the "
                    f"{generation.tic_phrase_threshold:.0%} limit. Replace them "
                    f"with actions and images specific to this character, this "
                    f"place, and this moment — do not swap one stock phrase for "
                    f"another. All offending sentences: "
                    f"{_list_offenders(tic_sentences, offender_list_chars)}"
                ),
                critic_source=CRITIC_SOURCE,
                whole_draft=True,
            )
        )

    # --- point-of-view guard -----------------------------------------------
    # First-person narration is valid when the author chose it, and third-person
    # pronouns are normal when a first-person narrator refers to other people.
    # The one reliable direction is therefore a declared third-person draft
    # slipping into first person. Dialogue is deliberately masked before the
    # pronoun scan: characters say "I" constantly in third-person fiction.
    pov_sentences: list[str] = []
    pov_intrusions: int | None = None
    if generation.narrative_person == "third":
        pov_sentences = first_person_narration_sentences(draft)
        pov_intrusions = len(pov_sentences)
        # Unlike a density breach, each POV intrusion is an independent,
        # locatable sentence. One span-local failure per sentence lets a single
        # span-mode revision correct every intrusion in this draft.
        for sentence in pov_sentences:
            failures.append(
                FailureObject(
                    error_code=POV_ERROR_CODE,
                    offending_text=sentence[:quote_chars],
                    suggested_fix=(
                        "First-person narration appears in this declared "
                        "third-person draft. Rewrite this sentence in the "
                        "established third-person point of view."
                    ),
                    critic_source=CRITIC_SOURCE,
                    whole_draft=False,
                )
            )

    log_node_event(
        "audit",
        event="audited",
        beat_index=beat_index,
        passive_density=f"{density:.3f}",
        threshold=threshold,
        passive_sentences=len(passives),
        paragraph_overlaps=len(overlaps),
        phrase_echoes=len(echoes),
        emotion_density=f"{emotion_density:.3f}",
        emotion_sentences=len(emotion_sentences),
        tic_density=f"{tic_density:.3f}",
        tic_sentences=len(tic_sentences),
        narrative_person=generation.narrative_person,
        pov_intrusions=pov_intrusions,
        failures=len(failures),
    )
    await bus.publish(
        "audit",
        {
            "beat_index": beat_index,
            "passive_density": round(density, 3),
            "threshold": threshold,
            "paragraph_overlaps": len(overlaps),
            "phrase_echoes": len(echoes),
            "emotion_density": round(emotion_density, 3),
            "tic_density": round(tic_density, 3),
            "narrative_person": generation.narrative_person,
            "pov_intrusions": pov_intrusions,
            "failures": [f.model_dump() for f in failures],
        },
    )

    return {
        "critic_failures": failures,
        "repetition_overlap_count": len(overlaps) + len(echoes),
    }
