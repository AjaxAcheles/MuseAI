"""Tests for museai.fsm.nodes.audit.

No LLM, no socket, no DB: the audit node is pure text analysis over
``current_draft_text`` plus one config read.
"""

from __future__ import annotations

import pytest

from museai.fsm.nodes.audit import (
    CRITIC_SOURCE,
    EMOTION_ERROR_CODE,
    ERROR_CODE,
    OVERLAP_ERROR_CODE,
    TIC_ERROR_CODE,
    audit,
    emotion_word_density,
    is_passive,
    paragraph_overlaps,
    passive_voice_density,
    split_paragraphs,
    split_sentences,
    _emotion_pattern,
)
from museai.fsm.nodes.deps import set_node_config
from museai.fsm.state import FSM_Pointer, make_initial_state

# Six sentences, five of them passive. Density 0.83, well over any sane gate.
PASSIVE_HEAVY = (
    "The door was opened by Mara. "
    "The letters had been sorted by the clerk. "
    "A lamp was quietly lit in the hallway. "
    "The seal was broken. "
    "The message was written in her own hand. "
    "She stared at it."
)

# Active throughout. "red" and "tired" end in -ed but follow no copula in the
# passive shape; "she needed" is a verb, not a participle.
CLEAN = (
    "Mara opened the door. "
    "The clerk sorted the letters and set them on the sill. "
    "She lit a lamp. "
    "She broke the seal. "
    "Her own hand had written the message. "
    "She stared at it, tired and red-eyed, and needed a moment."
)


@pytest.fixture(autouse=True)
def _config(config_factory):
    set_node_config(config_factory(passive_voice_threshold=0.25))


def state_with(draft: str):
    return make_initial_state(
        "test-project",
        FSM_Pointer(arc_id="arc-1", chapter_id="arc-1-c01", beat_index=0),
        current_draft_text=draft,
    )


class TestSentenceSplitting:
    def test_terminal_punctuation_splits(self):
        assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]

    def test_empty_text_has_no_sentences(self):
        assert split_sentences("   \n  ") == []

    def test_a_closing_quote_stays_with_its_sentence(self):
        assert split_sentences('"Go," she said. He went.') == [
            '"Go," she said.',
            "He went.",
        ]


class TestPassiveDetection:
    @pytest.mark.parametrize(
        "sentence",
        [
            "The door was opened by Mara.",
            "The letters had been sorted.",
            "A lamp was quietly lit.",
            "The seal is broken.",
            "The message was never sent.",
            "The vase got smashed.",
            "The truth will be understood.",
        ],
    )
    def test_passive_clauses_are_detected(self, sentence):
        assert is_passive(sentence)

    @pytest.mark.parametrize(
        "sentence",
        [
            "Mara opened the door.",
            "The lamp is green.",
            "He needed a moment.",
            "She was often late.",
            "The clerk sorted the letters.",
        ],
    )
    def test_active_and_copular_clauses_are_not_flagged(self, sentence):
        assert not is_passive(sentence)

    @pytest.mark.parametrize(
        "sentence",
        [
            "She was tired.",
            "He was worried about the letter.",
            "They were interested in the seal.",
            "She was not surprised.",
        ],
    )
    def test_predicate_adjectives_are_excused_not_flagged_as_passive(self, sentence):
        """"was tired" is a copula plus adjective; the gate must not fault it."""
        assert not is_passive(sentence)


class TestDensity:
    def test_a_passive_heavy_passage_scores_high(self):
        density, passives = passive_voice_density(PASSIVE_HEAVY)
        assert density == pytest.approx(5 / 6)
        assert len(passives) == 5

    def test_a_clean_passage_scores_zero(self):
        density, passives = passive_voice_density(CLEAN)
        assert density == 0.0
        assert passives == []

    def test_an_empty_draft_has_no_density(self):
        assert passive_voice_density("") == (0.0, [])


class TestAuditNode:
    async def test_a_passive_heavy_passage_breaches_the_threshold(self):
        delta = await audit(state_with(PASSIVE_HEAVY))

        failures = delta["critic_failures"]
        assert len(failures) == 1
        failure = failures[0]
        assert failure.error_code == ERROR_CODE
        assert failure.critic_source == CRITIC_SOURCE
        assert failure.offending_text == "The door was opened by Mara."
        assert "83%" in failure.suggested_fix
        assert "25%" in failure.suggested_fix

    async def test_a_clean_passage_does_not_breach(self):
        assert await audit(state_with(CLEAN)) == {"critic_failures": []}

    async def test_one_failure_is_raised_for_the_beat_not_one_per_sentence(self):
        delta = await audit(state_with(PASSIVE_HEAVY))
        assert len(delta["critic_failures"]) == 1

    async def test_the_threshold_is_read_from_config(self, config_factory):
        # Under a permissive gate the same passive-heavy prose passes.
        set_node_config(config_factory(passive_voice_threshold=0.9))
        assert await audit(state_with(PASSIVE_HEAVY)) == {"critic_failures": []}

    async def test_exactly_at_the_threshold_does_not_breach(self, config_factory):
        # Four sentences, one passive: density 0.25, equal to the gate.
        set_node_config(config_factory(passive_voice_threshold=0.25))
        draft = "The seal was broken. She stood. She read it. She left."
        assert await audit(state_with(draft)) == {"critic_failures": []}

    async def test_an_empty_draft_yields_no_failures(self):
        assert await audit(state_with("")) == {"critic_failures": []}

    async def test_no_drift_or_stylometric_metric_is_reported(self):
        delta = await audit(state_with(PASSIVE_HEAVY))
        assert set(delta) == {"critic_failures"}


# --------------------------------------------------------------- repetition guard

# Two paragraphs, each three sentences, so both clear the min-run size gate.
PARA_A = (
    "Mara pressed her back to the cold stone. Her jaw locked hard. "
    "The lamp guttered and did not catch."
)
PARA_B = (
    "Tomas turned from the fractured bracket. He said nothing for a long moment. "
    "Then he pointed at the crack."
)
# A one-line refrain: below the size gate, so a deliberate repeat is never faulted.
REFRAIN = "The lantern must never go dark."


def state_with_package(
    draft: str, *, recent_prose=None, committed_prose=None, intended_refrain=None
):
    package = {
        "recent_prose": list(recent_prose or []),
        "beat": {"intended_refrain": list(intended_refrain or [])},
    }
    if committed_prose is not None:
        package["committed_prose"] = list(committed_prose)
    return make_initial_state(
        "test-project",
        FSM_Pointer(arc_id="arc-1", chapter_id="arc-1-c01", beat_index=0),
        current_draft_text=draft,
        active_context_package=package,
    )


class TestParagraphOverlap:
    def test_a_verbatim_copied_paragraph_is_flagged(self):
        draft = f"{PARA_A}\n\nA wholly new paragraph moves the beat onward from here."
        offenders = paragraph_overlaps(
            draft, [f"Earlier prose.\n\n{PARA_A}"],
            threshold=0.9, min_sentences=3, allowlist=[],
        )
        assert offenders == [PARA_A]

    def test_fresh_prose_is_not_flagged(self):
        draft = f"{PARA_B}\n\nAnother fresh paragraph, three short sentences. She moved. She spoke."
        offenders = paragraph_overlaps(
            draft, [PARA_A], threshold=0.9, min_sentences=3, allowlist=[]
        )
        assert offenders == []

    def test_a_short_refrain_stays_under_the_size_gate(self):
        # The refrain is one sentence; even repeated verbatim it never trips.
        draft = f"{REFRAIN}\n\nShe crossed to the window and watched the dark water."
        offenders = paragraph_overlaps(
            draft, [f"He had said it once: {REFRAIN}"],
            threshold=0.9, min_sentences=3, allowlist=[],
        )
        assert offenders == []

    def test_an_allowlisted_phrase_exempts_a_long_repeat(self):
        long_refrain = f"{REFRAIN} {REFRAIN} {REFRAIN}"  # now 3 sentences, over the gate
        draft = f"{long_refrain}\n\nFresh prose that carries on. She moved. She spoke."
        without = paragraph_overlaps(
            draft, [long_refrain], threshold=0.9, min_sentences=3, allowlist=[]
        )
        assert without == [long_refrain]
        with_allow = paragraph_overlaps(
            draft, [long_refrain], threshold=0.9, min_sentences=3, allowlist=[REFRAIN]
        )
        assert with_allow == []

    def test_intra_draft_repetition_is_caught(self):
        draft = f"{PARA_A}\n\nSomething different happens here. She turned. She left.\n\n{PARA_A}"
        offenders = paragraph_overlaps(
            draft, [], threshold=0.9, min_sentences=3, allowlist=[]
        )
        assert offenders == [PARA_A]  # the second occurrence duplicates the first

    def test_committed_passages_are_split_into_paragraphs(self):
        # A committed beat is one multi-paragraph string; the match is per-paragraph.
        committed = f"{PARA_B}\n\n{PARA_A}"
        draft = f"{PARA_A}\n\nNew prose entirely. She rose. She went."
        offenders = paragraph_overlaps(
            draft, [committed], threshold=0.9, min_sentences=3, allowlist=[]
        )
        assert offenders == [PARA_A]


class TestRepetitionAuditNode:
    async def test_overlap_becomes_a_failure_object(self):
        draft = f"{PARA_A}\n\nFresh continuation. She rose. She went to the door."
        delta = await audit(state_with_package(draft, recent_prose=[PARA_A]))
        overlaps = [f for f in delta["critic_failures"] if f.error_code == OVERLAP_ERROR_CODE]
        assert len(overlaps) == 1
        assert overlaps[0].critic_source == CRITIC_SOURCE
        assert overlaps[0].offending_text == PARA_A[:240]

    async def test_a_declared_refrain_exempts_the_paragraph(self):
        long_refrain = f"{REFRAIN} {REFRAIN} {REFRAIN}"
        draft = f"{long_refrain}\n\nFresh continuation. She rose. She left."
        state = state_with_package(
            draft, recent_prose=[long_refrain], intended_refrain=[REFRAIN]
        )
        delta = await audit(state)
        assert [f for f in delta["critic_failures"] if f.error_code == OVERLAP_ERROR_CODE] == []

    async def test_no_package_means_no_overlap_check(self):
        # Called without a context package (e.g. a bare unit invocation): the
        # overlap check simply finds no corpus and raises nothing.
        delta = await audit(state_with(f"{PARA_A}\n\n{PARA_B}"))
        assert [f for f in delta["critic_failures"] if f.error_code == OVERLAP_ERROR_CODE] == []

    async def test_a_copy_of_a_distant_chapter_is_caught(self):
        """The corpus is the whole committed manuscript, not the drafter's
        recent-prose window: a beat that copies a passage long since pruned
        from that window is still faulted."""
        draft = f"{PARA_A}\n\nFresh continuation. She rose. She went to the door."
        state = state_with_package(
            draft,
            recent_prose=[PARA_B],  # the copy source is NOT in the window
            committed_prose=[PARA_A, PARA_B],
        )
        delta = await audit(state)
        overlaps = [f for f in delta["critic_failures"] if f.error_code == OVERLAP_ERROR_CODE]
        assert len(overlaps) == 1
        assert overlaps[0].offending_text == PARA_A[:240]


# --------------------------------------------------------------- emotion-tell guard

EMOTION_HEAVY = (
    "Pure panic seized her. "
    "Absolute horror filled the room. "
    "He crossed to the desk and picked up the wrench. "
    "Visceral dread coiled in her chest."
)
EMOTION_SHOWN = (
    "Her hands shook as she reached for the seal. "
    "She crossed to the desk and picked up the wrench. "
    "Her jaw tightened until it ached. "
    "She did not look at the door."
)


class TestEmotionDensity:
    def test_named_emotions_are_counted(self):
        pattern = _emotion_pattern(["panic", "horror", "dread"])
        density, offenders = emotion_word_density(EMOTION_HEAVY, pattern)
        assert density == pytest.approx(3 / 4)
        assert len(offenders) == 3

    def test_shown_emotion_scores_zero(self):
        pattern = _emotion_pattern(["panic", "horror", "dread", "rage"])
        density, offenders = emotion_word_density(EMOTION_SHOWN, pattern)
        assert density == 0.0
        assert offenders == []

    def test_no_vocabulary_is_never_a_breach(self):
        assert emotion_word_density(EMOTION_HEAVY, _emotion_pattern([])) == (0.0, [])

    def test_whole_word_only(self):
        # "spite" must not fire on "respite"; word boundaries matter.
        pattern = _emotion_pattern(["spite"])
        density, offenders = emotion_word_density(
            "She worked without respite. She took no break.", pattern
        )
        assert offenders == []


class TestEmotionAuditNode:
    async def test_emotion_heavy_draft_breaches(self, config_factory):
        set_node_config(config_factory(emotion_word_threshold=0.3))
        delta = await audit(state_with(EMOTION_HEAVY))
        tells = [f for f in delta["critic_failures"] if f.error_code == EMOTION_ERROR_CODE]
        assert len(tells) == 1
        assert tells[0].critic_source == CRITIC_SOURCE

    async def test_shown_emotion_does_not_breach(self, config_factory):
        set_node_config(config_factory(emotion_word_threshold=0.3))
        delta = await audit(state_with(EMOTION_SHOWN))
        assert [f for f in delta["critic_failures"] if f.error_code == EMOTION_ERROR_CODE] == []


# --------------------------------------------------------------- style-tic guard

# Four sentences, three leaning on stock gestures or abstract shorthand from
# the default vocabulary: "deep breath", "trembling", "the weight of".
TIC_HEAVY = (
    "She took a deep breath and steadied herself. "
    "Her hands were trembling as she reached for the latch. "
    "He crossed to the desk and picked up the wrench. "
    "The weight of it all pressed down on her shoulders."
)
# The same dramatic work carried by specific actions and images instead.
TIC_FREE = (
    "She counted the latch screws twice before touching them. "
    "Her thumbnail found the old groove in the brass and stopped there. "
    "He crossed to the desk and picked up the wrench. "
    "She reread the last line until the words stopped meaning anything."
)


class TestTicDensity:
    def test_multi_word_phrases_are_matched_whole(self):
        pattern = _emotion_pattern(["deep breath", "the weight of"])
        density, offenders = emotion_word_density(TIC_HEAVY, pattern)
        assert density == pytest.approx(2 / 4)
        assert len(offenders) == 2

    def test_a_phrase_does_not_fire_inside_a_longer_word(self):
        # "deep breath" must not fire on "deep breaths…" mid-word expansions;
        # boundaries hold at both ends of the phrase.
        pattern = _emotion_pattern(["breath"])
        density, offenders = emotion_word_density(
            "She breathed once. Her breathing slowed.", pattern
        )
        assert offenders == []


class TestTicAuditNode:
    async def test_a_tic_heavy_draft_breaches(self, config_factory):
        set_node_config(config_factory(tic_phrase_threshold=0.3))
        delta = await audit(state_with(TIC_HEAVY))
        tics = [f for f in delta["critic_failures"] if f.error_code == TIC_ERROR_CODE]
        assert len(tics) == 1
        assert tics[0].critic_source == CRITIC_SOURCE
        assert "stock gesture" in tics[0].suggested_fix

    async def test_specific_prose_does_not_breach(self, config_factory):
        set_node_config(config_factory(tic_phrase_threshold=0.3))
        delta = await audit(state_with(TIC_FREE))
        assert [f for f in delta["critic_failures"] if f.error_code == TIC_ERROR_CODE] == []

    async def test_the_threshold_is_read_from_config(self, config_factory):
        # Under a permissive gate the same tic-heavy prose passes.
        set_node_config(config_factory(tic_phrase_threshold=0.9))
        delta = await audit(state_with(TIC_HEAVY))
        assert [f for f in delta["critic_failures"] if f.error_code == TIC_ERROR_CODE] == []
