"""Tests for museai.fsm.nodes.audit.

No LLM, no socket, no DB: the audit node is pure text analysis over
``current_draft_text`` plus one config read.
"""

from __future__ import annotations

import pytest

from museai.fsm.nodes.audit import (
    CRITIC_SOURCE,
    ERROR_CODE,
    audit,
    is_passive,
    passive_voice_density,
    split_sentences,
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
