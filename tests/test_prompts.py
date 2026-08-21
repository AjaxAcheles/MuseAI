"""Tests for museai.llm.prompts and museai.llm.structured."""

from __future__ import annotations

import json

import pytest

from museai.fsm.state import FailureObject
from museai.llm.prompts import (
    PROMPT_DIR,
    PromptError,
    render,
    render_messages,
)
from museai.llm.structured import StructuredOutputError, parse_failure_objects

TEMPLATES = [
    "chapter_planner",
    "beat_planner",
    "drafter",
    "continuity_critic",
    "reviser",
]

# Capabilities v1 does not have. No template may hint at any of them.
FORBIDDEN_PHRASES = [
    "knowledge graph",
    "drift",
    "scene",
    "vector",
    "raptor",
    "summariser",
    "summarizer",
    "stylometric",
    "escalation",
    "not implemented",
    "coming soon",
]


PROJECT = {
    "genre": "literary thriller",
    "premise": "A forged ledger surfaces.",
    "setting": "A guild archive in a city living off its last trade route.",
}
ARC = {"description": "Mira uncovers the forgery and loses her patron."}
CHAPTER = {
    "description": "Mira searches the archive.",
    "obligations": ["Mira finds the ledger.", "Mira lies to Vaun about it."],
}
THREADS = [
    {"id": "t1", "description": "Who forged the ledger?", "status": "open"},
    {"id": "t2", "description": "Vaun's debt to the guild.", "status": "progressing"},
]
CHARACTERS = [
    {
        "id": "c1",
        "name": "Mira",
        "description": "An archivist who trusts documents more than people.",
        "pad": {"pleasure": -0.2, "arousal": 0.5, "dominance": 0.1},
    },
    {
        "id": "c2",
        "name": "Vaun",
        "description": "Her patron, and the reason she is afraid.",
        "pad": {"pleasure": 0.0, "arousal": -0.1, "dominance": 0.8},
    },
]
BEAT = {
    "intent": "Mira finds the ledger and hides it.",
    "entry_state": "Mira is alone in the archive after hours.",
    "exit_state": "Mira has the ledger under her coat, and Vaun is at the door.",
    "required_change": "The forgery stops being a rumour and becomes evidence in Mira's hands.",
    "observable_event": "Mira pulls the ledger from the shelf and hides it under her coat.",
    "beat_function": "discovery",
    "discharges": ["Mira finds the ledger."],
}
RECENT_PROSE = ["The archive smelled of dust and vinegar.", "Vaun had not called."]
DRAFT_TEXT = "Mira pocketed the ledger. The archive was bright with noon sun."
FAILURE = {
    "error_code": "CONTRADICTS_PRIOR_PROSE",
    "offending_text": "The archive was bright with noon sun.",
    "suggested_fix": "The prior passage puts this after hours, in the dark.",
    "critic_source": "continuity_critic",
}


def context_for(template: str) -> dict:
    """The full context each template requires (StrictUndefined: no gaps allowed)."""
    common = {"threads": THREADS, "characters": CHARACTERS, "research_mode": False}
    if template == "chapter_planner":
        return {**common, "project": PROJECT, "arc": ARC}
    if template == "beat_planner":
        return {
            **common,
            "chapter": CHAPTER,
            "min_beats_per_chapter": 3,
            "story_position": {
                "arc_description": ARC["description"],
                "chapter_ordering": 2,
                "chapter_count": 3,
            },
            "sibling_chapters": [
                {
                    "ordering": 1,
                    "description": "Mira is hired into the archive.",
                    "obligations": ["Mira gains archive access."],
                    "is_current": False,
                    "status": "completed",
                },
                {
                    "ordering": 2,
                    "description": CHAPTER["description"],
                    "obligations": CHAPTER["obligations"],
                    "is_current": True,
                    "status": "active",
                },
            ],
            "already_dramatized": [
                {
                    "ordering": 1,
                    "description": "Mira is hired into the archive.",
                    "intents": ["Mira accepts the post despite her unease."],
                }
            ],
            "recent_prose": RECENT_PROSE,
        }
    if template == "drafter":
        return {
            **common,
            "beat": BEAT,
            "project": PROJECT,
            "chapter": CHAPTER,
            "recent_prose": RECENT_PROSE,
            "pad_constraint": "Guarded, alert, and quietly in control.",
        }
    if template == "continuity_critic":
        return {
            **common,
            "beat": BEAT,
            "project": PROJECT,
            "chapter": CHAPTER,
            "recent_prose": RECENT_PROSE,
            "draft_text": DRAFT_TEXT,
            "repetition_overlap_count": 0,
        }
    if template == "reviser":
        return {
            **common,
            "beat": BEAT,
            "chapter": CHAPTER,
            "recent_prose": RECENT_PROSE,
            "pad_constraint": "Guarded, alert, and quietly in control.",
            "draft_text": DRAFT_TEXT,
            "failures": [FAILURE],
            "mode": "span",
            "span_text": "The archive was bright with noon sun.",
        }
    raise AssertionError(f"no context defined for {template}")


class TestTemplatesExist:
    def test_exactly_the_five_v1_templates_are_present(self):
        """No template for an absent feature (no scene planner, no summariser)."""
        found = sorted(p.name for p in PROMPT_DIR.glob("*.xml.j2"))
        assert found == sorted(f"{name}.xml.j2" for name in TEMPLATES)


class TestReviserModes:
    def test_span_mode_asks_for_one_passage(self):
        text = render("reviser", **context_for("reviser"))
        assert "<passage_to_rewrite>" in text
        assert "The archive was bright with noon sun." in text

    def test_full_mode_asks_for_the_whole_beat(self):
        context = {**context_for("reviser"), "mode": "full", "span_text": ""}
        text = render("reviser", **context)
        assert "<passage_to_rewrite>" not in text
        assert "Rewrite this beat" in text

    def test_both_modes_carry_the_critic_findings(self):
        for mode in ("span", "full"):
            text = render("reviser", **{**context_for("reviser"), "mode": mode})
            assert FAILURE["offending_text"] in text
            assert FAILURE["suggested_fix"] in text


class TestRendering:
    @pytest.mark.parametrize("template", TEMPLATES)
    def test_renders_non_empty_text(self, template):
        text = render(template, **context_for(template))
        assert text.strip()

    @pytest.mark.parametrize("template", TEMPLATES)
    def test_splits_into_system_and_user_messages(self, template):
        messages = render_messages(template, **context_for(template))

        assert [m["role"] for m in messages] == ["system", "user"]
        assert all(m["content"].strip() for m in messages)
        # The section tags themselves must not survive into the message bodies.
        for message in messages:
            assert "<system>" not in message["content"]
            assert "<user>" not in message["content"]

    @pytest.mark.parametrize("template", TEMPLATES)
    def test_context_is_actually_injected(self, template):
        rendered = render(template, **context_for(template))
        assert "Mira" in rendered
        assert "Who forged the ledger?" in rendered

    @pytest.mark.parametrize("template", TEMPLATES)
    def test_no_unrendered_jinja_syntax_remains(self, template):
        rendered = render(template, **context_for(template))
        assert "{{" not in rendered
        assert "{%" not in rendered

    @pytest.mark.parametrize("template", TEMPLATES)
    def test_promises_no_absent_capability(self, template):
        lowered = render(template, **context_for(template)).lower()
        for phrase in FORBIDDEN_PHRASES:
            assert phrase not in lowered, f"{template} references absent feature {phrase!r}"

    def test_missing_context_key_is_fatal(self):
        """StrictUndefined: a typo'd context key fails loudly, not silently blank."""
        from jinja2 import UndefinedError

        with pytest.raises(UndefinedError):
            render("chapter_planner", project=PROJECT, arc=ARC, threads=THREADS)

    def test_empty_collections_render_a_none_marker(self):
        rendered = render(
            "chapter_planner", project=PROJECT, arc=ARC, threads=[], characters=[],
            research_mode=False,
        )
        assert "<none/>" in rendered

    def test_drafter_carries_the_pad_constraint_and_no_word_target(self):
        messages = render_messages("drafter", **context_for("drafter"))
        user = messages[1]["content"]
        assert "Guarded, alert, and quietly in control." in user
        # Pacing is the drafter's call now; no numeric length target survives.
        assert "word_target" not in user

    def test_beat_planner_receives_the_configured_decomposition_floor(self):
        system = render_messages("beat_planner", **context_for("beat_planner"))[0]["content"]
        assert "at least 3 beats" in system

    @pytest.mark.parametrize("template", TEMPLATES)
    def test_every_agent_gets_story_tools_by_default(self, template):
        messages = render_messages(template, **context_for(template))
        assert "read-only story tools" in messages[0]["content"], template

    @pytest.mark.parametrize("template", TEMPLATES)
    def test_web_search_is_offered_only_in_research_mode(self, template):
        off = render_messages(template, **context_for(template))
        assert "web_search" not in off[0]["content"], template

        on = render_messages(
            template, **{**context_for(template), "research_mode": True}
        )
        assert "web_search" in on[0]["content"], template

    def test_critic_scopes_the_check(self):
        messages = render_messages("continuity_critic", **context_for("continuity_critic"))
        system = messages[0]["content"]
        assert "scoped to the material" in system
        assert "continuity_critic" in messages[1]["content"]

    def test_critic_requires_draft_only_quotes(self):
        messages = render_messages("continuity_critic", **context_for("continuity_critic"))
        rendered = "\n".join(message["content"] for message in messages)
        assert '"offending_text" must be copied verbatim from inside <draft_beat>' in rendered
        assert "Do not quote <recent_committed_prose>" in rendered

    def test_critic_forbids_copying_context_into_suggested_fix(self):
        rendered = render("continuity_critic", **context_for("continuity_critic"))
        assert "Never copy a sentence or phrase out of" in rendered
        assert "<recent_committed_prose> or any other context block into \"suggested_fix\"" in rendered

    def test_critic_sees_the_beat_goal(self):
        """The critic must be shown the mandate it is asked to enforce."""
        messages = render_messages("continuity_critic", **context_for("continuity_critic"))
        user = messages[1]["content"]
        assert "<beat_goal>" in user
        assert BEAT["intent"] in user
        assert BEAT["exit_state"] in user
        assert BEAT["required_change"] in user
        assert BEAT["observable_event"] in user
        assert "Mira finds the ledger." in user  # the discharged obligation
        assert "UNFULFILLED_OBLIGATION" in messages[0]["content"]

    def test_critic_sees_planned_thread_updates(self):
        beat = {**BEAT, "thread_updates": [{"id": "t1", "status": "progressing"}]}
        context = {**context_for("continuity_critic"), "beat": beat}
        user = render_messages("continuity_critic", **context)[1]["content"]
        assert '<update thread="t1" new_status="progressing"/>' in user

    def test_drafter_reads_the_change_before_the_manner(self):
        """The plot mandate leads the context; PAD is a modifier on it."""
        user = render_messages("drafter", **context_for("drafter"))[1]["content"]
        assert user.index("<this_beat_must_deliver>") < user.index("<manner>")
        assert user.index(BEAT["required_change"]) < user.index("<manner>")
        assert BEAT["exit_state"] in user
        assert BEAT["observable_event"] in user
        assert "Mira finds the ledger." in user  # the discharged obligation

    def test_reviser_reads_the_change_before_the_manner(self):
        """The reviser preserves the beat's required change; manner stays subordinate."""
        for mode in ("span", "full"):
            context = {**context_for("reviser"), "mode": mode}
            messages = render_messages("reviser", **context)
            user = messages[1]["content"]
            assert user.index("<this_beat_must_deliver>") < user.index("<manner>")
            assert BEAT["required_change"] in user
            assert "<focal_character_constraint>" not in user
            assert "Preserve the beat's required change" in messages[0]["content"]

    def test_drafter_sees_planned_thread_updates(self):
        beat = {**BEAT, "thread_updates": [{"id": "t2", "status": "resolved"}]}
        context = {**context_for("drafter"), "beat": beat}
        user = render_messages("drafter", **context)[1]["content"]
        assert '<update thread="t2" new_status="resolved"/>' in user

    def test_drafter_sees_the_story_world(self):
        """Premise and setting reach the drafter; the mandate still leads."""
        user = render_messages("drafter", **context_for("drafter"))[1]["content"]
        assert "<story_world>" in user
        assert PROJECT["premise"] in user
        assert PROJECT["setting"] in user
        assert user.index("<this_beat_must_deliver>") < user.index("<story_world>")

    def test_critic_sees_the_story_world_and_guards_canon(self):
        """The critic can only defend the premise, names, and central objects
        it has been told to check."""
        messages = render_messages("continuity_critic", **context_for("continuity_critic"))
        system, user = messages[0]["content"], messages[1]["content"]
        assert "<story_world>" in user
        assert PROJECT["premise"] in user
        assert PROJECT["setting"] in user
        assert "CONTRADICTS_PREMISE" in user
        assert "wrong or misspelled name" in system
        assert "central object" in system

    def test_a_setting_free_project_renders_no_setting_element(self):
        """Projects seeded before `setting` existed must render cleanly."""
        project = {"genre": PROJECT["genre"], "premise": PROJECT["premise"]}
        for template in ("drafter", "continuity_critic", "chapter_planner"):
            context = {**context_for(template), "project": project}
            user = render_messages(template, **context)[1]["content"]
            assert "<setting>" not in user, template
            assert PROJECT["premise"] in user, template


class TestRenderMessagesParser:
    def test_missing_sections_raise(self, tmp_path, monkeypatch):
        import museai.llm.prompts as prompts_module
        from jinja2 import Environment, FileSystemLoader, StrictUndefined

        (tmp_path / "broken.xml.j2").write_text("<system>only a system section</system>")
        (tmp_path / "empty_user.xml.j2").write_text("<system>hi</system><user>  </user>")

        monkeypatch.setattr(
            prompts_module,
            "_env",
            Environment(
                loader=FileSystemLoader(str(tmp_path)),
                autoescape=False,
                undefined=StrictUndefined,
            ),
        )

        with pytest.raises(PromptError, match="<user>"):
            render_messages("broken")
        with pytest.raises(PromptError, match="empty"):
            render_messages("empty_user")


CLEAN = "[]"
WELL_FORMED = json.dumps(
    [
        {
            "error_code": "CONTRADICTS_PRIOR_PROSE",
            "offending_text": "bright with noon sun",
            "suggested_fix": "It is after hours; make it dark.",
            "critic_source": "continuity_critic",
        }
    ]
)


class TestParseFailureObjects:
    def test_empty_array_is_a_clean_result(self):
        assert parse_failure_objects(CLEAN) == []

    def test_well_formed_array(self):
        findings = parse_failure_objects(WELL_FORMED)
        assert len(findings) == 1
        assert isinstance(findings[0], FailureObject)
        assert findings[0].error_code == "CONTRADICTS_PRIOR_PROSE"
        assert findings[0].critic_source == "continuity_critic"

    def test_fenced_array_is_extracted(self):
        raw = f"Here is what I found:\n\n```json\n{WELL_FORMED}\n```\n\nHope that helps."
        findings = parse_failure_objects(raw)
        assert len(findings) == 1
        assert findings[0].offending_text == "bright with noon sun"

    def test_bare_fence_without_language_hint(self):
        raw = f"```\n{CLEAN}\n```"
        assert parse_failure_objects(raw) == []

    def test_prose_preamble_without_a_fence(self):
        raw = f"I found one issue. {WELL_FORMED}"
        assert len(parse_failure_objects(raw)) == 1

    def test_single_object_is_rejected(self):
        raw = json.loads(WELL_FORMED)[0]
        with pytest.raises(StructuredOutputError, match="expected a JSON array"):
            parse_failure_objects(json.dumps(raw))

    def test_brackets_inside_strings_do_not_end_the_span(self):
        payload = [
            {
                "error_code": "CONTRADICTS_CHARACTER",
                "offending_text": 'she said "] is not a door"',
                "suggested_fix": "Remove the stray ] bracket.",
                "critic_source": "continuity_critic",
            }
        ]
        raw = f"```json\n{json.dumps(payload)}\n```"
        findings = parse_failure_objects(raw)
        assert findings[0].offending_text == 'she said "] is not a door"'

    def test_malformed_junk_is_rejected(self):
        with pytest.raises(StructuredOutputError, match="could not extract"):
            parse_failure_objects("the draft looks fine to me!")

    def test_truncated_json_is_rejected(self):
        with pytest.raises(StructuredOutputError):
            parse_failure_objects('[{"error_code": "X", "offending')

    def test_missing_required_field_is_rejected(self):
        raw = json.dumps([{"error_code": "X", "offending_text": "y"}])
        with pytest.raises(StructuredOutputError, match="missing required fields"):
            parse_failure_objects(raw)

    def test_extra_field_is_rejected(self):
        """FailureObject forbids extra keys; a hallucinated field is a hard failure."""
        payload = json.loads(WELL_FORMED)
        payload[0]["severity"] = "high"
        with pytest.raises(StructuredOutputError, match="unexpected fields: severity"):
            parse_failure_objects(json.dumps(payload))

    def test_non_array_json_is_rejected(self):
        with pytest.raises(StructuredOutputError, match="expected a JSON array"):
            parse_failure_objects('"clean"')

    def test_empty_response_is_rejected(self):
        with pytest.raises(StructuredOutputError, match="empty response"):
            parse_failure_objects("   ")

    def test_critic_source_is_optional_and_defaults(self):
        # B4, 2026-07-25 postmortem: FailureObject already defaults critic_source
        # (v1 has exactly one critic), so requiring the model to echo it back cost
        # a whole re-prompt whenever it forgot, for a field the parser supplies
        # itself. Omitting it must not be a schema deviation.
        raw = json.dumps(
            [{"error_code": "CONTRADICTS_THREAD", "offending_text": "y", "suggested_fix": "z"}]
        )
        findings = parse_failure_objects(raw)
        assert findings[0].critic_source == "continuity_critic"

    def test_the_error_carries_the_validation_detail(self):
        """The message is fed back to the model verbatim, so it must name the field."""
        raw = json.dumps([{"error_code": "X", "offending_text": "y"}])
        with pytest.raises(StructuredOutputError, match="suggested_fix"):
            parse_failure_objects(raw)


class TestLenientMode:
    def test_lenient_ignores_an_unknown_key(self):
        payload = json.loads(WELL_FORMED)
        payload[0]["severity"] = "high"
        findings = parse_failure_objects(json.dumps(payload), lenient=True)
        assert len(findings) == 1
        assert not hasattr(findings[0], "severity")

    def test_lenient_skips_rather_than_fabricates_a_missing_field(self):
        """An empty offending_text would make revise.locate() rewrite the whole draft."""
        payload = [
            {"error_code": "X", "offarming_text": "typo'd key", "suggested_fix": "z"},
            json.loads(WELL_FORMED)[0],
        ]
        findings = parse_failure_objects(json.dumps(payload), lenient=True)
        assert len(findings) == 1
        assert findings[0].offending_text == "bright with noon sun"

    def test_lenient_returns_empty_when_every_element_is_unreadable(self):
        payload = [{"error_code": "X", "offarming_text": "nope"}]
        assert parse_failure_objects(json.dumps(payload), lenient=True) == []

    def test_lenient_does_not_rescue_unparseable_text(self):
        """Element validation loosens; extraction never does."""
        with pytest.raises(StructuredOutputError, match="could not extract"):
            parse_failure_objects("the draft looks fine to me!", lenient=True)
