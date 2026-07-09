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


PROJECT = {"genre": "literary thriller", "premise": "A forged ledger surfaces."}
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
    "word_target": 600,
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
    common = {"threads": THREADS, "characters": CHARACTERS}
    if template == "chapter_planner":
        return {**common, "project": PROJECT, "arc": ARC}
    if template == "beat_planner":
        return {
            **common,
            "chapter": CHAPTER,
            "recent_prose": RECENT_PROSE,
            "beat_word_target": 600,
        }
    if template == "drafter":
        return {
            **common,
            "beat": BEAT,
            "chapter": CHAPTER,
            "recent_prose": RECENT_PROSE,
            "pad_constraint": "Guarded, alert, and quietly in control.",
        }
    if template == "continuity_critic":
        return {
            **common,
            "chapter": CHAPTER,
            "recent_prose": RECENT_PROSE,
            "draft_text": DRAFT_TEXT,
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
            "chapter_planner", project=PROJECT, arc=ARC, threads=[], characters=[]
        )
        assert "<none/>" in rendered

    def test_drafter_carries_the_pad_constraint_and_word_target(self):
        messages = render_messages("drafter", **context_for("drafter"))
        user = messages[1]["content"]
        assert "Guarded, alert, and quietly in control." in user
        assert "600" in user

    def test_critic_offers_the_web_search_tool_and_scopes_the_check(self):
        messages = render_messages("continuity_critic", **context_for("continuity_critic"))
        system = messages[0]["content"]
        assert "web_search" in system
        assert "scoped to the material" in system
        assert "continuity_critic" in messages[1]["content"]


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
        assert parse_failure_objects(CLEAN, retry_cap=3) == []

    def test_well_formed_array(self):
        findings = parse_failure_objects(WELL_FORMED, retry_cap=3)
        assert len(findings) == 1
        assert isinstance(findings[0], FailureObject)
        assert findings[0].error_code == "CONTRADICTS_PRIOR_PROSE"
        assert findings[0].critic_source == "continuity_critic"

    def test_fenced_array_is_extracted(self):
        raw = f"Here is what I found:\n\n```json\n{WELL_FORMED}\n```\n\nHope that helps."
        findings = parse_failure_objects(raw, retry_cap=3)
        assert len(findings) == 1
        assert findings[0].offending_text == "bright with noon sun"

    def test_bare_fence_without_language_hint(self):
        raw = f"```\n{CLEAN}\n```"
        assert parse_failure_objects(raw, retry_cap=3) == []

    def test_prose_preamble_without_a_fence(self):
        raw = f"I found one issue. {WELL_FORMED}"
        assert len(parse_failure_objects(raw, retry_cap=3)) == 1

    def test_single_object_is_accepted_as_one_finding(self):
        raw = json.loads(WELL_FORMED)[0]
        findings = parse_failure_objects(json.dumps(raw), retry_cap=3)
        assert len(findings) == 1

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
        findings = parse_failure_objects(raw, retry_cap=3)
        assert findings[0].offending_text == 'she said "] is not a door"'

    def test_malformed_junk_is_rejected(self):
        with pytest.raises(StructuredOutputError, match="could not extract"):
            parse_failure_objects("the draft looks fine to me!", retry_cap=3)

    def test_truncated_json_is_rejected(self):
        with pytest.raises(StructuredOutputError):
            parse_failure_objects('[{"error_code": "X", "offending', retry_cap=3)

    def test_missing_required_field_is_rejected(self):
        raw = json.dumps([{"error_code": "X", "offending_text": "y"}])
        with pytest.raises(StructuredOutputError, match="not a valid failure object"):
            parse_failure_objects(raw, retry_cap=3)

    def test_extra_field_is_rejected(self):
        """FailureObject forbids extra keys; a hallucinated field is a hard failure."""
        payload = json.loads(WELL_FORMED)
        payload[0]["severity"] = "high"
        with pytest.raises(StructuredOutputError, match="not a valid failure object"):
            parse_failure_objects(json.dumps(payload), retry_cap=3)

    def test_non_array_json_is_rejected(self):
        with pytest.raises(StructuredOutputError, match="expected a JSON array"):
            parse_failure_objects('"clean"', retry_cap=3)

    def test_empty_response_is_rejected(self):
        with pytest.raises(StructuredOutputError, match="empty response"):
            parse_failure_objects("   ", retry_cap=3)

    def test_retry_cap_must_be_positive(self):
        with pytest.raises(ValueError, match="retry_cap must be at least 1"):
            parse_failure_objects(CLEAN, retry_cap=0)

    def test_retry_cap_appears_in_the_error(self):
        with pytest.raises(StructuredOutputError, match="retry_cap=2"):
            parse_failure_objects("nonsense", retry_cap=2)
