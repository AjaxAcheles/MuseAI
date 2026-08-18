"""Tests for the agent tool registry and the story-canon tools.

Every DB tool must be read-only, scoped to the active project, and must answer
a bad argument with an error *payload* the model can read — never an exception,
because the agent loop would stringify a traceback and the model recovers
better from a named mistake. web_search is opt-in: it appears in a roster only
when ``generation.research_mode`` is on.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from museai.fsm.nodes.deps import set_node_config
from museai.fsm.tools.check_draft import check_draft
from museai.fsm.tools.check_plan_node import check_plan_node
from museai.fsm.tools.find_repetition import find_repetition
from museai.fsm.tools.get_canonical_state import get_canonical_state
from museai.fsm.tools.get_chapter_context import get_chapter_context
from museai.fsm.tools.get_character_emotion_history import (
    get_character_emotion_history,
)
from museai.fsm.tools.get_character_sheet import get_character_sheet
from museai.fsm.tools.get_current_pointer_context import get_current_pointer_context
from museai.fsm.tools.get_full_outline import get_full_outline
from museai.fsm.tools.get_recent_commits import get_recent_commits
from museai.fsm.tools.get_seed_contract import get_seed_contract
from museai.fsm.tools.get_thread_history import get_thread_history
from museai.fsm.tools.get_thread_status import get_thread_status
from museai.fsm.tools.project_db import project_connection
from museai.fsm.tools.registry import (
    AGENT_TOOLS,
    TOOL_IMPLS,
    TOOL_SPECS,
    tool_impls_for,
    tool_specs_for,
)
from museai.fsm.tools.search_manuscript import search_manuscript
from museai.fsm.tools.verify_replacement import verify_replacement
from museai.memory import db

PROJECT_ID = "tools-test"

# Three sentences (>= repetition_min_run) so it can trip the overlap guard when
# a draft copies it.
COMMITTED_PARAGRAPH = (
    "Mara set the brass lantern on the sill. She trimmed the wick with two "
    "fingers. The harbour answered with its own small light."
)

DIALOGUE_PROSE = (
    'Mara turned from the window. "The letters stop the year the light '
    'failed," she said. "Every one of them."\n\n'
    "The room kept its silence.\n\n"
    'Downstairs, Mara called out again: "Bring the ledger up with you."'
)

ACTIVE_BEAT_SPEC = {
    "intent": "The reply arrives.",
    "entry_state": "The ledger is open.",
    "exit_state": "Mara has read the reply.",
    "target_pad": {"pleasure": -0.1, "arousal": 0.4, "dominance": 0.2},
    "focal_character_id": "char-mara",
}


def _seed_world(config) -> None:
    conn = db.connect_db(config.db_path)
    with conn:
        db.upsert_project(conn, id=PROJECT_ID, genre="mystery", premise="p",
                          word_count_target=None)
        db.upsert_arc(conn, id="arc-1", project_id=PROJECT_ID, ordering=1,
                      description="The letters arc.", status="active")
        db.upsert_chapter(conn, id="arc-1-c01", arc_id="arc-1", ordering=1,
                          description="Mara catalogs the letters.",
                          obligations=json.dumps(["Mara dates the earliest letter."]),
                          status="completed")
        db.upsert_chapter(conn, id="arc-1-c02", arc_id="arc-1", ordering=2,
                          description="The reply.", status="active")
        db.upsert_character(conn, id="char-mara", project_id=PROJECT_ID,
                            name="Mara Voss", description="The lantern keeper.")
        db.upsert_character_emotions(conn, character_id="char-mara",
                                     pleasure=0.2, arousal=-0.4, dominance=0.1)
        db.upsert_thread(conn, id="thread-letters", project_id=PROJECT_ID,
                         description="Who writes the letters?",
                         status="progressing", priority_score=0.9)
        db.upsert_thread(conn, id="thread-light", project_id=PROJECT_ID,
                         description="Why did the light fail?",
                         status="open", priority_score=0.5)
        db.upsert_beat(
            conn, id="arc-1-c01-b01", chapter_id="arc-1-c01", ordering=1,
            beat_spec=json.dumps({
                "intent": "Mara finds the letter.",
                "exit_state": "The letter is locked away.",
                "focal_character_id": "char-mara",
                "target_pad": {"pleasure": -0.2, "arousal": 0.6, "dominance": 0.1},
                "thread_updates": [{"id": "thread-letters", "status": "progressing"}],
            }),
            prose=DIALOGUE_PROSE, word_count=40, status="completed",
        )
        db.upsert_beat(
            conn, id="arc-1-c01-b02", chapter_id="arc-1-c01", ordering=2,
            beat_spec=json.dumps({
                "intent": "The lantern is lit.",
                "exit_state": "The light answers.",
                "focal_character_id": "char-mara",
                "target_pad": {"pleasure": 0.1, "arousal": -0.3, "dominance": 0.4},
            }),
            prose=COMMITTED_PARAGRAPH, word_count=28, status="completed",
        )
        db.upsert_beat(conn, id="arc-1-c02-b01", chapter_id="arc-1-c02",
                       ordering=1, beat_spec=json.dumps(ACTIVE_BEAT_SPEC),
                       pad_constraint="Steady hands, short sentences.",
                       status="active")

        # A second project in the same database: nothing below may leak out.
        db.upsert_project(conn, id="other-project", genre="g", premise="p")
        db.upsert_arc(conn, id="other-arc", project_id="other-project",
                      ordering=1, description="other", status="active")
        db.upsert_chapter(conn, id="other-c01", arc_id="other-arc", ordering=1,
                          description="other chapter", status="active")
        db.upsert_beat(conn, id="other-b01", chapter_id="other-c01", ordering=1,
                       prose="A different brass lantern burned elsewhere.",
                       word_count=7, status="completed")
        db.upsert_thread(conn, id="other-thread", project_id="other-project",
                         description="other", status="open", priority_score=0.1)
    conn.close()


@pytest.fixture
def world(config_factory):
    config = config_factory(project_id=PROJECT_ID)
    set_node_config(config)
    db.init_db(config.db_path)
    _seed_world(config)
    return config


class TestRegistry:
    def test_specs_and_impls_cover_the_same_tools(self):
        assert set(TOOL_SPECS) == set(TOOL_IMPLS)
        for name, spec in TOOL_SPECS.items():
            assert spec["type"] == "function"
            assert spec["function"]["name"] == name
            assert callable(TOOL_IMPLS[name])

    def test_every_agent_roster_resolves_and_matches(self, world):
        for agent in AGENT_TOOLS:
            specs = tool_specs_for(agent)
            impls = tool_impls_for(agent)
            assert [s["function"]["name"] for s in specs] == list(impls)
            assert impls, agent  # every LLM call has at least one tool

    def test_web_search_is_not_a_default_tool(self, world):
        for agent, roster in AGENT_TOOLS.items():
            assert "web_search" not in roster, agent
            assert "web_search" not in tool_impls_for(agent), agent

    def test_research_mode_appends_web_search_to_every_roster(
        self, config_factory
    ):
        config = config_factory(project_id=PROJECT_ID, research_mode=True)
        set_node_config(config)
        for agent in AGENT_TOOLS:
            names = [s["function"]["name"] for s in tool_specs_for(agent)]
            assert names[-1] == "web_search", agent
            assert "web_search" in tool_impls_for(agent), agent

    def test_an_unknown_agent_is_a_loud_error(self, world):
        with pytest.raises(ValueError, match="no tool roster"):
            tool_specs_for("dialogue_critic")


class TestProjectConnection:
    def test_the_connection_is_read_only(self, world):
        with project_connection() as (conn, project_id):
            assert project_id == PROJECT_ID
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("DELETE FROM Beats")


class TestSearchManuscript:
    def test_a_phrase_finds_the_beat_that_wrote_it(self, world):
        results = search_manuscript("brass lantern")
        assert results
        assert results[0]["beat_id"] == "arc-1-c01-b02"
        assert "brass lantern" in results[0]["snippet"].lower()

    def test_results_never_leak_another_project(self, world):
        beat_ids = {r["beat_id"] for r in search_manuscript("brass lantern", limit=10)}
        assert "other-b01" not in beat_ids

    def test_a_blank_query_and_a_miss_both_return_empty(self, world):
        assert search_manuscript("") == []
        assert search_manuscript("zeppelin invoice") == []

    def test_limit_caps_the_results(self, world):
        assert len(search_manuscript("the", limit=1)) == 1

    def test_an_unsupported_scope_is_an_error_payload(self, world):
        result = search_manuscript("lantern", scope="drafts")
        assert "error" in result
        assert result["supported_scopes"] == ["committed_only"]


class TestGetSeedContract:
    def test_the_contract_carries_project_cast_and_threads(self, world):
        contract = get_seed_contract()
        assert contract["project"] == {
            "id": PROJECT_ID, "genre": "mystery", "premise": "p",
            "setting": "", "word_count_target": None,
        }
        assert contract["characters"] == [
            {"id": "char-mara", "name": "Mara Voss",
             "description": "The lantern keeper."}
        ]
        assert {t["id"] for t in contract["threads"]} == {
            "thread-letters", "thread-light"
        }


class TestGetCurrentPointerContext:
    def test_the_active_rows_come_back_with_the_beat_spec(self, world):
        context = get_current_pointer_context()
        assert context["arc"]["id"] == "arc-1"
        assert context["chapter"]["id"] == "arc-1-c02"
        beat = context["beat"]
        assert beat["id"] == "arc-1-c02-b01"
        assert beat["intent"] == "The reply arrives."
        assert beat["exit_state"] == "Mara has read the reply."
        assert beat["pad_constraint"] == "Steady hands, short sentences."

    def test_nothing_active_reads_as_none_not_an_error(self, config_factory):
        config = config_factory(project_id="empty-project")
        set_node_config(config)
        db.init_db(config.db_path)
        assert get_current_pointer_context() == {
            "arc": None, "chapter": None, "beat": None
        }


class TestGetFullOutline:
    def test_the_outline_carries_status_obligations_and_progress(self, world):
        outline = get_full_outline()
        assert [arc["id"] for arc in outline] == ["arc-1"]
        chapters = outline[0]["chapters"]
        assert [c["id"] for c in chapters] == ["arc-1-c01", "arc-1-c02"]
        first = chapters[0]
        assert first["status"] == "completed"
        assert first["obligations"] == ["Mara dates the earliest letter."]
        assert first["beats_completed"] == 2
        assert chapters[1]["beats_completed"] == 0
        assert chapters[1]["beats_planned"] == 1


class TestGetChapterContext:
    def test_the_context_is_structure_not_prose(self, world):
        context = get_chapter_context("arc-1-c01")
        assert context["obligations"] == ["Mara dates the earliest letter."]
        assert [b["beat_id"] for b in context["beats"]] == [
            "arc-1-c01-b01", "arc-1-c01-b02"
        ]
        assert context["beats"][0]["intent"] == "Mara finds the letter."
        assert context["beats"][0]["exit_state"] == "The letter is locked away."
        assert context["committed_word_count"] == 68
        # The whole point: no prose dump.
        assert "prose" not in context
        assert COMMITTED_PARAGRAPH not in json.dumps(context)

    def test_another_projects_chapter_is_not_reachable(self, world):
        result = get_chapter_context("other-c01")
        assert "error" in result
        assert result["known_chapters"] == ["arc-1-c01", "arc-1-c02"]


class TestGetCanonicalState:
    def test_beats_come_back_as_metadata_without_prose(self, world):
        result = get_canonical_state("beats")
        assert result["scope"] == "beats"
        assert [r["id"] for r in result["records"]] == [
            "arc-1-c01-b01", "arc-1-c01-b02", "arc-1-c02-b01"
        ]
        assert COMMITTED_PARAGRAPH not in json.dumps(result)

    def test_ids_filter_and_name_the_missing(self, world):
        result = get_canonical_state("threads", ids=["thread-light", "thread-typo"])
        assert [r["id"] for r in result["records"]] == ["thread-light"]
        assert result["missing_ids"] == ["thread-typo"]

    def test_characters_carry_their_current_pad(self, world):
        result = get_canonical_state("characters")
        assert result["records"][0]["current_pad"] == {
            "pleasure": 0.2, "arousal": -0.4, "dominance": 0.1
        }

    def test_an_unknown_scope_names_the_real_ones(self, world):
        result = get_canonical_state("scenes")
        assert "error" in result
        assert "beats" in result["supported_scopes"]


class TestThreadTools:
    def test_thread_history_reads_the_declared_advances(self, world):
        history = get_thread_history("thread-letters")
        assert history["thread"]["status"] == "progressing"
        assert history["advanced_by"] == [
            {
                "beat_id": "arc-1-c01-b01",
                "chapter_id": "arc-1-c01",
                "intent": "Mara finds the letter.",
                "status_set": "progressing",
            }
        ]

    def test_an_unadvanced_thread_has_an_empty_history(self, world):
        assert get_thread_history("thread-light")["advanced_by"] == []

    def test_an_unknown_thread_names_the_real_ones(self, world):
        result = get_thread_history("thread-typo")
        assert "error" in result
        assert result["known_threads"] == ["thread-letters", "thread-light"]

    def test_thread_status_lists_only_this_projects_threads(self, world):
        rows = get_thread_status()
        assert {row["id"] for row in rows} == {"thread-letters", "thread-light"}


class TestCharacterTools:
    def test_emotion_history_returns_current_pad_and_trajectory(self, world):
        result = get_character_emotion_history("char-mara")
        assert result["current_pad"] == {
            "pleasure": 0.2, "arousal": -0.4, "dominance": 0.1
        }
        assert [h["beat_id"] for h in result["history"]] == [
            "arc-1-c01-b01", "arc-1-c01-b02"
        ]
        assert result["history"][0]["target_pad"]["arousal"] == 0.6

    def test_a_character_name_resolves_too(self, world):
        assert get_character_emotion_history("mara voss")["character_id"] == "char-mara"

    def test_an_unknown_character_names_the_cast(self, world):
        result = get_character_emotion_history("nobody")
        assert "error" in result
        assert result["known_characters"] == [{"id": "char-mara", "name": "Mara Voss"}]

    def test_the_sheet_samples_committed_dialogue(self, world):
        sheet = get_character_sheet("Mara Voss")
        assert sheet["description"] == "The lantern keeper."
        assert "The letters stop the year the light failed," in sheet["dialogue_samples"]
        assert "Bring the ledger up with you." in sheet["dialogue_samples"]

    def test_a_first_name_resolves_when_unambiguous(self, world):
        assert get_character_sheet("Mara")["id"] == "char-mara"


class TestGetRecentCommits:
    def test_recent_commits_are_metadata_plus_closing_words(self, world):
        commits = get_recent_commits()
        assert [c["beat_id"] for c in commits] == ["arc-1-c01-b01", "arc-1-c01-b02"]
        last = commits[-1]
        assert last["intent"] == "The lantern is lit."
        assert last["closing_words"].endswith("its own small light.")
        # Closing words, not the whole beat.
        assert len(last["closing_words"].split()) <= 40

    def test_limit_keeps_only_the_newest(self, world):
        assert [c["beat_id"] for c in get_recent_commits(limit=1)] == ["arc-1-c01-b02"]


class TestCheckPlanNode:
    def test_a_well_formed_beat_is_valid(self, world):
        verdict = check_plan_node({
            "intent": "Mara answers the letter.",
            "entry_state": "The reply is unread.",
            "exit_state": "The reply is sent.",
            "required_change": "Mara commits herself in writing.",
            "observable_event": "Mara seals and posts the reply.",
            "target_pad": {"pleasure": 0.2, "arousal": 0.3, "dominance": 0.5},
            "focal_character_id": "Mara Voss",
            "thread_updates": [{"id": "thread-letters", "status": "closed"}],
        })
        assert verdict == {"valid": True, "node_type": "beat", "problems": []}

    def test_a_beat_without_a_change_or_event_is_named(self, world):
        verdict = check_plan_node({
            "intent": "Mara broods.",
            "entry_state": "a",
            "exit_state": "b",
            "target_pad": {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0},
            "focal_character_id": "Mara Voss",
        })
        assert verdict["valid"] is False
        problems = " | ".join(verdict["problems"])
        assert "required_change" in problems
        assert "observable_event" in problems

    def test_a_beat_with_bad_pad_and_unknown_thread_is_named(self, world):
        verdict = check_plan_node({
            "intent": "x",
            "entry_state": "a",
            "exit_state": "b",
            "target_pad": {"pleasure": 3.0, "arousal": "high", "dominance": 0.0},
            "focal_character_id": "char-nobody",
            "thread_updates": [{"id": "thread-typo", "status": "resolved"}],
        })
        assert verdict["valid"] is False
        problems = " | ".join(verdict["problems"])
        assert "target_pad.pleasure" in problems
        assert "target_pad.arousal" in problems
        assert "char-nobody" in problems
        assert "thread-typo" in problems
        assert "'resolved'" in problems

    def test_a_chapter_needs_description_and_obligations(self, world):
        verdict = check_plan_node({"description": "", "obligations": []})
        assert verdict["node_type"] == "chapter"
        assert verdict["valid"] is False
        assert len(verdict["problems"]) == 2

    def test_a_well_formed_chapter_is_valid(self, world):
        verdict = check_plan_node({
            "description": "Mara rows to the mainland.",
            "obligations": ["Mara meets Idris."],
        })
        assert verdict == {"valid": True, "node_type": "chapter", "problems": []}

    def test_a_placeholder_chapter_obligation_is_named(self, world):
        verdict = check_plan_node({
            "description": "Mara rows to the mainland.",
            "obligations": ["An event that must occur"],
        })
        assert verdict["valid"] is False
        assert "output-format placeholder" in verdict["problems"][0]


class TestFindRepetition:
    def test_a_copied_paragraph_is_found(self, world):
        result = find_repetition(COMMITTED_PARAGRAPH + " She waited.")
        assert result["matches"]
        top = result["matches"][0]
        assert top["beat_id"] == "arc-1-c01-b02"
        assert top["similarity"] >= 0.9

    def test_a_short_phrase_is_matched_verbatim(self, world):
        result = find_repetition("the brass lantern on the sill")
        assert result["matches"][0]["similarity"] == 1.0

    def test_phrase_limit_is_honoured(self, world, config_factory):
        config = config_factory(project_id=PROJECT_ID)
        set_node_config(
            config.model_copy(
                update={
                    "tools": config.tools.model_copy(
                        update={"repetition_phrase_max_words": 4}
                    )
                }
            )
        )

        result = find_repetition("the brass lantern on")

        assert result["matches"][0]["similarity"] == 1.0
        assert find_repetition("the brass lantern on the")["matches"] == []

    def test_fresh_prose_matches_nothing(self, world):
        result = find_repetition(
            "Idris waited at the ferry slip with two tickets and no plan. "
            "He read the timetable twice. Nothing about it changed."
        )
        assert result["matches"] == []

    def test_an_unknown_scope_is_an_error_payload(self, world):
        result = find_repetition("anything", scope="everywhere")
        assert "error" in result
        assert result["supported_scopes"] == ["project", "recent"]


class TestCheckDraft:
    def test_clean_active_prose_passes(self, world):
        verdict = check_draft(
            "Mara lifted the sack and shook the letters onto the table. "
            "She read the first one twice. The handwriting matched her own."
        )
        assert verdict == {"passes": True, "findings": []}

    def test_passive_density_is_flagged_with_the_sentences(self, world):
        verdict = check_draft(
            "The door was opened by the wind. The letters were scattered "
            "across the floor. The lamp was lit by a stranger."
        )
        codes = [f["error_code"] for f in verdict["findings"]]
        assert verdict["passes"] is False
        assert "PASSIVE_VOICE_DENSITY" in codes
        finding = verdict["findings"][codes.index("PASSIVE_VOICE_DENSITY")]
        assert finding["offending"]

    def test_copying_committed_prose_is_flagged(self, world):
        verdict = check_draft(
            COMMITTED_PARAGRAPH + "\n\nThen she climbed down to the water."
        )
        assert "PARAGRAPH_OVERLAP" in [f["error_code"] for f in verdict["findings"]]

    def test_naming_emotions_is_flagged(self, world):
        verdict = check_draft(
            "Mara felt pure panic. The dread rose in her chest. "
            "Terror pinned her to the sill."
        )
        assert "EMOTION_TELL" in [f["error_code"] for f in verdict["findings"]]

    def test_empty_text_is_an_error_payload(self, world):
        assert "error" in check_draft("   ")


class TestVerifyReplacement:
    DRAFT = (
        "The post came up the path in a canvas sack. Mara signed for it "
        "without a word. The keeper's ledger stayed open on the table."
    )

    def test_a_sane_rewrite_is_approved(self, world):
        result = verify_replacement(
            self.DRAFT,
            "Mara signed for it without a word.",
            "Mara scrawled her name and said nothing.",
        )
        assert result["ok"] is True

    def test_a_ballooned_rewrite_is_rejected(self, world):
        result = verify_replacement(
            self.DRAFT, "Mara signed for it without a word.", "x" * 600
        )
        assert result["ok"] is False
        assert "grew" in result["reason"]

    def test_an_echo_of_surrounding_prose_is_rejected(self, world):
        result = verify_replacement(
            self.DRAFT,
            "Mara signed for it without a word.",
            "Mara nodded. The post came up the path in a canvas sack. Mara",
        )
        assert result["ok"] is False
        assert "repeats surrounding prose" in result["reason"]

    def test_a_span_that_is_not_in_the_draft_is_rejected(self, world):
        result = verify_replacement(self.DRAFT, "An invented sentence entirely.", "New.")
        assert result["ok"] is False
        assert "not found" in result["reason"]
