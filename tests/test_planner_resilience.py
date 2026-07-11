"""The planner failure modes that killed two runs, and the fixes for them.

Two regressions live here, both taken from real crashes rather than imagined ones:

1. **Unparseable planner JSON.** On 2026-07-10 the beat planner wrote dialogue
   with unescaped double quotes, `parse_json_array` raised, and the run died
   after seven committed beats. `PRODUCTION_BEAT_PLAN` is that reply, verbatim
   from `data/chat.jsonl`.

2. **Re-planning destroys committed prose.** Every run enters at `plan_chapter`,
   which re-planned the arc and upserted beats with `prose=NULL`. Replaying the
   real upserts against a copy of the production database erased 979 of 2745
   committed words. The planners now reuse a plan that already exists.
"""

from __future__ import annotations

import json
import logging

import pytest

from museai.core.stream_bus import bus
from museai.fsm.nodes.deps import set_node_config
from museai.fsm.nodes.plan_beat import beat_id_for, plan_beat
from museai.fsm.nodes.plan_chapter import chapter_id_for, plan_chapter
from museai.fsm.state import FSM_Pointer, make_initial_state
from museai.llm import planning as planning_module
from museai.llm.planning import call_llm_for_json_array
from museai.llm.structured import (
    StructuredOutputError,
    parse_json_array,
    repair_json_text,
)
from museai.memory.db import (
    connect_db,
    get_beats_for_chapter,
    get_chapters_for_arc,
    init_db,
    upsert_arc,
    upsert_beat,
    upsert_chapter,
    upsert_character,
    upsert_project,
)

from conftest import patch_planner_llm

ARC_ID = "arc-1"
PROJECT_ID = "test-project"

# The exact reply that killed the run, from data/chat.jsonl. Note beat 2's
# exit_state: `regret ("It was meant to be stronger") while` — three bare quotes
# inside a JSON string value.
PRODUCTION_BEAT_PLAN = """```json
[
  {
    "ordering": 1,
    "intent": "Establish Mara's paralyzing internal struggle.",
    "entry_state": "Mara is suspended in silent paralysis.",
    "exit_state": "Mara traces a hairline fracture in the metalwork.",
    "word_target": 350,
    "focal_character_id": "char-mara",
    "target_pad": {"pleasure": -0.6, "arousal": 0.7, "dominance": -0.2}
  },
  {
    "ordering": 2,
    "intent": "Mara seeks an external justification for her fear.",
    "entry_state": "Mara reacts to the evidence, seeking a scapegoat.",
    "exit_state": "Mara expresses vague regret ("It was meant to be stronger") while trying to contain the chaotic energy within her.",
    "word_target": 380,
    "focal_character_id": "char-mara",
    "target_pad": {"pleasure": -0.4, "arousal": 0.8, "dominance": 0.1}
  }
]
```"""

VALID_BEAT_PLAN = """```json
[
  {"ordering": 1, "intent": "The letter arrives.",
   "entry_state": "A routine morning.", "exit_state": "Mara holds her handwriting.",
   "word_target": 400, "focal_character_id": "char-mara",
   "target_pad": {"pleasure": -0.6, "arousal": 0.8, "dominance": -0.4}}
]
```"""

VALID_CHAPTER_PLAN = """```json
[{"ordering": 1, "description": "Mara catalogs the letters.", "obligations": ["She dates one."]}]
```"""


class _Response:
    """What the planning helper and the agent loop read off an ``LLMResponse``."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.tool_calls: list[dict] = []


@pytest.fixture
def seeded(config_factory):
    """A config whose DB holds a project, one arc, and one character."""
    config = config_factory(project_id=PROJECT_ID)
    init_db(config.db_path)
    conn = connect_db(config.db_path)
    with conn:
        upsert_project(
            conn, id=PROJECT_ID, genre="mystery", premise="Letters arrive.",
            word_count_target=40000,
        )
        upsert_arc(
            conn, id=ARC_ID, project_id=PROJECT_ID, ordering=1,
            description="Mara traces the postmarks.", status="planned",
        )
        upsert_character(
            conn, id="char-mara", project_id=PROJECT_ID, name="Mara",
            description="The keeper.",
        )
    conn.close()
    set_node_config(config)
    return config


def _state(chapter_id: str = "") -> dict:
    return make_initial_state(
        PROJECT_ID, FSM_Pointer(arc_id=ARC_ID, chapter_id=chapter_id, beat_index=0)
    )


@pytest.fixture
def scripted(monkeypatch):
    """Install a fake ``call_llm`` inside the planning helper itself.

    The helper owns the retry ladder, so the fake has to sit *under* it. Returns
    the list of message-lists each call received, so a test can inspect the
    correction turns.
    """

    def _install(replies: list[str]) -> list:
        seen: list = []

        async def fake(endpoint, messages, **kwargs):
            seen.append(list(messages))
            return _Response(replies[min(len(seen) - 1, len(replies) - 1)])

        monkeypatch.setattr(planning_module, "call_llm", fake)
        return seen

    return _install


@pytest.fixture
def published(monkeypatch):
    """Capture every event published to the stream bus during a test."""
    events: list = []
    original = bus.publish

    async def spy(event_type, data):
        events.append((event_type, data))
        await original(event_type, data)

    monkeypatch.setattr(bus, "publish", spy)
    return events


# --------------------------------------------------------------------------- #
# repair_json_text                                                             #
# --------------------------------------------------------------------------- #

class TestRepairJsonText:
    def test_the_production_payload_is_unparseable_until_repaired(self):
        with pytest.raises(StructuredOutputError):
            parse_json_array(PRODUCTION_BEAT_PLAN, what="beats")

        beats = parse_json_array(PRODUCTION_BEAT_PLAN, what="beats", repair=True)
        assert len(beats) == 2
        assert beats[1]["exit_state"].startswith("Mara expresses vague regret (")
        # The quotes survive as *content*, which is the whole point of repairing
        # rather than stripping.
        assert '"It was meant to be stronger"' in beats[1]["exit_state"]

    def test_valid_json_is_returned_byte_identical(self):
        valid = json.dumps(
            [{"ordering": 1, "intent": "A plain intent."}], indent=2
        )
        assert repair_json_text(valid) == valid

    def test_a_quote_before_a_comma_is_not_mistaken_for_the_string_end(self):
        """The failure mode a lookahead-based scanner would have."""
        nasty = '[\n  {\n    "d": "he said "yes", then left"\n  }\n]'
        assert json.loads(repair_json_text(nasty)) == [
            {"d": 'he said "yes", then left'}
        ]

    def test_already_escaped_quotes_are_not_double_escaped(self):
        text = '[\n  {\n    "d": "he said \\"yes\\""\n  }\n]'
        assert json.loads(repair_json_text(text)) == [{"d": 'he said "yes"'}]

    def test_structural_and_non_string_lines_are_untouched(self):
        text = '[\n  {\n    "n": 5,\n    "pad": {"pleasure": 0.1}\n  }\n]'
        assert repair_json_text(text) == text

    def test_repair_does_not_rescue_genuinely_broken_json(self):
        with pytest.raises(StructuredOutputError):
            parse_json_array("the model refused to answer", what="beats", repair=True)


# --------------------------------------------------------------------------- #
# call_llm_for_json_array                                                      #
# --------------------------------------------------------------------------- #

async def _plan_beats(config, *, retries: int) -> list[dict]:
    return await call_llm_for_json_array(
        config.endpoint, [{"role": "user", "content": "plan"}],
        what="beats", agent="beat_planner", node="plan_beat", retries=retries,
    )


class TestPlannerRetryLadder:
    async def test_a_clean_first_reply_costs_one_call(self, seeded, scripted):
        seen = scripted([VALID_BEAT_PLAN])
        beats = await _plan_beats(seeded, retries=2)
        assert len(beats) == 1
        assert len(seen) == 1

    async def test_a_correction_turn_carries_the_error_and_recovers(
        self, seeded, scripted
    ):
        seen = scripted([PRODUCTION_BEAT_PLAN, VALID_BEAT_PLAN])
        beats = await _plan_beats(seeded, retries=2)
        assert len(beats) == 1
        assert len(seen) == 2

        # The second call replayed the model's bad reply and told it what broke.
        assert seen[1][-2]["role"] == "assistant"
        correction = seen[1][-1]
        assert correction["role"] == "user"
        assert "could not be parsed" in correction["content"]
        assert "escape every double quote" in correction["content"]

    async def test_the_model_gets_exactly_the_configured_number_of_retries(
        self, seeded, scripted
    ):
        seen = scripted(["not json at all"])
        with pytest.raises(StructuredOutputError):
            await _plan_beats(seeded, retries=2)
        assert len(seen) == 3  # the first attempt plus two retries

    async def test_zero_retries_is_a_single_call(self, seeded, scripted):
        seen = scripted(["not json at all"])
        with pytest.raises(StructuredOutputError):
            await _plan_beats(seeded, retries=0)
        assert len(seen) == 1

    async def test_a_stubborn_model_is_repaired_and_the_repair_is_announced(
        self, seeded, scripted, published, caplog
    ):
        seen = scripted([PRODUCTION_BEAT_PLAN])
        with caplog.at_level(logging.WARNING, logger="museai"):
            beats = await _plan_beats(seeded, retries=1)

        assert len(beats) == 2
        assert len(seen) == 2  # it never produced valid JSON on its own

        repaired = [data for event, data in published if event == "planner_repaired"]
        assert len(repaired) == 1
        assert repaired[0]["count"] == 2
        assert repaired[0]["what"] == "beats"

        messages = [record.getMessage() for record in caplog.records]
        assert any("event=json_repaired" in message for message in messages)
        assert any("event=parse_failed" in message for message in messages)

    async def test_a_recovered_plan_announces_nothing(
        self, seeded, scripted, published
    ):
        """A model that fixed itself was not repaired, and must not be reported as such."""
        scripted([PRODUCTION_BEAT_PLAN, VALID_BEAT_PLAN])
        await _plan_beats(seeded, retries=2)
        assert not [event for event, _ in published if event == "planner_repaired"]

    async def test_an_unrepairable_reply_still_kills_the_run(self, seeded, scripted):
        """A planner cannot degrade: there is no honest empty plan."""
        scripted(["I would rather not."])
        with pytest.raises(StructuredOutputError, match="beats"):
            await _plan_beats(seeded, retries=1)


# --------------------------------------------------------------------------- #
# The data-loss regression                                                     #
# --------------------------------------------------------------------------- #

def _commit_a_beat(config, chapter_id: str, beat_id: str) -> None:
    """A beat as `commit` leaves it: prose written, status completed."""
    conn = connect_db(config.db_path)
    with conn:
        upsert_beat(
            conn,
            id=beat_id,
            chapter_id=chapter_id,
            ordering=1,
            beat_spec=json.dumps(
                {
                    "intent": "The original intent.",
                    "entry_state": "in",
                    "exit_state": "out",
                    "target_pad": {"pleasure": 0.1, "arousal": 0.2, "dominance": 0.3},
                    "focal_character_id": "char-mara",
                }
            ),
            pad_constraint="Some constraint.",
            prose="The committed prose that must survive.",
            word_count=6,
            status="completed",
        )
    conn.close()


class TestUpsertNeverUnwritesProse:
    def test_a_planner_style_upsert_does_not_null_committed_prose(self, seeded):
        chapter_id = chapter_id_for(ARC_ID, 1)
        conn = connect_db(seeded.db_path)
        with conn:
            upsert_chapter(
                conn, id=chapter_id, arc_id=ARC_ID, ordering=1,
                description="One.", status="completed",
            )
        conn.close()
        beat_id = beat_id_for(chapter_id, 1)
        _commit_a_beat(seeded, chapter_id, beat_id)

        # Exactly what plan_beat used to do on a fresh Generate press.
        conn = connect_db(seeded.db_path)
        with conn:
            upsert_beat(
                conn, id=beat_id, chapter_id=chapter_id, ordering=1,
                beat_spec="{}", pad_constraint=None,
                status="planned",
            )
        row = conn.execute("SELECT * FROM Beats WHERE id=?", (beat_id,)).fetchone()
        conn.close()

        assert row["prose"] == "The committed prose that must survive."
        assert row["word_count"] == 6
        assert row["status"] == "completed"

    def test_a_completed_chapter_is_never_downgraded(self, seeded):
        chapter_id = chapter_id_for(ARC_ID, 1)
        conn = connect_db(seeded.db_path)
        with conn:
            upsert_chapter(
                conn, id=chapter_id, arc_id=ARC_ID, ordering=1,
                description="One.", status="completed",
            )
            upsert_chapter(
                conn, id=chapter_id, arc_id=ARC_ID, ordering=1,
                description="A different plan entirely.", status="planned",
            )
        row = conn.execute("SELECT * FROM Chapters WHERE id=?", (chapter_id,)).fetchone()
        conn.close()
        assert row["status"] == "completed"

    def test_commit_may_still_write_prose_over_a_completed_beat(self, seeded):
        """`Regenerate` after review recommits the same beat; that must work."""
        chapter_id = chapter_id_for(ARC_ID, 1)
        beat_id = beat_id_for(chapter_id, 1)
        conn = connect_db(seeded.db_path)
        with conn:
            upsert_chapter(
                conn, id=chapter_id, arc_id=ARC_ID, ordering=1,
                description="One.", status="active",
            )
        conn.close()
        _commit_a_beat(seeded, chapter_id, beat_id)

        conn = connect_db(seeded.db_path)
        with conn:
            upsert_beat(
                conn, id=beat_id, chapter_id=chapter_id, ordering=1,
                prose="A better draft.", word_count=3, status="completed",
            )
        row = conn.execute("SELECT * FROM Beats WHERE id=?", (beat_id,)).fetchone()
        conn.close()
        assert row["prose"] == "A better draft."
        assert row["word_count"] == 3


class TestPlannersReuseAnExistingPlan:
    async def test_plan_chapter_reuses_chapters_and_never_calls_the_model(
        self, seeded, monkeypatch
    ):
        calls: list = []

        async def fake_chapter_llm(endpoint, messages, **kwargs):
            calls.append(messages)
            return _Response(VALID_CHAPTER_PLAN)

        patch_planner_llm(monkeypatch, chapter=fake_chapter_llm)

        await plan_chapter(_state())
        assert len(calls) == 1

        # Finish chapter 1, then re-enter the node exactly as a restart would.
        conn = connect_db(seeded.db_path)
        with conn:
            conn.execute(
                "UPDATE Chapters SET status='completed' WHERE id=?",
                (chapter_id_for(ARC_ID, 1),),
            )
        conn.close()

        await plan_chapter(_state())
        assert len(calls) == 1, "a second run re-planned the arc"

        conn = connect_db(seeded.db_path)
        rows = get_chapters_for_arc(conn, ARC_ID)
        conn.close()
        assert [row["description"] for row in rows] == ["Mara catalogs the letters."]
        assert rows[0]["status"] == "completed"

    async def test_plan_chapter_points_at_the_first_unfinished_chapter(
        self, seeded, monkeypatch
    ):
        conn = connect_db(seeded.db_path)
        with conn:
            for ordering, status in ((1, "completed"), (2, "completed"), (3, "planned")):
                upsert_chapter(
                    conn, id=chapter_id_for(ARC_ID, ordering), arc_id=ARC_ID,
                    ordering=ordering, description=f"Chapter {ordering}.",
                    obligations="[]", status=status,
                )
        conn.close()

        async def explode(endpoint, messages, **kwargs):
            raise AssertionError("the model must not be called")

        patch_planner_llm(monkeypatch, chapter=explode)
        delta = await plan_chapter(_state())
        assert delta["fsm_pointer"].chapter_id == chapter_id_for(ARC_ID, 3)

    async def test_plan_beat_reuses_beats_and_resumes_at_the_unfinished_one(
        self, seeded, monkeypatch
    ):
        chapter_id = chapter_id_for(ARC_ID, 1)
        conn = connect_db(seeded.db_path)
        with conn:
            upsert_chapter(
                conn, id=chapter_id, arc_id=ARC_ID, ordering=1,
                description="One.", obligations="[]", status="active",
            )
        conn.close()
        _commit_a_beat(seeded, chapter_id, beat_id_for(chapter_id, 1))

        conn = connect_db(seeded.db_path)
        with conn:
            upsert_beat(
                conn, id=beat_id_for(chapter_id, 2), chapter_id=chapter_id, ordering=2,
                beat_spec=json.dumps(
                    {
                        "intent": "The second beat.",
                        "entry_state": "in",
                        "exit_state": "out",
                        "target_pad": {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0},
                        "focal_character_id": "char-mara",
                    }
                ),
                pad_constraint="c", status="planned",
            )
        conn.close()

        async def explode(endpoint, messages, **kwargs):
            raise AssertionError("the model must not be called")

        patch_planner_llm(monkeypatch, beat=explode)
        delta = await plan_beat(_state(chapter_id))

        # 0-based: beat 1 is committed, so the run resumes at beat 2.
        assert delta["fsm_pointer"].beat_index == 1

        conn = connect_db(seeded.db_path)
        rows = get_beats_for_chapter(conn, chapter_id)
        conn.close()
        assert rows[0]["prose"] == "The committed prose that must survive."
        assert rows[0]["status"] == "completed"
        assert rows[1]["status"] == "active"

    async def test_reuse_does_not_republish_a_stale_pad_target(
        self, seeded, monkeypatch, published
    ):
        """Re-applying an old beat's PAD would drag the character backwards."""
        chapter_id = chapter_id_for(ARC_ID, 1)
        conn = connect_db(seeded.db_path)
        with conn:
            upsert_chapter(
                conn, id=chapter_id, arc_id=ARC_ID, ordering=1,
                description="One.", obligations="[]", status="active",
            )
        conn.close()
        _commit_a_beat(seeded, chapter_id, beat_id_for(chapter_id, 1))
        conn = connect_db(seeded.db_path)
        with conn:
            upsert_beat(
                conn, id=beat_id_for(chapter_id, 2), chapter_id=chapter_id, ordering=2,
                beat_spec=json.dumps(
                    {
                        "intent": "Second.",
                        "entry_state": "in",
                        "exit_state": "out",
                        "target_pad": {"pleasure": 0.9, "arousal": 0.9, "dominance": 0.9},
                        "focal_character_id": "char-mara",
                    }
                ),
                pad_constraint="c", status="planned",
            )
        conn.close()

        async def explode(endpoint, messages, **kwargs):
            raise AssertionError("the model must not be called")

        patch_planner_llm(monkeypatch, beat=explode)
        await plan_beat(_state(chapter_id))

        events = [event for event, _ in published]
        assert "pad_update" not in events
        assert "beats_planned" in events
