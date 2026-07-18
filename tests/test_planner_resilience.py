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
from museai.fsm.state import FSM_Pointer, OrchestratorState, make_initial_state
from museai.llm import planning as planning_module
from museai.llm.planning import call_llm_for_json_array
from museai.llm.structured import (
    FakeToolCallTextError,
    StructuredOutputError,
    TruncatedResponseError,
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
    "focal_character_id": "char-mara",
    "target_pad": {"pleasure": -0.6, "arousal": 0.7, "dominance": -0.2}
  },
  {
    "ordering": 2,
    "intent": "Mara seeks an external justification for her fear.",
    "entry_state": "Mara reacts to the evidence, seeking a scapegoat.",
    "exit_state": "Mara expresses vague regret ("It was meant to be stronger") while trying to contain the chaotic energy within her.",
    "focal_character_id": "char-mara",
    "target_pad": {"pleasure": -0.4, "arousal": 0.8, "dominance": 0.1}
  }
]
```"""

VALID_BEAT_PLAN = """```json
[
  {"ordering": 1, "intent": "The letter arrives.",
   "entry_state": "A routine morning.", "exit_state": "Mara holds her handwriting.",
   "focal_character_id": "char-mara",
   "target_pad": {"pleasure": -0.6, "arousal": 0.8, "dominance": -0.4}}
]
```"""

VALID_CHAPTER_PLAN = """```json
[{"ordering": 1, "description": "Mara catalogs the letters.", "obligations": ["She dates one."]}]
```"""

TOOL_CALL_CHAPTER_REPLY = """```
{
  "name": "get_full_outline",
  "parameters": {
    "key": "arc"
  }
}
```"""

TOOL_CALL_BEAT_REPLY = """```
{"name": "get_chapter_context", "parameters": {"chapter_id": "<chapter>"}}
{"name": "get_character_emotion_history", "parameters": {"character_id": "char-mara"}}
```"""

RECENT_FAKE_TOOL_CHAIN = """```
{"name": "get_full_outline", "parameters": {"format": "array"}}
{"name": "check_plan_node", "parameters": {"plan_node": "{'description': 'One.', 'obligations': ['an event that must occur']}"}}
{"name": "get_canonical_state", "parameters": {"scope": "chapters"}}
{"name": "check_plan_node", "parameters": {"plan_node": "{\"ordering\": 1, \"description\": \"One.\"}"}}
```"""


class _Response:
    """What the planning helper and the agent loop read off an ``LLMResponse``."""

    def __init__(self, text: str, finish_reason: str | None = None) -> None:
        self.text = text
        self.finish_reason = finish_reason
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


def _state(chapter_id: str = "") -> OrchestratorState:
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

    def _install(replies: list) -> list:
        seen: list = []

        async def fake(endpoint, messages, **kwargs):
            seen.append(list(messages))
            reply = replies[min(len(seen) - 1, len(replies) - 1)]
            return reply if isinstance(reply, _Response) else _Response(reply)

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

class TestNestedArrayUnwrap:
    """qwen2.5:3b wrapped a live beat plan once more: `[[{...}]]` (v1.17)."""

    def test_a_doubly_nested_array_is_flattened_one_level(self):
        nested = json.dumps([[{"ordering": 1, "intent": "a"},
                              {"ordering": 2, "intent": "b"}]])
        beats = parse_json_array(nested, what="beats")
        assert [b["ordering"] for b in beats] == [1, 2]

    def test_two_wrapped_groups_flatten_in_order(self):
        nested = json.dumps([[{"n": 1}], [{"n": 2}]])
        assert [b["n"] for b in parse_json_array(nested, what="beats")] == [1, 2]

    def test_deeper_nesting_is_still_a_shape_error(self):
        with pytest.raises(StructuredOutputError, match="not a JSON object"):
            parse_json_array(json.dumps([[[{"n": 1}]]]), what="beats")

    def test_a_mixed_array_is_still_a_shape_error(self):
        with pytest.raises(StructuredOutputError, match="not a JSON object"):
            parse_json_array(json.dumps([{"n": 1}, [{"n": 2}]]), what="beats")


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


class TestToolCallShapedPlannerReplies:
    """Small models sometimes answer with tool-call JSON instead of a plan array."""

    def test_a_tool_call_object_is_not_a_chapter_plan(self):
        with pytest.raises(FakeToolCallTextError, match="tools were not executed"):
            parse_json_array(TOOL_CALL_CHAPTER_REPLY, what="chapters")

    def test_multiple_tool_call_objects_are_not_a_beat_plan(self):
        with pytest.raises(FakeToolCallTextError, match="tools were not executed"):
            parse_json_array(TOOL_CALL_BEAT_REPLY, what="beats", repair=True)

    def test_recent_fake_tool_chain_is_reported_as_tool_protocol_confusion(self):
        with pytest.raises(FakeToolCallTextError) as excinfo:
            parse_json_array(RECENT_FAKE_TOOL_CHAIN, what="chapters", repair=True)

        message = str(excinfo.value)
        assert "tool calls as plain text" in message
        assert "get_full_outline" in message
        assert "check_plan_node" in message

    def test_planner_extraction_prefers_an_array_over_an_earlier_object(self):
        raw = """The first thing is not the answer:
{"note": "this object should be ignored"}

```json
[{"ordering": 1, "intent": "The real beat plan."}]
```"""

        beats = parse_json_array(raw, what="beats")
        assert beats == [{"ordering": 1, "intent": "The real beat plan."}]


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

    async def test_fake_tool_text_gets_a_specific_correction_without_replaying_it(
        self, seeded, scripted, caplog
    ):
        seen = scripted([RECENT_FAKE_TOOL_CHAIN, VALID_BEAT_PLAN])
        with caplog.at_level(logging.WARNING, logger="museai"):
            beats = await _plan_beats(seeded, retries=2)

        assert len(beats) == 1
        assert len(seen) == 2
        assert seen[1][-2]["role"] == "assistant"
        assert "omitted" in seen[1][-2]["content"]
        assert "get_full_outline" not in seen[1][-2]["content"]
        correction = seen[1][-1]["content"]
        assert "tool calls as plain text" in correction
        assert "use the provided tool-call channel" in correction
        assert any("event=fake_tool_call_text" in r.getMessage() for r in caplog.records)

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


class TestTruncatedAndEmptyReplies:
    """finish_reason == "length" is truncation, not a quoting mistake.

    The 2026-07-12 run: the endpoint's default token cap cut every plan
    mid-JSON, and the quote-escaping correction was sent three times for a
    problem no amount of escaping can fix.
    """

    # A reply cut off after an inner array — the shape that once parsed as
    # "the plan" (see test_structured.py).
    TRUNCATED = _Response(
        '[{"intent": "x", "thread_updates": [{"id": "t1", "status": "open"}]',
        finish_reason="length",
    )

    async def test_a_truncated_reply_gets_a_truncation_correction(
        self, seeded, scripted, caplog
    ):
        seen = scripted([self.TRUNCATED, VALID_BEAT_PLAN])
        with caplog.at_level(logging.WARNING, logger="museai"):
            beats = await _plan_beats(seeded, retries=2)

        assert len(beats) == 1
        assert len(seen) == 2
        # The truncated garbage is not replayed, and the advice is "shorter",
        # never "escape your quotes".
        assistant = seen[1][-2]
        assert assistant["role"] == "assistant"
        assert "cut off" in assistant["content"]
        correction = seen[1][-1]["content"]
        assert "cut off" in correction
        assert "shorter" in correction
        assert "escape every double quote" not in correction
        # The operator sees the real cause in the log.
        messages = [record.getMessage() for record in caplog.records]
        assert any("event=truncated" in message for message in messages)
        assert not any("event=parse_failed" in message for message in messages)

    async def test_persistent_truncation_raises_and_names_the_knob(
        self, seeded, scripted
    ):
        scripted([self.TRUNCATED])
        with pytest.raises(TruncatedResponseError, match="max_output_tokens"):
            await _plan_beats(seeded, retries=1)

    async def test_persistent_empty_truncation_names_the_context_window(
        self, seeded, scripted
    ):
        """Empty + length is context-window exhaustion, not an output-cap hit,
        so the raised error must point at the window (num_ctx), not the cap."""
        scripted([_Response("", finish_reason="length")])
        with pytest.raises(TruncatedResponseError, match="num_ctx"):
            await _plan_beats(seeded, retries=1)

    async def test_an_empty_truncated_reply_resets_instead_of_growing(
        self, seeded, scripted, caplog
    ):
        """Empty text with finish_reason "length" means the prompt filled the
        context window, leaving no room to generate. Asking for a "shorter"
        reply is futile — there is no output to shorten — and appending a
        correction only enlarges the prompt. The retry drops back to the
        original messages instead of growing them, and the log flags empty=True
        so the operator sees the real cause (context window, not output cap)."""
        seen = scripted([_Response("", finish_reason="length"), VALID_BEAT_PLAN])
        with caplog.at_level(logging.WARNING, logger="museai"):
            beats = await _plan_beats(seeded, retries=2)

        assert len(beats) == 1
        # No "shorter" correction, and the retry is no larger than the first
        # attempt — the ladder does not pile tokens onto an already-full window.
        assert "shorter" not in seen[1][-1]["content"]
        assert len(seen[1]) == len(seen[0])
        messages = [record.getMessage() for record in caplog.records]
        assert any("event=truncated" in message for message in messages)
        assert any("empty=True" in message for message in messages)
        assert not any("event=empty_reply" in message for message in messages)

    async def test_a_truncated_last_reply_is_never_quote_repaired(
        self, seeded, scripted, published
    ):
        """Repairing half a written array puts words in the model's mouth."""
        scripted([PRODUCTION_BEAT_PLAN, self.TRUNCATED])
        with pytest.raises(TruncatedResponseError):
            await _plan_beats(seeded, retries=1)
        assert not [event for event, _ in published if event == "planner_repaired"]

    async def test_an_empty_reply_gets_its_own_correction_not_an_empty_turn(
        self, seeded, scripted, caplog
    ):
        seen = scripted([_Response(""), VALID_BEAT_PLAN])
        with caplog.at_level(logging.WARNING, logger="museai"):
            beats = await _plan_beats(seeded, retries=2)

        assert len(beats) == 1
        assistant = seen[1][-2]
        assert assistant["role"] == "assistant"
        assert assistant["content"], "an empty assistant turn teaches the model nothing"
        assert "omitted" in assistant["content"]
        correction = seen[1][-1]["content"]
        assert "empty" in correction
        assert any("event=empty_reply" in r.getMessage() for r in caplog.records)


class TestRetriesKeepToolResults:
    async def test_a_retry_after_tool_calls_does_not_rerun_the_tools(
        self, seeded, monkeypatch
    ):
        from museai.fsm.tools import loop as loop_module

        tool_runs: list[dict] = []

        def lookup(**kwargs):
            tool_runs.append(kwargs)
            return {"result": "the gathered context"}

        tool_call_reply = _Response("")
        tool_call_reply.tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ]
        replies = iter([tool_call_reply, _Response("not json at all"),
                        _Response(VALID_BEAT_PLAN)])
        seen: list = []

        async def fake(endpoint, messages, **kwargs):
            seen.append(list(messages))
            return next(replies)

        monkeypatch.setattr(loop_module, "call_llm", fake)

        beats = await call_llm_for_json_array(
            seeded.endpoint,
            [{"role": "user", "content": "plan"}],
            what="beats",
            agent="beat_planner",
            node="plan_beat",
            retries=2,
            tools=[{"type": "function", "function": {"name": "lookup"}}],
            tool_impls={"lookup": lookup},
            max_tool_iterations=3,
        )

        assert len(beats) == 1
        assert len(tool_runs) == 1, "the retry re-ran a tool whose result it already had"
        # The retry's conversation carried the first attempt's tool result.
        retry_messages = seen[-1]
        assert any(m.get("role") == "tool" for m in retry_messages)
        # And the correction turns landed after it.
        assert retry_messages[-1]["role"] == "user"
        assert "could not be parsed" in retry_messages[-1]["content"]


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
