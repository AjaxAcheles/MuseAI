"""Tests for the chapter and beat planning nodes, and the PAD lookup table.

No test opens a socket: ``call_llm`` is replaced in each node's namespace with a
fake that returns canned JSON. The PAD translation is a pure table lookup, so it
needs no mocking at all.
"""

from __future__ import annotations

import json
from itertools import product
from types import SimpleNamespace

import pytest

from museai.core.stream_bus import bus
from museai.fsm.nodes import plan_beat as plan_beat_module
from museai.fsm.nodes import plan_chapter as plan_chapter_module
from museai.fsm.nodes.deps import PlanningError, set_node_config
from museai.fsm.nodes.plan_beat import _validate_beat_item, plan_beat
from museai.fsm.nodes.plan_chapter import (
    _validate_chapter_item,
    chapter_id_for,
    plan_chapter,
)
from museai.fsm.pad import (
    PAD_BAND_THRESHOLD,
    PAD_BASELINES_PATH,
    load_pad_baselines,
    pad_key,
    pad_keys,
    quantize_axis,
    resolve_pad_constraint,
)
from museai.fsm.state import FSM_Pointer, make_initial_state
from museai.llm.structured import StructuredOutputError
from museai.memory.db import (
    connect_db,
    get_beats_for_chapter,
    get_chapters_for_arc,
    init_db,
    upsert_arc,
    upsert_chapter,
    upsert_character,
    upsert_character_emotions,
    upsert_project,
    upsert_thread,
)

from conftest import patch_planner_llm

ARC_ID = "arc-1"
PROJECT_ID = "test-project"

CHAPTERS_JSON = """Here is the plan:

```json
[
  {"ordering": 1, "description": "Mara catalogs the letters.",
   "obligations": ["Mara dates the earliest letter.", "Mara hides it from Idris."]},
  {"ordering": 2, "description": "Mara rows to the mainland.",
   "obligations": ["Mara meets Idris."]},
  {"ordering": 3, "description": "The postmarks contradict each other.",
   "obligations": ["Idris admits he has seen the handwriting before."]}
]
```
"""

BEATS_JSON = """```json
[
  {"ordering": 1, "intent": "Mara finds the letter in the day's post.",
   "entry_state": "A routine morning.", "exit_state": "Mara is holding her own handwriting.",
   "required_change": "Mara can no longer treat the post as routine.",
   "observable_event": "Mara opens a letter written in her own hand.",
   "beat_function": "discovery",
   "discharges": ["Mara dates the earliest letter.", "Not a real obligation."],
   "thread_updates": [{"id": "thread-1", "status": "progressing"}],
   "focal_character_id": "char-mara",
   "target_pad": {"pleasure": -0.6, "arousal": 0.8, "dominance": -0.4}},
  {"ordering": 2, "intent": "Mara files the letter and says nothing.",
   "entry_state": "Mara is holding the letter.", "exit_state": "The letter is locked away.",
   "focal_character_id": "char-idris",
   "target_pad": {"pleasure": -0.3, "arousal": -0.5, "dominance": 0.6}}
]
```
"""


def _response(text: str) -> SimpleNamespace:
    """The only attribute the planners read off an ``LLMResponse``."""
    return SimpleNamespace(text=text, tool_calls=[])


@pytest.fixture
def seeded(config_factory):
    """A config whose DB holds a project, one arc, threads, and two characters."""
    config = config_factory(project_id=PROJECT_ID)
    init_db(config.db_path)

    conn = connect_db(config.db_path)
    with conn:
        upsert_project(
            conn,
            id=PROJECT_ID,
            genre="literary mystery",
            premise="Letters arrive in the keeper's own hand.",
            word_count_target=40000,
        )
        upsert_arc(
            conn,
            id=ARC_ID,
            project_id=PROJECT_ID,
            ordering=1,
            description="Mara traces the postmarks.",
            status="planned",
        )
        upsert_thread(
            conn,
            id="thread-1",
            project_id=PROJECT_ID,
            description="Who is writing the letters?",
            status="open",
            priority_score=0.9,
        )
        upsert_character(
            conn, id="char-mara", project_id=PROJECT_ID, name="Mara", description="The keeper."
        )
        upsert_character(
            conn, id="char-idris", project_id=PROJECT_ID, name="Idris", description="An archivist."
        )
        upsert_character_emotions(
            conn, character_id="char-mara", pleasure=-0.2, arousal=0.1, dominance=0.3
        )
    conn.close()

    set_node_config(config)
    return config


def _state(chapter_id: str = "") -> dict:
    return make_initial_state(
        PROJECT_ID,
        FSM_Pointer(arc_id=ARC_ID, chapter_id=chapter_id, beat_index=0),
    )


def _seed_active_chapter(config, obligations: list[str] | None = None) -> str:
    chapter_id = chapter_id_for(ARC_ID, 1)
    conn = connect_db(config.db_path)
    with conn:
        upsert_chapter(
            conn,
            id=chapter_id,
            arc_id=ARC_ID,
            ordering=1,
            description="Mara catalogs the letters.",
            obligations=json.dumps(obligations or ["Mara dates the earliest letter."]),
            status="active",
        )
    conn.close()
    return chapter_id


# --------------------------------------------------------------------------- #
# plan_chapter                                                                #
# --------------------------------------------------------------------------- #

async def test_plan_chapter_writes_ordered_rows_and_advances_pointer(seeded, monkeypatch):
    async def fake_call_llm(endpoint, messages, **kwargs):
        return _response(CHAPTERS_JSON)

    patch_planner_llm(monkeypatch, chapter=fake_call_llm)

    delta = await plan_chapter(_state())

    conn = connect_db(seeded.db_path)
    rows = get_chapters_for_arc(conn, ARC_ID)
    arc = conn.execute("SELECT status FROM Arcs WHERE id=?", (ARC_ID,)).fetchone()
    conn.close()

    assert [row["ordering"] for row in rows] == [1, 2, 3]
    assert [row["description"] for row in rows] == [
        "Mara catalogs the letters.",
        "Mara rows to the mainland.",
        "The postmarks contradict each other.",
    ]
    assert [row["status"] for row in rows] == ["active", "planned", "planned"]
    assert json.loads(rows[0]["obligations"]) == [
        "Mara dates the earliest letter.",
        "Mara hides it from Idris.",
    ]
    assert arc["status"] == "active"

    pointer = delta["fsm_pointer"]
    assert pointer.chapter_id == rows[0]["id"] == chapter_id_for(ARC_ID, 1)
    assert pointer.arc_id == ARC_ID
    assert pointer.beat_index == 0


async def test_plan_chapter_is_idempotent_across_replans(seeded, monkeypatch):
    async def fake_call_llm(endpoint, messages, **kwargs):
        return _response(CHAPTERS_JSON)

    patch_planner_llm(monkeypatch, chapter=fake_call_llm)

    await plan_chapter(_state())
    await plan_chapter(_state())

    conn = connect_db(seeded.db_path)
    rows = get_chapters_for_arc(conn, ARC_ID)
    conn.close()
    assert len(rows) == 3


async def test_plan_chapter_publishes_phase_change_and_summary(seeded, monkeypatch):
    async def fake_call_llm(endpoint, messages, **kwargs):
        return _response(CHAPTERS_JSON)

    patch_planner_llm(monkeypatch, chapter=fake_call_llm)

    queue = bus.subscribe()
    try:
        await plan_chapter(_state())
        events = [queue.get_nowait() for _ in range(queue.qsize())]
    finally:
        bus.unsubscribe(queue)

    by_type = {event["type"]: event["data"] for event in events}
    assert by_type["phase_change"]["phase"] == "Planning"
    assert by_type["chapters_planned"]["chapter_count"] == 3
    assert by_type["chapters_planned"]["active_chapter_id"] == chapter_id_for(ARC_ID, 1)


async def test_plan_chapter_raises_on_unparseable_plan(seeded, monkeypatch):
    async def fake_call_llm(endpoint, messages, **kwargs):
        return _response("I'd rather not plan this arc.")

    patch_planner_llm(monkeypatch, chapter=fake_call_llm)

    with pytest.raises(StructuredOutputError):
        await plan_chapter(_state())

    conn = connect_db(seeded.db_path)
    rows = get_chapters_for_arc(conn, ARC_ID)
    conn.close()
    assert rows == []


async def test_plan_chapter_raises_on_empty_array(seeded, monkeypatch):
    async def fake_call_llm(endpoint, messages, **kwargs):
        return _response("[]")

    patch_planner_llm(monkeypatch, chapter=fake_call_llm)

    with pytest.raises(StructuredOutputError, match="empty array"):
        await plan_chapter(_state())


async def test_plan_chapter_rejects_an_unknown_arc(seeded, monkeypatch):
    async def fake_call_llm(endpoint, messages, **kwargs):
        raise AssertionError("the endpoint must not be reached")

    patch_planner_llm(monkeypatch, chapter=fake_call_llm)

    state = make_initial_state(
        PROJECT_ID, FSM_Pointer(arc_id="arc-nope", chapter_id="", beat_index=0)
    )
    with pytest.raises(PlanningError, match="arc-nope"):
        await plan_chapter(state)


# --------------------------------------------------------------------------- #
# plan_beat                                                                   #
# --------------------------------------------------------------------------- #

async def test_plan_beat_writes_ordered_rows_with_pad_constraints(seeded, monkeypatch):
    chapter_id = _seed_active_chapter(seeded)
    calls = []

    async def fake_call_llm(endpoint, messages, **kwargs):
        calls.append(messages)
        return _response(BEATS_JSON)

    patch_planner_llm(monkeypatch, beat=fake_call_llm)

    delta = await plan_beat(_state(chapter_id))

    conn = connect_db(seeded.db_path)
    rows = get_beats_for_chapter(conn, chapter_id)
    conn.close()

    assert [row["ordering"] for row in rows] == [1, 2]
    assert [row["status"] for row in rows] == ["active", "planned"]

    # The beat plan is the only model call: PAD translation is a table lookup.
    assert len(calls) == 1

    baselines = load_pad_baselines()
    assert rows[0]["pad_constraint"] == baselines["neg_pos_neg"]
    # Beat 2's pleasure of -0.3 sits inside the ±0.33 band, so it reads "neu".
    assert rows[1]["pad_constraint"] == baselines["neu_neg_pos"]

    spec = json.loads(rows[0]["beat_spec"])
    assert spec["intent"] == "Mara finds the letter in the day's post."
    assert spec["entry_state"] == "A routine morning."
    assert spec["exit_state"] == "Mara is holding her own handwriting."
    assert spec["focal_character_id"] == "char-mara"
    assert spec["target_pad"] == {"pleasure": -0.6, "arousal": 0.8, "dominance": -0.4}

    # The concrete plot mandate is stored with the spec.
    assert spec["required_change"] == "Mara can no longer treat the post as routine."
    assert spec["observable_event"] == "Mara opens a letter written in her own hand."
    assert spec["beat_function"] == "discovery"
    # Only entries matching a chapter obligation survive into discharges.
    assert spec["discharges"] == ["Mara dates the earliest letter."]
    assert spec["thread_updates"] == [{"id": "thread-1", "status": "progressing"}]

    # A beat that declared no required_change falls back to its exit state, so
    # every stored beat carries a non-empty change.
    spec2 = json.loads(rows[1]["beat_spec"])
    assert spec2["required_change"] == "The letter is locked away."
    assert "observable_event" not in spec2
    assert "discharges" not in spec2

    pointer = delta["fsm_pointer"]
    assert pointer.chapter_id == chapter_id
    assert pointer.beat_index == 0


async def test_plan_beat_publishes_an_obligation_gap(seeded, monkeypatch):
    """An obligation no beat discharges is surfaced loudly, not silently lost."""
    chapter_id = _seed_active_chapter(
        seeded,
        obligations=["Mara dates the earliest letter.", "Mara hides it from Idris."],
    )

    async def fake_call_llm(endpoint, messages, **kwargs):
        return _response(BEATS_JSON)

    patch_planner_llm(monkeypatch, beat=fake_call_llm)

    queue = bus.subscribe()
    try:
        await plan_beat(_state(chapter_id))
        events = [queue.get_nowait() for _ in range(queue.qsize())]
    finally:
        bus.unsubscribe(queue)

    by_type = {event["type"]: event["data"] for event in events}
    assert by_type["planner_obligation_gap"]["unassigned"] == ["Mara hides it from Idris."]
    assert by_type["planner_obligation_gap"]["chapter_id"] == chapter_id


async def test_plan_beat_only_plans_the_active_chapter(seeded, monkeypatch):
    active = _seed_active_chapter(seeded)
    other = chapter_id_for(ARC_ID, 2)
    conn = connect_db(seeded.db_path)
    with conn:
        upsert_chapter(
            conn,
            id=other,
            arc_id=ARC_ID,
            ordering=2,
            description="Mara rows to the mainland.",
            obligations="[]",
            status="planned",
        )
    conn.close()

    async def fake_call_llm(endpoint, messages, **kwargs):
        return _response(BEATS_JSON)

    patch_planner_llm(monkeypatch, beat=fake_call_llm)

    # The pointer names no chapter, so the node must find the active one.
    await plan_beat(_state())

    conn = connect_db(seeded.db_path)
    assert len(get_beats_for_chapter(conn, active)) == 2
    assert get_beats_for_chapter(conn, other) == []
    conn.close()


async def test_plan_beat_publishes_summary_and_focal_pad_update(seeded, monkeypatch):
    chapter_id = _seed_active_chapter(seeded)

    async def fake_call_llm(endpoint, messages, **kwargs):
        return _response(BEATS_JSON)

    patch_planner_llm(monkeypatch, beat=fake_call_llm)

    queue = bus.subscribe()
    try:
        await plan_beat(_state(chapter_id))
        events = [queue.get_nowait() for _ in range(queue.qsize())]
    finally:
        bus.unsubscribe(queue)

    by_type = {event["type"]: event["data"] for event in events}
    assert by_type["beats_planned"]["beat_count"] == 2
    assert by_type["beats_planned"]["chapter_id"] == chapter_id
    assert by_type["pad_update"]["character_id"] == "char-mara"
    assert by_type["pad_update"]["target_pad"] == {
        "pleasure": -0.6,
        "arousal": 0.8,
        "dominance": -0.4,
    }


async def test_plan_beat_raises_on_unparseable_plan(seeded, monkeypatch):
    chapter_id = _seed_active_chapter(seeded)

    async def fake_call_llm(endpoint, messages, **kwargs):
        return _response("No beats today.")

    patch_planner_llm(monkeypatch, beat=fake_call_llm)

    with pytest.raises(StructuredOutputError):
        await plan_beat(_state(chapter_id))

    conn = connect_db(seeded.db_path)
    assert get_beats_for_chapter(conn, chapter_id) == []
    conn.close()


# --------------------------------------------------------------------------- #
# plan_beat: pacing context, threads, refrains, intensity                     #
# --------------------------------------------------------------------------- #

# A beat that closes the open thread. thread_updates was a dead field before:
# the planner never emitted it and plan_beat stripped it. Both are fixed now.
BEATS_CLOSING_THREAD = """```json
[
  {"ordering": 1, "intent": "Mara admits who wrote the letters.",
   "entry_state": "Denial.", "exit_state": "Confession.",
   "focal_character_id": "char-mara",
   "target_pad": {"pleasure": -0.2, "arousal": 0.3, "dominance": 0.1},
   "thread_updates": [{"id": "thread-1", "status": "closed"}]}
]
```"""

BEATS_WITH_REFRAIN = """```json
[
  {"ordering": 1, "intent": "Establish the keeper's creed.",
   "entry_state": "Dawn.", "exit_state": "The creed spoken.",
   "focal_character_id": "char-mara",
   "target_pad": {"pleasure": 0.1, "arousal": -0.2, "dominance": 0.4},
   "intended_refrain": ["The lantern must never go dark."]}
]
```"""

# Three beats, every one at high arousal: a flat, exhausting arc.
FLAT_HOT_BEATS = """```json
[
  {"ordering": 1, "intent": "Terror one.", "entry_state": "a", "exit_state": "b",
   "focal_character_id": "char-mara",
   "target_pad": {"pleasure": -0.5, "arousal": 0.9, "dominance": -0.3}},
  {"ordering": 2, "intent": "Terror two.", "entry_state": "b", "exit_state": "c",
   "focal_character_id": "char-mara",
   "target_pad": {"pleasure": -0.6, "arousal": 0.9, "dominance": -0.4}},
  {"ordering": 3, "intent": "Terror three.", "entry_state": "c", "exit_state": "d",
   "focal_character_id": "char-mara",
   "target_pad": {"pleasure": -0.7, "arousal": 0.95, "dominance": -0.5}}
]
```"""

# The varied re-plan: one peak, two calmer beats. Distinct intents so the
# test can tell which plan was accepted.
VARIED_BEATS = """```json
[
  {"ordering": 1, "intent": "A quiet opening.", "entry_state": "a", "exit_state": "b",
   "focal_character_id": "char-mara",
   "target_pad": {"pleasure": 0.1, "arousal": -0.3, "dominance": 0.2}},
  {"ordering": 2, "intent": "The peak.", "entry_state": "b", "exit_state": "c",
   "focal_character_id": "char-mara",
   "target_pad": {"pleasure": -0.6, "arousal": 0.9, "dominance": -0.4}},
  {"ordering": 3, "intent": "The settling.", "entry_state": "c", "exit_state": "d",
   "focal_character_id": "char-mara",
   "target_pad": {"pleasure": 0.0, "arousal": 0.1, "dominance": 0.3}}
]
```"""


async def test_plan_beat_stores_thread_updates_in_the_spec(seeded, monkeypatch):
    chapter_id = _seed_active_chapter(seeded)

    async def fake_call_llm(endpoint, messages, **kwargs):
        return _response(BEATS_CLOSING_THREAD)

    patch_planner_llm(monkeypatch, beat=fake_call_llm)
    await plan_beat(_state(chapter_id))

    conn = connect_db(seeded.db_path)
    rows = get_beats_for_chapter(conn, chapter_id)
    conn.close()
    spec = json.loads(rows[0]["beat_spec"])
    # Exactly the shape commit's _apply_thread_updates reads off beat_spec.
    assert spec["thread_updates"] == [{"id": "thread-1", "status": "closed"}]


async def test_thread_update_round_trips_planner_to_commit_to_db(seeded, monkeypatch):
    from museai.fsm.nodes.commit import commit_transaction

    chapter_id = _seed_active_chapter(seeded)

    async def fake_call_llm(endpoint, messages, **kwargs):
        return _response(BEATS_CLOSING_THREAD)

    patch_planner_llm(monkeypatch, beat=fake_call_llm)
    delta = await plan_beat(_state(chapter_id))

    # Commit the beat the planner just made active; its thread_update must apply.
    commit_state = make_initial_state(
        PROJECT_ID, delta["fsm_pointer"], current_draft_text="She said it plainly."
    )
    await commit_transaction(commit_state)

    conn = connect_db(seeded.db_path)
    thread = conn.execute("SELECT status FROM Threads WHERE id=?", ("thread-1",)).fetchone()
    conn.close()
    assert thread["status"] == "closed"


async def test_plan_beat_stores_and_announces_an_intended_refrain(seeded, monkeypatch):
    chapter_id = _seed_active_chapter(seeded)

    async def fake_call_llm(endpoint, messages, **kwargs):
        return _response(BEATS_WITH_REFRAIN)

    patch_planner_llm(monkeypatch, beat=fake_call_llm)

    queue = bus.subscribe()
    try:
        await plan_beat(_state(chapter_id))
        events = [queue.get_nowait() for _ in range(queue.qsize())]
    finally:
        bus.unsubscribe(queue)

    conn = connect_db(seeded.db_path)
    rows = get_beats_for_chapter(conn, chapter_id)
    conn.close()
    spec = json.loads(rows[0]["beat_spec"])
    assert spec["intended_refrain"] == ["The lantern must never go dark."]

    # Never silent: a human must be able to see the exemption and veto it.
    refrains = [e["data"] for e in events if e["type"] == "planner_refrain"]
    assert len(refrains) == 1
    assert refrains[0]["phrase"] == "The lantern must never go dark."


async def test_plan_beat_reprompts_once_for_a_flat_hot_arc(seeded, monkeypatch):
    chapter_id = _seed_active_chapter(seeded)
    replies = [FLAT_HOT_BEATS, VARIED_BEATS]

    async def fake_call_llm(endpoint, messages, **kwargs):
        return _response(replies.pop(0))

    patch_planner_llm(monkeypatch, beat=fake_call_llm)

    queue = bus.subscribe()
    try:
        await plan_beat(_state(chapter_id))
        events = [queue.get_nowait() for _ in range(queue.qsize())]
    finally:
        bus.unsubscribe(queue)

    conn = connect_db(seeded.db_path)
    rows = get_beats_for_chapter(conn, chapter_id)
    conn.close()

    # The flat-hot plan was re-prompted, and the varied re-plan is what got
    # stored — not the flat one ("Terror one." etc).
    assert replies == []  # both replies consumed: one re-prompt happened
    intents = [json.loads(row["beat_spec"])["intent"] for row in rows]
    assert intents == ["A quiet opening.", "The peak.", "The settling."]
    assert any(e["type"] == "planner_intensity" for e in events)


async def test_plan_beat_accepts_a_varied_arc_without_reprompting(seeded, monkeypatch):
    chapter_id = _seed_active_chapter(seeded)
    calls = []

    async def fake_call_llm(endpoint, messages, **kwargs):
        calls.append(messages)
        return _response(VARIED_BEATS)

    patch_planner_llm(monkeypatch, beat=fake_call_llm)
    await plan_beat(_state(chapter_id))

    # A varied plan is accepted on the first call; no intensity re-prompt.
    assert len(calls) == 1


async def test_plan_beat_gives_the_planner_story_position_and_threads(seeded, monkeypatch):
    chapter_id = _seed_active_chapter(seeded)
    captured = []

    async def fake_call_llm(endpoint, messages, **kwargs):
        captured.append(messages)
        return _response(BEATS_JSON)

    patch_planner_llm(monkeypatch, beat=fake_call_llm)
    await plan_beat(_state(chapter_id))

    user = captured[0][1]["content"]
    assert "Chapter 1 of 1" in user            # positional context
    assert 'current="true"' in user            # this chapter marked among siblings
    assert "Who is writing the letters?" in user  # the open thread, with status


# --------------------------------------------------------------------------- #
# The PAD baseline table                                                      #
# --------------------------------------------------------------------------- #

def test_pad_baselines_defines_all_27_regions():
    baselines = json.loads(PAD_BASELINES_PATH.read_text(encoding="utf-8"))

    assert len(baselines) == 27  # the full 3x3x3 grid
    assert set(baselines) == set(pad_keys())

    for region, baseline in baselines.items():
        assert isinstance(baseline, str), region
        assert baseline.strip(), region


def test_resolve_pad_constraint_covers_every_axis_combination():
    readings = (-0.8, 0.0, 0.8)
    resolved = {}

    for pleasure, arousal, dominance in product(readings, repeat=3):
        constraint = resolve_pad_constraint(pleasure, arousal, dominance)
        assert constraint.strip()
        resolved[pad_key(pleasure, arousal, dominance)] = constraint

    # Three readings per axis land on three distinct bands, so the 27 sampled
    # coordinates cover the table exactly once each.
    assert set(resolved) == set(pad_keys())
    assert len(set(resolved.values())) == 27


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-1.0, "neg"),
        (-0.8, "neg"),
        (-PAD_BAND_THRESHOLD - 0.01, "neg"),
        (-PAD_BAND_THRESHOLD, "neu"),
        (0.0, "neu"),
        (PAD_BAND_THRESHOLD, "neu"),
        (PAD_BAND_THRESHOLD + 0.01, "pos"),
        (0.8, "pos"),
        (1.0, "pos"),
    ],
)
def test_quantize_axis_bands(value, expected):
    assert quantize_axis(value) == expected


def test_pad_key_orders_axes_pleasure_arousal_dominance():
    assert pad_key(0.8, 0.0, -0.8) == "pos_neu_neg"


async def test_plan_beat_resolves_focal_character_by_name(seeded, monkeypatch):
    """A focal id given as a character *name* resolves to the seeded id; an
    unknown one attributes PAD to nobody rather than an arbitrary character."""
    _seed_active_chapter(seeded)
    beats = json.dumps([
        {"ordering": 1, "intent": "Named by name.",
         "focal_character_id": "Mara",
         "target_pad": {"pleasure": 0.1, "arousal": 0.1, "dominance": 0.1}},
        {"ordering": 2, "intent": "Named by nobody known.",
         "focal_character_id": "the-mysterious-stranger",
         "target_pad": {"pleasure": 0.1, "arousal": 0.1, "dominance": 0.1}},
    ])

    async def fake_call_llm(endpoint, messages, **kwargs):
        return _response(beats)

    patch_planner_llm(monkeypatch, beat=fake_call_llm)
    await plan_beat(_state())

    conn = connect_db(seeded.db_path)
    rows = get_beats_for_chapter(conn, chapter_id_for(ARC_ID, 1))
    conn.close()

    specs = [json.loads(row["beat_spec"]) for row in rows]
    assert specs[0]["focal_character_id"] == "char-mara"
    assert specs[1]["focal_character_id"] == ""


class TestResolveFocalCharacter:
    """The values below are the real ones a live run produced, from logs/fsm.log.

    Every beat's PAD was dropped because none of them resolved, so
    `CharacterEmotions` never moved off its seeded values.
    """

    CHARACTERS = [
        {"id": "love-thy-doppelganger-char-1", "name": "Chloe Evans"},
        {"id": "love-thy-doppelganger-char-2", "name": "Liam Hayes"},
    ]

    def resolve(self, raw: str) -> str:
        from museai.fsm.nodes.plan_beat import resolve_focal_character

        return resolve_focal_character(raw, self.CHARACTERS)

    def test_an_exact_id_passes_through(self):
        assert self.resolve("love-thy-doppelganger-char-1") == "love-thy-doppelganger-char-1"

    def test_the_schema_key_fused_to_the_value_is_stripped(self):
        """`"focal_character_id": "character-id=<id>"` — observed in production."""
        assert self.resolve("character-id=love-thy-doppelganger-char-1") == "love-thy-doppelganger-char-1"
        assert self.resolve("focal_character_id: love-thy-doppelganger-char-2") == "love-thy-doppelganger-char-2"

    def test_a_case_folded_id_resolves(self):
        assert self.resolve("LOVE-THY-DOPPELGANGER-CHAR-2") == "love-thy-doppelganger-char-2"

    def test_a_character_name_resolves(self):
        assert self.resolve("Chloe Evans") == "love-thy-doppelganger-char-1"
        assert self.resolve("  liam hayes ") == "love-thy-doppelganger-char-2"

    def test_surrounding_quotes_are_stripped(self):
        assert self.resolve('"love-thy-doppelganger-char-1"') == "love-thy-doppelganger-char-1"

    def test_an_invented_abbreviation_does_not_resolve(self):
        """`ch-1` / `cl-2` — the model made these up. Guessing would corrupt PAD state."""
        assert self.resolve("ch-1") == ""
        assert self.resolve("cl-2") == ""

    def test_an_ambiguous_string_resolves_to_nobody(self):
        """Two known ids embedded: attributing to either would be a coin flip."""
        raw = "love-thy-doppelganger-char-1 and love-thy-doppelganger-char-2"
        assert self.resolve(raw) == ""

    def test_empty_and_whitespace_resolve_to_nobody(self):
        assert self.resolve("") == ""
        assert self.resolve("   ") == ""


# --------------------------------------------------------------------------- #
# tool rosters                                                                #
# --------------------------------------------------------------------------- #

async def test_each_planner_is_offered_its_own_tool_roster(seeded, monkeypatch):
    offered: dict[str, list] = {}

    async def fake_chapter(endpoint, messages, **kwargs):
        offered["chapter_planner"] = kwargs.get("tools")
        return _response(CHAPTERS_JSON)

    async def fake_beat(endpoint, messages, **kwargs):
        offered["beat_planner"] = kwargs.get("tools")
        return _response(BEATS_JSON)

    patch_planner_llm(monkeypatch, chapter=fake_chapter, beat=fake_beat)

    await plan_chapter(_state())
    chapter_id = chapter_id_for(ARC_ID, 1)
    await plan_beat(_state(chapter_id))

    assert [t["function"]["name"] for t in offered["chapter_planner"]] == [
        "get_seed_contract", "get_full_outline", "get_thread_history",
        "get_thread_status", "get_canonical_state", "check_plan_node",
    ]
    assert [t["function"]["name"] for t in offered["beat_planner"]] == [
        "get_seed_contract", "get_current_pointer_context", "get_chapter_context",
        "get_character_emotion_history", "get_canonical_state", "check_plan_node",
    ]


async def test_plan_chapter_retries_placeholder_obligations_before_writing(seeded, monkeypatch):
    placeholder = json.dumps([
        {
            "ordering": 1,
            "description": "Mara catalogs the letters.",
            "obligations": ["An event that must occur"],
        }
    ])
    replies = [placeholder, CHAPTERS_JSON]
    calls = 0

    async def fake_call_llm(endpoint, messages, **kwargs):
        nonlocal calls
        reply = replies[min(calls, len(replies) - 1)]
        calls += 1
        return _response(reply)

    patch_planner_llm(monkeypatch, chapter=fake_call_llm)
    await plan_chapter(_state())

    conn = connect_db(seeded.db_path)
    rows = get_chapters_for_arc(conn, ARC_ID)
    conn.close()
    assert calls == 2
    assert json.loads(rows[0]["obligations"]) == [
        "Mara dates the earliest letter.", "Mara hides it from Idris."
    ]


async def test_plan_chapter_does_not_persist_stubborn_placeholder_plan(seeded, monkeypatch):
    placeholder = json.dumps([
        {
            "ordering": 1,
            "description": "Mara catalogs the letters.",
            "obligations": ["An event that must occur"],
        }
    ])

    async def fake_call_llm(endpoint, messages, **kwargs):
        return _response(placeholder)

    patch_planner_llm(monkeypatch, chapter=fake_call_llm)
    with pytest.raises(StructuredOutputError, match="output-format placeholder"):
        await plan_chapter(_state())

    conn = connect_db(seeded.db_path)
    assert get_chapters_for_arc(conn, ARC_ID) == []
    conn.close()


class TestPlannerItemSchemas:
    def test_chapter_fields_are_exact_and_strictly_typed(self):
        with pytest.raises(StructuredOutputError, match="unexpected fields"):
            _validate_chapter_item(
                {
                    "ordering": 1,
                    "description": "A chapter.",
                    "obligations": [],
                    "summary": "renamed description",
                },
                1,
            )
        with pytest.raises(StructuredOutputError, match="ordering must be an integer"):
            _validate_chapter_item(
                {"ordering": "1", "description": "A chapter.", "obligations": []},
                1,
            )

    @pytest.mark.parametrize("value", [
        "An event that must occur",
        "  A CONCRETE EVENT THAT MUST OCCUR  ",
    ])
    def test_chapter_placeholder_obligations_are_rejected(self, value):
        with pytest.raises(StructuredOutputError, match="output-format placeholder"):
            _validate_chapter_item(
                {"ordering": 1, "description": "A chapter.", "obligations": [value]}, 1
            )

    def test_chapter_needs_at_least_one_obligation(self):
        with pytest.raises(StructuredOutputError, match="non-empty array"):
            _validate_chapter_item(
                {"ordering": 1, "description": "A chapter.", "obligations": []}, 1
            )

    @pytest.mark.parametrize("value", ["0.5", True, None, float("nan"), float("inf")])
    def test_beat_pad_values_are_finite_json_numbers(self, value):
        item = {
            "ordering": 1,
            "intent": "A change occurs.",
            "target_pad": {"pleasure": value, "arousal": 0.0, "dominance": 0.0},
        }
        with pytest.raises(StructuredOutputError, match="target_pad.pleasure"):
            _validate_beat_item(item, 1)

    def test_renamed_beat_key_is_rejected(self):
        item = {
            "ordering": 1,
            "Intent": "wrong casing",
            "target_pad": {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0},
        }
        with pytest.raises(StructuredOutputError, match="unexpected fields: Intent"):
            _validate_beat_item(item, 1)

    def test_missing_core_beat_field_is_rejected(self):
        item = {
            "ordering": 1,
            "target_pad": {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0},
        }
        with pytest.raises(StructuredOutputError, match="missing required fields: intent"):
            _validate_beat_item(item, 1)

    @pytest.mark.parametrize("status", ["open", "resolved", "Progressing", True])
    def test_thread_update_status_is_an_exact_enum(self, status):
        item = {
            "ordering": 1,
            "intent": "A change occurs.",
            "target_pad": {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0},
            "thread_updates": [{"id": "thread-1", "status": status}],
        }
        with pytest.raises(StructuredOutputError, match="invalid status"):
            _validate_beat_item(item, 1)
