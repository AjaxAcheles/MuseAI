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
from museai.fsm.nodes.plan_beat import plan_beat
from museai.fsm.nodes.plan_chapter import chapter_id_for, plan_chapter
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
   "word_target": 550, "focal_character_id": "char-mara",
   "target_pad": {"pleasure": -0.6, "arousal": 0.8, "dominance": -0.4}},
  {"ordering": 2, "intent": "Mara files the letter and says nothing.",
   "entry_state": "Mara is holding the letter.", "exit_state": "The letter is locked away.",
   "word_target": 700, "focal_character_id": "char-idris",
   "target_pad": {"pleasure": -0.3, "arousal": -0.5, "dominance": 0.6}}
]
```
"""


def _response(text: str) -> SimpleNamespace:
    """The only attribute the planners read off an ``LLMResponse``."""
    return SimpleNamespace(text=text)


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

    monkeypatch.setattr(plan_chapter_module, "call_llm", fake_call_llm)

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

    monkeypatch.setattr(plan_chapter_module, "call_llm", fake_call_llm)

    await plan_chapter(_state())
    await plan_chapter(_state())

    conn = connect_db(seeded.db_path)
    rows = get_chapters_for_arc(conn, ARC_ID)
    conn.close()
    assert len(rows) == 3


async def test_plan_chapter_publishes_phase_change_and_summary(seeded, monkeypatch):
    async def fake_call_llm(endpoint, messages, **kwargs):
        return _response(CHAPTERS_JSON)

    monkeypatch.setattr(plan_chapter_module, "call_llm", fake_call_llm)

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

    monkeypatch.setattr(plan_chapter_module, "call_llm", fake_call_llm)

    with pytest.raises(StructuredOutputError):
        await plan_chapter(_state())

    conn = connect_db(seeded.db_path)
    rows = get_chapters_for_arc(conn, ARC_ID)
    conn.close()
    assert rows == []


async def test_plan_chapter_raises_on_empty_array(seeded, monkeypatch):
    async def fake_call_llm(endpoint, messages, **kwargs):
        return _response("[]")

    monkeypatch.setattr(plan_chapter_module, "call_llm", fake_call_llm)

    with pytest.raises(StructuredOutputError, match="empty array"):
        await plan_chapter(_state())


async def test_plan_chapter_rejects_an_unknown_arc(seeded, monkeypatch):
    async def fake_call_llm(endpoint, messages, **kwargs):
        raise AssertionError("the endpoint must not be reached")

    monkeypatch.setattr(plan_chapter_module, "call_llm", fake_call_llm)

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

    monkeypatch.setattr(plan_beat_module, "call_llm", fake_call_llm)

    delta = await plan_beat(_state(chapter_id))

    conn = connect_db(seeded.db_path)
    rows = get_beats_for_chapter(conn, chapter_id)
    conn.close()

    assert [row["ordering"] for row in rows] == [1, 2]
    assert [row["status"] for row in rows] == ["active", "planned"]
    assert [row["word_target"] for row in rows] == [550, 700]

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

    pointer = delta["fsm_pointer"]
    assert pointer.chapter_id == chapter_id
    assert pointer.beat_index == 0


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

    monkeypatch.setattr(plan_beat_module, "call_llm", fake_call_llm)

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

    monkeypatch.setattr(plan_beat_module, "call_llm", fake_call_llm)

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

    monkeypatch.setattr(plan_beat_module, "call_llm", fake_call_llm)

    with pytest.raises(StructuredOutputError):
        await plan_beat(_state(chapter_id))

    conn = connect_db(seeded.db_path)
    assert get_beats_for_chapter(conn, chapter_id) == []
    conn.close()


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
