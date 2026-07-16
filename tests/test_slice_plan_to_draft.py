"""End-to-end slice: seed → plan_chapter → plan_beat → assemble_context → draft_prose.

Every node is real and every write lands in a real SQLite file. Only the endpoint
is faked: the two planners get canned JSON, the drafter gets a canned stream.
This is the isolation check that the four nodes actually compose — that each
one's state delta is what the next one reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from museai.core.runtime import init_resources
from museai.core.stream_bus import bus
from museai.fsm.nodes import draft_prose as draft_prose_module
from museai.fsm.nodes import plan_beat as plan_beat_module
from museai.fsm.nodes import plan_chapter as plan_chapter_module
from museai.fsm.nodes.assemble_context import assemble_context
from museai.fsm.nodes.deps import set_node_config
from museai.fsm.nodes.draft_prose import draft_prose
from museai.fsm.nodes.plan_beat import plan_beat
from museai.fsm.nodes.plan_chapter import plan_chapter
from museai.fsm.pad import load_pad_baselines
from museai.fsm.state import FSM_Pointer, make_initial_state
from museai.memory.db import connect_db, get_beats_for_chapter, get_chapters_for_arc
from museai.seed.loader import load_seed

from conftest import patch_planner_llm

SEED_PATH = Path(__file__).resolve().parent.parent / "seeds" / "example.json"

CHAPTERS_JSON = """```json
[
  {"ordering": 1, "description": "Mara catalogs the impossible letters.",
   "obligations": ["Mara dates the earliest letter."]},
  {"ordering": 2, "description": "Mara rows to the mainland.",
   "obligations": ["Mara meets Idris."]}
]
```"""

BEATS_JSON = """```json
[
  {"ordering": 1, "intent": "The letter arrives in the day's post.",
   "entry_state": "A routine morning.", "exit_state": "Mara holds her own handwriting.",
   "focal_character_id": "lantern-keeper-char-1",
   "target_pad": {"pleasure": -0.7, "arousal": 0.8, "dominance": -0.5}},
  {"ordering": 2, "intent": "Mara files the letter and tells no one.",
   "entry_state": "Mara holds the letter.", "exit_state": "The letter is locked away.",
   "focal_character_id": "lantern-keeper-char-1",
   "target_pad": {"pleasure": -0.2, "arousal": -0.6, "dominance": 0.5}}
]
```"""

PROSE_TOKENS = ["The post came ", "up the path ", "in a canvas sack, ", "and she knew."]


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tool_calls: list[dict] = []
        self.tokens_out = len(text.split())
        self.finish_reason = "stop"


@pytest.fixture
def project(config_factory):
    """The example seed, loaded into a temp DB, with the nodes pointed at it."""
    config = config_factory(project_id="lantern-keeper")
    init_resources(config)
    load_seed(json.loads(SEED_PATH.read_text(encoding="utf-8")), config)
    set_node_config(config)
    return config


async def test_the_plan_to_draft_slice_composes(project, monkeypatch):
    async def fake_chapter_llm(endpoint, messages, **kwargs):
        return _Response(CHAPTERS_JSON)

    async def fake_beat_llm(endpoint, messages, **kwargs):
        return _Response(BEATS_JSON)

    async def fake_draft_loop(
        endpoint, messages, tools, tool_impls, max_iterations, *, on_token=None, **kw
    ):
        for token in PROSE_TOKENS:
            await on_token(token)
        return _Response("".join(PROSE_TOKENS))

    patch_planner_llm(monkeypatch, chapter=fake_chapter_llm, beat=fake_beat_llm)
    monkeypatch.setattr(draft_prose_module, "run_agent_loop", fake_draft_loop)

    # The seed marks its first arc active.
    arc_id = "lantern-keeper-arc-1"
    state = make_initial_state(
        "lantern-keeper", FSM_Pointer(arc_id=arc_id, chapter_id="", beat_index=0)
    )

    queue = bus.subscribe()
    try:
        state.update(await plan_chapter(state))
        state.update(await plan_beat(state))
        state.update(await assemble_context(state))
        state.update(await draft_prose(state))
        events = [queue.get_nowait() for _ in range(queue.qsize())]
    finally:
        bus.unsubscribe(queue)

    # The pointer walked from the arc down to beat 0 of chapter 1.
    pointer = state["fsm_pointer"]
    chapter_id = f"{arc_id}-c01"
    assert pointer.chapter_id == chapter_id
    assert pointer.beat_index == 0

    conn = connect_db(project.db_path)
    chapters = get_chapters_for_arc(conn, arc_id)
    beats = get_beats_for_chapter(conn, chapter_id)
    conn.close()

    assert [c["ordering"] for c in chapters] == [1, 2]
    assert [c["status"] for c in chapters] == ["active", "planned"]

    assert [b["ordering"] for b in beats] == [1, 2]
    assert [b["status"] for b in beats] == ["active", "planned"]
    assert [b["id"] for b in beats] == [f"{chapter_id}-b01", f"{chapter_id}-b02"]

    # PAD came from the static table, not from a model.
    baselines = load_pad_baselines()
    assert beats[0]["pad_constraint"] == baselines["neg_pos_neg"]
    assert beats[1]["pad_constraint"] == baselines["neu_neg_pos"]

    # The context package fed the drafter the beat it was pointed at.
    package = state["active_context_package"]
    assert package["beat"]["id"] == f"{chapter_id}-b01"
    assert package["beat"]["intent"] == "The letter arrives in the day's post."
    assert package["chapter"]["obligations"] == ["Mara dates the earliest letter."]
    assert package["recent_prose"] == []  # nothing committed yet

    # The prose is non-empty and is exactly what the stream delivered.
    assert state["current_draft_text"] == "".join(PROSE_TOKENS)
    assert state["current_draft_text"].strip()
    assert state["streaming_buffer"] == state["current_draft_text"]

    phases = [e["data"]["phase"] for e in events if e["type"] == "phase_change"]
    assert phases == ["Planning", "Drafting", "Auditing"]

    streamed = [e["data"]["text"] for e in events if e["type"] == "token"]
    assert streamed == PROSE_TOKENS
