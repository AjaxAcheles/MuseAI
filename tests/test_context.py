"""Tests for museai.fsm.nodes.assemble_context.

The node is model-free, so no test mocks an endpoint — it reads SQLite and
counts tokens. The budget is driven down to force pruning rather than up to
avoid it, so the priority order is exercised, not merely declared.
"""

from __future__ import annotations

import json
import logging

import pytest

from museai.core.stream_bus import bus
from museai.fsm.nodes.assemble_context import assemble_context, drafter_messages
from museai.fsm.nodes.deps import PlanningError, set_node_config
from museai.fsm.state import FSM_Pointer, make_initial_state
from museai.llm.tokenizer import count_message_tokens
from museai.memory.db import (
    connect_db,
    init_db,
    upsert_arc,
    upsert_beat,
    upsert_chapter,
    upsert_character,
    upsert_character_emotions,
    upsert_project,
    upsert_thread,
)

PROJECT_ID = "test-project"
ARC_ID = "arc-1"
CHAPTER_ID = "arc-1-c01"
BEAT_ID = "arc-1-c01-b01"

OBLIGATIONS = ["Mara dates the earliest letter.", "Mara lies to Idris."]
PAD_CONSTRAINT = "Energy with nowhere to go. Checking exits, restarting sentences."

BEAT_SPEC = {
    "intent": "Mara finds the letter in the day's post.",
    "entry_state": "A routine morning.",
    "exit_state": "Mara is holding her own handwriting.",
    "target_pad": {"pleasure": -0.6, "arousal": 0.8, "dominance": -0.4},
    "focal_character_id": "char-mara",
}

# Committed prose, oldest first. Each passage is distinctive enough to assert on.
PASSAGES = [
    "OLDEST. " + "The lamp turned through the fog. " * 40,
    "MIDDLE. " + "She counted the rocks below the rail. " * 40,
    "NEWEST. " + "The post came up the path in a canvas sack. " * 40,
]

THREADS = [
    ("thread-hi", "Who is writing the letters?", 0.9),
    ("thread-mid", "The automation grant expires.", 0.6),
    ("thread-lo", "Whether Idris can be trusted.", 0.2),
]


def _seed(config, *, passages=PASSAGES, threads=THREADS) -> None:
    init_db(config.db_path)
    conn = connect_db(config.db_path)
    with conn:
        upsert_project(conn, id=PROJECT_ID, genre="mystery", premise="Letters arrive.")
        upsert_arc(
            conn, id=ARC_ID, project_id=PROJECT_ID, ordering=1,
            description="Mara traces the postmarks.", status="active",
        )
        upsert_chapter(
            conn, id=CHAPTER_ID, arc_id=ARC_ID, ordering=1,
            description="Mara catalogs the letters.",
            obligations=json.dumps(OBLIGATIONS), status="active",
        )
        upsert_beat(
            conn, id=BEAT_ID, chapter_id=CHAPTER_ID, ordering=1,
            beat_spec=json.dumps(BEAT_SPEC), pad_constraint=PAD_CONSTRAINT,
            word_target=600, status="active",
        )
        for thread_id, description, priority in threads:
            upsert_thread(
                conn, id=thread_id, project_id=PROJECT_ID, description=description,
                status="open", priority_score=priority,
            )
        upsert_character(
            conn, id="char-mara", project_id=PROJECT_ID, name="Mara",
            description="The keeper.",
        )
        upsert_character_emotions(
            conn, character_id="char-mara", pleasure=-0.2, arousal=0.1, dominance=0.3
        )

        # Committed beats of an earlier chapter, so they read back as recent prose.
        upsert_chapter(
            conn, id="arc-1-c00", arc_id=ARC_ID, ordering=0,
            description="Before.", obligations="[]", status="completed",
        )
        for index, prose in enumerate(passages, start=1):
            upsert_beat(
                conn, id=f"arc-1-c00-b{index:02d}", chapter_id="arc-1-c00",
                ordering=index, beat_spec="{}", pad_constraint=PAD_CONSTRAINT,
                word_target=600, prose=prose, word_count=len(prose.split()),
                status="completed",
            )
    conn.close()


def _state(beat_index: int = 0) -> dict:
    return make_initial_state(
        PROJECT_ID,
        FSM_Pointer(arc_id=ARC_ID, chapter_id=CHAPTER_ID, beat_index=beat_index),
    )


def _tokens(package, config) -> int:
    return count_message_tokens(
        drafter_messages(package),
        config.endpoint.tokenizer_family,
        config.endpoint.model_name,
    )


def _assert_protected(package) -> None:
    """The sections a draft cannot be written without."""
    assert package["beat"]["intent"] == BEAT_SPEC["intent"]
    assert package["beat"]["entry_state"] == BEAT_SPEC["entry_state"]
    assert package["beat"]["exit_state"] == BEAT_SPEC["exit_state"]
    assert package["pad_constraint"] == PAD_CONSTRAINT
    assert package["chapter"]["obligations"] == OBLIGATIONS


async def test_assembles_every_section_when_the_budget_is_ample(config_factory):
    config = config_factory(project_id=PROJECT_ID)
    _seed(config)
    set_node_config(config)

    delta = await assemble_context(_state())
    package = delta["active_context_package"]

    _assert_protected(package)
    assert package["beat"]["id"] == BEAT_ID
    assert package["beat"]["word_target"] == 600
    assert [t["id"] for t in package["threads"]] == ["thread-hi", "thread-mid", "thread-lo"]
    assert package["characters"][0]["pad"] == {
        "pleasure": -0.2, "arousal": 0.1, "dominance": 0.3
    }
    # Oldest -> newest, nothing dropped.
    assert package["recent_prose"] == PASSAGES
    assert package["budget"]["dropped_prose_passages"] == 0
    assert package["budget"]["dropped_threads"] == 0
    assert package["budget"]["over_budget"] is False


async def test_over_budget_drops_the_oldest_prose_first(config_factory):
    # A budget that fits the protected core plus roughly one passage.
    config = config_factory(project_id=PROJECT_ID, context_token_budget=700)
    _seed(config)
    set_node_config(config)

    package = (await assemble_context(_state()))["active_context_package"]

    _assert_protected(package)
    assert package["budget"]["dropped_prose_passages"] > 0
    # Whatever survived is a suffix of the original: the newest passages.
    surviving = package["recent_prose"]
    assert surviving == PASSAGES[len(PASSAGES) - len(surviving):]
    assert "OLDEST." not in "".join(surviving)
    assert _tokens(package, config) <= 700


async def test_prose_is_exhausted_before_any_thread_is_dropped(config_factory):
    config = config_factory(project_id=PROJECT_ID, context_token_budget=700)
    _seed(config)
    set_node_config(config)

    package = (await assemble_context(_state()))["active_context_package"]

    # The threads are small; dropping prose alone should have been enough.
    assert package["budget"]["dropped_threads"] == 0
    assert len(package["threads"]) == 3


async def test_threads_are_dropped_lowest_priority_first(config_factory):
    # Tight enough that all prose goes and the thread list must be trimmed too.
    config = config_factory(project_id=PROJECT_ID, context_token_budget=260)
    _seed(config)
    set_node_config(config)

    package = (await assemble_context(_state()))["active_context_package"]

    _assert_protected(package)
    assert package["recent_prose"] == []
    assert package["budget"]["dropped_threads"] > 0

    surviving = [t["id"] for t in package["threads"]]
    # Survivors are the highest-priority prefix; thread-lo goes before thread-mid.
    assert surviving == ["thread-hi", "thread-mid", "thread-lo"][: len(surviving)]
    assert "thread-lo" not in surviving


async def test_protected_core_survives_an_impossible_budget(
    config_factory, caplog, monkeypatch
):
    config = config_factory(project_id=PROJECT_ID, context_token_budget=1)
    _seed(config)
    set_node_config(config)

    # The museai tree owns fsm.log and does not propagate; caplog sits on root.
    monkeypatch.setattr(logging.getLogger("museai"), "propagate", True)
    with caplog.at_level("WARNING", logger="museai.fsm"):
        package = (await assemble_context(_state()))["active_context_package"]

    # Everything droppable is gone, and the beat's instructions are still intact.
    assert package["recent_prose"] == []
    assert package["threads"] == []
    _assert_protected(package)
    assert package["budget"]["over_budget"] is True
    assert "exceeds budget" in caplog.text


async def test_publishes_the_drafting_phase_change(config_factory):
    config = config_factory(project_id=PROJECT_ID)
    _seed(config)
    set_node_config(config)

    queue = bus.subscribe()
    try:
        await assemble_context(_state())
        events = [queue.get_nowait() for _ in range(queue.qsize())]
    finally:
        bus.unsubscribe(queue)

    by_type = {event["type"]: event["data"] for event in events}
    assert by_type["phase_change"]["phase"] == "Drafting"
    assert by_type["phase_change"]["beat_id"] == BEAT_ID


async def test_a_beat_with_no_pad_constraint_is_a_hard_failure(config_factory):
    config = config_factory(project_id=PROJECT_ID)
    _seed(config)
    set_node_config(config)

    conn = connect_db(config.db_path)
    with conn:
        conn.execute("UPDATE Beats SET pad_constraint=NULL WHERE id=?", (BEAT_ID,))
    conn.close()

    with pytest.raises(PlanningError, match="pad_constraint"):
        await assemble_context(_state())


async def test_an_unplanned_chapter_is_a_hard_failure(config_factory):
    config = config_factory(project_id=PROJECT_ID)
    _seed(config)
    set_node_config(config)

    conn = connect_db(config.db_path)
    with conn:
        conn.execute("DELETE FROM Beats WHERE chapter_id=?", (CHAPTER_ID,))
    conn.close()

    with pytest.raises(PlanningError, match="no planned beats"):
        await assemble_context(_state())
