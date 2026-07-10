"""LangGraph wiring for MuseAI v1.

The compiled graph starts at ``plan_chapter`` for a freshly seeded project: the
seed loader marks the first arc active, but chapters and beats do not exist yet.
From there the graph plans chapters, plans beats, drafts, audits, runs the single
continuity critic, revises as needed, commits, and routes from SQLite ground
truth until export.

Interactive review is a safe boundary. The ``review`` node records
``review_requested=True``, publishes the review payload, and ends the current
graph run. It never waits for a human inside the FSM; the generation manager
owns explicit resumption.
"""

from __future__ import annotations

import logging
from typing import Literal, cast

from langgraph.graph import END, StateGraph

from museai.core.config import AppConfig
from museai.core.logging_setup import log_node_event
from museai.core.stream_bus import bus
from museai.fsm.nodes.assemble_context import assemble_context
from museai.fsm.nodes.audit import audit
from museai.fsm.nodes.commit import commit_transaction
from museai.fsm.nodes.critics import adversarial_critics
from museai.fsm.nodes.deps import get_node_config, set_node_config
from museai.fsm.nodes.draft_prose import draft_prose
from museai.fsm.nodes.plan_beat import plan_beat
from museai.fsm.nodes.plan_chapter import plan_chapter
from museai.fsm.nodes.revise import revise_prose
from museai.fsm.routers.commit_router import (
    ASSEMBLE,
    EXPORT,
    PLAN_BEAT,
    PLAN_CHAPTER,
    commit_router,
)
from museai.fsm.routers.mode_selector import COMMIT, REVIEW, REVISE, mode_selector
from museai.fsm.state import FSM_Pointer, OrchestratorState
from museai.memory.db import connect_db

GraphEntry = Literal["plan_chapter", "plan_beat", "assemble", "draft", "audit", "critics", "revise", "commit"]


def _active_arc_pointer(state: OrchestratorState) -> FSM_Pointer:
    config = get_node_config()
    pointer = state["fsm_pointer"]
    conn = connect_db(config.db_path)
    try:
        arc = conn.execute(
            "SELECT * FROM Arcs WHERE id=? AND status='active'", (pointer.arc_id,)
        ).fetchone()
        if arc is None:
            arc = conn.execute(
                """
                SELECT * FROM Arcs
                WHERE project_id=? AND status='active'
                ORDER BY ordering ASC
                LIMIT 1
                """,
                (state["project_id"],),
            ).fetchone()
        if arc is None:
            arc = conn.execute(
                """
                SELECT * FROM Arcs
                WHERE project_id=?
                ORDER BY ordering ASC
                LIMIT 1
                """,
                (state["project_id"],),
            ).fetchone()
        if arc is None:
            return pointer
        return FSM_Pointer(arc_id=arc["id"], chapter_id="", beat_index=0)
    finally:
        conn.close()


def _active_chapter_pointer(state: OrchestratorState) -> FSM_Pointer:
    config = get_node_config()
    pointer = state["fsm_pointer"]
    conn = connect_db(config.db_path)
    try:
        arc_id = pointer.arc_id
        active_arc = conn.execute(
            "SELECT * FROM Arcs WHERE id=? AND status='active'", (arc_id,)
        ).fetchone()
        if active_arc is None:
            active_arc = conn.execute(
                """
                SELECT * FROM Arcs
                WHERE project_id=? AND status='active'
                ORDER BY ordering ASC
                LIMIT 1
                """,
                (state["project_id"],),
            ).fetchone()
            if active_arc is not None:
                arc_id = active_arc["id"]

        chapter = conn.execute(
            """
            SELECT * FROM Chapters
            WHERE arc_id=? AND status='active'
            ORDER BY ordering ASC
            LIMIT 1
            """,
            (arc_id,),
        ).fetchone()
        if chapter is None:
            return FSM_Pointer(arc_id=arc_id, chapter_id=pointer.chapter_id, beat_index=0)
        return FSM_Pointer(arc_id=arc_id, chapter_id=chapter["id"], beat_index=0)
    finally:
        conn.close()


def _active_beat_pointer(state: OrchestratorState) -> FSM_Pointer:
    config = get_node_config()
    chapter_pointer = _active_chapter_pointer(state)
    conn = connect_db(config.db_path)
    try:
        beat = conn.execute(
            """
            SELECT * FROM Beats
            WHERE chapter_id=? AND status='active'
            ORDER BY ordering ASC
            LIMIT 1
            """,
            (chapter_pointer.chapter_id,),
        ).fetchone()
        if beat is None:
            return chapter_pointer
        return FSM_Pointer(
            arc_id=chapter_pointer.arc_id,
            chapter_id=chapter_pointer.chapter_id,
            beat_index=int(beat["ordering"]) - 1,
        )
    finally:
        conn.close()


async def _plan_chapter_node(state: OrchestratorState) -> dict:
    synced = cast(OrchestratorState, {**state, "fsm_pointer": _active_arc_pointer(state)})
    return await plan_chapter(synced)


async def _plan_beat_node(state: OrchestratorState) -> dict:
    synced = cast(OrchestratorState, {**state, "fsm_pointer": _active_chapter_pointer(state)})
    return await plan_beat(synced)


async def _assemble_node(state: OrchestratorState) -> dict:
    pointer = _active_beat_pointer(state)
    synced = cast(OrchestratorState, {**state, "fsm_pointer": pointer})
    delta = await assemble_context(synced)
    return {"fsm_pointer": pointer, **delta}


async def _commit_node(state: OrchestratorState) -> dict:
    pointer = _active_beat_pointer(state)
    synced = cast(OrchestratorState, {**state, "fsm_pointer": pointer})
    delta = await commit_transaction(synced)
    return {"fsm_pointer": pointer, **delta}


async def review(state: OrchestratorState) -> dict:
    """Park the graph at the interactive review boundary and end this run."""
    pointer = state["fsm_pointer"]
    payload = {
        "project_id": state["project_id"],
        "fsm_pointer": {
            "arc_id": pointer.arc_id,
            "chapter_id": pointer.chapter_id,
            "beat_index": pointer.beat_index,
        },
        "best_seen_draft": state["best_seen_draft"],
        "best_seen_failure_count": state["best_seen_failure_count"],
        "failures": [failure.model_dump() for failure in state["critic_failures"]],
    }
    # WARNING: the revision loop spent its budget and never got the beat clean.
    # The run is now stopped until a person decides what to do with it.
    log_node_event(
        "review",
        level=logging.WARNING,
        event="review_needed",
        arc_id=pointer.arc_id,
        chapter_id=pointer.chapter_id,
        beat_index=pointer.beat_index,
        failures=len(state["critic_failures"]),
    )
    await bus.publish("review_needed", payload)
    return {"review_requested": True}


def build_graph(config: AppConfig, entry_point: GraphEntry = "plan_chapter"):
    """Build and compile the v1 orchestration graph.

    ``entry_point`` defaults to ``plan_chapter`` for a freshly seeded project.
    The generation manager uses the same graph with ``commit`` or ``assemble`` as
    explicit resume entries after human review.
    """
    set_node_config(config)

    graph = StateGraph(OrchestratorState)
    graph.add_node("plan_chapter", _plan_chapter_node)
    graph.add_node("plan_beat", _plan_beat_node)
    graph.add_node("assemble", _assemble_node)
    graph.add_node("draft", draft_prose)
    graph.add_node("audit", audit)
    graph.add_node("critics", adversarial_critics)
    graph.add_node("revise", revise_prose)
    graph.add_node("commit", _commit_node)
    graph.add_node("review", review)

    graph.set_entry_point(entry_point)
    graph.add_edge("plan_chapter", "plan_beat")
    graph.add_edge("plan_beat", "assemble")
    graph.add_edge("assemble", "draft")
    graph.add_edge("draft", "audit")
    graph.add_edge("audit", "critics")
    graph.add_conditional_edges(
        "critics",
        mode_selector,
        {COMMIT: "commit", REVISE: "revise", REVIEW: "review"},
    )
    graph.add_edge("revise", "audit")
    graph.add_conditional_edges(
        "commit",
        commit_router,
        {
            ASSEMBLE: "assemble",
            PLAN_BEAT: "plan_beat",
            PLAN_CHAPTER: "plan_chapter",
            EXPORT: END,
        },
    )
    graph.add_edge("review", END)

    return graph.compile()
