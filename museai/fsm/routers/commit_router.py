"""Ground-truth router after a beat commit.

Routing is decided from SQLite, not in-memory counters. When the router advances
the outline pointer it also writes the newly-active row's status, then mutates
``state['fsm_pointer']`` explicitly so the following node receives the pointer it
is meant to operate on.
"""

from __future__ import annotations

from museai.core.logging_setup import log_node_event
from museai.fsm.nodes.deps import get_node_config
from museai.fsm.state import FSM_Pointer, OrchestratorState
from museai.memory.db import (
    connect_db,
    get_arcs,
    get_beats_for_chapter,
    get_chapters_for_arc,
    get_project,
    upsert_arc,
    upsert_beat,
    upsert_chapter,
)

ASSEMBLE = "assemble"
PLAN_BEAT = "plan_beat"
PLAN_CHAPTER = "plan_chapter"
EXPORT = "export"


class CommitRoutingError(RuntimeError):
    """The commit router could not resolve a valid project outline."""


def _committed_words(conn, project_id: str) -> int:
    row = conn.execute(
        """
        SELECT COALESCE(SUM(Beats.word_count), 0) AS total
        FROM Beats
        JOIN Chapters ON Beats.chapter_id = Chapters.id
        JOIN Arcs ON Chapters.arc_id = Arcs.id
        WHERE Arcs.project_id=? AND Beats.status='completed'
        """,
        (project_id,),
    ).fetchone()
    return int(row["total"] or 0)


def _planned_beat_in_active_chapter(conn, pointer: FSM_Pointer):
    chapter = conn.execute(
        "SELECT * FROM Chapters WHERE id=? AND status='active'",
        (pointer.chapter_id,),
    ).fetchone()
    if chapter is None:
        chapter = conn.execute(
            """
            SELECT * FROM Chapters
            WHERE arc_id=? AND status='active'
            ORDER BY ordering ASC
            LIMIT 1
            """,
            (pointer.arc_id,),
        ).fetchone()
    if chapter is None:
        return None, None

    beat = conn.execute(
        """
        SELECT * FROM Beats
        WHERE chapter_id=? AND status='planned'
        ORDER BY ordering ASC
        LIMIT 1
        """,
        (chapter["id"],),
    ).fetchone()
    return chapter, beat


def _planned_chapter_without_beats(conn, arc_id: str):
    return conn.execute(
        """
        SELECT Chapters.*
        FROM Chapters
        LEFT JOIN Beats ON Beats.chapter_id = Chapters.id
        WHERE Chapters.arc_id=? AND Chapters.status='planned'
        GROUP BY Chapters.id
        HAVING COUNT(Beats.id)=0
        ORDER BY Chapters.ordering ASC
        LIMIT 1
        """,
        (arc_id,),
    ).fetchone()


def _all_outline_completed(conn, project_id: str) -> bool:
    unfinished = conn.execute(
        """
        SELECT
            (SELECT COUNT(*) FROM Arcs WHERE project_id=? AND status!='completed') +
            (SELECT COUNT(*) FROM Chapters
             JOIN Arcs ON Chapters.arc_id = Arcs.id
             WHERE Arcs.project_id=? AND Chapters.status!='completed') +
            (SELECT COUNT(*) FROM Beats
             JOIN Chapters ON Beats.chapter_id = Chapters.id
             JOIN Arcs ON Chapters.arc_id = Arcs.id
             WHERE Arcs.project_id=? AND Beats.status!='completed') AS n
        """,
        (project_id, project_id, project_id),
    ).fetchone()["n"]
    return int(unfinished or 0) == 0


def commit_router(state: OrchestratorState) -> str:
    """Choose the next node after commit; first matching SQLite branch wins."""
    config = get_node_config()
    pointer = state["fsm_pointer"]
    project_id = state["project_id"]

    conn = connect_db(config.db_path)
    try:
        project = get_project(conn, project_id)
        if project is None:
            raise CommitRoutingError(f"project {project_id!r} is not in the database")

        chapter, beat = _planned_beat_in_active_chapter(conn, pointer)
        if chapter is not None and beat is not None:
            with conn:
                upsert_beat(
                    conn,
                    id=beat["id"],
                    chapter_id=beat["chapter_id"],
                    ordering=beat["ordering"],
                    beat_spec=beat["beat_spec"],
                    pad_constraint=beat["pad_constraint"],
                    prose=beat["prose"],
                    word_count=beat["word_count"],
                    status="active",
                )
            state["fsm_pointer"] = FSM_Pointer(
                arc_id=pointer.arc_id,
                chapter_id=chapter["id"],
                beat_index=int(beat["ordering"]) - 1,
            )
            log_node_event(
                "commit_router",
                event="route",
                destination=ASSEMBLE,
                arc_id=pointer.arc_id,
                chapter_id=chapter["id"],
                beat_id=beat["id"],
                beat_index=int(beat["ordering"]) - 1,
            )
            return ASSEMBLE

        active_arc = conn.execute(
            "SELECT * FROM Arcs WHERE id=? AND status='active'", (pointer.arc_id,)
        ).fetchone()
        if active_arc is None:
            active_arc = conn.execute(
                """
                SELECT * FROM Arcs
                WHERE project_id=? AND status='active'
                ORDER BY ordering ASC
                LIMIT 1
                """,
                (project_id,),
            ).fetchone()

        if active_arc is not None:
            next_chapter = _planned_chapter_without_beats(conn, active_arc["id"])
            if next_chapter is not None:
                with conn:
                    upsert_chapter(
                        conn,
                        id=next_chapter["id"],
                        arc_id=next_chapter["arc_id"],
                        ordering=next_chapter["ordering"],
                        description=next_chapter["description"],
                        obligations=next_chapter["obligations"],
                        status="active",
                    )
                state["fsm_pointer"] = FSM_Pointer(
                    arc_id=active_arc["id"], chapter_id=next_chapter["id"], beat_index=0
                )
                log_node_event(
                    "commit_router",
                    event="route",
                    destination=PLAN_BEAT,
                    arc_id=active_arc["id"],
                    chapter_id=next_chapter["id"],
                    beat_index=0,
                )
                return PLAN_BEAT

        next_arc = conn.execute(
            """
            SELECT * FROM Arcs
            WHERE project_id=? AND status='planned'
            ORDER BY ordering ASC
            LIMIT 1
            """,
            (project_id,),
        ).fetchone()
        if next_arc is not None:
            with conn:
                upsert_arc(
                    conn,
                    id=next_arc["id"],
                    project_id=next_arc["project_id"],
                    ordering=next_arc["ordering"],
                    description=next_arc["description"],
                    status="active",
                )
            state["fsm_pointer"] = FSM_Pointer(
                arc_id=next_arc["id"], chapter_id="", beat_index=0
            )
            log_node_event(
                "commit_router",
                event="route",
                destination=PLAN_CHAPTER,
                arc_id=next_arc["id"],
                chapter_id="",
                beat_index=0,
            )
            return PLAN_CHAPTER

        # A falsy target (NULL or 0) means "no word limit": the outline alone
        # decides when the manuscript is done.
        target = project["word_count_target"]
        total = _committed_words(conn, project_id)
        if (target and total >= int(target)) or _all_outline_completed(conn, project_id):
            log_node_event(
                "commit_router",
                event="route",
                destination=EXPORT,
                project_total=total,
                word_count_target=target,
            )
            return EXPORT

        log_node_event(
            "commit_router",
            event="route",
            destination=EXPORT,
            project_total=total,
            word_count_target=target,
            reason="no_remaining_planned_rows",
        )
        return EXPORT
    finally:
        conn.close()
