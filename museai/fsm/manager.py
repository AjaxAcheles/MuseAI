"""Background lifecycle manager for the MuseAI generation graph.

The manager owns graph execution and human-review resumption. The LangGraph run
never blocks waiting for a browser: the ``review`` node ends the current run, the
manager records ``status='review'``, and an explicit ``resolve_review`` call
starts the next run at either ``commit`` or ``assemble``.
"""

from __future__ import annotations

import asyncio
from typing import Literal, cast

from museai.core.config import AppConfig
from museai.core.logging_setup import log_node_event
from museai.core.stream_bus import bus
from museai.fsm.export import committed_word_count, export_manuscript
from museai.fsm.graph import GraphEntry, build_graph
from museai.fsm.nodes.deps import set_node_config
from museai.fsm.state import FSM_Pointer, OrchestratorState, accumulate_or_reset, make_initial_state
from museai.memory.db import connect_db

RunStatus = Literal["idle", "running", "paused", "review", "stopped", "done"]
ReviewDecision = Literal["accept", "regenerate"]


class GenerationManagerError(RuntimeError):
    """The generation manager cannot perform the requested lifecycle action."""


class GenerationManager:
    """Run the compiled graph in a background asyncio task."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        set_node_config(config)
        self.graph = build_graph(config)
        self.state: OrchestratorState | None = None
        self.status: RunStatus = "idle"
        self._task: asyncio.Task | None = None
        self._paused_entry: GraphEntry = "assemble"

    async def start(self, project_id: str) -> None:
        """Build the initial state and start generation in the background."""
        if self._task is not None and not self._task.done():
            raise GenerationManagerError("generation is already running")
        pointer = self._initial_pointer(project_id)
        self.state = make_initial_state(project_id, pointer)
        await self._start_run("plan_chapter")

    def pause(self) -> None:
        """Request a pause at the next safe boundary between graph nodes."""
        if self.state is not None:
            self.state["pause_requested"] = True
        if self.status == "running":
            log_node_event("manager", event="pause_requested")

    async def resume(self) -> None:
        """Resume a paused run from the next safe graph entry."""
        if self.status != "paused":
            raise GenerationManagerError(f"cannot resume while status is {self.status!r}")
        if self.state is None:
            raise GenerationManagerError("cannot resume without state")
        self.state["pause_requested"] = False
        await self._start_run(self._paused_entry)

    def stop(self) -> None:
        """Request a hard stop and cancel any in-flight background task."""
        if self.state is not None:
            self.state["hard_stop_asserted"] = True
        self.status = "stopped"
        if self._task is not None and not self._task.done():
            self._task.cancel()
        log_node_event("manager", event="stop_requested")

    async def resolve_review(
        self, decision: ReviewDecision, edited_text: str | None = None
    ) -> None:
        """Resolve the parked review state and resume at the requested boundary."""
        if self.status != "review":
            raise GenerationManagerError(f"cannot resolve review while status is {self.status!r}")
        if self.state is None:
            raise GenerationManagerError("cannot resolve review without state")

        if decision == "accept":
            accepted = edited_text or self.state["best_seen_draft"]
            if not accepted:
                raise GenerationManagerError("accept requires an edited_text or a best_seen_draft")
            self.state.update(
                {
                    "current_draft_text": accepted,
                    "streaming_buffer": accepted,
                    "critic_failures": [],
                    "review_requested": False,
                    "pause_requested": False,
                    "hard_stop_asserted": False,
                }
            )
            await self._start_run("commit")
            return

        if decision == "regenerate":
            self.state.update(
                {
                    "retry_count": 0,
                    "critic_failures": [],
                    "review_requested": False,
                    "pause_requested": False,
                    "hard_stop_asserted": False,
                }
            )
            await self._start_run("assemble")
            return

        raise GenerationManagerError(f"unknown review decision: {decision!r}")

    async def wait(self) -> RunStatus:
        """Wait for the current background run to park or finish; useful for tests."""
        task = self._task
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                self.status = "stopped"
        return self.status

    def _initial_pointer(self, project_id: str) -> FSM_Pointer:
        conn = connect_db(self.config.db_path)
        try:
            arc = conn.execute(
                """
                SELECT * FROM Arcs
                WHERE project_id=? AND status='active'
                ORDER BY ordering ASC
                LIMIT 1
                """,
                (project_id,),
            ).fetchone()
            if arc is None:
                arc = conn.execute(
                    """
                    SELECT * FROM Arcs
                    WHERE project_id=?
                    ORDER BY ordering ASC
                    LIMIT 1
                    """,
                    (project_id,),
                ).fetchone()
            if arc is None:
                raise GenerationManagerError(f"project {project_id!r} has no seeded arcs")
            return FSM_Pointer(arc_id=arc["id"], chapter_id="", beat_index=0)
        finally:
            conn.close()

    async def _start_run(self, entry_point: GraphEntry) -> None:
        if self.state is None:
            raise GenerationManagerError("cannot start a graph run without state")
        self.graph = build_graph(self.config, entry_point=entry_point)
        self.status = "running"
        await bus.publish("run_status", {"status": self.status, "entry_point": entry_point})
        self._task = asyncio.create_task(self._run(entry_point))
        await asyncio.sleep(0)

    async def _run(self, entry_point: GraphEntry) -> None:
        assert self.state is not None
        log_node_event("manager", event="run_started", entry_point=entry_point)
        try:
            async for chunk in self.graph.astream(self.state):
                for node_name, delta in chunk.items():
                    if delta:
                        self._merge_delta(delta)
                    await bus.publish(
                        "run_status",
                        {"status": self.status, "node": node_name, "entry_point": entry_point},
                    )
                    if node_name == "review":
                        self.status = "review"
                        await bus.publish("run_status", {"status": self.status, "node": node_name})
                        log_node_event("manager", event="run_parked_for_review")
                        return
                    if self.state["hard_stop_asserted"]:
                        self.status = "stopped"
                        await bus.publish("run_status", {"status": self.status, "node": node_name})
                        return
                    if self.state["pause_requested"]:
                        self.status = "paused"
                        self._paused_entry = self._next_entry_after(node_name)
                        await bus.publish(
                            "run_status",
                            {"status": self.status, "next_entry": self._paused_entry},
                        )
                        return
            if self.status == "running":
                self.status = "done"
                manuscript_path = export_manuscript(self.config)
                final_word_count = committed_word_count(self.config)
                await bus.publish(
                    "manuscript_ready",
                    {"path": str(manuscript_path), "word_count": final_word_count},
                )
                await bus.publish("run_status", {"status": self.status})
                log_node_event(
                    "manager",
                    event="run_done",
                    manuscript_path=manuscript_path,
                    final_word_count=final_word_count,
                )
        except asyncio.CancelledError:
            self.status = "stopped"
            await bus.publish("run_status", {"status": self.status})
            raise

    def _merge_delta(self, delta: dict) -> None:
        assert self.state is not None
        for key, value in delta.items():
            if key == "critic_failures":
                self.state[key] = accumulate_or_reset(self.state[key], value)
            else:
                self.state[key] = value

    def _next_entry_after(self, node_name: str) -> GraphEntry:
        next_entry = {
            "plan_chapter": "plan_beat",
            "plan_beat": "assemble",
            "assemble": "draft",
            "draft": "audit",
            "audit": "critics",
            "critics": "revise",
            "revise": "audit",
            "commit": "assemble",
        }.get(node_name, "assemble")
        return cast(GraphEntry, next_entry)
