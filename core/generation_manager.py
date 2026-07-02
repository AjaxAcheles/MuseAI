"""Module: M01 (Coordinator & State Machine)

Background driver for generation runs; lives in core to share the server event loop.

This is the minimal linear orchestrator for the local end-to-end path. It runs the
real nodes in dependency order as a plain async sequence:

    plan_global -> plan_arc -> plan_structure
      -> for each planned beat: assemble_context -> draft_prose -> commit

rather than the full LangGraph conditional graph (``fsm/graph.py::compile_graph``,
still to be built), so there are no reducer channels and each node's returned delta
is merged directly into a single state dict. Quality passes (critics, audit,
revision) and the branching recovery edges are not wired here.

``start`` spawns the run as a background ``asyncio.Task`` and returns immediately
with a ``run_id``; lifecycle and streamed prose are published to the injected
:class:`~core.stream_bus.StreamBus` for the dashboard SSE surface. Per-run status is
queryable via :meth:`status`.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import core.runtime as runtime
from core.stream_bus import StreamBus
from fsm.nodes.node_assemble_context import node_assemble_context
from fsm.nodes.node_commit_transaction import node_commit_transaction
from fsm.nodes.node_draft_prose import node_draft_prose
from fsm.nodes.node_plan_arc import node_plan_arc
from fsm.nodes.node_plan_global import node_plan_global
from fsm.nodes.node_plan_structure import node_plan_structure
from fsm.state import FSM_Pointer, make_initial_state

# Absolute ceiling on beats generated in one run — a runaway-loop backstop far above
# any real short story / novel structure.
_MAX_BEATS = 4096

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class GenerationManager:
    """Coordinate background generation runs and stream their progress."""

    def __init__(self, stream_bus: StreamBus | None = None) -> None:
        self.stream_bus = stream_bus or StreamBus()
        self._runs: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def start(self, project_metadata: dict[str, Any] | None = None) -> str:
        """Begin a generation run in the background and return its ``run_id``.

        ``project_metadata`` may carry ``premise_seed``, ``genre``,
        ``target_word_count``, and ``beat_word_target``.
        """
        run_id = uuid.uuid4().hex[:12]
        project_id = f"proj_{run_id}"
        self._runs[run_id] = {
            "run_id": run_id,
            "project_id": project_id,
            "status": "starting",
            "beats_committed": 0,
            "word_count": 0,
            "error": None,
        }
        task = asyncio.ensure_future(
            self._run(run_id, project_id, dict(project_metadata or {}))
        )
        self._tasks[run_id] = task
        return run_id

    def status(self, run_id: str) -> dict[str, Any] | None:
        """Return the status record for ``run_id`` (or ``None`` if unknown)."""
        record = self._runs.get(run_id)
        return dict(record) if record is not None else None

    async def _run(
        self, run_id: str, project_id: str, project_metadata: dict[str, Any]
    ) -> None:
        """Drive one generation run end to end, publishing progress to the bus."""
        record = self._runs[run_id]
        bus = self.stream_bus
        try:
            from core.config_loader import load_config

            config = load_config(CONFIG_PATH)
            runtime.init_resources(config)

            # `app_config` / `project_metadata` are node-consumed extras, not
            # OrchestratorState fields, so they are attached after construction
            # (make_initial_state rejects unknown override kwargs).
            state = make_initial_state(
                project_id,
                FSM_Pointer(arc_id="", chapter_id="", scene_id="", beat_index=0),
                planning_snapshot_id=f"snap_{project_id}",
            )
            state["app_config"] = config
            state["project_metadata"] = project_metadata

            record["status"] = "planning"
            bus.publish(run_id, {"type": "status", "status": "planning"})

            state.update(await node_plan_global(state) or state)
            state.update(await node_plan_arc(state) or state)
            state.update(await node_plan_structure(state) or state)

            beat_order = state.get("beat_order") or []
            bus.publish(
                run_id,
                {"type": "structure", "beat_count": len(beat_order), "status": "drafting"},
            )
            record["status"] = "drafting"
            record["beats_planned"] = len(beat_order)

            def _emit(token: str) -> None:
                bus.publish(run_id, {"type": "token", "text": token})

            beats_done = 0
            while not state.get("generation_complete") and beats_done < _MAX_BEATS:
                beat_id = getattr(state["fsm_pointer"], "beat_id", "")
                bus.publish(run_id, {"type": "beat_start", "beat_id": beat_id})

                state.update(await node_assemble_context(state) or state)
                state.update(await node_draft_prose(state, on_token=_emit) or {})
                state.update(await node_commit_transaction(state) or {})

                beats_done += 1
                record["beats_committed"] = beats_done
                record["word_count"] = state.get("committed_word_count", 0)
                bus.publish(
                    run_id,
                    {
                        "type": "beat_committed",
                        "beat_id": beat_id,
                        "prose": state.get("last_committed_prose", ""),
                        "word_count": record["word_count"],
                    },
                )

            record["status"] = "complete"
            bus.publish(
                run_id,
                {"type": "complete", "beats": beats_done, "word_count": record["word_count"]},
            )
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI + status
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            bus.publish(run_id, {"type": "error", "message": record["error"]})
        finally:
            bus.close(run_id)
