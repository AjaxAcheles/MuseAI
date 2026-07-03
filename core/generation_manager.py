"""Module: M01 (Coordinator & State Machine)
Background driver for generation runs; lives in core to share the server event loop.

Owns one background task per vertical-slice run and the run-status lifecycle
(``idle → starting → running → paused/blocked/completed/error/stopped``). All
run telemetry goes through the shared :class:`~core.stream_bus.StreamBus`
(stamped with a per-run ``run_id``); all uncaught task exceptions are captured,
stored on status, and published as an ``error`` event — never swallowed.

The start payload is the product's Project Setup form: a required premise plus
optional title/genre/word target and per-run execution/approval mode overrides
(validated with the same cross-rules as config §3a). The macro-outline approval
gate is real: the runner parks on :meth:`wait_for_approval` and
:meth:`approve_plan` (from ``POST /plan/approve``) releases it.

Pause remains limited: it takes effect at the next planner-node boundary, not
mid-token. The status surface says so explicitly.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import uuid
from typing import Any

from core.app_log import truncate_user_content
from fsm.planning_runner import LLM_MODES, run_planning_vertical_slice
from fsm.state import FSM_Pointer, make_initial_state

logger = logging.getLogger(__name__)

RUN_STATUSES = (
    "idle",
    "starting",
    "running",
    "paused",
    "blocked",
    "completed",
    "error",
    "stopped",
)

PAUSE_NOTE = "Pause takes effect at the next planner-node boundary."

# Statuses during which the run counts as "running" for display purposes.
_ACTIVE_STATUSES = frozenset({"starting", "running", "paused"})

_DEFAULT_PROJECT_ID = "vertical_slice"
_PROJECT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# The only browser-suppliable start fields: plain-string creative seeds, sizing,
# per-run mode overrides, and the model mode. No code, paths, shell commands,
# or endpoint overrides.
_ALLOWED_PAYLOAD_KEYS = {
    "llm_mode",
    "project_id",
    "title",
    "premise",
    "genre",
    "target_word_count",
    "planning_execution_mode",
    "approval_mode",
}
_MAX_PREMISE_LENGTH = 2000
_MAX_SHORT_FIELD_LENGTH = 200
_EXECUTION_MODES = ("macro_outline_before_draft", "rolling")
_APPROVAL_MODES = ("off", "macro_outline")


class GenerationManager:
    """Coordinate the single background vertical-slice run."""

    def __init__(self, resources: Any) -> None:
        self._resources = resources
        self._task: asyncio.Task | None = None
        self._status = "idle"
        self._message = ""
        self._last_error: str | None = None
        self._llm_mode: str | None = None
        self._run_id: str | None = None
        self._final_state: dict[str, Any] | None = None
        self._run_state: dict[str, Any] | None = None
        self._resume_event = asyncio.Event()
        self._resume_event.set()
        self._approval_event = asyncio.Event()
        self._awaiting_approval = False

    # -- status ------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Manager-owned run status as a JSON-safe dict."""
        return {
            "status": self._status,
            "running": self._status in _ACTIVE_STATUSES,
            # "active" = the background task is alive (includes a run parked at
            # the approval gate); "running" is the display-facing flag.
            "active": self._task_active(),
            "message": self._message,
            "last_error": self._last_error,
            "llm_mode": self._llm_mode,
            "run_id": self._run_id,
            "awaiting_approval": self._awaiting_approval,
            "pause_note": PAUSE_NOTE,
        }

    @property
    def final_state(self) -> dict[str, Any] | None:
        """The last completed/blocked run's final orchestrator state (or None)."""
        return self._final_state

    @property
    def planning_snapshot_id(self) -> str | None:
        """The active/last run's PlanningSnapshot id, if a run has produced one."""
        state = self._run_state if self._run_state is not None else self._final_state
        if state is None:
            return None
        return state.get("planning_snapshot_id")

    @property
    def run_pointer(self) -> dict[str, Any] | None:
        """The active/last run's FSM pointer as a dict (or None)."""
        state = self._run_state if self._run_state is not None else self._final_state
        pointer = state.get("fsm_pointer") if state else None
        return pointer.model_dump() if pointer is not None else None

    def _task_active(self) -> bool:
        return self._task is not None and not self._task.done()

    async def join(self) -> None:
        """Await the current run task, if any (used by tests and shutdown)."""
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def _log(self, event: str, **fields: Any) -> None:
        self._resources.log(event, run_id=self._run_id, **fields)

    async def _set_status(self, status: str, message: str = "") -> None:
        if status not in RUN_STATUSES:
            raise ValueError(f"unknown run status {status!r}")
        self._status = status
        self._message = message
        self._log("run_status", status=status, message=message)
        await self._resources.event_bus.publish(
            "status",
            {
                "status": status,
                "running": status in _ACTIVE_STATUSES,
                "message": message,
            },
        )

    # -- controls ----------------------------------------------------------

    async def start(self, initial_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Start one background vertical-slice run; reject a duplicate start."""
        if self._task_active():
            return {
                "ok": False,
                "status": self._status,
                "message": "A run is already active; stop it before starting another.",
            }
        try:
            payload = self._sanitize_payload(initial_payload or {})
        except ValueError as exc:
            self._log("start_rejected", reason=str(exc))
            return {"ok": False, "status": self._status, "message": str(exc)}

        self._last_error = None
        self._final_state = None
        self._run_state = None
        self._llm_mode = payload["llm_mode"]
        self._run_id = uuid.uuid4().hex[:12]
        self._resources.event_bus.set_run_id(self._run_id)
        self._resume_event.set()
        self._approval_event.clear()
        self._awaiting_approval = False
        self._log(
            "run_start",
            payload={
                **{k: payload[k] for k in payload if k != "premise"},
                "premise": truncate_user_content(payload["premise"]),
            },
        )
        await self._set_status("starting", "Starting planning vertical slice.")
        # Created from a running event loop (Quart handler / test coroutine);
        # never at import time.
        self._task = asyncio.create_task(self._run(payload), name="vertical-slice-run")
        return {
            "ok": True,
            "status": self._status,
            "message": "Started planning vertical slice.",
        }

    async def pause(self) -> dict[str, Any]:
        """Request a pause at the next planner-node boundary (limited pause)."""
        if self._status != "running" or not self._task_active():
            return {
                "ok": False,
                "status": self._status,
                "message": "Pause is only available while a run is active.",
            }
        self._resume_event.clear()
        await self._set_status("paused", f"Paused. {PAUSE_NOTE}")
        return {"ok": True, "status": self._status, "message": f"Pause requested. {PAUSE_NOTE}"}

    async def resume(self) -> dict[str, Any]:
        """Resume a paused run."""
        if self._status != "paused":
            return {
                "ok": False,
                "status": self._status,
                "message": "Resume is only available while paused.",
            }
        self._resume_event.set()
        await self._set_status("running", "Resumed planning vertical slice.")
        return {"ok": True, "status": self._status, "message": "Resumed."}

    async def stop(self) -> dict[str, Any]:
        """Cancel the running task cleanly, or clear a blocked run's presentation."""
        if self._task_active():
            task = self._task
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            self._resume_event.set()
            self._awaiting_approval = False
            self._log("run_stopped")
            await self._set_status("stopped", "Run stopped.")
            await self._resources.event_bus.publish("phase_change", {
                "phase": "idle",
                "message": "Run stopped; pipeline idle.",
            })
            return {"ok": True, "status": self._status, "message": "Run stopped."}
        if self._status == "blocked":
            await self._set_status("stopped", "Blocked run cleared.")
            return {"ok": True, "status": self._status, "message": "Blocked run cleared."}
        return {"ok": False, "status": self._status, "message": "No active run to stop."}

    async def wait_if_paused(self) -> None:
        """Hold the runner while paused (called between planner nodes)."""
        await self._resume_event.wait()
        # Cooperative yield so SSE subscribers drain between nodes even when
        # the deterministic path never otherwise suspends.
        await asyncio.sleep(0)

    # -- macro approval gate -------------------------------------------------

    async def wait_for_approval(self) -> None:
        """Park the runner at the macro-outline approval gate until approved.

        Cancellable: Stop cancels the run task and the wait unwinds cleanly.
        The gate is a safe-boundary block, distinct from pause and error.
        """
        self._approval_event.clear()
        self._awaiting_approval = True
        self._log("approval_reached")
        await self._set_status("blocked", "Awaiting macro-outline approval.")
        try:
            await self._approval_event.wait()
        finally:
            self._awaiting_approval = False
        self._log("approval_resumed")
        await self._set_status("running", "Macro outline approved; resuming planning.")

    async def approve_plan(self) -> dict[str, Any]:
        """Approve the macro outline and release the waiting runner.

        Refused unless a run task is actually parked at the approval gate —
        never a silent no-op on an idle/completed/errored run.
        """
        if not self._task_active() or not self._awaiting_approval:
            return {
                "ok": False,
                "status": self._status,
                "message": "No run is awaiting macro-outline approval.",
            }
        self._log("approval_granted")
        self._approval_event.set()
        return {
            "ok": True,
            "status": self._status,
            "message": "Macro outline approved; resuming into scene/beat planning.",
        }

    # -- run body ----------------------------------------------------------

    def _sanitize_payload(self, raw: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError("Start payload must be a JSON object.")
        unknown = set(raw) - _ALLOWED_PAYLOAD_KEYS
        if unknown:
            raise ValueError(f"Unknown start field(s): {', '.join(sorted(unknown))}.")

        llm_mode = raw.get("llm_mode", "deterministic")
        if llm_mode not in LLM_MODES:
            raise ValueError(f"llm_mode must be one of {list(LLM_MODES)}.")

        project_id = raw.get("project_id", _DEFAULT_PROJECT_ID)
        if not isinstance(project_id, str) or not _PROJECT_ID_PATTERN.match(project_id):
            raise ValueError("project_id must be a short [A-Za-z0-9_-] slug.")

        premise = raw.get("premise", "")
        if not isinstance(premise, str) or not premise.strip():
            raise ValueError("premise is required: enter a story premise before starting.")
        if len(premise) > _MAX_PREMISE_LENGTH:
            raise ValueError(f"premise must be at most {_MAX_PREMISE_LENGTH} characters.")

        short_fields: dict[str, str] = {}
        for key in ("title", "genre"):
            value = raw.get(key, "")
            if not isinstance(value, str) or len(value) > _MAX_SHORT_FIELD_LENGTH:
                raise ValueError(
                    f"{key} must be a string of at most {_MAX_SHORT_FIELD_LENGTH} chars."
                )
            short_fields[key] = value.strip()

        config = self._resources.config
        raw_words = raw.get("target_word_count", config.runtime.word_count_target)
        try:
            target_word_count = int(raw_words)
        except (TypeError, ValueError):
            raise ValueError("target_word_count must be a positive integer.") from None
        if target_word_count <= 0:
            raise ValueError("target_word_count must be a positive integer.")

        execution_mode = raw.get(
            "planning_execution_mode", config.planning.execution_mode
        )
        if execution_mode not in _EXECUTION_MODES:
            raise ValueError(
                f"planning_execution_mode must be one of {list(_EXECUTION_MODES)}."
            )
        approval_mode = raw.get("approval_mode", config.planning.approval_mode)
        if approval_mode not in _APPROVAL_MODES:
            raise ValueError(f"approval_mode must be one of {list(_APPROVAL_MODES)}.")
        # Same cross-rule config validation enforces at boot (§3a).
        if approval_mode == "macro_outline" and execution_mode != "macro_outline_before_draft":
            raise ValueError(
                "approval_mode 'macro_outline' is valid only with "
                "planning_execution_mode 'macro_outline_before_draft'."
            )

        return {
            "llm_mode": llm_mode,
            "project_id": project_id,
            "premise": premise.strip(),
            "target_word_count": target_word_count,
            "planning_execution_mode": execution_mode,
            "approval_mode": approval_mode,
            **short_fields,
        }

    def _build_initial_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        resources = self._resources
        # The run-start mode snapshot comes from the validated start payload
        # (config values are its defaults); nodes read modes from state only.
        state = make_initial_state(
            payload["project_id"],
            FSM_Pointer(arc_id="", chapter_id="", scene_id="", beat_index=0),
            planning_execution_mode=payload["planning_execution_mode"],
            approval_mode=payload["approval_mode"],
        )
        state["app_config"] = resources.config
        state["sqlite_db_path"] = resources.stores["sqlite"].path
        state["event_log_path"] = resources.stores["event_log"].path
        state["provisional_store_path"] = resources.stores["provisional"].path
        state["project_metadata"] = {
            "title": payload["title"],
            "genre": payload["genre"],
            "premise_seed": payload["premise"],
            "target_word_count": payload["target_word_count"],
        }
        state["world_rules"] = []
        state["llm_mode"] = payload["llm_mode"]
        return state

    async def _run(self, payload: dict[str, Any]) -> None:
        bus = self._resources.event_bus
        try:
            await self._set_status("running", "Planning vertical slice is running.")
            state = self._build_initial_state(payload)
            self._run_state = state
            final_state = await run_planning_vertical_slice(
                state, self._resources, bus.publish
            )
            self._final_state = final_state
            block_reason = final_state.get("planning_block_reason")
            if block_reason or final_state.get("awaiting_planning_approval"):
                self._log("run_blocked", reason=block_reason)
                await self._set_status(
                    "blocked", f"Planning is blocked: {block_reason or 'awaiting approval'}."
                )
            else:
                self._log("run_completed")
                await self._set_status("completed", "Planning vertical slice completed.")
        except asyncio.CancelledError:
            # stop() awaits us and then publishes the "stopped" status.
            logger.info("vertical-slice run cancelled")
            raise
        except Exception as exc:  # noqa: BLE001 - surface, never swallow
            logger.exception("vertical-slice run failed")
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._log("run_error", error=self._last_error)
            await bus.publish(
                "error",
                {
                    "message": self._last_error,
                    "where": "vertical_slice_run",
                },
            )
            await bus.publish(
                "phase_change", {"phase": "error", "message": "Run failed; see error card."}
            )
            await self._set_status("error", self._last_error)
        finally:
            self._run_state = None
