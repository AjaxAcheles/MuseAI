"""Module: M06 (Drafting & Generation)

Level-1 drafting node. Turns the assembled ``active_context_package`` plus the active
beat's plan (read from its beat ``PlanningNode``, keyed by ``state["fsm_pointer"]``)
into a single rendered prompt, fires a streamed generation call, accumulates the
streamed text, and passes the completed text through the M11 anti-slop black-box
contract (``detect_slop``/``resolve_slop`` — today an honest no-op passthrough) before
writing it to ``current_draft_text``. Model contact is only through the injected async
``call_seam`` (default: the real ``call_llm`` against ``config.endpoints.drafter``,
``stream=True``, ``timeout_seconds`` from ``config.runtime.inference_timeout_seconds``);
stream chunks are forwarded only through the injected ``publisher`` (default: no-op —
the real SSE bus is ``core/stream_bus.py``, a Build 19 concern this node never imports).

The node performs no store writes (the draft is state-only until M10 commits it), no
graph routing, and no waiting on a human. On a call-seam failure, the error surfaces to
the graph after state is confirmed clean: ``current_draft_text`` is left unchanged and
``streaming_buffer`` never carries a half-written draft as the working text.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Awaitable, Callable

import core.runtime as runtime
from core.antislop import detect_slop, resolve_slop
from core.logger import get_logger, log_node_event
from llm.call_llm import call_llm
from memory import sqlite_db
from prompts.prompt_loader import PromptLoader

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
_NODE_NAME = "node_draft_prose"

CallSeam = Callable[..., Awaitable[str]]
Publisher = Callable[[str], Any]


def _resolve_config(state: dict[str, Any]) -> Any:
    """Return the active typed config (mirrors the built planner nodes' resolution)."""
    config = state.get("app_config") or state.get("config")
    if config is not None:
        return config
    from core.config_loader import load_config

    return load_config(state.get("config_path", CONFIG_PATH))


def _parse_purpose(node: dict | None) -> dict:
    """Parse a PlanningNode's ``purpose`` JSON into a dict ({} on absent/invalid)."""
    if not node:
        return {}
    try:
        parsed = json.loads(node.get("purpose") or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _active_beat_plan(db_path: Any, snapshot_id: str | None, beat_id: str) -> dict:
    """Return the active beat's validated plan fields from its PlanningNode.

    Reads the sibling-node ``purpose`` JSON written by ``node_plan_beat``'s
    ``upsert_planning_node`` call (07.00/07.13 convention) — the same read pattern the
    planner nodes use for already-planned siblings. ``{}`` when the snapshot/beat is not
    yet planned (the graph wiring that guarantees planning-before-drafting is Build 14).
    """
    if not snapshot_id or not beat_id:
        return {}
    node_id = f"{snapshot_id}:beat:{beat_id}"
    node = next(
        (
            n
            for n in sqlite_db.get_planning_nodes(db_path, snapshot_id, level="beat")
            if n["node_id"] == node_id
        ),
        None,
    )
    return _parse_purpose(node)


def _prior_prose_tail(package: dict[str, Any]) -> str:
    """The last committed beat's prose in this scene, or "" if none is committed yet."""
    committed = (package.get("relational") or {}).get("committed_beats") or []
    if not committed:
        return ""
    return committed[-1].get("prose") or ""


def _target_word_count(package: dict[str, Any]) -> int:
    """Remaining scene word budget as this beat's guidance (never a hardcoded literal).

    The scene's ``word_budget`` and already-committed beat word counts both come from
    the context package; no per-beat allocation exists yet, so the remaining budget is
    the most honest guidance available without inventing a new mechanism.
    """
    relational = package.get("relational") or {}
    scene = relational.get("current_scene") or {}
    word_budget = int(scene.get("word_budget") or 0)
    committed_words = sum(
        int(b.get("word_count") or 0) for b in relational.get("committed_beats") or []
    )
    remaining = word_budget - committed_words
    return remaining if remaining > 0 else word_budget


async def _maybe_await(result: Any) -> None:
    """Await ``result`` if it is awaitable; a plain sync return is a no-op."""
    if inspect.isawaitable(result):
        await result


def _default_call_seam(endpoint: Any) -> CallSeam:
    """Build the default call seam: the real streamed ``call_llm`` against ``endpoint``."""

    async def _seam(
        messages: list[dict[str, str]], *, on_token: Callable[[str], Any], timeout_seconds: float
    ) -> str:
        response = await call_llm(
            messages,
            endpoint,
            stream=True,
            on_token=on_token,
            timeout_seconds=timeout_seconds,
        )
        return response.text

    return _seam


async def node_draft_prose(
    state: dict[str, Any],
    *,
    call_seam: CallSeam | None = None,
    publisher: Publisher | None = None,
) -> dict[str, Any]:
    """Draft prose for the active beat and accumulate the streamed text.

    ``call_seam`` and ``publisher`` are injectable seams: ``call_seam=None`` builds the
    real streamed ``call_llm`` call against ``config.endpoints.drafter``;
    ``publisher=None`` skips stream publishing entirely.
    """
    pointer = state.get("fsm_pointer")
    package = state.get("active_context_package")
    if not package:
        # Context assembly runs before drafting in the wired graph (Build 14); with no
        # package there is nothing to render, so nothing runs.
        return state

    config = _resolve_config(state)
    db_path = state.get("sqlite_db_path", runtime.SQLITE_DB_PATH)
    endpoint = config.endpoints.drafter
    timeout_seconds = config.runtime.inference_timeout_seconds

    snapshot_id = state.get("planning_snapshot_id")
    beat_id = getattr(pointer, "beat_id", "") if pointer is not None else ""
    beat_plan = _active_beat_plan(db_path, snapshot_id, beat_id)

    ctx = {
        "relational_facts": package.get("relational", {}),
        "epistemic_beliefs": package.get("coreference_candidates", []),
        "summary_context": package.get("summaries", {}),
        "flavour_passages": package.get("flavour", []),
        "temporal_context": package.get("temporal", {}),
        "macro_constraints": package.get("macro_constraints") or {},
        "beat_immediate_objective": beat_plan.get("immediate_objective", ""),
        "beat_physical_constraints": beat_plan.get("physical_constraints") or [],
        "active_continuity": {
            "entry_condition": beat_plan.get("entry_condition", ""),
            "exit_condition": beat_plan.get("exit_condition", ""),
            "continuity_facts": beat_plan.get("asserted_facts", []),
        },
        "pad_behavioral_constraint": beat_plan.get("behavioral_constraint", ""),
        # The beat's chained entry_condition already resolves to the scene's entry_state
        # for a scene's first beat (node_plan_beat's chaining convention) — reused here
        # rather than re-deriving it from a second store read.
        "prior_prose_tail": _prior_prose_tail(package),
        "scene_entry_state": beat_plan.get("entry_condition", ""),
        "target_word_count": _target_word_count(package),
    }

    rendered = PromptLoader().render(_NODE_NAME, ctx)
    messages = [{"role": "user", "content": rendered}]

    seam = call_seam if call_seam is not None else _default_call_seam(endpoint)
    publish_failed = False

    async def _on_token(chunk: str) -> None:
        nonlocal publish_failed
        state["streaming_buffer"] = state.get("streaming_buffer", "") + chunk
        if publisher is not None and not publish_failed:
            try:
                await _maybe_await(publisher(chunk))
            except Exception:  # noqa: BLE001 - a publisher failure must not break generation
                publish_failed = True

    started = perf_counter()
    try:
        raw_text = await seam(messages, on_token=_on_token, timeout_seconds=timeout_seconds)
    except Exception as exc:
        # Failure path: current_draft_text is untouched (never assigned above), so no
        # half-written draft becomes the working text. No retries — that policy belongs
        # to M07/M09 routing, not the drafter.
        duration_ms = (perf_counter() - started) * 1000
        logger = get_logger(_NODE_NAME)
        log_node_event(
            logger,
            pointer.model_dump() if hasattr(pointer, "model_dump") else (pointer or {}),
            duration_ms,
            "failure",
            error=str(exc),
        )
        raise

    # Anti-slop is a black box: detect then resolve, called unconditionally even though
    # M11's contract is an honest no-op today — the node must not change when its
    # internals are replaced.
    findings = detect_slop(raw_text)
    resolved_text = resolve_slop(raw_text, findings)
    state["current_draft_text"] = resolved_text
    # Draft complete: no stream is in flight, so the buffer clears (current_draft_text
    # is the single source of truth for the finished text).
    state["streaming_buffer"] = ""
    return state
