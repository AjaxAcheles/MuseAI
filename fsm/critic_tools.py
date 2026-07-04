"""Module: M07 (Quality Gauntlet)

Read-only, runtime-fenced tool registry the adversarial critics use to ground their
verdicts in committed Shared-Truth (Module_Ability_Specification.md §7's "agentic
tool-use verification") before issuing a `FailureObject`. Mirrors the shape of the
`07.03` planner tool registry (typed dispatch, bounded/truncated/secret-redacted trace
records, `_safe_store_read` degradation) but is a **separate, unrelated registry**:

  - **Read-only.** Every tool in `DISPATCH` wraps a `memory/sqlite_db.py` read helper or
    a still-stubbed store's read seam. There is no write-shaped tool at all — no
    insert/update/delete surface, no shell, no filesystem, no network — so there is
    nothing to widen into a write path by construction, not by convention alone.
  - **Committed truth only.** Beats are exposed only once `committed_at` is set;
    `current_draft_text` and any provisional/proposal state are never reachable through
    a tool — the draft under audit reaches the critic as prompt content, never a tool
    result.
  - **Runtime-scoped fencing** (Module_Ability_Specification.md §7): a `CriticToolRegistry`
    is constructed once per investigation with a `fence` — the active narrative identity
    (`project_id`, `planning_snapshot_id`, and the active `arc_id`/`chapter_id`/
    `scene_id`/`beat_id`) captured by the critic node from state, never supplied or
    overridable by the decider. A tool call whose explicit scope argument contradicts
    the bound fence is refused (a traced result, never an exception) before any store is
    touched — a critic cannot ground a verdict in an alternate timeline, another branch,
    or an out-of-scope entity. (This codebase has no separate "branch" identity table
    yet; `project_id` + the `fsm_pointer` ancestry is the whole of "narrative identity"
    available to fence against today.) Omitted scope arguments default transparently to
    the fence's own value, so the common case needs no argument at all.
  - **Planning constraints are read from committed plan data, not recompiled.** The
    active chapter's constraints come from the chapter `PlanningNode.purpose` JSON that
    `node_plan_chapter` already validated and persisted — never from re-running
    `compile_planning_constraints`, which can itself write a `needs_clarification`
    annotation update on a detected hard-vs-hard conflict. That write, however rare or
    idempotent, is exactly the write surface this registry must not expose.

`run_critic_investigation` (Prompt 2) is the bounded decider-drives-tools loop the
critic node runs before issuing a verdict: hard-capped at `MAX_AGENT_ITERATIONS`,
de-duplicating identical tool requests within one investigation, and always returning a
verdict-ready evidence bundle — even when the cap fires on a decider that never
finalizes.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Awaitable, Callable
from uuid import uuid4

from llm.call_llm import _redact_secrets
from memory import sqlite_db
from memory.graphiti_client import GraphitiClient

# --- fence -------------------------------------------------------------------

# Scope dimensions a tool call may be checked/defaulted against. Not every tool uses
# every key; a key absent from `fence` (None) is simply never enforced.
_SCOPE_KEYS: tuple[str, ...] = (
    "project_id",
    "planning_snapshot_id",
    "arc_id",
    "chapter_id",
    "scene_id",
    "beat_id",
)


def _fence_violation(args: dict[str, Any], fence: dict[str, Any]) -> str | None:
    """None if `args` is within `fence`; else a human-readable refusal reason.

    Only scope keys the caller explicitly supplied are checked — an explicit value that
    disagrees with the bound fence is a violation; an omitted key is not (it is filled
    in from the fence by `_scoped` instead).
    """
    for key in _SCOPE_KEYS:
        if key in args and fence.get(key) is not None and args[key] != fence[key]:
            return (
                f"requested {key}={args[key]!r} is outside the active narrative "
                f"identity ({key}={fence[key]!r})"
            )
    return None


def _scoped(args: dict[str, Any], fence: dict[str, Any], key: str) -> Any:
    """The caller's explicit value for `key` if given (already fence-checked), else the
    fence's own value — the "injected behind the tool boundary" default."""
    return args[key] if key in args else fence.get(key)


# --- tool result shape (mirrors fsm/planning_tools.py's convention) ----------


def _tool_result(
    tool: str, data: Any, *, available: bool = True, reason: str = "ok"
) -> dict[str, Any]:
    """Uniform serializable tool result: data plus a not-built-aware availability flag."""
    return {"tool": tool, "available": available, "data": data, "reason": reason}


def _safe_store_read(
    tool: str, read_callable: Callable[[], Any], *, empty: Any
) -> dict[str, Any]:
    """Run a store query, degrading only for documented not-yet-built stores.

    A `NotImplementedError` from a deferred store stub becomes an empty result marked
    `available=False`; any other exception propagates (a real bug must not be masked as
    "unavailable").
    """
    try:
        data = read_callable()
    except NotImplementedError:
        return _tool_result(
            tool, empty, available=False, reason=f"{tool} store not yet implemented"
        )
    return _tool_result(tool, data if data is not None else empty)


def _parse_purpose(node: dict | None) -> dict:
    """Parse a PlanningNode's `purpose` JSON into a dict (`{}` on absent/invalid)."""
    if not node:
        return {}
    try:
        parsed = json.loads(node.get("purpose") or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


# --- read tools (route to memory/sqlite_db.py read helpers) ------------------


def _read_open_threads(db_path: Any, args: dict, fence: dict) -> dict[str, Any]:
    del args, fence  # global, unscoped canonical truth — no owning column to scope by
    return _tool_result("read_open_threads", sqlite_db.get_open_threads(db_path))


def _read_committed_scenes(db_path: Any, args: dict, fence: dict) -> dict[str, Any]:
    chapter_id = _scoped(args, fence, "chapter_id")
    scenes = sqlite_db.get_scenes_for_chapter_ordered(db_path, chapter_id) if chapter_id else []
    return _tool_result("read_committed_scenes", scenes)


def _read_committed_beats(db_path: Any, args: dict, fence: dict) -> dict[str, Any]:
    scene_id = _scoped(args, fence, "scene_id")
    beats = sqlite_db.get_beats_for_scene_ordered(db_path, scene_id) if scene_id else []
    # Committed truth only: a structural beat row created by the planner (prose/
    # committed_at still NULL — including the very beat under audit) is never surfaced
    # as if it were committed.
    committed = [b for b in beats if b.get("committed_at") is not None]
    return _tool_result("read_committed_beats", committed)


def _read_latest_pad(db_path: Any, args: dict, fence: dict) -> dict[str, Any]:
    character_id = _scoped(args, fence, "character_id") if "character_id" in args else None
    if character_id:
        data = sqlite_db.get_latest_pad_for_character(db_path, character_id)
    else:
        scene_id = _scoped(args, fence, "scene_id")
        data = sqlite_db.get_latest_pad_for_scene(db_path, scene_id) if scene_id else []
    return _tool_result("read_latest_pad", data)


def _read_chapter_constraints(db_path: Any, args: dict, fence: dict) -> dict[str, Any]:
    snapshot_id = _scoped(args, fence, "planning_snapshot_id")
    chapter_id = _scoped(args, fence, "chapter_id")
    if not snapshot_id or not chapter_id:
        return _tool_result("read_chapter_constraints", {})
    node_id = f"{snapshot_id}:chapter:{chapter_id}"
    node = next(
        (
            n
            for n in sqlite_db.get_planning_nodes(db_path, snapshot_id, level="chapter")
            if n["node_id"] == node_id
        ),
        None,
    )
    plan = _parse_purpose(node)
    # Committed plan data only — never a live recompile. `compile_planning_constraints`
    # can itself write a `needs_clarification` annotation update on a hard-vs-hard
    # conflict; that write, however rare, is a write this read-only registry must never
    # reach for.
    constraints = {
        "dramatic_function": plan.get("dramatic_function", ""),
        "scene_planning_constraints": plan.get("scene_planning_constraints") or [],
        "thread_obligations": plan.get("thread_obligations") or [],
    }
    return _tool_result("read_chapter_constraints", constraints)


def _read_continuity_facts(db_path: Any, args: dict, fence: dict) -> dict[str, Any]:
    del db_path  # Graphiti is its own store, not the relational hub
    query_args = {
        "scene_id": _scoped(args, fence, "scene_id"),
        "chapter_id": _scoped(args, fence, "chapter_id"),
    }
    return _safe_store_read(
        "read_continuity_facts", lambda: GraphitiClient().query(**query_args), empty=[]
    )


# Tool name -> callable(db_path, args, fence) -> tool result. No write-shaped tool is
# registered here, ever — that is the whole of the "no write surface" guarantee.
DISPATCH: dict[str, Callable[[Any, dict, dict], dict[str, Any]]] = {
    "read_open_threads": _read_open_threads,
    "read_committed_scenes": _read_committed_scenes,
    "read_committed_beats": _read_committed_beats,
    "read_latest_pad": _read_latest_pad,
    "read_chapter_constraints": _read_chapter_constraints,
    "read_continuity_facts": _read_continuity_facts,
}


# --- traced, fenced registry ---------------------------------------------------

# Provisional fallback cap for the bounded trace/result summary — same convention as
# fsm/planning_tools.py's _DEFAULT_TRACE_SUMMARY_MAX_CHARS. A length guard, not a
# tunable narrative threshold, so no config key is added for it.
_DEFAULT_TRACE_SUMMARY_MAX_CHARS = 1000


def _truncate(text: str, max_chars: int | None) -> str:
    """Bound `text` to `max_chars`, appending an explicit elision marker if cut."""
    if max_chars is None or len(text) <= max_chars:
        return text
    dropped = len(text) - max_chars
    return f"{text[:max_chars]}…[+{dropped} chars truncated]"


class CriticToolRegistry:
    """Read-only, fenced tool registry for the adversarial critics' investigation loop.

    `fence` is bound once at construction — the active narrative identity, injected by
    the critic node (never the decider). `call(tool_name, args)` is the single
    execution surface: unknown tools are rejected, fence-violating calls are refused,
    and every attempt (executed, degraded, rejected, or refused) is appended to
    `self.trace` as a bounded, truncated, secret-redacted in-memory record — never
    persisted to any store (critics get no SQLite trace table; that is the planner's).
    """

    def __init__(
        self,
        db_path: Any,
        fence: dict[str, Any] | None = None,
        *,
        trace_summary_max_chars: int | None = _DEFAULT_TRACE_SUMMARY_MAX_CHARS,
    ) -> None:
        self.db_path = db_path
        self.fence = dict(fence or {})
        self.trace_summary_max_chars = trace_summary_max_chars
        self.trace: list[dict[str, Any]] = []

    def _summary(self, obj: Any) -> str:
        """Redact secrets, JSON-encode, then length-bound for trace/evidence display."""
        encoded = json.dumps(_redact_secrets(obj), separators=(",", ":"), default=str)
        return _truncate(encoded, self.trace_summary_max_chars)

    def _record_trace(
        self, *, tool_name: str, args: dict, outcome: str, result: dict | None, error: str | None
    ) -> dict[str, Any]:
        """Append one bounded, in-memory trace record; return it (carries `trace_id`)."""
        row = {
            "trace_id": uuid4().hex,
            "tool_name": tool_name,
            "args_summary": self._summary(args),
            "outcome": outcome,
            "result_summary": (
                self._summary(result["data"])
                if result is not None and result.get("data") is not None
                else None
            ),
            "error": _truncate(error, self.trace_summary_max_chars) if error else None,
        }
        self.trace.append(row)
        return row

    def call(self, tool_name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate, fence-check, execute, and trace one critic tool call.

        Returns a result dict carrying `outcome` (`executed` / `degraded` / `rejected` /
        `refused`), `available`, `data`, `reason`, and `trace_id`. Unknown tools are
        rejected; fence-violating calls are refused — both without executing, both
        still traced, both returned as a normal result rather than an exception, so an
        investigation loop is never killed by a bad request.
        """
        args = dict(args or {})

        if tool_name not in DISPATCH:
            error = "rejected: unknown tool"
            row = self._record_trace(
                tool_name=tool_name, args=args, outcome="rejected", result=None, error=error
            )
            return {
                "tool": tool_name, "outcome": "rejected", "available": False,
                "data": None, "reason": error, "trace_id": row["trace_id"],
            }

        violation = _fence_violation(args, self.fence)
        if violation is not None:
            error = f"refused: {violation}"
            row = self._record_trace(
                tool_name=tool_name, args=args, outcome="refused", result=None, error=error
            )
            return {
                "tool": tool_name, "outcome": "refused", "available": False,
                "data": None, "reason": error, "trace_id": row["trace_id"],
            }

        result = DISPATCH[tool_name](self.db_path, args, self.fence)
        available = bool(result.get("available", True))
        outcome = "executed" if available else "degraded"
        row = self._record_trace(
            tool_name=tool_name, args=args, outcome=outcome, result=result,
            error=None if available else f"unavailable: {result.get('reason')}",
        )
        return {**result, "outcome": outcome, "trace_id": row["trace_id"]}


# --- bounded investigation loop ------------------------------------------------

# Fixed structural ceiling on a critic's tool-use reasoning loop (Configuration_Reference
# .md §4, "Agentic critic loop ceiling") — a named module constant, not a config key, so
# it can never be relaxed or skipped by configuration.
MAX_AGENT_ITERATIONS = 5

Decider = Callable[[dict, list, list], "dict[str, Any] | Awaitable[dict[str, Any]]"]


def _call_cache_key(tool_name: str, args: dict[str, Any]) -> str:
    return json.dumps({"tool": tool_name, "args": args}, sort_keys=True, default=str)


async def run_critic_investigation(
    decider: Decider,
    registry: CriticToolRegistry,
    draft_context: dict[str, Any],
) -> dict[str, Any]:
    """Bounded decider-drives-tools investigation loop.

    Each iteration the injected `decider(draft_context, evidence, trace)` either
    requests one tool call (`{"action": "call_tool", "tool_name": ..., "args": {...}}`)
    or declares readiness to finalize (`{"action": "finalize"}`); this helper never
    calls an LLM itself — the decider is the only model contact, injected by the
    caller. Hard-capped at `MAX_AGENT_ITERATIONS`: on cap exhaustion the evidence
    gathered so far is returned with `cap_exhausted=True` rather than raising, so the
    critic can still issue a verdict from partial evidence. A repeated identical
    `(tool_name, args)` request within one investigation is served from the first
    call's cached result — `registry.call` (and any store it wraps) is never invoked
    twice for the same request.
    """
    evidence: list[dict[str, Any]] = []
    call_cache: dict[str, dict[str, Any]] = {}

    for _ in range(MAX_AGENT_ITERATIONS):
        action = decider(draft_context, evidence, registry.trace)
        if inspect.isawaitable(action):
            action = await action

        if action.get("action") == "finalize":
            return {"evidence": evidence, "trace": list(registry.trace), "cap_exhausted": False}

        tool_name = action.get("tool_name")
        args = action.get("args") or {}
        cache_key = _call_cache_key(tool_name, args)
        if cache_key in call_cache:
            cached_result = call_cache[cache_key]
            evidence.append(cached_result)
            # Still visible to 10.T2/the frontend rendering the investigation, even
            # though the underlying tool is not dispatched again.
            registry.trace.append(
                {
                    "trace_id": uuid4().hex,
                    "tool_name": tool_name,
                    "args_summary": registry._summary(args),
                    "outcome": "deduped",
                    "result_summary": None,
                    "error": None,
                    "duplicate_of_trace_id": cached_result.get("trace_id"),
                }
            )
            continue

        result = registry.call(tool_name, args)
        call_cache[cache_key] = result
        evidence.append(result)

    return {"evidence": evidence, "trace": list(registry.trace), "cap_exhausted": True}
