"""Module: M05 (Hierarchical Planning Cascade)
Per-level planner tool permission matrix (Data_Structures.md §1.5) and the dispatch
table mapping tool names to the real M02 read/write helpers, with graceful degradation
for the still-stubbed temporal/RAPTOR/flavour stores.

Tool scope narrows as the planner level approaches prose: a global planner cannot read
individual beats; a beat planner cannot rewrite the global plan. Planner tools may write
canonical truth only through the validated plan-time write helpers, and this registry
provides **no** shell, arbitrary filesystem, network, or external-service tools — its
only side effects are the M02 store reads/writes it wraps.

Scope: this module owns the permission matrix, `permitted()`, and the dispatch table.
Per-loop cap counting belongs to the deliberation loop (07.04) and is not enforced here;
PlannerToolCallTrace logging is a later increment (Prompt 2). Continuity-fact (Graphiti),
RAPTOR-summary, and Chroma-flavour query tools route through a not-yet-built seam that
returns an empty result marked `available=False` instead of crashing, mirroring
`fsm/nodes/node_assemble_context._safe_store_read`. A few §1.5 categories have no M02
helper yet (project-metadata / world-rules reads; the global-plan writer) and likewise
degrade honestly rather than fake a backing store.
"""

import json
from typing import Any, Callable
from uuid import uuid4

from llm.call_llm import _redact_secrets
from memory import sqlite_db
from memory.chroma_client import ChromaClient
from memory.graphiti_client import GraphitiClient
from memory.raptor import RaptorStore

# Planner levels, coarsest → finest (Data_Structures.md §1.5 column order).
PLANNER_LEVELS: tuple[str, ...] = ("global", "arc", "chapter", "scene", "beat")

# Access qualifiers stored in the matrix. "limited" = read-only summary access without
# full detail (§1.5); both "full" and "limited" mean the tool is permitted.
FULL = "full"
LIMITED = "limited"

# The per-level permission matrix from Data_Structures.md §1.5, kept as inspectable data
# (level → {tool_name: access}). A tool absent from a level's dict is denied at that
# level. Dispatch logic reads this table; it is never hardcoded inside the callables.
PERMISSION_MATRIX: dict[str, dict[str, str]] = {
    "global": {
        "read_project_metadata": FULL,
        "read_world_rules": FULL,
        "read_arcs": FULL,
        "read_open_threads": FULL,
        "read_character_state": LIMITED,
        "query_continuity_facts": LIMITED,
        "query_raptor_summaries": FULL,
        "write_global_plan": FULL,
    },
    "arc": {
        "read_project_metadata": FULL,
        "read_world_rules": FULL,
        "read_arcs": FULL,
        "read_chapters": FULL,
        "read_open_threads": FULL,
        "read_character_state": FULL,
        "query_continuity_facts": FULL,
        "query_raptor_summaries": FULL,
        "query_chroma_flavour": LIMITED,
        "write_arc_plan": FULL,
    },
    "chapter": {
        "read_project_metadata": FULL,
        "read_world_rules": FULL,
        "read_arcs": FULL,
        "read_chapters": FULL,
        "read_scenes": FULL,
        "read_open_threads": FULL,
        "read_character_state": FULL,
        "query_continuity_facts": FULL,
        "query_raptor_summaries": FULL,
        "query_chroma_flavour": LIMITED,
        "write_chapter_plan": FULL,
    },
    "scene": {
        "read_project_metadata": FULL,
        "read_world_rules": FULL,
        "read_arcs": FULL,
        "read_chapters": FULL,
        "read_scenes": FULL,
        "read_beats": FULL,
        "read_open_threads": FULL,
        "read_character_state": FULL,
        "query_continuity_facts": FULL,
        "query_raptor_summaries": FULL,
        "query_chroma_flavour": FULL,
        "write_scene_plan": FULL,
    },
    "beat": {
        "read_project_metadata": FULL,
        "read_world_rules": FULL,
        "read_arcs": FULL,
        "read_chapters": FULL,
        "read_scenes": FULL,
        "read_beats": FULL,
        "read_open_threads": FULL,
        "read_character_state": FULL,
        "query_continuity_facts": FULL,
        "query_raptor_summaries": LIMITED,
        "query_chroma_flavour": FULL,
        "write_beat_plan": FULL,
    },
}


def permitted(level: str, tool_name: str) -> bool:
    """True iff `tool_name` is permitted (full or limited) for `level` per §1.5."""

    return tool_name in PERMISSION_MATRIX.get(level, {})


def access_for(level: str, tool_name: str) -> str | None:
    """Return the access qualifier ('full'/'limited') for a permitted tool, else None."""

    return PERMISSION_MATRIX.get(level, {}).get(tool_name)


# --- tool result shape ------------------------------------------------------


def _tool_result(
    tool: str, data: Any, *, available: bool = True, reason: str = "ok"
) -> dict[str, Any]:
    """Uniform serializable tool result: data plus a not-built-aware availability flag."""

    return {"tool": tool, "available": available, "data": data, "reason": reason}


def _safe_store_read(
    tool: str, read_callable: Callable[[], Any], *, empty: Any
) -> dict[str, Any]:
    """Run a store query, degrading only for documented not-yet-built stores.

    Mirrors `node_assemble_context._safe_store_read`: a `NotImplementedError` from a
    deferred store stub becomes an empty result marked `available=False`; any other
    exception propagates (a real bug must not be masked as "unavailable").
    """

    try:
        data = read_callable()
    except NotImplementedError:
        return _tool_result(
            tool, empty, available=False, reason=f"{tool} store not yet implemented"
        )
    return _tool_result(tool, data if data is not None else empty)


# --- read tools (route to memory/sqlite_db.py read helpers) -----------------


def _read_project_metadata(db_path: Any, args: dict) -> dict[str, Any]:
    # No project-metadata table/read helper exists in M02 yet; degrade honestly.
    return _tool_result(
        "read_project_metadata", {}, available=False,
        reason="no project-metadata read helper built yet",
    )


def _read_world_rules(db_path: Any, args: dict) -> dict[str, Any]:
    # No world-rules table/read helper exists in M02 yet; degrade honestly.
    return _tool_result(
        "read_world_rules", [], available=False,
        reason="no world-rules read helper built yet",
    )


def _read_arcs(db_path: Any, args: dict) -> dict[str, Any]:
    return _tool_result("read_arcs", sqlite_db.get_arc(db_path, args.get("arc_id")))


def _read_chapters(db_path: Any, args: dict) -> dict[str, Any]:
    return _tool_result(
        "read_chapters", sqlite_db.get_chapters_for_arc(db_path, args.get("arc_id"))
    )


def _read_scenes(db_path: Any, args: dict) -> dict[str, Any]:
    return _tool_result(
        "read_scenes",
        sqlite_db.get_scenes_for_chapter_ordered(db_path, args.get("chapter_id")),
    )


def _read_beats(db_path: Any, args: dict) -> dict[str, Any]:
    return _tool_result(
        "read_beats", sqlite_db.get_beats_for_scene_ordered(db_path, args.get("scene_id"))
    )


def _read_open_threads(db_path: Any, args: dict) -> dict[str, Any]:
    return _tool_result("read_open_threads", sqlite_db.get_open_threads(db_path))


def _read_character_state(db_path: Any, args: dict) -> dict[str, Any]:
    # Character state is the persisted PAD record: per-character when a character_id is
    # given, else the latest per-character PAD across the active scene.
    if args.get("character_id"):
        data = sqlite_db.get_latest_pad_for_character(db_path, args["character_id"])
    else:
        data = sqlite_db.get_latest_pad_for_scene(db_path, args.get("scene_id"))
    return _tool_result("read_character_state", data)


# --- query tools (graceful degradation seam over deferred stores) -----------


def _query_continuity_facts(db_path: Any, args: dict) -> dict[str, Any]:
    return _safe_store_read(
        "query_continuity_facts", lambda: GraphitiClient().query(**args), empty=[]
    )


def _query_raptor_summaries(db_path: Any, args: dict) -> dict[str, Any]:
    return _safe_store_read(
        "query_raptor_summaries", lambda: RaptorStore().summarize(**args), empty=[]
    )


def _query_chroma_flavour(db_path: Any, args: dict) -> dict[str, Any]:
    return _safe_store_read(
        "query_chroma_flavour", lambda: ChromaClient().query(**args), empty=[]
    )


# --- write tools (route to the 07.00 plan-time outline writers) -------------


def _write_global_plan(db_path: Any, args: dict) -> dict[str, Any]:
    # No dedicated global-plan writer exists in M02 yet (07.00 added arc/chapter/scene/
    # beat writers only); degrade honestly rather than misroute to an arc writer.
    return _tool_result(
        "write_global_plan", None, available=False,
        reason="no global-plan writer built yet",
    )


def _write_arc_plan(db_path: Any, args: dict) -> dict[str, Any]:
    return _tool_result("write_arc_plan", sqlite_db.upsert_arc_plan(db_path, **args))


def _write_chapter_plan(db_path: Any, args: dict) -> dict[str, Any]:
    return _tool_result("write_chapter_plan", sqlite_db.upsert_chapter_plan(db_path, **args))


def _write_scene_plan(db_path: Any, args: dict) -> dict[str, Any]:
    return _tool_result("write_scene_plan", sqlite_db.upsert_scene_plan(db_path, **args))


def _write_beat_plan(db_path: Any, args: dict) -> dict[str, Any]:
    return _tool_result("write_beat_plan", sqlite_db.upsert_beat_plan(db_path, **args))


# Tool name → callable(db_path, args) -> tool result. Every name here also appears in
# PERMISSION_MATRIX for at least one level; permission is checked separately by callers.
DISPATCH: dict[str, Callable[[Any, dict], dict[str, Any]]] = {
    "read_project_metadata": _read_project_metadata,
    "read_world_rules": _read_world_rules,
    "read_arcs": _read_arcs,
    "read_chapters": _read_chapters,
    "read_scenes": _read_scenes,
    "read_beats": _read_beats,
    "read_open_threads": _read_open_threads,
    "read_character_state": _read_character_state,
    "query_continuity_facts": _query_continuity_facts,
    "query_raptor_summaries": _query_raptor_summaries,
    "query_chroma_flavour": _query_chroma_flavour,
    "write_global_plan": _write_global_plan,
    "write_arc_plan": _write_arc_plan,
    "write_chapter_plan": _write_chapter_plan,
    "write_scene_plan": _write_scene_plan,
    "write_beat_plan": _write_beat_plan,
}


def call_tool(
    level: str, tool_name: str, args: dict | None = None, *, db_path: Any
) -> dict[str, Any]:
    """Permission-checked dispatch of a single planner tool (no caps, no tracing here).

    Fails closed: a tool not permitted for `level` (or with no registered
    implementation) raises `ValueError` rather than executing. Returns the tool result
    dict from DISPATCH. Cap counting (07.04) and PlannerToolCallTrace logging (Prompt 2)
    wrap this elsewhere.
    """

    if not permitted(level, tool_name):
        raise ValueError(f"tool {tool_name!r} is not permitted for planner level {level!r}")
    fn = DISPATCH.get(tool_name)
    if fn is None:
        raise ValueError(f"no dispatch implementation for tool {tool_name!r}")
    return fn(db_path, args or {})


# ---------------------------------------------------------------------------
# Traced, permission-checked registry (Data_Structures.md §1.5 + §2.7).
#
# `PlanningToolRegistry.call(...)` is the surface the deliberation loop uses. It
# validates against PERMISSION_MATRIX, executes the mapped tool (or surfaces its
# graceful-degradation marker), and writes a PlannerToolCallTrace for *every* attempt —
# executed, rejected, or degraded. Trace arg/result summaries are length-bounded and
# secret-redacted so a trace can never persist unbounded text or leak a credential.
# ---------------------------------------------------------------------------

# Provisional fallback cap for the bounded trace summary. There is no dedicated config
# key for this yet; the cap is an injectable constructor parameter so the loop (07.04)
# can pass `config.planning.<key>` once such a key is added. Until then this labelled
# default is used — it is a length guard, not a tunable narrative threshold.
_DEFAULT_TRACE_SUMMARY_MAX_CHARS = 1000


def _truncate(text: str, max_chars: int | None) -> str:
    """Bound `text` to `max_chars`, appending an explicit elision marker if cut."""

    if max_chars is None or len(text) <= max_chars:
        return text
    dropped = len(text) - max_chars
    return f"{text[:max_chars]}…[+{dropped} chars truncated]"


class PlanningToolRegistry:
    """Permission-checked, traced planner tool registry over the §1.5 matrix.

    `call(level, tool_name, args, snapshot_id, loop_index)` is the single execution
    surface: it rejects unknown/disallowed tools (never executing them), runs permitted
    tools, and writes a PlannerToolCallTrace for executed, rejected, and degraded calls
    alike. Its only side effects are the wrapped M02 read/write and the trace insert —
    no shell, filesystem, or network. Per-loop cap enforcement is **not** done here; the
    deliberation loop (07.04) counts calls and caps using the returned outcome + traces.
    """

    def __init__(
        self,
        db_path: Any,
        *,
        trace_summary_max_chars: int | None = _DEFAULT_TRACE_SUMMARY_MAX_CHARS,
    ) -> None:
        self.db_path = db_path
        self.trace_summary_max_chars = trace_summary_max_chars

    def permitted(self, level: str, tool_name: str) -> bool:
        """Matrix-only permission check (delegates to module `permitted`)."""

        return permitted(level, tool_name)

    def access_for(self, level: str, tool_name: str) -> str | None:
        return access_for(level, tool_name)

    def _summary(self, obj: Any) -> str:
        """Redact secrets, JSON-encode, then length-bound for trace persistence."""

        encoded = json.dumps(_redact_secrets(obj), separators=(",", ":"), default=str)
        return _truncate(encoded, self.trace_summary_max_chars)

    def _record_trace(
        self,
        *,
        level: str,
        tool_name: str,
        args: dict,
        snapshot_id: str | None,
        loop_index: int,
        success: bool,
        result: Any,
        error: str | None,
    ) -> str:
        """Persist one PlannerToolCallTrace row, returning its trace_id."""

        trace_id = uuid4().hex
        result_summary = None
        if result is not None and result.get("data") is not None:
            result_summary = self._summary(result["data"])
        insert_kwargs: dict[str, Any] = {
            "trace_id": trace_id,
            "planner_level": level,
            "loop_index": loop_index,
            "tool_name": tool_name,
            "success": success,
            "snapshot_id": snapshot_id,
            "tool_args_json": self._summary(args),
            "result_summary": result_summary,
            "error": _truncate(error, self.trace_summary_max_chars) if error else None,
        }
        sqlite_db.insert_planner_tool_call_trace(self.db_path, **insert_kwargs)
        return trace_id

    def call(
        self,
        level: str,
        tool_name: str,
        args: dict | None,
        snapshot_id: str | None,
        loop_index: int,
    ) -> dict[str, Any]:
        """Validate, execute, and trace one planner tool call.

        Returns a result dict carrying `outcome` (`executed` / `degraded` / `rejected`),
        `available`, `data`, `reason`, and the `trace_id`. Unknown or disallowed tools
        are rejected (never executed) and still traced.
        """

        args = args or {}

        # Unknown tool: reject without executing, but trace the attempt.
        if tool_name not in DISPATCH:
            error = "rejected: unknown tool"
            trace_id = self._record_trace(
                level=level, tool_name=tool_name, args=args, snapshot_id=snapshot_id,
                loop_index=loop_index, success=False, result=None, error=error,
            )
            return {
                "tool": tool_name, "outcome": "rejected", "available": False,
                "data": None, "reason": error, "trace_id": trace_id,
            }

        # Disallowed for this level: reject without executing, but trace the attempt.
        if not permitted(level, tool_name):
            error = f"rejected: tool not permitted for level {level!r}"
            trace_id = self._record_trace(
                level=level, tool_name=tool_name, args=args, snapshot_id=snapshot_id,
                loop_index=loop_index, success=False, result=None, error=error,
            )
            return {
                "tool": tool_name, "outcome": "rejected", "available": False,
                "data": None, "reason": error, "trace_id": trace_id,
            }

        # Permitted: execute the mapped tool (read/write or a degradation marker).
        try:
            result = DISPATCH[tool_name](self.db_path, args)
        except Exception as exc:  # real execution fault — trace then re-raise, never mask.
            self._record_trace(
                level=level, tool_name=tool_name, args=args, snapshot_id=snapshot_id,
                loop_index=loop_index, success=False, result=None,
                error=f"execution_error: {exc!r}",
            )
            raise

        available = bool(result.get("available", True))
        outcome = "executed" if available else "degraded"
        trace_id = self._record_trace(
            level=level, tool_name=tool_name, args=args, snapshot_id=snapshot_id,
            loop_index=loop_index, success=available, result=result,
            error=None if available else f"unavailable: {result.get('reason')}",
        )
        return {**result, "outcome": outcome, "trace_id": trace_id}
