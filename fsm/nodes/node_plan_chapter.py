"""Module: M05 (Hierarchical Planning Cascade)

Level-3 (chapter) planner node. Expands one chapter stub from the approved arc plan into
chapter obligations + scene-planning constraints by running the shared bounded
deliberation loop and persisting only a validator-passed chapter plan. It never creates
final scene plans or Scene rows — the runtime Scene Planner (``node_plan_scene``)
consumes the constraints written here. A THREAD_PARADOX handed up from recovery arrives
as an ordinary hard constraint in the compiled annotation package; the plan is
structured around it with no dedicated code path.

Flow per invocation:
  1. resolve the run's ``PlanningSnapshot`` and the active chapter target — the first
     unplanned chapter ``PlanningNode`` (preferring ``fsm_pointer.chapter_id`` when it
     names one), walking the snapshot's arcs in order;
  2. ``compile_planning_constraints`` against the chapter node — if two hard annotations
     contradict, set the clarification block and return WITHOUT running the loop;
  3. assemble the chapter ``base_context`` (approved arc + global plans, the active
     chapter stub, genre, recent chapter summaries, open-thread priority queue),
     degrading gracefully for not-yet-built stores;
  4. build an M04-backed decider via ``make_planner_decider`` and run
     ``run_planner_loop`` with a deterministic minimal-valid chapter baseline as the
     loop's floor;
  5. map the ``LoopOutcome`` through ``persist_loop_outcome`` — whose ``persist_plan_fn``
     writes the ``Chapters`` row + the chapter obligations / scene-planning constraints
     into the chapter ``PlanningNode`` proposal surface (never Scene rows) — then, on
     success, advance ``fsm_pointer.chapter_id`` and fold snapshot/revision/block state
     back;
  6. as the LAST macro planning level: when the run's mode (read from the STATE snapshot
     fields ``planning_execution_mode`` / ``approval_mode``, never from config at node
     time) is ``macro_outline_before_draft`` and no unplanned chapter remains, set
     ``macro_outline_ready`` — and, when ``approval_mode == "macro_outline"``, set the
     approval-gate state (``awaiting_planning_approval`` / ``planning_block_reason`` /
     snapshot status). The node only SETS that state and returns — it never blocks or
     waits (Module_Ability_Specification §5: a headless run must never wait
     indefinitely, and 07.0a config validation guarantees ``headless_mode`` cannot
     coexist with ``approval_mode != "off"``). The graph edge that actually holds
     drafting on ``awaiting_planning_approval`` is Build 14. In ``rolling`` mode the
     readiness/pause state is never touched.

The node makes **no** direct ``call_llm``: all model contact is through the injected
decider seam, so it is fully testable by injecting a synthetic decider. It writes no
prose and wires no graph route / approval edge (that is Build 14).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import core.runtime as runtime
from fsm.planning_annotations import compile_planning_constraints
from fsm.planning_loop import run_planner_loop
from fsm.planning_node_support import make_planner_decider, persist_loop_outcome
from fsm.planning_tools import PlanningToolRegistry
from memory import sqlite_db

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
_LEVEL = "chapter"
_NODE_NAME = "node_plan_chapter"
_MACRO_MODE = "macro_outline_before_draft"

# Human-readable, provisional planning-block reason for a loop that exhausted its caps and
# fallback ladder without a valid plan. The actual route to failure recovery is Build 14.
_ESCALATION_BLOCK_REASON = "planning_escalation"

# Documented block reason while the FSM sits at the macro-outline approval gate
# (Data_Structures §1.1) — a safe-boundary interruption, not a failure.
_AWAITING_APPROVAL_BLOCK_REASON = "awaiting_macro_approval"

# The §2.7 snapshot-status vocabulary has no dedicated 'awaiting approval' value: a
# completed, unapproved snapshot is presented for review as 'draft' (the approval
# surface, routes/plan.py, later moves it to 'approved'/'rejected'). The transition is
# idempotent for a snapshot still in 'draft'.
_AWAITING_APPROVAL_SNAPSHOT_STATUS = "draft"


def _resolve_config(state: dict[str, Any]) -> Any:
    """Return the active typed config (mirrors the other planner nodes' resolution).

    Used for loop caps / required-check names / endpoint resolution only — the
    execution/approval modes are read from the STATE snapshot fields, not from config.
    """
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


def _id_suffix(node_id: str, marker: str) -> str | None:
    """Extract the trailing id from a ``{snapshot}:{level}:{id}`` PlanningNode id."""
    if marker in node_id:
        return node_id.rsplit(marker, 1)[1] or None
    return None


def _is_chapter_planned(chapter_node: dict) -> bool:
    """True when the chapter PlanningNode already carries real obligations.

    The arc planner writes chapter stubs whose ``purpose`` obligations are all None;
    both the 07.00 obligations JSON and this node's full-plan JSON carry a truthy
    ``dramatic_function`` once the chapter has actually been planned.
    """
    return bool(_parse_purpose(chapter_node).get("dramatic_function"))


def _recent_chapter_summaries(db_path: Any) -> list[str]:
    """Recent chapter summaries for the template's context slot.

    The design reads the last two chapter summaries from the RAPTOR tree; the RAPTOR
    clustering/summarization pipeline is not built yet and defines no level convention
    to query, so this degrades to an empty list (the template renders its documented
    empty-state branch). Wired to the real store when that module lands.
    """
    del db_path  # no RAPTOR chapter-summary convention exists to read yet
    return []


def _select_chapter_target(
    db_path: Any, snapshot_id: str, pointer: Any
) -> dict[str, Any] | None:
    """Resolve the active chapter target: the first unplanned chapter stub, in order.

    Walks the snapshot's arc PlanningNodes (ordering ASC) and their chapter children.
    If ``fsm_pointer.chapter_id`` names an unplanned chapter, that one is preferred;
    otherwise the first unplanned chapter in cascade order is chosen (this is how one
    node invocation per chapter sweeps the whole macro scope — the self-loop routing is
    Build 14). Returns None when every chapter is planned or no stubs exist yet.
    """
    unplanned: list[dict[str, Any]] = []
    for arc_node in sqlite_db.get_planning_nodes(db_path, snapshot_id, level="arc"):
        arc_plan = _parse_purpose(arc_node)
        arc_id = arc_plan.get("arc_id") or _id_suffix(arc_node["node_id"], ":arc:") or ""
        children = sqlite_db.get_planning_nodes_by_parent(
            db_path, snapshot_id, arc_node["node_id"]
        )
        for chapter_node in children:
            if chapter_node.get("level") != _LEVEL or _is_chapter_planned(chapter_node):
                continue
            chapter_id = (
                _id_suffix(chapter_node["node_id"], ":chapter:")
                or chapter_node.get("title")
                or ""
            )
            stub = next(
                (
                    c.get("stub")
                    for c in arc_plan.get("chapters", [])
                    if isinstance(c, dict) and c.get("chapter_id") == chapter_id and c.get("stub")
                ),
                None,
            ) or (chapter_node.get("title") or chapter_node.get("summary") or "")
            unplanned.append(
                {
                    "chapter_id": chapter_id,
                    "node_id": chapter_node["node_id"],
                    "arc_id": arc_id,
                    "arc_node_id": arc_node["node_id"],
                    "arc_plan": arc_plan,
                    "stub": stub,
                    "ordering": chapter_node.get("ordering", 0),
                }
            )
    if not unplanned:
        return None
    wanted = getattr(pointer, "chapter_id", None) if pointer is not None else None
    if wanted:
        for candidate in unplanned:
            if candidate["chapter_id"] == wanted:
                return candidate
    return unplanned[0]


def _macro_chapter_scope_complete(db_path: Any, snapshot_id: str) -> bool:
    """True iff chapter PlanningNodes exist and none remain unplanned (macro scope done)."""
    chapters = sqlite_db.get_planning_nodes(db_path, snapshot_id, level=_LEVEL)
    return bool(chapters) and all(_is_chapter_planned(n) for n in chapters)


def _build_chapter_baseline(base_context: dict[str, Any]) -> Any:
    """Build the loop's deterministic baseline: a minimal, single-pass valid chapter plan.

    The fallback floor when the model never produces a validating plan. Its shape
    matches the authoritative chapter validators (``schema``/``no_drafting``/
    ``chapter_function``/``pacing``/``annotation_satisfaction``) and the depth contract
    (``obligations`` present, no ``scenes`` key). Hard annotations are satisfied by a
    real mechanism, not a claim: each hard requirement's text is forwarded verbatim as a
    binding scene-planning constraint (so the downstream Scene Planner inherits it), and
    only then is its id recorded ``applied`` in ``annotation_outcomes``.
    """
    active = base_context.get("active_chapter") or {}
    arc_plan = base_context.get("arc_plan") or {}
    open_threads = base_context.get("open_threads") or []

    def _baseline(level: str, target_node: Any, constraints: dict) -> dict:
        chapter_id = active.get("chapter_id") or "chapter_1"
        stub = active.get("stub") or "advance the active arc"
        arc_id = arc_plan.get("arc_id") or "the active arc"
        thread_obligations = [
            {
                "thread_id": (t.get("id") or t.get("thread_id") or f"thread_{k + 1}")
                if isinstance(t, dict)
                else str(t),
                "required_progress": "make concrete progress toward this thread's target resolution",
            }
            for k, t in enumerate(open_threads)
        ]
        scene_constraints = [
            f"the chapter's scenes must together accomplish: {stub}",
            "open consistent with the previous chapter's closing state and close having "
            "delivered this chapter's causal deliverables",
        ]
        annotation_outcomes: dict[str, str] = {}
        hard = (constraints or {}).get("hard_annotations", []) or []
        for ann in hard:
            ann_id = ann.get("annotation_id") if isinstance(ann, dict) else None
            if ann_id:
                scene_constraints.append(
                    f"hard user requirement (must hold): {ann.get('text', '')}"
                )
                annotation_outcomes[ann_id] = "applied"
        return {
            "chapter_id": chapter_id,
            "dramatic_function": f"advance {arc_id}: {stub}",
            "expected_emotional_shift": "engagement steady at the open, tension raised by the close",
            "pacing": "measured build serving the arc's escalation",
            "obligations": {
                "thread_obligations": thread_obligations,
                "causal_prerequisites": [
                    "the preceding chapters' committed outcomes hold as this chapter opens"
                ],
                "causal_deliverables": [f"the chapter's stated purpose is accomplished: {stub}"],
            },
            "annotation_outcomes": annotation_outcomes,
            "scene_planning_constraints": scene_constraints,
        }

    return _baseline


def _chapter_persist_plan_fn(db_path: Any, snapshot_id: str, target: dict[str, Any]) -> Any:
    """Return the level-specific writer: the Chapters row + the chapter PlanningNode.

    Identity is harness-owned: rows are keyed by the TARGET chapter (never by whatever
    ``chapter_id`` the plan claims). The 07.00 ``upsert_chapter_plan`` writes the
    ``Chapters`` structural row and the paired chapter ``PlanningNode`` (obligations
    JSON in ``purpose``) in one atomic transaction; a second idempotent
    ``upsert_planning_node`` then upgrades ``purpose`` to the FULL validated plan JSON —
    a superset of the obligations — matching the global/arc convention (full plan in
    ``purpose``) so the Scene Planner reads one authoritative shape. No Scene rows are
    written here (that is ``node_plan_scene``'s job).
    """
    chapter_id = target["chapter_id"]
    node_id = target["node_id"]

    def _persist(plan: dict) -> None:
        obligations = plan.get("obligations") or {}
        scene_constraints = plan.get("scene_planning_constraints")
        dramatic_function = plan.get("dramatic_function") or target["stub"] or chapter_id
        sqlite_db.upsert_chapter_plan(
            db_path,
            chapter_id=chapter_id,
            arc_id=target["arc_id"],
            description=dramatic_function,
            status="planned",
            snapshot_id=snapshot_id,
            node_id=node_id,
            parent_node_id=target["arc_node_id"],
            ordering=target["ordering"],
            dramatic_function=plan.get("dramatic_function"),
            expected_emotional_shift=(
                plan.get("expected_emotional_shift") or plan.get("pacing")
            ),
            required_thread_progress=json.dumps(obligations, sort_keys=True),
            scene_planning_constraints=(
                json.dumps(scene_constraints, sort_keys=True)
                if scene_constraints is not None
                else None
            ),
        )
        sqlite_db.upsert_planning_node(
            db_path,
            node_id=node_id,
            snapshot_id=snapshot_id,
            level=_LEVEL,
            status="planned",
            parent_id=target["arc_node_id"],
            ordering=target["ordering"],
            title=target["stub"] or chapter_id,
            summary=plan.get("dramatic_function"),
            purpose=json.dumps(plan, sort_keys=True),
        )

    return _persist


def _apply_macro_readiness(state: dict[str, Any], db_path: Any, snapshot_id: str) -> None:
    """Mark the macro outline complete; in approval mode, SET (never enter) the pause.

    The node only sets state and returns — it never blocks or waits
    (Module_Ability_Specification §5: the system never blocks waiting on a human; a
    headless run must not wait indefinitely — and 07.0a config validation guarantees
    ``headless_mode`` cannot coexist with ``approval_mode != "off"``, so headless always
    takes the no-pause branch). The graph edge that actually holds drafting on
    ``awaiting_planning_approval`` until ``macro_outline_approved`` is Build 14; the
    approval surface (routes/plan.py, Build 19) later flips approval and moves the
    snapshot to 'approved'/'rejected'. The ``macro_outline_approved`` guard keeps a
    re-run after approval from re-pausing (or clearing the snapshot's approval stamp).
    """
    state["macro_outline_ready"] = True
    if state.get("approval_mode") == "macro_outline" and not state.get(
        "macro_outline_approved"
    ):
        state["awaiting_planning_approval"] = True
        state["planning_block_reason"] = _AWAITING_APPROVAL_BLOCK_REASON
        sqlite_db.transition_snapshot_status(
            db_path, snapshot_id, status=_AWAITING_APPROVAL_SNAPSHOT_STATUS
        )


async def node_plan_chapter(
    state: dict[str, Any],
    *,
    decider: Any = None,
    registry: Any = None,
) -> dict[str, Any]:
    """Plan the active chapter's obligations + scene-planning constraints and persist them.

    Returns the (mutated) orchestrator state. ``decider`` and ``registry`` are injectable
    seams; when ``decider`` is None the production M04-backed decider is built via
    ``make_planner_decider`` and awaited by the loop.
    """
    config = _resolve_config(state)
    db_path = state.get("sqlite_db_path", runtime.SQLITE_DB_PATH)
    event_log_path = state.get("event_log_path", runtime.EVENT_LOG_PATH)
    project_id = state["project_id"]
    trace = state.setdefault("planner_deliberation_trace", [])

    # 1. resolve the snapshot (idempotent). Execution mode comes from the STATE snapshot
    # field (captured at run start), never re-read from config at node time.
    execution_mode = state.get("planning_execution_mode") or _MACRO_MODE
    snapshot_id = state.get("planning_snapshot_id") or f"snap_{project_id}"
    sqlite_db.create_planning_snapshot(
        db_path, snapshot_id=snapshot_id, project_id=project_id, mode=execution_mode
    )
    state["planning_snapshot_id"] = snapshot_id

    # 2. resolve the active chapter target from the arc plans' chapter stubs.
    target = _select_chapter_target(db_path, snapshot_id, state.get("fsm_pointer"))
    if target is None:
        # Nothing to expand: either the arc level has not produced chapter stubs yet, or
        # every chapter is already planned (idempotent completion — refresh macro
        # readiness so a re-run after approval-state loss converges to the same state).
        if execution_mode == _MACRO_MODE and _macro_chapter_scope_complete(
            db_path, snapshot_id
        ):
            _apply_macro_readiness(state, db_path, snapshot_id)
            trace.append(
                {"level": _LEVEL, "phase": "select_target", "outcome": "already_complete"}
            )
        else:
            trace.append(
                {"level": _LEVEL, "phase": "select_target", "outcome": "no_chapter_stubs"}
            )
        return state

    # 3. compile constraints against the chapter node — bail to clarification on a
    # hard-vs-hard contradiction (a THREAD_PARADOX constraint arrives here as ordinary
    # hard input; only a genuine hard-vs-hard conflict blocks).
    constraints = compile_planning_constraints(snapshot_id, target["node_id"], db_path=db_path)
    if constraints.get("needs_clarification"):
        state["planning_block_reason"] = constraints.get(
            "block_reason", "unresolved_hard_conflict"
        )
        trace.append(
            {
                "level": _LEVEL,
                "phase": "compile_constraints",
                "outcome": "needs_clarification",
                "hard_conflicts": constraints.get("hard_conflicts", []),
            }
        )
        return state

    # 4. assemble the chapter base_context — exactly the six base variables the P1
    # template documents (degrade gracefully for not-yet-built stores).
    project_metadata = state.get("project_metadata") or {}
    global_nodes = sqlite_db.get_planning_nodes(db_path, snapshot_id, level="global")
    global_plan = _parse_purpose(global_nodes[0]) if global_nodes else {}
    try:
        open_threads = sqlite_db.get_open_threads(db_path)
    except Exception:  # noqa: BLE001 - degrade to empty if the store is unavailable
        open_threads = []
    base_context = {
        "arc_plan": target["arc_plan"],
        "active_chapter": {"chapter_id": target["chapter_id"], "stub": target["stub"]},
        "global_plan": global_plan,
        "genre": project_metadata.get("genre", ""),
        "recent_chapter_summaries": _recent_chapter_summaries(db_path),
        "open_threads": list(open_threads),
    }
    continuity: dict[str, Any] = {}

    # 5. registry + decider (model contact only through the injected decider seam).
    registry = registry or PlanningToolRegistry(db_path)
    if decider is None:
        decider = make_planner_decider(_NODE_NAME, base_context, config)

    # 6. run the bounded deliberation loop with a deterministic baseline floor.
    outcome = await run_planner_loop(
        level=_LEVEL,
        snapshot_id=snapshot_id,
        target_node=target["node_id"],
        constraints=constraints,
        continuity=continuity,
        registry=registry,
        config=config,
        planner_decider=decider,
        deterministic_baseline=_build_chapter_baseline(base_context),
    )
    if outcome.records:
        trace.extend(outcome.records)

    # 7. map the outcome to persistence + state (never persists an invalid plan). The
    # plan's recorded annotation outcomes feed the revision diff.
    prior_nodes = sqlite_db.get_planning_nodes(db_path, snapshot_id)
    annotation_outcomes = (
        outcome.plan.get("annotation_outcomes") if isinstance(outcome.plan, dict) else None
    )
    result = persist_loop_outcome(
        outcome,
        level=_LEVEL,
        snapshot_id=snapshot_id,
        target_node=target["node_id"],
        persist_plan_fn=_chapter_persist_plan_fn(db_path, snapshot_id, target),
        db_path=db_path,
        event_log_path=event_log_path,
        prior_nodes=prior_nodes,
        annotation_outcomes=annotation_outcomes,
    )

    state["planning_snapshot_id"] = result.planning_snapshot_id
    state["active_planning_revision_id"] = result.active_planning_revision_id
    if result.outcome == "needs_clarification":
        state["planning_block_reason"] = result.planning_block_reason
    elif result.outcome == "escalate":
        # Loop exhausted caps + fallback ladder with no valid plan: signal recovery.
        # Expected to be rare — the deterministic baseline is the validated floor. No
        # readiness state is touched and the pointer does not advance.
        state["planning_block_reason"] = _ESCALATION_BLOCK_REASON
    else:
        # finalized / fallback_baseline: advance the pointer to the planned chapter so
        # the cascade can proceed (arc_id kept consistent when the sweep crosses arcs).
        pointer = state.get("fsm_pointer")
        if pointer is not None:
            state["fsm_pointer"] = pointer.model_copy(
                update={"arc_id": target["arc_id"], "chapter_id": target["chapter_id"]}
            )
        # 8. last macro level: readiness/approval state, read from STATE mode fields.
        # Only fires when this chapter completed the macro scope; in "rolling" mode the
        # readiness/pause fields are never touched and the node never pauses.
        if execution_mode == _MACRO_MODE and _macro_chapter_scope_complete(
            db_path, snapshot_id
        ):
            _apply_macro_readiness(state, db_path, snapshot_id)
    return state
