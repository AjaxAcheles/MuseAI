"""Module: M17 (Web UI & Real-time Observer Surface)
Planning read surface + macro-outline approval endpoint for the product shell.

Normalizes the REAL persisted planning state (PlanningSnapshot / PlanningNode
rows written by the M05 planner nodes, whose ``purpose`` column holds each
validated plan's full JSON) into frontend-friendly timeline data. Nothing here
is invented client-side fixture data: an empty store returns an honest empty
response. This is the vertical-slice ancestor of the Build-19 ``/plan``
timeline, not the final approval UI.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from quart import Blueprint, jsonify

from core.runtime import get_resources
from memory.sqlite_db import (
    get_annotations_for_snapshot,
    get_planning_nodes,
    get_planning_snapshot,
    get_planning_snapshots,
    get_revisions_for_snapshot,
)

logger = logging.getLogger(__name__)

# Readable-summary field preferences per planning level: (title keys, summary keys).
# These read the validated plan JSON the nodes persisted in `purpose`.
_LEVEL_FIELDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "global": (("premise",), ("central_conflict", "ending_target")),
    "arc": (("title",), ("function",)),
    "chapter": (("dramatic_function",), ("expected_emotional_shift", "stub")),
    "scene": (("scene_function",), ("setting", "exit_state")),
    "beat": (("immediate_objective",), ("exit_condition", "physical_constraints")),
}

# Fields surfaced in the readable detail panel, per level and in display order.
_DETAIL_FIELDS: dict[str, tuple[str, ...]] = {
    "global": ("premise", "central_conflict", "ending_target", "arcs", "promises"),
    "chapter": (
        "dramatic_function",
        "expected_emotional_shift",
        "pacing",
        "obligations",
        "scene_planning_constraints",
    ),
    "arc": ("title", "function", "character_milestones", "chapters"),
    "scene": (
        "scene_function",
        "setting",
        "participants",
        "entry_state",
        "exit_state",
        "conflict_turn",
        "continuity_constraints",
        "word_budget",
    ),
    "beat": (
        "immediate_objective",
        "physical_constraints",
        "entry_condition",
        "exit_condition",
        "pad_target",
        "behavioral_constraint",
    ),
}


def _parse_purpose(row: dict[str, Any]) -> dict[str, Any]:
    try:
        parsed = json.loads(row.get("purpose") or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _first_str(plan: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = plan.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_node(row: dict[str, Any]) -> dict[str, Any]:
    """One PlanningNode row → readable tree node (no raw JSON blob)."""
    level = row.get("level") or ""
    plan = _parse_purpose(row)
    title_keys, summary_keys = _LEVEL_FIELDS.get(level, ((), ()))
    title = (
        (row.get("title") or "").strip()
        or _first_str(plan, title_keys)
        or row.get("node_id", "")
    )
    summary = (row.get("summary") or "").strip() or _first_str(plan, summary_keys)
    # An arc-written chapter stub has no dramatic_function yet — label it so
    # the timeline distinguishes planned chapters from pending stubs.
    planned = bool(plan.get("dramatic_function")) if level == "chapter" else bool(plan)
    return {
        "node_id": row.get("node_id"),
        "level": level,
        "title": title,
        "summary": summary,
        "status": "planned" if planned else "stub",
        "ordering": row.get("ordering"),
        "children": [],
    }


def _resolve_snapshot_id(db_path: Any) -> str | None:
    """Active/last run's snapshot first; else the newest persisted snapshot."""
    manager = get_resources().generation_manager
    snapshot_id = manager.planning_snapshot_id if manager is not None else None
    if snapshot_id:
        return snapshot_id
    rows = get_planning_snapshots(db_path)
    return rows[0]["snapshot_id"] if rows else None


def create_blueprint() -> Blueprint:
    """Create the plan blueprint (/plan/*)."""
    bp = Blueprint("plan", __name__)

    @bp.get("/plan/snapshot")
    async def plan_snapshot() -> Any:
        resources = get_resources()
        db_path = resources.stores["sqlite"].path
        snapshot_id = _resolve_snapshot_id(db_path)
        if snapshot_id is None:
            return jsonify(
                {
                    "ok": True,
                    "snapshot_id": None,
                    "tree": None,
                    "runtime": None,
                    "message": "No planning snapshot exists yet — start a run.",
                }
            )
        snapshot = get_planning_snapshot(db_path, snapshot_id)
        if snapshot is None:
            return jsonify(
                {
                    "ok": True,
                    "snapshot_id": None,
                    "tree": None,
                    "runtime": None,
                    "message": "No planning snapshot exists yet — start a run.",
                }
            )

        rows = get_planning_nodes(db_path, snapshot_id)
        by_id = {row["node_id"]: _normalize_node(row) for row in rows}
        root = None
        scenes: list[dict[str, Any]] = []
        beats: list[dict[str, Any]] = []
        for row in rows:
            node = by_id[row["node_id"]]
            level = node["level"]
            if level in ("scene", "beat"):
                # Runtime levels render in their own section, not the macro tree.
                (scenes if level == "scene" else beats).append(node)
                continue
            parent = by_id.get(row.get("parent_id") or "")
            if parent is not None and parent["level"] not in ("scene", "beat"):
                parent["children"].append(node)
            elif level == "global":
                root = node

        manager = resources.generation_manager
        pointer = manager.run_pointer if manager is not None else None
        awaiting = bool(
            manager is not None and manager.status().get("awaiting_approval")
        )
        revisions = get_revisions_for_snapshot(db_path, snapshot_id)
        return jsonify(
            {
                "ok": True,
                "snapshot_id": snapshot_id,
                "status": snapshot.get("status"),
                "mode": snapshot.get("mode"),
                "active_revision_id": snapshot.get("active_revision_id"),
                "approved_at": snapshot.get("approved_at"),
                "revision_count": len(revisions),
                "awaiting_approval": awaiting,
                "tree": root,
                "runtime": {
                    "scenes": scenes,
                    "beats": beats,
                    "current_scene": (pointer or {}).get("scene_id") or None,
                    "current_beat": (pointer or {}).get("beat_id") or None,
                },
                "message": f"Snapshot {snapshot_id} ({snapshot.get('status')}).",
            }
        )

    @bp.get("/plan/node/<path:node_id>")
    async def plan_node(node_id: str) -> Any:
        resources = get_resources()
        db_path = resources.stores["sqlite"].path
        snapshot_id = _resolve_snapshot_id(db_path)
        row = None
        if snapshot_id is not None:
            row = next(
                (
                    candidate
                    for candidate in get_planning_nodes(db_path, snapshot_id)
                    if candidate["node_id"] == node_id
                ),
                None,
            )
        if row is None:
            return (
                jsonify(
                    {"ok": False, "status": "error", "message": f"Unknown node {node_id!r}."}
                ),
                404,
            )
        plan = _parse_purpose(row)
        normalized = _normalize_node(row)
        fields = [
            {"name": name, "value": plan[name]}
            for name in _DETAIL_FIELDS.get(normalized["level"], ())
            if name in plan
        ]
        return jsonify(
            {
                "ok": True,
                "node": {
                    **{k: normalized[k] for k in normalized if k != "children"},
                    "parent_id": row.get("parent_id"),
                    "db_status": row.get("status"),
                    "fields": fields,
                    "raw": plan,  # collapsible debug block only
                },
            }
        )

    @bp.get("/plan/debug/raw")
    async def plan_debug_raw() -> Any:
        resources = get_resources()
        db_path = resources.stores["sqlite"].path
        snapshot_id = _resolve_snapshot_id(db_path)
        if snapshot_id is None:
            return jsonify({"ok": True, "snapshot": None, "nodes": [], "revisions": [],
                            "annotations": []})
        return jsonify(
            {
                "ok": True,
                "snapshot": get_planning_snapshot(db_path, snapshot_id),
                "nodes": get_planning_nodes(db_path, snapshot_id),
                "revisions": get_revisions_for_snapshot(db_path, snapshot_id),
                "annotations": get_annotations_for_snapshot(db_path, snapshot_id),
            }
        )

    @bp.post("/plan/approve")
    async def plan_approve() -> Any:
        manager = get_resources().generation_manager
        if manager is None:
            return (
                jsonify(
                    {
                        "ok": False,
                        "status": "error",
                        "message": "Generation manager is not initialized.",
                    }
                ),
                503,
            )
        try:
            result = await manager.approve_plan()
        except Exception:  # noqa: BLE001 - never leak a traceback to the browser
            logger.exception("plan approval failed")
            return (
                jsonify(
                    {"ok": False, "status": "error", "message": "Approval failed; see logs."}
                ),
                500,
            )
        return jsonify(result)

    return bp
