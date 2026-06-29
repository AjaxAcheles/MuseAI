"""Module: M05 (Hierarchical Planning Cascade)
The annotation compiler: turn the user's structured node annotations into a clean,
planner-ready constraint package (Context_Injection_Algorithms.md Phase 1 preamble;
Data_Structures.md §2.7).

User revisions to a planning snapshot are **structured node annotations, not chat**. Before
a planner level deliberates over a node, `compile_planning_constraints(snapshot_id,
target_node)` gathers the annotations that apply to that node and hands the planner a
single tidy package instead of raw scattered comments. It:

1. **collects** the annotations attached directly to the target node,
2. **inherits** applicable ancestor / global-scope annotations — respecting each
   annotation's `scope` field (global → every node; subtree → the node and its
   descendants; children → direct children; sibling_sequence → same-parent siblings;
   this_node → only that node),
3. applies **precedence** — hard requirements (`priority == 'hard'`) rank above soft
   preferences, and an annotation attached to a `locked_pinned` node is flagged as
   protected so a softer annotation cannot drop it,
4. **emits** a serializable constraint package (a plain dict) listing the effective hard
   requirements and soft preferences for the target node.

This module reads the planning store and may update an annotation's status to
`needs_clarification` (its only write); it never mutates planner output or canonical
prose, never calls an LLM, and never writes to the narrative tables. The package is
serializable and carries only structured annotation fields (no conversation text).

Conflict detection (Phase 1 step 4): when two **hard** annotations carry mutually
exclusive structural directives for the same planning node (e.g. a hard `pin` and a hard
`remove`, or a `remove`/`move` aimed at a `locked_pinned` node), the compiler marks the
involved annotations `status='needs_clarification'` via the M02 status-update helper,
omits both from the emitted package, and raises a blocking signal — never feeding a
contradictory hard pair into the planner loop. This is the **one** store mutation this
module performs (annotation status only). An unresolved hard conflict is a blocking
condition for drafting when `approval_mode` requires it (Module_Ability_Specification §5 /
A13, `planning_block_reason='unresolved_hard_conflict'`); this module only **detects,
marks, and signals** — the graph edge-blocking is M01/Build 14 and the surfacing UI is
M17/Build 19.

Deterministic, schema-grounded detection: contradiction is decided from the §2.7
`note_type` directives and `PlanningNode.locked_pinned`, never by an LLM or fragile
free-text parsing. Richer semantic contradiction (e.g. opposing continuity claims in the
annotation `text`) needs structured directives or a model check and is out of scope here.

`compute_revision_diff` produces the serializable `diff_json` for a `PlanningRevision`
(added/removed/moved/modified nodes + per-annotation outcomes). It only computes the diff
and never discards prior state; the actual `PlanningRevision` insert + `active_revision_id`
advance is the planner node's job via the 07.00 helper (revisions are never silent
overwrites — §2.7).
"""

from typing import Any

from memory import sqlite_db

# §2.7 CHECK vocabularies — referenced, never extended here.
_SCOPES = ("this_node", "children", "subtree", "sibling_sequence", "global")
_PRIORITIES = ("low", "normal", "high", "hard")

# Only these annotation statuses are active constraints. `rejected`/`superseded` no longer
# apply; `needs_clarification` is an unresolved hard conflict and is deliberately NOT sent
# to the planner (Phase 1 step 4) — Prompt 2 owns setting/resolving that status.
_ACTIVE_STATUSES = frozenset({"pending", "applied", "partially_applied"})

# Precedence ranking for ordering soft preferences (hard lives in its own bucket above).
_PRIORITY_RANK = {"low": 0, "normal": 1, "high": 2, "hard": 3}


def _ancestor_ids(target_id: str, node_index: dict[str, dict]) -> list[str]:
    """Return the target node's ancestor node_ids, nearest-first (cycle-safe)."""

    ancestors: list[str] = []
    seen: set[str] = {target_id}
    current = node_index.get(target_id)
    while current is not None:
        parent_id = current.get("parent_id")
        if not parent_id or parent_id in seen:
            break
        seen.add(parent_id)
        ancestors.append(parent_id)
        current = node_index.get(parent_id)
    return ancestors


def _applies_to_target(
    annotation: dict,
    *,
    target_id: str,
    target_parent_id: str | None,
    ancestor_ids: set[str],
    node_index: dict[str, dict],
) -> bool:
    """Decide whether an annotation reaches the target node, honoring its `scope`.

    `source` is the node the annotation is attached to (`target_node_id`); the relationship
    between that source and the target node — combined with `scope` — decides reach.
    """

    source = annotation.get("target_node_id")
    scope = annotation.get("scope")

    if scope == "global":
        return True
    if scope == "this_node":
        return source == target_id
    if scope == "subtree":
        # The annotation's node, or any ancestor of the target → target is in its subtree.
        return source == target_id or source in ancestor_ids
    if scope == "children":
        # Direct children only: the target is a direct child of the annotation's node.
        return source == target_parent_id
    if scope == "sibling_sequence":
        # The annotation's node and its same-parent siblings (includes the node itself).
        source_node = node_index.get(source)
        source_parent = source_node.get("parent_id") if source_node else None
        return source == target_id or source_parent == target_parent_id
    # Unknown scope (cannot occur given the §2.7 CHECK) — ignore rather than guess.
    return False


def _entry(annotation: dict, *, origin: str, from_pinned_node: bool) -> dict[str, Any]:
    """Build a lean, serializable constraint entry from a stored annotation row."""

    return {
        "annotation_id": annotation.get("annotation_id"),
        "source_node_id": annotation.get("target_node_id"),
        "source_level": annotation.get("target_level"),
        "note_type": annotation.get("note_type"),
        "scope": annotation.get("scope"),
        "priority": annotation.get("priority"),
        "text": annotation.get("text"),
        "status": annotation.get("status"),
        "created_at": annotation.get("created_at"),
        "origin": origin,  # "direct" (attached to the target) | "inherited" (ancestor/global)
        "from_pinned_node": from_pinned_node,
    }


# Mutually-exclusive hard structural directives on the SAME planning node. These are the
# deterministic, schema-grounded contradictions (§2.7 note_type vocabulary): you cannot
# both keep (`pin`) and delete (`remove`) the same node, etc.
_MUTUALLY_EXCLUSIVE_NOTE_TYPES: tuple[frozenset[str], ...] = (
    frozenset({"pin", "remove"}),
    frozenset({"pin", "move"}),
    frozenset({"remove", "move"}),
)


def _detect_hard_conflicts(
    hard_rows: list[dict], node_index: dict[str, dict]
) -> list[dict]:
    """Find hard-vs-hard contradictions among annotations on the same planning node.

    Deterministic and schema-grounded: a conflict is mutually exclusive `note_type`
    directives on one node (pin/remove/move combinations), or a hard `remove`/`move`
    directive aimed at a `locked_pinned` node (which cannot be removed/moved without
    resolution — §2.7). Returns one group per conflicting node with the involved
    annotation ids, the offending note_types, and a human-readable reason.
    """

    by_source: dict[str, list[dict]] = {}
    for annotation in hard_rows:
        by_source.setdefault(annotation.get("target_node_id"), []).append(annotation)

    conflicts: list[dict] = []
    for source_id, group in by_source.items():
        types_present = {a.get("note_type") for a in group}
        exclusive_hit: set[str] = set()
        for pair in _MUTUALLY_EXCLUSIVE_NOTE_TYPES:
            if pair <= types_present:
                exclusive_hit |= pair
        pinned = bool(node_index.get(source_id, {}).get("locked_pinned"))
        if pinned:
            # A hard remove/move directive cannot apply to a user-pinned node.
            exclusive_hit |= types_present & {"remove", "move"}
        if not exclusive_hit:
            continue
        involved = [a for a in group if a.get("note_type") in exclusive_hit]
        note_types = sorted(exclusive_hit)
        reason = (
            f"conflicting hard directives on planning node {source_id!r}: {note_types}"
            + (" (node is locked_pinned)" if pinned else "")
        )
        conflicts.append(
            {
                "source_node_id": source_id,
                "annotation_ids": sorted(a.get("annotation_id") for a in involved),
                "note_types": note_types,
                "reason": reason,
                "newly_marked": True,
            }
        )
    return sorted(conflicts, key=lambda c: c["source_node_id"] or "")


def compile_planning_constraints(
    snapshot_id: str,
    target_node: dict | str,
    *,
    db_path: Any,
) -> dict[str, Any]:
    """Compile the planner-ready constraint package for one target node.

    `target_node` may be a PlanningNode row (dict) or a `node_id` string. Reads the
    snapshot's nodes and annotations via the M02 (07.00) helpers, filters annotations to
    those that are active and that reach the target node (respecting `scope`), ranks hard
    requirements above soft preferences, flags constraints from `locked_pinned` nodes as
    protected, and returns a serializable package. Hard-vs-hard contradictions are detected
    deterministically: the involved hard annotations are marked `needs_clarification` (the
    only store write), excluded from the package, and surfaced via a blocking signal — a
    contradictory hard pair is never sent to the planner. No LLM, no canonical-narrative
    writes. The graph edge-block (Build 14) and UI (Build 19) are out of scope; this only
    detects, marks, and signals.
    """

    if isinstance(target_node, dict):
        target_id = target_node.get("node_id")
    elif isinstance(target_node, str):
        target_id = target_node
    else:
        raise TypeError("target_node must be a PlanningNode dict or a node_id string")
    if not target_id:
        raise ValueError("target_node must carry a node_id")

    nodes = sqlite_db.get_planning_nodes(db_path, snapshot_id)
    node_index = {n["node_id"]: n for n in nodes}
    target_row = node_index.get(target_id)
    if target_row is None and isinstance(target_node, dict):
        target_row = target_node
    if target_row is None:
        raise ValueError(
            f"target node {target_id!r} not found in snapshot {snapshot_id!r}"
        )

    target_level = target_row.get("level")
    target_parent_id = target_row.get("parent_id")
    target_locked_pinned = bool(target_row.get("locked_pinned"))
    ancestor_ids = set(_ancestor_ids(target_id, node_index))

    annotations = sqlite_db.get_annotations_for_snapshot(db_path, snapshot_id)

    # Partition the annotations that REACH this target (by scope) by status.
    active_hard_rows: list[dict] = []
    active_soft_rows: list[dict] = []
    unresolved_nc_ids: list[str] = []
    skipped_inactive = 0

    for annotation in annotations:
        if not _applies_to_target(
            annotation,
            target_id=target_id,
            target_parent_id=target_parent_id,
            ancestor_ids=ancestor_ids,
            node_index=node_index,
        ):
            continue
        status = annotation.get("status")
        if status == "needs_clarification":
            # A pre-existing unresolved hard conflict still blocks (survives re-runs).
            unresolved_nc_ids.append(annotation.get("annotation_id"))
            continue
        if status not in _ACTIVE_STATUSES:
            skipped_inactive += 1
            continue
        if annotation.get("priority") == "hard":
            active_hard_rows.append(annotation)
        else:
            active_soft_rows.append(annotation)

    # Hard-vs-hard conflict detection: never emit a contradictory hard pair. Mark the
    # involved annotations `needs_clarification` (the one store mutation here) so the
    # approval gate / planner node can surface them; exclude them from the package.
    conflict_groups = _detect_hard_conflicts(active_hard_rows, node_index)
    conflicting_ids = {aid for group in conflict_groups for aid in group["annotation_ids"]}
    for group in conflict_groups:
        for aid in group["annotation_ids"]:
            sqlite_db.update_annotation_status(
                db_path, aid, status="needs_clarification",
                planner_response=group["reason"],
            )

    def _build(rows: list[dict]) -> list[dict]:
        built: list[dict] = []
        for annotation in rows:
            source = annotation.get("target_node_id")
            origin = "direct" if source == target_id else "inherited"
            from_pinned_node = bool(node_index.get(source, {}).get("locked_pinned"))
            built.append(_entry(annotation, origin=origin, from_pinned_node=from_pinned_node))
        return built

    hard_annotations = _build(
        [a for a in active_hard_rows if a.get("annotation_id") not in conflicting_ids]
    )
    soft_preferences = _build(active_soft_rows)
    pinned_annotations = [
        e for e in (hard_annotations + soft_preferences) if e["from_pinned_node"]
    ]

    # Precedence ordering: hard requirements (all 'hard') by creation order; soft
    # preferences by descending priority then creation order — deterministic and stable.
    hard_annotations.sort(key=lambda e: (e["created_at"] or "", e["annotation_id"] or ""))
    soft_preferences.sort(
        key=lambda e: (
            -_PRIORITY_RANK.get(e["priority"], 0),
            e["created_at"] or "",
            e["annotation_id"] or "",
        )
    )

    snapshot = sqlite_db.get_planning_snapshot(db_path, snapshot_id)
    unresolved_ids = sorted(set(unresolved_nc_ids) | conflicting_ids)
    needs_clarification = bool(conflict_groups) or bool(unresolved_nc_ids)

    return {
        "snapshot_id": snapshot_id,
        "target_node_id": target_id,
        "target_level": target_level,
        "target_locked_pinned": target_locked_pinned,
        # `hard_annotations` is the key the validators consume (validate_user_annotations);
        # hard ranks above soft by living in its own bucket.
        "hard_annotations": hard_annotations,
        "soft_preferences": soft_preferences,
        # Constraints from locked_pinned nodes: protected, cannot be dropped by a softer
        # annotation (also present in their hard/soft bucket above).
        "pinned_annotations": pinned_annotations,
        # Blocking signal: True when a hard conflict (newly detected or a pre-existing
        # unresolved one) applies to this target. The graph edge-block is Build 14; the
        # surfacing UI is Build 19 — this only signals.
        "needs_clarification": needs_clarification,
        "block_reason": "unresolved_hard_conflict" if needs_clarification else None,
        "hard_conflicts": conflict_groups,
        "unresolved_conflict_annotation_ids": unresolved_ids,
        "meta": {
            "compiled_by": "fsm.planning_annotations.compile_planning_constraints",
            "snapshot_status": snapshot.get("status") if snapshot else None,
            "ancestor_node_ids": sorted(ancestor_ids),
            "counts": {
                "hard": len(hard_annotations),
                "soft": len(soft_preferences),
                "pinned": len(pinned_annotations),
                "skipped_inactive": skipped_inactive,
                "hard_conflicts": len(conflict_groups),
                "unresolved_needs_clarification": len(unresolved_nc_ids),
            },
            "conflict_detection": "structural_note_type_and_pin_state",
        },
    }


# --- revision diff ----------------------------------------------------------

# Node fields whose change is a content modification vs a structural move.
_NODE_CONTENT_FIELDS: tuple[str, ...] = (
    "title",
    "summary",
    "purpose",
    "status",
    "locked_pinned",
)
_NODE_POSITION_FIELDS: tuple[str, ...] = ("parent_id", "ordering")


def _node_summary(node: dict) -> dict[str, Any]:
    """Compact, serializable identity of a node for added/removed listings."""

    return {
        "node_id": node.get("node_id"),
        "level": node.get("level"),
        "title": node.get("title"),
    }


def _normalize_annotation_outcomes(annotation_outcomes: Any) -> list[dict]:
    """Normalize {id: outcome} or [{annotation_id|id, outcome}] into a sorted list."""

    outcomes: list[dict] = []
    if isinstance(annotation_outcomes, dict):
        for ann_id, outcome in annotation_outcomes.items():
            outcomes.append({"annotation_id": ann_id, "outcome": outcome})
    elif isinstance(annotation_outcomes, (list, tuple)):
        for entry in annotation_outcomes:
            if isinstance(entry, dict):
                ann_id = entry.get("annotation_id", entry.get("id"))
                outcomes.append({"annotation_id": ann_id, "outcome": entry.get("outcome")})
    return sorted(outcomes, key=lambda o: str(o.get("annotation_id")))


def compute_revision_diff(
    previous_nodes: list[dict] | None,
    new_nodes: list[dict] | None,
    annotation_outcomes: Any = None,
) -> dict[str, Any]:
    """Compute the serializable ``diff_json`` between two planning-node states (§2.7).

    Returns added / removed / moved (``parent_id`` or ``ordering`` changed) / modified
    (a content field changed) nodes plus normalized per-annotation outcomes (applied /
    partially_applied / rejected / overridden — caller-supplied). Pure and deterministic:
    it never mutates its inputs and never discards prior state (removed nodes are listed
    explicitly), so revision history is preserved. The caller (planner node) inserts the
    PlanningRevision and advances ``active_revision_id`` via the 07.00 helper; this only
    computes the diff.
    """

    prev_index = {n.get("node_id"): n for n in (previous_nodes or [])}
    new_index = {n.get("node_id"): n for n in (new_nodes or [])}
    prev_ids = set(prev_index)
    new_ids = set(new_index)

    modified: list[dict] = []
    moved: list[dict] = []
    unchanged = 0
    for node_id in sorted(prev_ids & new_ids, key=str):
        before = prev_index[node_id]
        after = new_index[node_id]
        changed_fields = {
            field: {"from": before.get(field), "to": after.get(field)}
            for field in _NODE_CONTENT_FIELDS
            if before.get(field) != after.get(field)
        }
        position_change = {
            field: {"from": before.get(field), "to": after.get(field)}
            for field in _NODE_POSITION_FIELDS
            if before.get(field) != after.get(field)
        }
        if changed_fields:
            modified.append({"node_id": node_id, "changed_fields": changed_fields})
        if position_change:
            moved.append({"node_id": node_id, "position_change": position_change})
        if not changed_fields and not position_change:
            unchanged += 1

    return {
        "added_nodes": [
            _node_summary(new_index[n]) for n in sorted(new_ids - prev_ids, key=str)
        ],
        "removed_nodes": [
            _node_summary(prev_index[n]) for n in sorted(prev_ids - new_ids, key=str)
        ],
        "modified_nodes": modified,
        "moved_nodes": moved,
        "unchanged_node_count": unchanged,
        "annotation_outcomes": _normalize_annotation_outcomes(annotation_outcomes),
    }
