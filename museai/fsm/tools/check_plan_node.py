"""Validate one planned chapter or beat before it is returned in the plan.

The same shape checks the planning nodes will apply, offered up front: a
planner that validates its own elements fixes them in the same turn instead of
burning a parse-retry or storing a beat whose focal character does not exist.
Purely deterministic — no model, no writes.
"""

from __future__ import annotations

from typing import Any

from museai.fsm.pad import PAD_AXES
from museai.fsm.plan_validation import validate_concrete_obligation
from museai.fsm.tools.project_db import project_connection
from museai.memory.db import get_characters, get_threads_for_project

_THREAD_STATUSES = ("open", "progressing", "closed")

CHECK_PLAN_NODE_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "check_plan_node",
        "description": (
            "Validate one planned chapter or beat object before you include "
            "it in your answer. Checks required fields, target_pad ranges, "
            "and that the focal character and thread ids actually exist. "
            "Returns the problems to fix; an empty list means the node is "
            "well-formed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "plan_node": {
                    "type": "object",
                    "description": (
                        "One element of your plan: a chapter "
                        "({description, obligations}) or a beat ({intent, "
                        "entry_state, exit_state, required_change, "
                        "observable_event, target_pad, focal_character_id, ...})."
                    ),
                },
            },
            "required": ["plan_node"],
        },
    },
}


def _check_chapter(node: dict) -> list[str]:
    problems: list[str] = []
    if not str(node.get("description") or "").strip():
        problems.append("a chapter needs a non-empty description")
    obligations = node.get("obligations")
    if not isinstance(obligations, list) or not obligations:
        problems.append("obligations must be a non-empty list of concrete promises")
    else:
        for index, item in enumerate(obligations, start=1):
            _, problem = validate_concrete_obligation(item)
            if problem:
                problems.append(f"obligation {index} {problem}")
    return problems


def _check_beat(node: dict, characters: list, threads: list) -> list[str]:
    problems: list[str] = []
    if not str(node.get("intent") or "").strip():
        problems.append("a beat needs a non-empty intent")
    for field in ("entry_state", "exit_state"):
        if not str(node.get(field) or "").strip():
            problems.append(f"a beat needs a non-empty {field}")
    if not str(node.get("required_change") or "").strip():
        problems.append(
            "a beat needs a non-empty required_change — the one meaningful "
            "change it produces (it may be interior: a realization, a decision)"
        )
    if not str(node.get("observable_event") or "").strip():
        problems.append(
            "a beat needs a non-empty observable_event — what the reader sees "
            "on the page that carries its change"
        )
    discharges = node.get("discharges")
    if discharges is not None:
        if not isinstance(discharges, list):
            problems.append("discharges must be a list of obligation strings")
        else:
            for index, entry in enumerate(discharges, start=1):
                if not str(entry or "").strip():
                    problems.append(f"discharges[{index}] is empty")

    pad = node.get("target_pad")
    if not isinstance(pad, dict):
        problems.append("target_pad must be an object with pleasure/arousal/dominance")
    else:
        for axis in PAD_AXES:
            value = pad.get(axis)
            try:
                number = float(value)
            except (TypeError, ValueError):
                problems.append(f"target_pad.{axis} is not a number: {value!r}")
                continue
            if not -1.0 <= number <= 1.0:
                problems.append(f"target_pad.{axis} must be in -1.0..1.0, got {number}")

    focal = str(node.get("focal_character_id") or "").strip()
    if focal:
        # The planners' own resolver, so this check agrees with what the node
        # will accept. Imported lazily: plan_beat imports the tool registry.
        from museai.fsm.nodes.plan_beat import resolve_focal_character

        if not resolve_focal_character(focal, characters):
            known = ", ".join(c["id"] for c in characters) or "none"
            problems.append(
                f"focal_character_id {focal!r} matches no character; known: {known}"
            )
    else:
        problems.append("a beat needs a focal_character_id from the cast")

    updates = node.get("thread_updates")
    if updates is not None:
        known_threads = {t["id"] for t in threads}
        if not isinstance(updates, list):
            problems.append("thread_updates must be a list of {id, status} objects")
        else:
            for index, entry in enumerate(updates, start=1):
                if not isinstance(entry, dict):
                    problems.append(f"thread_updates[{index}] is not an object")
                    continue
                thread_id = str(entry.get("id") or entry.get("thread_id") or "").strip()
                status = str(entry.get("status") or "").strip()
                if thread_id not in known_threads:
                    problems.append(
                        f"thread_updates[{index}] names unknown thread {thread_id!r}"
                    )
                if status not in _THREAD_STATUSES:
                    problems.append(
                        f"thread_updates[{index}] status must be one of "
                        f"{', '.join(_THREAD_STATUSES)}; got {status!r}"
                    )
    return problems


def check_plan_node(plan_node: dict) -> dict:
    """``{"valid", "node_type", "problems"}`` for one plan element."""
    if not isinstance(plan_node, dict):
        return {
            "valid": False,
            "node_type": "unknown",
            "problems": ["plan_node must be a JSON object"],
        }

    is_beat = "intent" in plan_node or "target_pad" in plan_node
    if is_beat:
        with project_connection() as (conn, project_id):
            characters = [dict(row) for row in get_characters(conn, project_id)]
            threads = [dict(row) for row in get_threads_for_project(conn, project_id)]
        problems = _check_beat(plan_node, characters, threads)
        node_type = "beat"
    else:
        problems = _check_chapter(plan_node)
        node_type = "chapter"

    return {"valid": not problems, "node_type": node_type, "problems": problems}
