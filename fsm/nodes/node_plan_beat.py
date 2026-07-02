"""Module: M05 (Hierarchical Planning Cascade)

Level-5 (beat) planner node. Partitions the active scene into actionable beats — ONE
beat per invocation — by running the shared bounded deliberation loop and persisting
only a validator-passed beat plan, and owns the PAD Grounded Translation Pipeline:

  1. **raw target** — the proposed affect comes from the approved scene plan's own
     affect key (``pad_target``/``pad``/``pad_state``) when the scene planner supplied
     one; otherwise the proposal is the prior state itself (affect inertia). No
     coordinate is ever invented here: the design's scene-intent→PAD interpretive step
     has no specified deterministic mechanism and ``pad_regions.json`` carries no
     reverse text→coordinate prototypes, so fabricating numbers would be a hidden
     tunable. The prior is the mean of the scene's latest committed per-character PAD
     rows (``get_latest_pad_for_scene``), degrading to the neutral origin when no
     history exists;
  2. **EWMA smoothing** — ``smooth_pad_target(prior, proposed, alpha)`` with ``alpha``
     read from the existing ``config.thresholds.pad_ewma_alpha`` key (never a new
     planning-scoped alpha, never hardcoded);
  3. **grounded translation** — ``translate_pad_to_behaviour(target, context,
     adapt_fn=...)``: the static/fallback rungs are pure ``pad_regions.json`` lookups
     and run with **no LLM**; the optional small-tier adaptation is the injected
     ``adapt_fn`` seam, passed through verbatim (``None`` → static floor). The
     translation rung is recorded in the deliberation trace (WARNING-level log wiring
     lands with node-logging integration);
  4. **persistence** — the tailored behavioural-constraint string is written into the
     beat's proposal surface by the 07.00 ``upsert_beat_plan``.

The harness owns the affect: the persisted plan's ``pad_target``/``behavioral_constraint``
are forced to the node-computed values regardless of what the model echoed.

After a successful persist the node computes the **volume-aware scene-stop signal**
(``state["scene_needs_more"]``): a scene may close only when its committed word volume
has met the scene's ``word_budget`` (SQLite ``Scenes`` row) AND at least
``config.runtime.beats_per_scene_min`` beats exist — otherwise the scene needs more
beats. (The conceptual spec's volume-aware stop criterion; the book-level
``runtime.word_count_target`` is not a scene-stop input.) Routing on the signal is
Build 14; the field is set on the state dict and consumed there.

The node makes **no** direct ``call_llm``: deliberation-path model contact is only the
injected decider seam, and the only other possible model contact is the optional,
skippable PAD ``adapt_fn``. It writes no prose and wires no graph route (Build 14).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import core.runtime as runtime
from fsm.pad_translation import PADTarget, smooth_pad_target, translate_pad_to_behaviour
from fsm.planning_annotations import compile_planning_constraints
from fsm.planning_loop import run_planner_loop
from fsm.planning_node_support import make_planner_decider, persist_loop_outcome
from fsm.planning_tools import PlanningToolRegistry
from memory import sqlite_db

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
_LEVEL = "beat"
_NODE_NAME = "node_plan_beat"

# Human-readable, provisional planning-block reason for a loop that exhausted its caps and
# fallback ladder without a valid plan. The actual route to failure recovery is Build 14.
_ESCALATION_BLOCK_REASON = "planning_escalation"


def _resolve_config(state: dict[str, Any]) -> Any:
    """Return the active typed config (mirrors the other planner nodes' resolution)."""
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


def _continuity_facts(db_path: Any) -> list[dict]:
    """Active continuity facts for the beat to honor; [] while the fact store is a stub."""
    del db_path  # no continuity-fact store exists to read yet
    return []


def _prior_pad(db_path: Any, scene_id: str) -> PADTarget:
    """The scene's prior affect state: mean of its latest per-character PAD rows.

    Degrades to the neutral origin (0, 0, 0) — the structural centre of the PAD space,
    not a tunable — when the scene has no committed PAD history (the common case at
    planning time) or the store is unavailable.
    """
    try:
        rows = sqlite_db.get_latest_pad_for_scene(db_path, scene_id)
    except Exception:  # noqa: BLE001 - degrade to neutral if the store is unavailable
        rows = []
    if not rows:
        return PADTarget(0.0, 0.0, 0.0)
    n = len(rows)
    return PADTarget(
        sum(float(r.get("pleasure") or 0.0) for r in rows) / n,
        sum(float(r.get("arousal") or 0.0) for r in rows) / n,
        sum(float(r.get("dominance") or 0.0) for r in rows) / n,
    )


def _proposed_pad(scene_plan: dict, prior: PADTarget) -> Any:
    """The raw proposed affect target for this beat.

    Uses the scene plan's own affect key when the scene planner declared one; otherwise
    proposes the prior itself (affect inertia — the EWMA then holds steady). Never
    invents coordinates from prose-level intent; that interpretive step is a documented
    future increment.
    """
    for key in ("pad_target", "pad", "pad_state"):
        value = scene_plan.get(key)
        if value:
            return value
    return prior


def _planned_beats(db_path: Any, snapshot_id: str, scene_node_id: str) -> list[dict]:
    """The scene's already-planned beats, in order, as lean plan summaries."""
    beats: list[dict] = []
    for node in sqlite_db.get_planning_nodes_by_parent(db_path, snapshot_id, scene_node_id):
        if node.get("level") != _LEVEL:
            continue
        plan = _parse_purpose(node)
        beats.append(
            {
                "beat_id": plan.get("beat_id") or node.get("title") or node["node_id"],
                "immediate_objective": plan.get("immediate_objective")
                or node.get("summary")
                or "",
                "exit_condition": plan.get("exit_condition") or "",
                "beat_index": node.get("ordering", 0),
            }
        )
    return beats


def _build_beat_baseline(base_context: dict[str, Any]) -> Any:
    """Build the loop's deterministic baseline: a minimal, single-pass valid beat plan.

    Validator-shaped for all five beat checks (``schema``/``no_drafting``/
    ``draftability``/``continuity``/``pad_grounding``): a concrete objective and
    staging drawn from the scene plan, entry chained from the last beat's exit (or the
    scene's entry state), the harness-computed affect values attached verbatim, and
    ``asserted_facts`` kept empty — the baseline asserts no continuity it cannot verify.
    """
    scene_plan = base_context.get("scene_plan") or {}
    planned = base_context.get("planned_beats") or []
    pad_target = base_context.get("pad_target")
    behavioural = base_context.get("pad_behavioral_constraint") or ""

    def _baseline(level: str, target_node: Any, constraints: dict) -> dict:
        if planned:
            entry = f"as the previous beat ended: {planned[-1].get('exit_condition') or 'its outcome stands'}"
        else:
            entry = scene_plan.get("entry_state") or "the scene opens on its established entry state"
        objective = (
            scene_plan.get("conflict_turn")
            or scene_plan.get("scene_function")
            or "advance the scene's function one concrete step"
        )
        participants = scene_plan.get("participants") or ["the scene's focal character"]
        setting = scene_plan.get("setting") or "the scene's established location"
        return {
            "immediate_objective": f"advance the scene toward its turn: {objective}",
            "physical_constraints": [
                f"present: {', '.join(str(p) for p in participants)}",
                f"location: {setting}",
            ],
            "entry_condition": entry,
            "exit_condition": (
                "the beat's objective has landed and the scene is one concrete step "
                "closer to its exit state"
            ),
            "pad_target": pad_target,
            "behavioral_constraint": behavioural,
            "asserted_facts": [],
        }

    return _baseline


def _beat_persist_plan_fn(
    db_path: Any,
    snapshot_id: str,
    scene_id: str,
    scene_node_id: str,
    pad_target_dict: dict[str, float],
    behavioural_constraint: str,
    allocated: dict[str, Any],
) -> Any:
    """Return the level-specific writer: the Beats row + the beat PlanningNode.

    ``beat_index`` is derived at write time from the scene's existing beats (``max + 1``,
    ``0`` for the first) and beat identity is harness-owned (a missing/colliding plan
    ``beat_id`` is replaced with a derived ``{scene_id}_b{n}``). The harness also owns
    the affect: the persisted plan's ``pad_target``/``behavioral_constraint`` are forced
    to the node-computed values. The 07.00 ``upsert_beat_plan`` writes the structural
    ``Beats`` row (never ``prose``/``word_count``/``committed_at``) plus the beat
    PlanningNode; a second idempotent ``upsert_planning_node`` upgrades ``purpose`` to
    the full validated plan JSON (the sibling-node convention). The allocated identity
    is reported back through ``allocated`` for the pointer update.
    """

    def _persist(plan: dict) -> None:
        existing = sqlite_db.get_beats_for_scene_ordered(db_path, scene_id)
        indexes = [b["beat_index"] for b in existing]
        beat_index = (max(indexes) + 1) if indexes else 0
        existing_ids = {b["id"] for b in existing}
        beat_id = plan.get("beat_id")
        if not beat_id or beat_id in existing_ids:
            n = beat_index + 1
            beat_id = f"{scene_id}_b{n}"
            while beat_id in existing_ids:
                n += 1
                beat_id = f"{scene_id}_b{n}"
        node_id = f"{snapshot_id}:beat:{beat_id}"
        physical = plan.get("physical_constraints")
        physical_str = physical if isinstance(physical, str) else json.dumps(physical or [])
        final_plan = {
            **plan,
            "beat_id": beat_id,
            "beat_index": beat_index,
            "pad_target": pad_target_dict,
            "behavioral_constraint": behavioural_constraint,
        }
        sqlite_db.upsert_beat_plan(
            db_path,
            beat_id=beat_id,
            scene_id=scene_id,
            beat_index=beat_index,
            status="planned",
            snapshot_id=snapshot_id,
            node_id=node_id,
            parent_node_id=scene_node_id,
            ordering=beat_index,
            title=plan.get("immediate_objective"),
            pad_constraint=behavioural_constraint,
            immediate_objective=plan.get("immediate_objective"),
            physical_constraints=physical_str,
        )
        sqlite_db.upsert_planning_node(
            db_path,
            node_id=node_id,
            snapshot_id=snapshot_id,
            level=_LEVEL,
            status="planned",
            parent_id=scene_node_id,
            ordering=beat_index,
            title=plan.get("immediate_objective"),
            summary=plan.get("immediate_objective"),
            purpose=json.dumps(final_plan, sort_keys=True),
        )
        allocated["beat_id"] = beat_id
        allocated["beat_index"] = beat_index

    return _persist


def _scene_chapter(db_path: Any, scene_id: str) -> str:
    """Resolve a scene's chapter_id from its Scenes row ('' when absent)."""
    with sqlite_db.connect_db(db_path) as conn:
        row = conn.execute(
            "SELECT chapter_id FROM Scenes WHERE id = ?", (scene_id,)
        ).fetchone()
    return row["chapter_id"] if row else ""


def _scene_needs_more(db_path: Any, scene_id: str, config: Any) -> bool:
    """The volume-aware scene-stop predicate (conceptual spec: volume AND min units).

    A scene may close only when BOTH hold: its committed word volume has met the
    scene's ``word_budget`` (from the ``Scenes`` row; plan-time beats count 0 words)
    AND at least ``config.runtime.beats_per_scene_min`` beats exist. Otherwise the
    scene needs more beats. Both targets are config/store values — nothing hardcoded.
    """
    beats = sqlite_db.get_beats_for_scene_ordered(db_path, scene_id)
    committed_words = sum(int(b.get("word_count") or 0) for b in beats)
    chapter_id = _scene_chapter(db_path, scene_id)
    scene_row = next(
        (s for s in sqlite_db.get_scenes_for_chapter_ordered(db_path, chapter_id) if s["id"] == scene_id),
        None,
    )
    word_budget = int(scene_row.get("word_budget") or 0) if scene_row else 0
    min_beats = int(config.runtime.beats_per_scene_min)
    volume_met = committed_words >= word_budget
    min_units_met = len(beats) >= min_beats
    return not (volume_met and min_units_met)


async def node_plan_beat(
    state: dict[str, Any],
    *,
    decider: Any = None,
    registry: Any = None,
    adapt_fn: Any = None,
) -> dict[str, Any]:
    """Plan the active scene's next beat, PAD-ground it, persist it, and set scene-stop.

    Returns the (mutated) orchestrator state. ``decider``, ``registry``, and
    ``adapt_fn`` are injectable seams: ``decider=None`` builds the production
    M04-backed decider; ``adapt_fn`` is passed to the PAD translation ladder verbatim
    (``None`` → the LLM-free static floor; wiring the default small-tier adapter —
    ``fsm.pad_translation.build_pad_adapt_fn`` — is deferred until its prompt template
    is authored).
    """
    config = _resolve_config(state)
    db_path = state.get("sqlite_db_path", runtime.SQLITE_DB_PATH)
    event_log_path = state.get("event_log_path", runtime.EVENT_LOG_PATH)
    project_id = state["project_id"]
    trace = state.setdefault("planner_deliberation_trace", [])

    # 1. resolve the snapshot (idempotent) + the active scene anchor.
    execution_mode = state.get("planning_execution_mode") or "macro_outline_before_draft"
    snapshot_id = state.get("planning_snapshot_id") or f"snap_{project_id}"
    sqlite_db.create_planning_snapshot(
        db_path, snapshot_id=snapshot_id, project_id=project_id, mode=execution_mode
    )
    state["planning_snapshot_id"] = snapshot_id

    pointer = state.get("fsm_pointer")
    scene_id = getattr(pointer, "scene_id", None) if pointer is not None else None
    scene_node_id = f"{snapshot_id}:scene:{scene_id}" if scene_id else None
    scene_node = None
    if scene_node_id:
        scene_node = next(
            (
                n
                for n in sqlite_db.get_planning_nodes(db_path, snapshot_id, level="scene")
                if n["node_id"] == scene_node_id
            ),
            None,
        )
    scene_plan = _parse_purpose(scene_node)
    if not scene_plan.get("scene_function"):
        # The active scene has not been planned: the beat planner has nothing to
        # partition. Record and return — routing back through node_plan_scene is Build 14.
        trace.append(
            {
                "level": _LEVEL,
                "phase": "resolve_anchor",
                "outcome": "no_planned_scene",
                "scene_id": scene_id,
            }
        )
        return state

    # 2. compile constraints against the scene anchor (the beat node does not exist
    # yet) — bail to clarification on a hard-vs-hard contradiction.
    constraints = compile_planning_constraints(snapshot_id, scene_node_id, db_path=db_path)
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

    # 3. PAD Grounded Translation Pipeline: prior → proposed → EWMA smooth → ground.
    # alpha reuses the existing thresholds key; the static/fallback rungs are LLM-free.
    prior = _prior_pad(db_path, scene_id)
    proposed = _proposed_pad(scene_plan, prior)
    alpha = config.thresholds.pad_ewma_alpha
    pad_target = smooth_pad_target(prior, proposed, alpha)
    translation = await translate_pad_to_behaviour(
        pad_target,
        {
            "scene_function": scene_plan.get("scene_function"),
            "setting": scene_plan.get("setting"),
        },
        adapt_fn=adapt_fn,
    )
    pad_target_dict = {
        "pleasure": pad_target.pleasure,
        "arousal": pad_target.arousal,
        "dominance": pad_target.dominance,
    }
    trace.append(
        {
            "level": _LEVEL,
            "phase": "pad_grounding",
            "rung": translation.rung,  # 'fallback' here is the WARNING-worthy degrade
            "region_key": translation.region_key,
            "pad_target": pad_target_dict,
        }
    )

    # 4. assemble the beat base_context — exactly the six P1-template variables.
    project_metadata = state.get("project_metadata") or {}
    base_context = {
        "scene_plan": scene_plan,
        "planned_beats": _planned_beats(db_path, snapshot_id, scene_node_id),
        "pad_target": pad_target_dict,
        "pad_behavioral_constraint": translation.text,
        "genre": project_metadata.get("genre", ""),
        "continuity_facts": _continuity_facts(db_path),
    }
    continuity = {"continuity_facts": base_context["continuity_facts"]}

    # 5. registry + decider (deliberation model contact only through the decider seam).
    registry = registry or PlanningToolRegistry(db_path)
    if decider is None:
        decider = make_planner_decider(_NODE_NAME, base_context, config)

    outcome = await run_planner_loop(
        level=_LEVEL,
        snapshot_id=snapshot_id,
        target_node=scene_node_id,
        constraints=constraints,
        continuity=continuity,
        registry=registry,
        config=config,
        planner_decider=decider,
        deterministic_baseline=_build_beat_baseline(base_context),
    )
    if outcome.records:
        trace.extend(outcome.records)

    # 6. map the outcome to persistence + state (never persists an invalid plan).
    allocated: dict[str, Any] = {}
    prior_nodes = sqlite_db.get_planning_nodes(db_path, snapshot_id)
    result = persist_loop_outcome(
        outcome,
        level=_LEVEL,
        snapshot_id=snapshot_id,
        target_node=scene_node_id,
        persist_plan_fn=_beat_persist_plan_fn(
            db_path,
            snapshot_id,
            scene_id,
            scene_node_id,
            pad_target_dict,
            translation.text,
            allocated,
        ),
        db_path=db_path,
        event_log_path=event_log_path,
        prior_nodes=prior_nodes,
    )

    state["planning_snapshot_id"] = result.planning_snapshot_id
    state["active_planning_revision_id"] = result.active_planning_revision_id
    if result.outcome == "needs_clarification":
        state["planning_block_reason"] = result.planning_block_reason
    elif result.outcome == "escalate":
        # Loop exhausted caps + fallback ladder with no valid plan: signal recovery.
        state["planning_block_reason"] = _ESCALATION_BLOCK_REASON
    else:
        # finalized / fallback_baseline: advance the pointer to the new beat and set
        # the volume-aware scene-stop signal (routing on it is Build 14).
        if allocated.get("beat_id") and pointer is not None:
            state["fsm_pointer"] = pointer.model_copy(
                update={
                    "beat_id": allocated["beat_id"],
                    "beat_index": allocated["beat_index"],
                }
            )
        state["scene_needs_more"] = _scene_needs_more(db_path, scene_id, config)
    return state
