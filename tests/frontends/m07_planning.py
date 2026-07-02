"""Module: M05 (Hierarchical Planning Cascade)
Streamlit frontend for the real planner cascade over temp stores.

Drives the production planner nodes (global/arc/chapter/scene/beat) end to end —
compile constraints -> bounded deliberation loop -> persist. An **LLM mode** toggle
at the top selects the decider: "Mocked LLM" (default) injects a *scripted* decider
(an editable list of PlannerActions) so no live model or network is ever contacted;
"Real LLM" leaves the decider seam empty so the node builds the production
M04-backed decider (`make_planner_decider` -> `call_llm_structured`) against the
endpoints in `config.yaml`, with per-endpoint secrets (and optional base-URL
overrides) loaded from the repo `.env`. The page renders every intermediate
planning structure AND the agentic loop itself: the per-iteration deliberation
trace, validator results per configured required check, the fallback rung taken,
the persisted PlanningNode + PlanningRevision diff, the PAD grounding pipeline
(beat level; `adapt_fn=None` in both modes — the small-tier adapter's prompt
template is not authored yet), and the PlannerToolCallTrace rows including
rejected/degraded calls.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import streamlit as st
import yaml

from shared import (
    PLANNING_PROJECT_ID,
    PLANNING_SNAPSHOT_ID,
    ROOT,
    output_label,
    page_intro,
    reset_workspace,
    section_header,
    seed_narrative_data,
    seed_planning_data,
    two_section_help,
    workspace,
)
from core.config_loader import load_config
from fsm.nodes.node_plan_arc import node_plan_arc
from fsm.nodes.node_plan_beat import node_plan_beat
from fsm.nodes.node_plan_chapter import _select_chapter_target, node_plan_chapter
from fsm.nodes.node_plan_global import node_plan_global
from fsm.nodes.node_plan_scene import node_plan_scene
from fsm.pad_translation import compose_baseline_string, load_pad_regions
from fsm.planning_actions import PlannerAction
from fsm.planning_annotations import compile_planning_constraints
from fsm.planning_validators import REQUIRED_CHECK_REGISTRY
from fsm.state import FSM_Pointer
from memory import event_log, sqlite_db

_LEVELS = ("global", "arc", "chapter", "scene", "beat")

# LLM-mode toggle: mocked (scripted decider, zero network) is ALWAYS the default.
_LLM_MODES = ("Mocked LLM", "Real LLM")
_LLM_MODE_WIDGET_KEY = "m07_llm_mode_widget"
# Widget-keyed session values are dropped when the widget unmounts (e.g. switching
# pages inside run_all.py), so the chosen mode is mirrored into this plain session
# key, which survives, and fed back as the widget's index on remount.
_LLM_MODE_STORE_KEY = "m07_llm_mode_store"

_ENV_PATH = ROOT / ".env"
# The five endpoint roles named in config.yaml / .env.example; used only for the
# frontend's optional {ROLE}_BASE_URL overrides (api_key env handling is owned by
# core.config_loader.load_config, not re-implemented here).
_ENDPOINT_ROLES = ("planner", "drafter", "critic", "pad_translator", "craft_consultant")

_PLANNER_NODES = {
    "global": node_plan_global,
    "arc": node_plan_arc,
    "chapter": node_plan_chapter,
    "scene": node_plan_scene,
    "beat": node_plan_beat,
}

# The five fallback rungs the runbook reports, in ladder order. The loop only ever
# returns the four LoopOutcome statuses; `best_valid` and `baseline` are the two
# provenances of `fallback_baseline` (LoopOutcome.baseline_source).
_RUNGS = (
    ("finalized", "the loop accepted a validator-passed finalize from the decider"),
    ("best_valid", "caps exhausted; the best validated candidate seen was used"),
    ("baseline", "caps exhausted; the deterministic baseline validated and was used"),
    ("needs_clarification", "a hard-vs-hard conflict blocked planning; nothing persisted"),
    ("escalate", "no valid plan anywhere on the ladder; nothing persisted"),
)

# --- default scripted decider sequences --------------------------------------
# Each default finalizes with a plan shaped for the level's configured required
# checks (config.yaml planning.planner_required_checks), so a stock run demonstrates
# the `finalized` rung. The global default also demonstrates a think turn and a
# permitted tool call so the agentic loop has more than one step to render.

_VALID_DECIDERS: dict[str, list[dict[str, Any]]] = {
    "global": [
        {
            "action_type": "continue_deliberation",
            "rationale": (
                "Weigh the premise against the seeded open threads before "
                "committing to an arc split."
            ),
        },
        {"action_type": "call_tool", "tool_name": "read_open_threads", "tool_args": {}},
        {
            "action_type": "finalize_plan",
            "final_plan": {
                "premise": (
                    "Elena Marchetti revives The Paper Petal bookshop in Willow "
                    "Creek and falls slowly for carpenter Marcus Hale."
                ),
                "central_conflict": (
                    "Elena must choose between a safe city career and the fragile "
                    "new life (and love) she is building."
                ),
                "ending_target": (
                    "Elena stays; she and Marcus choose each other openly as the "
                    "shop finds its footing."
                ),
                "arcs": [
                    {
                        "arc_id": "arc-1",
                        "title": "The Paper Petal",
                        "function": (
                            "establish Elena in Willow Creek and kindle the "
                            "slow-burn romance"
                        ),
                        "word_allocation": 50000,
                    },
                    {
                        "arc_id": "arc-2",
                        "title": "Staying",
                        "function": (
                            "test the romance and the shop, then resolve both "
                            "through Elena's choice to stay"
                        ),
                        "word_allocation": 30000,
                    },
                ],
                "promises": [
                    {
                        "id": "promise-slow-burn",
                        "promise": (
                            "Elena and Marcus's slow-burn attraction will be answered."
                        ),
                        "payoff": (
                            "They speak their feelings under the finished shelves "
                            "in the final chapter."
                        ),
                    }
                ],
            },
        },
    ],
    # The arc default introduces a NEW arc (arc-2) with fresh chapter stubs, so it
    # never overwrites the seeded, already-planned chapter-1 node.
    "arc": [
        {"action_type": "call_tool", "tool_name": "read_open_threads", "tool_args": {}},
        {
            "action_type": "finalize_plan",
            "final_plan": {
                "arcs": [
                    {
                        "arc_id": "arc-2",
                        "title": "Staying",
                        "function": (
                            "test the romance and the shop, then resolve both "
                            "through Elena's choice to stay"
                        ),
                        "character_milestones": [
                            "Elena stops keeping a city exit plan",
                            "Marcus lets himself be seen",
                        ],
                        "chapters": [
                            {
                                "chapter_id": "arc-2_ch1",
                                "stub": "the ledger's verdict and the pulled-back hand",
                            },
                            {
                                "chapter_id": "arc-2_ch2",
                                "stub": "an apology shaped like a bookcase; the choice to stay",
                            },
                        ],
                    }
                ],
                "milestones": [
                    {"arc_id": "arc-2", "label": "the misunderstanding lands", "tension": 2},
                    {"arc_id": "arc-2", "label": "the shop nearly fails", "tension": 3},
                    {"arc_id": "arc-2", "label": "Elena chooses to stay", "tension": 4},
                ],
                "thread_distribution": [
                    {"thread_id": "thread-slow-burn", "in_arcs": ["arc-2"]},
                    {"thread_id": "thread-shop-finances", "in_arcs": ["arc-2"]},
                ],
            },
        },
    ],
    "chapter": [
        {
            "action_type": "finalize_plan",
            "final_plan": {
                "dramatic_function": (
                    "the misunderstanding pulls Marcus back as the shop's finances strain"
                ),
                "expected_emotional_shift": "warmth curdles into doubt; Elena's resolve wavers",
                "pacing": "tightening; shorter scenes as pressure mounts",
                "obligations": {
                    "thread_obligations": [
                        {
                            "thread_id": "thread-shop-finances",
                            "required_progress": "the ledger's red tide becomes undeniable",
                        }
                    ],
                    "causal_prerequisites": [
                        "Rosa's visit has left Marcus doubting where he stands"
                    ],
                    "causal_deliverables": [
                        "Elena discovers how deep the shop's losses run"
                    ],
                },
                "annotation_outcomes": {},
                "scene_planning_constraints": [
                    "Marcus withdraws without a confrontation scene"
                ],
            },
        }
    ],
    "scene": [
        {
            "action_type": "finalize_plan",
            "final_plan": {
                "scene_function": "Elena finds the ledger deep in the red and doubts her choice",
                "setting": "The Paper Petal back office, late night",
                "participants": ["char-elena"],
                "entry_state": (
                    "the shop is closed; Elena sits down with the ledger after a slow week"
                ),
                "exit_state": (
                    "Elena has seen the full depth of the losses and resolves to tell no one yet"
                ),
                "conflict_turn": "hope collides with arithmetic",
                "asserted_facts": [],
                "word_budget": 450,
                "pad_target": {"pleasure": -0.35, "arousal": 0.55, "dominance": 0.25},
            },
        }
    ],
    # The harness owns beat affect: whatever pad_target/behavioral_constraint the
    # decider claims, the persisted values are forced to the node-computed ones.
    "beat": [
        {
            "action_type": "finalize_plan",
            "final_plan": {
                "immediate_objective": "Elena opens the ledger and confronts the numbers",
                "physical_constraints": [
                    "present: char-elena",
                    "location: the back-office desk at The Paper Petal",
                ],
                "entry_condition": "the scene opens on its established entry state",
                "exit_condition": (
                    "the ledger's verdict is on the table and Elena chooses silence"
                ),
                "pad_target": {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0},
                "behavioral_constraint": (
                    "placeholder; the harness replaces this with the PAD-grounded string"
                ),
                "asserted_facts": [],
            },
        }
    ],
}

# A finalize that passes no level's configured checks: schema accepts a non-empty
# dict, but every level-specific check (arc_coverage / escalation / chapter_function /
# scene_function / draftability, ...) fails. When the sequence is exhausted the
# scripted decider raises, so the loop burns wasted turns to its cap and the
# fallback ladder (best_valid -> baseline -> escalate) takes over.
_INVALID_DECIDER: list[dict[str, Any]] = [
    {
        "action_type": "finalize_plan",
        "final_plan": {"placeholder": "an intentionally invalid plan for the ladder demo"},
    }
]

_DECIDER_PRESETS = ("valid finalize (default)", "always-invalid finalize")


def _default_decider_json(level: str, preset: str) -> str:
    actions = _INVALID_DECIDER if preset == _DECIDER_PRESETS[1] else _VALID_DECIDERS[level]
    return json.dumps(actions, indent=2)


def _planning_visual_config(execution_mode: str, approval_mode: str) -> SimpleNamespace:
    """Config-shaped object for the planner nodes, read from config.yaml.

    Mirrors `visual_config()`: real config.yaml values (planning caps, required
    checks, pad_ewma_alpha, beats_per_scene_min) without requiring endpoint secrets —
    the decider is injected, so `config.endpoints` is never touched. The execution
    mode / approval mode selected in the UI override the yaml values so both paths
    are exercisable.
    """
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
    planning = dict(raw["planning"])
    planning["execution_mode"] = execution_mode
    planning["approval_mode"] = approval_mode
    return SimpleNamespace(
        planning=SimpleNamespace(**planning),
        thresholds=SimpleNamespace(**dict(raw["thresholds"])),
        runtime=SimpleNamespace(**dict(raw["runtime"])),
    )


def _load_env_file(path: Any) -> list[str]:
    """Load KEY=VALUE lines from ``path`` into ``os.environ`` (existing env wins).

    Minimal dotenv semantics — blank lines and ``#`` comments skipped, surrounding
    quotes stripped — so the frontend needs no extra dependency. Returns the keys it
    actually set, for display. Variables already present in the environment are never
    overwritten (the shell stays authoritative, matching production expectations).
    """
    loaded: list[str] = []
    path = Path(path)
    if not path.exists():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def _build_real_config() -> tuple[Any, dict[str, Any]]:
    """Build the real validated AppConfig for Real-LLM runs, plus a status dict.

    Loads ``.env`` (per-endpoint ``{ROLE}_API_KEY`` secrets, exactly the variables
    ``.env.example`` documents), then runs the production ``core.config_loader
    .load_config`` — strict validation included, so a missing or empty secret fails
    here with the loader's own error rather than mid-run. ``{ROLE}_BASE_URL`` env
    values, when set, override the yaml ``base_url`` afterwards (a frontend
    convenience mirroring ``.env.example``; the production loader keeps base_url in
    config.yaml).
    """
    loaded_keys = _load_env_file(_ENV_PATH)
    try:
        config = load_config(ROOT / "config.yaml")
    except Exception as exc:
        hint = (
            f" — Real LLM mode needs the per-endpoint secrets from .env "
            f"({_ENV_PATH}{'' if _ENV_PATH.exists() else ' does not exist; copy .env.example'}). "
        )
        raise RuntimeError(f"config load failed{hint}{exc}") from exc
    overridden: list[str] = []
    for role in _ENDPOINT_ROLES:
        env_url = os.environ.get(f"{role.upper()}_BASE_URL")
        if env_url:
            getattr(config.endpoints, role).base_url = env_url
            overridden.append(role)
    info = {
        "env_file": str(_ENV_PATH),
        "env_file_found": _ENV_PATH.exists(),
        "env_keys_loaded": loaded_keys,
        "base_url_overrides": overridden,
        "planner_endpoint": {
            "base_url": config.endpoints.planner.base_url,
            "model_name": config.endpoints.planner.model_name,
            "grammar_constraint_strategy": config.endpoints.planner.grammar_constraint_strategy,
        },
    }
    return config, info


def _make_scripted_decider(actions: list[PlannerAction]):
    """One PlannerAction per loop turn, in order; raises when exhausted.

    An exhausted (raising) decider is exactly what the loop treats as a wasted,
    non-finalizing turn, so the run degrades through the real fallback ladder
    instead of crashing — the same seam production uses for a flaky model.
    """
    remaining = list(actions)

    def decider(loop_state: Any) -> PlannerAction:
        if not remaining:
            raise RuntimeError("scripted decider sequence exhausted")
        return remaining.pop(0)

    return decider


def _resolve_compile_target(
    level: str, db_path: str, snapshot_id: str, pointer: FSM_Pointer
) -> str | None:
    """The node_id each level's node compiles constraints against (production logic).

    Mirrors the nodes exactly: global/arc anchor on the global node; chapter uses the
    production `_select_chapter_target` sweep; scene/beat anchor on the pointer's
    chapter/scene PlanningNode.
    """
    if level == "global":
        return f"{snapshot_id}:global"
    if level == "arc":
        nodes = sqlite_db.get_planning_nodes(db_path, snapshot_id, level="global")
        return nodes[0]["node_id"] if nodes else f"{snapshot_id}:global"
    if level == "chapter":
        target = _select_chapter_target(db_path, snapshot_id, pointer)
        return target["node_id"] if target else None
    if level == "scene":
        return f"{snapshot_id}:chapter:{pointer.chapter_id}" if pointer.chapter_id else None
    return f"{snapshot_id}:scene:{pointer.scene_id}" if pointer.scene_id else None


def _count_events(path: Any) -> int:
    try:
        return sum(1 for _ in event_log.iter_events(path))
    except ValueError:
        return 0


def _run_level(
    level: str,
    paths: dict,
    pointer: FSM_Pointer,
    execution_mode: str,
    approval_mode: str,
    actions: list[PlannerAction] | None,
    llm_mode: str = _LLM_MODES[0],
) -> dict[str, Any]:
    """Drive one real planner-node invocation and gather everything the runbook shows.

    ``llm_mode`` selects the decider seam: "Mocked LLM" injects the scripted
    ``actions`` sequence (no model, no network); "Real LLM" passes ``decider=None``
    so the node builds the production M04-backed decider against the validated
    real config (endpoints + secrets from config.yaml/.env). In both modes the PAD
    ladder runs ``adapt_fn=None`` (its adapter template is not authored yet).
    """
    db_path = str(paths["db"])
    log_path = str(paths["event_log"])
    real_llm = llm_mode == _LLM_MODES[1]
    llm_info: dict[str, Any] | None = None
    if real_llm:
        config, llm_info = _build_real_config()
    else:
        config = _planning_visual_config(execution_mode, approval_mode)

    state: dict[str, Any] = {
        "project_id": PLANNING_PROJECT_ID,
        "fsm_pointer": pointer,
        "app_config": config,
        "sqlite_db_path": db_path,
        "event_log_path": log_path,
        "planning_snapshot_id": PLANNING_SNAPSHOT_ID,
        "planning_execution_mode": execution_mode,
        "approval_mode": approval_mode,
        "project_metadata": {
            "genre": "small-town slow-burn romance",
            "premise_seed": (
                "Elena Marchetti revives The Paper Petal bookshop in Willow Creek "
                "and falls slowly for carpenter Marcus Hale."
            ),
            "target_word_count": getattr(config.runtime, "word_count_target", 0),
        },
    }

    pre_nodes = {
        n["node_id"]: n for n in sqlite_db.get_planning_nodes(db_path, PLANNING_SNAPSHOT_ID)
    }
    pre_event_count = _count_events(log_path)
    target = _resolve_compile_target(level, db_path, PLANNING_SNAPSHOT_ID, pointer)

    node_fn = _PLANNER_NODES[level]
    kwargs: dict[str, Any] = {}
    if not real_llm:
        # Mocked mode: the scripted sequence replaces the model behind the seam.
        kwargs["decider"] = _make_scripted_decider(actions or [])
    # Real mode passes no decider: the node builds the production M04-backed decider
    # (make_planner_decider -> call_llm_structured on config.endpoints.planner).
    if level == "beat":
        kwargs["adapt_fn"] = None  # static PAD floor; adapter template not authored yet
    result_state = asyncio.run(node_fn(state, **kwargs))

    # Re-compile the constraints the node just planned against, for display. The
    # compiler is the production one; a conflict the node hit is now surfaced as a
    # pre-existing unresolved needs_clarification, which is the same blocking signal.
    constraints: dict[str, Any] | None = None
    constraints_error: str | None = None
    if target is not None:
        try:
            constraints = compile_planning_constraints(
                PLANNING_SNAPSHOT_ID, target, db_path=db_path
            )
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI, never crashes it
            constraints_error = f"{type(exc).__name__}: {exc}"

    all_events = list(event_log.iter_events(log_path))
    new_events = all_events[pre_event_count:]
    commit_event = next(
        (e for e in reversed(new_events) if e.get("event_type") == "planning_commit"),
        None,
    )

    post_nodes = sqlite_db.get_planning_nodes(db_path, PLANNING_SNAPSHOT_ID)
    changed_nodes = [
        n for n in post_nodes if pre_nodes.get(n["node_id"]) != n
    ]

    revision = None
    if commit_event is not None:
        revision = next(
            (
                r
                for r in sqlite_db.get_revisions_for_snapshot(db_path, PLANNING_SNAPSHOT_ID)
                if r["revision_id"] == commit_event.get("revision_id")
            ),
            None,
        )

    return {
        "level": level,
        "target": target,
        "state": result_state,
        "config": config,
        "llm_mode": llm_mode,
        "llm_info": llm_info,
        "constraints": constraints,
        "constraints_error": constraints_error,
        "commit_event": commit_event,
        "new_events": new_events,
        "changed_nodes": changed_nodes,
        "revision": revision,
        "rung": _taken_rung(result_state, commit_event),
    }


def _taken_rung(result_state: dict[str, Any], commit_event: dict | None) -> str | None:
    """Map the run's outcome to one of the five reported fallback rungs."""
    block = result_state.get("planning_block_reason")
    if block == "unresolved_hard_conflict":
        return "needs_clarification"
    if block == "planning_escalation":
        return "escalate"
    if commit_event is not None:
        if commit_event.get("outcome") == "finalized":
            return "finalized"
        if commit_event.get("outcome") == "fallback_baseline":
            return (
                "best_valid"
                if commit_event.get("baseline_source") == "best_valid"
                else "baseline"
            )
    return None  # nothing planned (no anchor / already complete) or awaiting approval


# --- rendering ----------------------------------------------------------------


def _trace_rows(trace: list[dict]) -> list[dict[str, Any]]:
    """Flatten the deliberation trace into uniform rows for the step table."""
    rows: list[dict[str, Any]] = []
    for i, record in enumerate(trace):
        if "phase" in record:
            kind, name = "phase", record["phase"]
            outcome = str(record.get("outcome") or record.get("rung") or "")
        else:
            kind = "action"
            name = str(record.get("action_type"))
            if name == "call_tool":
                if record.get("reused"):
                    outcome = "reused cached result"
                elif record.get("accepted"):
                    outcome = str(record.get("outcome") or "executed")
                else:
                    outcome = "refused"
            elif name == "revise_plan":
                outcome = "valid" if record.get("validation_passes") else "invalid"
                if record.get("early_finalize"):
                    outcome += " (early finalize)"
                elif record.get("no_change"):
                    outcome += " (no change)"
            elif name == "finalize_plan":
                outcome = "accepted" if record.get("accepted") else "rejected"
            elif name == "raise_conflict":
                outcome = f"{record.get('conflicts', 0)} conflict(s)"
            elif name == "continue_deliberation":
                outcome = "think turn"
            else:
                outcome = "wasted turn"
        detail = {
            k: v
            for k, v in record.items()
            if k not in ("phase", "action_type", "loop_index")
        }
        rows.append(
            {
                "step": i,
                "loop": record.get("loop_index", ""),
                "kind": kind,
                "what": name,
                "outcome": outcome,
                "detail": json.dumps(detail, default=str),
            }
        )
    return rows


def _final_validation_record(trace: list[dict]) -> dict | None:
    """The last trace record that carried a validation verdict."""
    return next(
        (r for r in reversed(trace) if "validation_passes" in r),
        None,
    )


def _persisted_plan(run: dict[str, Any]) -> dict | None:
    """Parse the run-level node's persisted full-plan JSON, when one exists.

    global/chapter/scene/beat persist the full validated plan into their
    PlanningNode `purpose`; the arc level splits the plan across per-arc nodes, so
    it has no single full-plan node and returns None (the trace-based validator
    summary still applies).
    """
    level = run["level"]
    state = run["state"]
    db_path = state.get("sqlite_db_path")
    pointer = state.get("fsm_pointer")
    if level == "global":
        node_id = f"{PLANNING_SNAPSHOT_ID}:global"
    elif level == "chapter":
        node_id = run["target"]
    elif level == "scene":
        scene_id = getattr(pointer, "scene_id", None)
        node_id = f"{PLANNING_SNAPSHOT_ID}:scene:{scene_id}" if scene_id else None
    elif level == "beat":
        beat_id = getattr(pointer, "beat_id", None)
        node_id = f"{PLANNING_SNAPSHOT_ID}:beat:{beat_id}" if beat_id else None
    else:
        return None
    if not node_id:
        return None
    node = next(
        (
            n
            for n in sqlite_db.get_planning_nodes(db_path, PLANNING_SNAPSHOT_ID)
            if n["node_id"] == node_id
        ),
        None,
    )
    if node is None:
        return None
    try:
        plan = json.loads(node.get("purpose") or "{}")
    except ValueError:
        return None
    return plan if isinstance(plan, dict) and plan else None


def _validator_rows(run: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Per-configured-check pass/fail rows plus a caption naming the source.

    Preferred source: re-run each configured required check live (the production
    REQUIRED_CHECK_REGISTRY callables) against the persisted plan. When no single
    full plan was persisted (arc level, or a needs_clarification/escalate run), fall
    back to the loop's final validation verdict from the trace.
    """
    level = run["level"]
    config = run["config"]
    names = list(config.planning.planner_required_checks[level])
    plan = _persisted_plan(run)
    constraints = run.get("constraints") or {}
    if run["rung"] in ("finalized", "best_valid", "baseline") and plan is not None:
        rows = []
        for name in names:
            fn = REQUIRED_CHECK_REGISTRY.get(name)
            if fn is None:
                rows.append({"check": name, "passes": None, "reason": "not registered"})
                continue
            result = fn(plan, level, constraints, {})
            rows.append(
                {
                    "check": name,
                    "passes": result.passes,
                    "reason": json.dumps(result.details.get(name, {}), default=str),
                }
            )
        return rows, "re-run live against the persisted plan (REQUIRED_CHECK_REGISTRY)"
    record = _final_validation_record(run["state"].get("planner_deliberation_trace", []))
    if record is None:
        return (
            [{"check": n, "passes": None, "reason": "no validation ran"} for n in names],
            "no validation-bearing turn occurred in this run",
        )
    failed = set(record.get("failed_checks", []))
    rows = [
        {
            "check": name,
            "passes": name not in failed,
            "reason": "failed in the loop's final validation" if name in failed else "passed",
        }
        for name in names
    ]
    return rows, "from the loop's final validation verdict (per-check reasons not retained)"


def _render_status(run: dict[str, Any]) -> None:
    state = run["state"]
    section_header(
        "Run outcome",
        two_section_help(
            "The headline result of the planner run: which fallback rung the loop landed on, which snapshot and revision are now active, and any block or readiness flags the node set on the orchestrator state.",
            "Rung is derived from planning_block_reason and the planning_commit event; the rest reads the returned state.",
        ),
    )
    cols = st.columns(5)
    cols[0].metric(
        "Rung taken",
        run["rung"] or "none",
        help=two_section_help(
            "Which rung of the ladder produced (or blocked) the plan: finalized, best_valid, baseline, needs_clarification, or escalate. 'none' means the node had nothing to plan (no anchor, or the scope was already complete).",
            "finalized/fallback_baseline come from the planning_commit event; the two block reasons come from state.planning_block_reason.",
        ),
    )
    cols[1].metric("Level", run["level"])
    cols[2].metric(
        "Active revision",
        str(state.get("active_planning_revision_id") or "—"),
        help=two_section_help(
            "The PlanningRevision now marked active on the snapshot. A persisted run advances it; a blocked run leaves it unchanged.",
            "PlanningSnapshot.active_revision_id after the run.",
        ),
    )
    cols[3].metric(
        "Block reason",
        str(state.get("planning_block_reason") or "—"),
        help=two_section_help(
            "Set when planning is blocked: unresolved_hard_conflict (clarification needed), planning_escalation (no valid plan), or awaiting_macro_approval (approval gate armed).",
            "state.planning_block_reason as set by the node.",
        ),
    )
    flags = {
        "macro_outline_ready": state.get("macro_outline_ready"),
        "awaiting_planning_approval": state.get("awaiting_planning_approval"),
        "scene_needs_more": state.get("scene_needs_more"),
    }
    cols[4].metric(
        "LLM mode",
        run.get("llm_mode", _LLM_MODES[0]),
        help=two_section_help(
            "Which decider produced this run: the scripted mock (deterministic, offline) or the real M04-backed decider making live structured-output calls to the planner endpoint.",
            "Mocked: injected action sequence. Real: make_planner_decider -> call_llm_structured; endpoint details in the expander below.",
        ),
    )
    with st.expander("Returned orchestrator-state fields"):
        pointer = state.get("fsm_pointer")
        st.json(
            {
                "fsm_pointer": pointer.model_dump() if pointer is not None else None,
                "planning_snapshot_id": state.get("planning_snapshot_id"),
                "active_planning_revision_id": state.get("active_planning_revision_id"),
                "planning_block_reason": state.get("planning_block_reason"),
                **{k: v for k, v in flags.items() if v is not None},
                **(
                    {"real_llm_endpoint": run["llm_info"]}
                    if run.get("llm_info")
                    else {}
                ),
            }
        )


def _render_trace(run: dict[str, Any]) -> None:
    trace = run["state"].get("planner_deliberation_trace", [])
    section_header(
        "Deliberation trace — the agentic loop",
        two_section_help(
            "Every turn of the bounded planner loop, in order: think turns, tool calls (with permission/cap refusals), plan revisions, finalize attempts and whether the harness validators accepted them, and any raised conflicts. Phase rows show the node's pre/post-loop steps (constraint compilation, PAD grounding, anchor resolution).",
            "state.planner_deliberation_trace: the node's phase records plus LoopOutcome.records from fsm/planning_loop.py.",
        ),
    )
    if not trace:
        st.info("The node recorded no deliberation trace (nothing to plan).")
        return
    st.dataframe(_trace_rows(trace), width='stretch', hide_index=True)
    with st.expander("Raw trace records (JSON)"):
        st.json(trace)


def _render_constraints_validators_rung(run: dict[str, Any]) -> None:
    col_a, col_b, col_c = st.columns([1.2, 1.2, 0.8])
    with col_a:
        output_label(
            "Compiled constraints",
            two_section_help(
                "The planner-ready package the annotation compiler built for this run's target node: hard requirements ranked above soft preferences, pinned-node protections, and any hard-vs-hard conflicts (which block planning entirely).",
                "compile_planning_constraints(snapshot, target) re-run after the node for display; conflicts the node hit appear as unresolved needs_clarification ids.",
            ),
        )
        if run["target"] is None:
            st.info("No compile target resolved for this level (nothing to plan against).")
        elif run["constraints_error"]:
            st.error(run["constraints_error"])
        else:
            st.json(run["constraints"])
    with col_b:
        output_label(
            "Validator results (required checks)",
            two_section_help(
                "Pass/fail for each validator this level is configured to require (config.yaml planning.planner_required_checks). These deterministic checks — not the model's self_check — are the only gate a finalize can pass.",
                "Each configured check re-run via REQUIRED_CHECK_REGISTRY against the persisted plan when one exists; otherwise the loop's final verdict.",
            ),
        )
        try:
            rows, source = _validator_rows(run)
            st.dataframe(rows, width='stretch', hide_index=True)
            st.caption(f"Source: {source}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"{type(exc).__name__}: {exc}")
    with col_c:
        output_label(
            "Fallback ladder",
            two_section_help(
                "The harness degrades in strict order and never persists an invalid plan: accept a validated finalize; else reuse the best validated candidate; else validate the deterministic baseline; a raised hard conflict stops everything for clarification; and escalation persists nothing.",
                "finalized -> best_valid -> baseline; needs_clarification short-circuits; escalate is the empty-handed end of the ladder.",
            ),
        )
        for rung, meaning in _RUNGS:
            marker = "🟢" if run["rung"] == rung else "⚪"
            st.markdown(f"{marker} **{rung}**")
            st.caption(meaning)


def _render_persisted(run: dict[str, Any]) -> None:
    section_header(
        "Persisted planning structures",
        two_section_help(
            "What the run actually wrote: the PlanningNode rows it created or changed (the validated plan JSON lives in `purpose`), and the PlanningRevision recording the change as a diff — history is preserved, never overwritten.",
            "Node changes are pre/post get_planning_nodes() comparisons; the revision is looked up by the planning_commit event's revision_id.",
        ),
    )
    col_a, col_b = st.columns(2)
    with col_a:
        output_label(
            "PlanningNode rows created/changed",
            two_section_help(
                "Every proposal-surface node this run added or modified. A blocked run (clarification/escalate) changes no plan content.",
                "Rows from get_planning_nodes() that differ from the pre-run snapshot.",
            ),
        )
        if run["changed_nodes"]:
            st.json(run["changed_nodes"])
        else:
            st.info("No PlanningNode rows changed in this run.")
    with col_b:
        output_label(
            "PlanningRevision + diff_json",
            two_section_help(
                "The revision written for this commit: its parent revision (the history chain) and the structured diff of added/removed/modified nodes plus annotation outcomes.",
                "insert_planning_revision also advanced the snapshot's active_revision_id; diff_json is compute_revision_diff output.",
            ),
        )
        revision = run["revision"]
        if revision is None:
            st.info("No revision was written (nothing persisted).")
        else:
            shown = dict(revision)
            try:
                shown["diff_json"] = json.loads(shown.get("diff_json") or "{}")
            except ValueError:
                pass
            st.json(shown)
    if run["commit_event"] is not None:
        with st.expander("planning_commit event (append-only log mirror)"):
            st.json(run["commit_event"])


def _render_pad(run: dict[str, Any]) -> None:
    trace = run["state"].get("planner_deliberation_trace", [])
    pad_record = next(
        (r for r in trace if r.get("phase") == "pad_grounding"), None
    )
    if run["level"] != "beat" or pad_record is None:
        return
    section_header(
        "PAD grounded translation (beat level)",
        two_section_help(
            "How the beat's emotional target was computed with no LLM: the scene's committed PAD history gives the prior, the scene plan proposes a target, EWMA smoothing blends them, and the smoothed coordinate is looked up in the static region table to produce the behavioural-constraint string the drafter will receive.",
            "prior = mean latest per-character scene PAD (neutral origin if none); smoothed = alpha*proposed + (1-alpha)*prior with alpha = thresholds.pad_ewma_alpha; adapt_fn=None means the static rung is the floor.",
        ),
    )
    pad = pad_record.get("pad_target") or {}
    region_key = pad_record.get("region_key")
    cols = st.columns(6)
    cols[0].metric("pleasure", f"{pad.get('pleasure', 0):.2f}")
    cols[1].metric("arousal", f"{pad.get('arousal', 0):.2f}")
    cols[2].metric("dominance", f"{pad.get('dominance', 0):.2f}")
    cols[3].metric(
        "Region",
        str(region_key),
        help=two_section_help(
            "The octant (or neutral zone) of PAD space the smoothed target falls in — the key into the static pad_regions.json lookup table.",
            "region_key_for_pad(): sign per axis, or 'neutral' inside the dead-zone band.",
        ),
    )
    cols[4].metric(
        "Rung",
        str(pad_record.get("rung")),
        help=two_section_help(
            "Which rung of the translation ladder produced the string: 'static' (no adapter — this page always runs adapt_fn=None), 'adapted', or 'fallback' (adapter tried and failed).",
            "BehaviourTranslation.rung from translate_pad_to_behaviour().",
        ),
    )
    alpha = getattr(run["config"].thresholds, "pad_ewma_alpha", None)
    cols[5].metric(
        "EWMA alpha",
        str(alpha),
        help=two_section_help(
            "How strongly the scene's proposed affect pulls the smoothed target away from the prior. Read from config.yaml, never hardcoded.",
            "config.thresholds.pad_ewma_alpha passed to smooth_pad_target().",
        ),
    )
    col_a, col_b = st.columns(2)
    with col_a:
        output_label(
            "Region entry + static baseline string",
            two_section_help(
                "The raw pad_regions.json entry for the resolved region and the composed baseline behavioural string (behaviour + physical cues + register).",
                "load_pad_regions()['regions'][region_key] and compose_baseline_string(region_key).",
            ),
        )
        try:
            st.json(load_pad_regions().get("regions", {}).get(region_key, {}))
            st.code(compose_baseline_string(region_key))
        except Exception as exc:  # noqa: BLE001
            st.error(f"{type(exc).__name__}: {exc}")
    with col_b:
        output_label(
            "Prior PAD rows + persisted behavioural constraint",
            two_section_help(
                "The committed per-character PAD rows whose mean formed the prior, and the behavioural-constraint string the harness actually persisted onto the beat (it overrides whatever the decider claimed).",
                "get_latest_pad_for_scene(scene_id) and the beat PlanningNode's purpose.behavioral_constraint.",
            ),
        )
        try:
            scene_id = getattr(run["state"].get("fsm_pointer"), "scene_id", None)
            rows = sqlite_db.get_latest_pad_for_scene(
                run["state"].get("sqlite_db_path"), scene_id
            )
            st.dataframe(rows, width='stretch', hide_index=True)
        except Exception as exc:  # noqa: BLE001
            st.error(f"{type(exc).__name__}: {exc}")
        plan = _persisted_plan(run)
        if plan and plan.get("behavioral_constraint"):
            st.code(str(plan["behavioral_constraint"]))


def _render_tool_traces(run: dict[str, Any]) -> None:
    section_header(
        "PlannerToolCallTrace rows",
        two_section_help(
            "Every tool call the registry saw during deliberation — executed, degraded (a not-yet-built store answering honestly 'unavailable'), and rejected (unknown tool, or a tool outside this level's permission matrix). Refused-over-cap calls never reach the registry, so they appear only in the trace above.",
            "insert_planner_tool_call_trace rows via get_traces_for_snapshot(); args/results are length-bounded and secret-redacted before persistence.",
        ),
    )
    try:
        rows = sqlite_db.get_traces_for_snapshot(
            run["state"].get("sqlite_db_path"), PLANNING_SNAPSHOT_ID
        )
        if rows:
            st.dataframe(rows, width='stretch', hide_index=True)
        else:
            st.info("No tool calls have been traced for this snapshot yet.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"{type(exc).__name__}: {exc}")


def _render_store_expander(db_path: str) -> None:
    with st.expander("Raw planning store (snapshot / nodes / annotations / revisions)"):
        try:
            st.json(
                {
                    "snapshot": sqlite_db.get_planning_snapshot(db_path, PLANNING_SNAPSHOT_ID),
                }
            )
            output_label(
                "PlanningNode",
                two_section_help(
                    "All proposal-surface nodes in the snapshot, across every level.",
                    "get_planning_nodes(db, snapshot_id).",
                ),
            )
            st.dataframe(
                sqlite_db.get_planning_nodes(db_path, PLANNING_SNAPSHOT_ID),
                width='stretch',
                hide_index=True,
            )
            output_label(
                "PlanningAnnotation",
                two_section_help(
                    "The user's structured node annotations, including any the compiler marked needs_clarification after detecting a hard-vs-hard conflict.",
                    "get_annotations_for_snapshot(db, snapshot_id).",
                ),
            )
            st.dataframe(
                sqlite_db.get_annotations_for_snapshot(db_path, PLANNING_SNAPSHOT_ID),
                width='stretch',
                hide_index=True,
            )
            output_label(
                "PlanningRevision",
                two_section_help(
                    "The full revision history, newest first — revisions are appended, never overwritten.",
                    "get_revisions_for_snapshot(db, snapshot_id).",
                ),
            )
            st.dataframe(
                sqlite_db.get_revisions_for_snapshot(db_path, PLANNING_SNAPSHOT_ID),
                width='stretch',
                hide_index=True,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"{type(exc).__name__}: {exc}")


def render() -> None:
    paths = workspace()
    page_intro(
        "M07 Hierarchical Planning Cascade",
        two_section_help(
            "This page runs the real five-level planner cascade — the bounded 'the model proposes, the harness disposes' loop — with a scripted decider standing in for the model, so you can watch constraint compilation, every deliberation turn, validator gating, the fallback ladder, and what actually gets persisted, all against throwaway temp stores.",
            "Drives node_plan_{global,arc,chapter,scene,beat}; the LLM-mode toggle picks the scripted decider (default) or the production M04-backed one; PAD runs with adapt_fn=None.",
        ),
    )

    # --- LLM mode toggle (mocked is the default) --------------------------------
    stored_mode = st.session_state.get(_LLM_MODE_STORE_KEY, _LLM_MODES[0])
    if stored_mode not in _LLM_MODES:
        stored_mode = _LLM_MODES[0]
    mode_cols = st.columns([1, 3])
    llm_mode = mode_cols[0].selectbox(
        "LLM mode",
        _LLM_MODES,
        index=_LLM_MODES.index(stored_mode),
        key=_LLM_MODE_WIDGET_KEY,
        help=two_section_help(
            "Mocked LLM (the default) feeds the loop your scripted PlannerAction sequence — fully deterministic, zero network. Real LLM bypasses the mock: the node builds its production decider and every deliberation turn is an actual structured-output call to the planner endpoint, with secrets (and optional base-URL overrides) loaded from the repo .env file.",
            "Mocked: decider=_make_scripted_decider(...). Real: decider=None -> make_planner_decider -> call_llm_structured on config.endpoints.planner; config via core.config_loader.load_config after reading .env.",
        ),
    )
    st.session_state[_LLM_MODE_STORE_KEY] = llm_mode
    real_llm = llm_mode == _LLM_MODES[1]
    with mode_cols[1]:
        if real_llm:
            if _ENV_PATH.exists():
                st.warning(
                    "Real LLM mode: deliberation turns will call the configured planner "
                    "endpoint. Secrets load from .env; an unreachable endpoint degrades "
                    "through retries to the fallback ladder (slow, not an error).",
                    icon="⚡",
                )
            else:
                st.error(
                    f"Real LLM mode needs `{_ENV_PATH}` (copy `.env.example` and fill the "
                    "`*_API_KEY` values). Runs will fail config validation until it exists.",
                    icon="🔑",
                )
        else:
            st.caption(
                "Mocked LLM: the scripted decider below stands in for the model — "
                "no credentials, no network."
            )

    action_cols = st.columns([1, 1, 2])
    if action_cols[0].button(
        "Seed planning stores",
        width='stretch',
        help=two_section_help(
            "Writes the narrative seed (arcs, scenes, committed beats with PAD history) and then layers a deterministic planning surface on top: a snapshot, planned global/arc/chapter-1/scene-1 nodes, unplanned chapter-2/chapter-3 stubs, a soft tone preference, a hard constraint, and a deliberate hard-vs-hard conflict pair on chapter-2.",
            "seed_narrative_data(db, provisional) then seed_planning_data(db, event_log_path=...); both idempotent, real store APIs only.",
        ),
    ):
        try:
            seed_narrative_data(paths["db"], paths["provisional"])
            seed_planning_data(paths["db"], event_log_path=paths["event_log"])
            st.success("Seeded narrative + planning stores.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"{type(exc).__name__}: {exc}")
    if action_cols[1].button(
        "Reset temp workspace",
        width='stretch',
        help=two_section_help(
            "Deletes the temporary stores and the last run's outputs. Use this to clear a marked conflict or a half-planned cascade and start fresh.",
            "Drops the session temp dir (shared.reset_workspace) and the stored m07 run.",
        ),
    ):
        st.session_state.pop("m07_run", None)
        reset_workspace()
    with action_cols[2]:
        output_label(
            "Temp workspace",
            two_section_help(
                "The planner reads and writes only inside this session-scoped temp directory — no production data is touched.",
                "workspace() paths: SQLite DB, append-only event log, provisional store.",
            ),
        )
        st.code(str(paths["root"]))

    section_header(
        "Configure and run one planner level",
        two_section_help(
            "Pick a cascade level, position the FSM pointer, choose the execution mode and approval gate, and script the decider — the exact sequence of PlannerActions the loop will be fed, one per turn. Then run the real node. Tip: the default pointer (chapter-1) makes a chapter run hit the seeded hard-vs-hard conflict on chapter-2; set chapter_id to chapter-3 for a clean chapter run.",
            "The node resolves its own target from the pointer and the planning surface; the decider JSON is validated against the strict PlannerAction schema before the run.",
        ),
    )
    controls_col, pointer_col, info_col = st.columns([1, 1, 1])
    with controls_col:
        level = st.selectbox(
            "planner level",
            _LEVELS,
            help=two_section_help(
                "Which of the five cascade levels to run. Each level anchors on the one above it: arc needs the global node, scene needs a planned chapter, beat needs a planned scene — seed first.",
                "Selects node_plan_<level>; targets resolve exactly as in production.",
            ),
        )
        execution_mode = st.selectbox(
            "planning.execution_mode",
            ["macro_outline_before_draft", "rolling"],
            help=two_section_help(
                "macro_outline_before_draft plans the whole outline before drafting and can arm the approval gate when the chapter scope completes; rolling plans just-in-time and never pauses.",
                "Written into the synthetic state snapshot (planning_execution_mode) and the config-shaped object.",
            ),
        )
        approval = st.toggle(
            "approval gate (macro_outline)",
            value=False,
            help=two_section_help(
                "When on (and in macro mode), completing the chapter scope sets awaiting_planning_approval instead of proceeding — the node only sets state; it never blocks or waits.",
                "state['approval_mode']='macro_outline'; only read by node_plan_chapter at macro completion.",
            ),
        )
        preset = st.selectbox(
            "decider preset",
            _DECIDER_PRESETS,
            disabled=real_llm,
            help=two_section_help(
                "'valid finalize' scripts a think turn / tool call / validator-passing finalize (level-dependent). 'always-invalid finalize' scripts a plan no validator accepts, so you can watch the loop burn its budget and the fallback ladder catch it — no invalid plan is ever persisted. Disabled in Real LLM mode (the real model decides).",
                "Presets only change the editable JSON below; edit freely before running.",
            ),
        )
    with pointer_col:
        arc_id = st.text_input(
            "arc_id", "arc-1",
            help=two_section_help(
                "The pointer's arc. The seeded planning surface lives under arc-1.",
                "FSM_Pointer.arc_id.",
            ),
        )
        chapter_id = st.text_input(
            "chapter_id", "chapter-1",
            help=two_section_help(
                "The pointer's chapter. chapter-1 is planned (scene runs work); chapter-2 carries the seeded hard-vs-hard conflict; chapter-3 is a clean unplanned stub for a successful chapter run.",
                "FSM_Pointer.chapter_id; the chapter planner prefers it when it names an unplanned stub.",
            ),
        )
        scene_id = st.text_input(
            "scene_id", "scene-1",
            help=two_section_help(
                "The pointer's scene. scene-1 is seeded with a planned scene node (incl. a declared pad_target) and committed PAD history, so beat runs demonstrate the full PAD pipeline.",
                "FSM_Pointer.scene_id; the beat planner anchors on snap:scene:<scene_id>.",
            ),
        )
        beat_index = st.number_input(
            "beat_index", min_value=0, value=0, step=1,
            help=two_section_help(
                "The pointer's in-scene position. Successful beat runs advance it to the newly planned beat.",
                "FSM_Pointer.beat_index.",
            ),
        )
    with info_col:
        output_label(
            "Real code under this page",
            two_section_help(
                "The run button calls the production node for the chosen level, which compiles constraints (fsm/planning_annotations), runs the bounded loop (fsm/planning_loop) with the selected decider — scripted (mocked) or the production M04-backed one (real) — gates finalizes on the deterministic validators (fsm/planning_validators), executes tools through the traced permission-matrix registry (fsm/planning_tools), and persists through the 07.00 helpers (fsm/planning_node_support).",
                "Injected pieces: the decider sequence in Mocked mode (none in Real mode) and adapt_fn=None in both.",
            ),
        )
        st.caption(
            "Cascade order for a full walkthrough: seed → global → arc → chapter "
            "(pointer at chapter-3, or an arc-2 stub) → scene → beat."
        )

    decider_text = st.text_area(
        "scripted decider sequence (JSON list of PlannerActions)",
        value=_default_decider_json(level, preset),
        height=260,
        key=f"m07_decider_{level}_{preset}",
        disabled=real_llm,
        help=two_section_help(
            "One PlannerAction per loop turn, in order. Five action types exist: call_tool, revise_plan, finalize_plan, raise_conflict, and continue_deliberation (a free 'think' turn). When the list runs out the decider raises, which the loop counts as wasted turns until its cap. Ignored (disabled) in Real LLM mode — the production decider proposes each turn instead.",
            "Each entry is validated with PlannerAction.model_validate (extra='forbid') before the run starts.",
        ),
    )
    if real_llm:
        st.caption(
            "Real LLM mode: the scripted sequence above is ignored; each turn is a live "
            "structured-output call constrained to the PlannerAction schema."
        )

    if st.button(
        "Run planner level",
        type="primary",
        help=two_section_help(
            "Builds the synthetic orchestrator state (pointer, mode, approval, temp paths, config-shaped object), then executes the real async planner node with your scripted decider and renders the full runbook below.",
            "asyncio.run(node_plan_<level>(state, decider=scripted[, adapt_fn=None])).",
        ),
    ):
        try:
            actions: list[PlannerAction] | None = None
            if not real_llm:
                raw_actions = json.loads(decider_text)
                if not isinstance(raw_actions, list):
                    raise ValueError("the decider sequence must be a JSON list")
                actions = [PlannerAction.model_validate(a) for a in raw_actions]
            pointer = FSM_Pointer(
                arc_id=arc_id,
                chapter_id=chapter_id,
                scene_id=scene_id,
                beat_index=int(beat_index),
            )
            spinner_text = (
                "Running planner level with REAL LLM calls (bounded by the level's "
                "deliberation caps; retries on a slow endpoint take a while)…"
                if real_llm
                else "Running planner level with the scripted decider…"
            )
            with st.spinner(spinner_text):
                st.session_state.m07_run = _run_level(
                    level,
                    paths,
                    pointer,
                    execution_mode,
                    "macro_outline" if approval else "off",
                    actions,
                    llm_mode=llm_mode,
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"{type(exc).__name__}: {exc}")

    run = st.session_state.get("m07_run")
    if not run:
        st.info("Seed the planning stores, script a decider, then run a planner level.")
        return

    _render_status(run)
    _render_trace(run)
    _render_constraints_validators_rung(run)
    _render_persisted(run)
    _render_pad(run)
    _render_tool_traces(run)
    _render_store_expander(run["state"].get("sqlite_db_path"))


def main() -> None:
    st.set_page_config(page_title="M07 Planning Cascade", layout="wide")
    render()


if __name__ == "__main__":
    main()
