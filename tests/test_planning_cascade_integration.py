"""Module: M05 (Hierarchical Planning Cascade)
INT-C-E integration tests for the planner cascade across M04, M05, and real M02 stores.

The deterministic path is always network-free: it loads the real config, resolves the
planner endpoint role, renders the real planner template through ``make_planner_decider``,
returns strict ``PlannerAction`` objects through the decider seam, then runs
``node_plan_global`` over temp SQLite/provisional/event-log stores. The live path uses the
same node with the real M04 structured-output boundary and skips explicitly when no usable
planner endpoint/credential is available.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

import core.llm_io_logger as llm_io_logger
from core.config_loader import AppConfig, EndpointConfig, load_config
from fsm.nodes.node_plan_global import node_plan_global
from fsm.planning_actions import PlannerAction
from fsm.planning_annotations import compile_planning_constraints
from fsm.planning_node_support import PlannerDeciderError, make_planner_decider
from fsm.planning_tools import PlanningToolRegistry
from fsm.state import FSM_Pointer, make_initial_state
from llm.call_llm import (
    LLMCallError,
    UnsupportedGrammarStrategyError,
    call_llm_structured,
)
from memory.event_log import init_event_log, iter_events
from memory.provisional_store import get_claim, init_provisional_store, upsert_claim
from memory.sqlite_db import (
    create_planning_snapshot,
    get_planning_nodes,
    get_planning_snapshot,
    get_revisions_for_snapshot,
    init_db,
    insert_planning_annotation,
    upsert_planning_node,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

ENDPOINT_NAMES = ("planner", "drafter", "critic", "pad_translator", "craft_consultant")
SKIP_ENV = "MUSEAI_SKIP_LIVE_LLM"
PLACEHOLDER_SECRET = "int-c-e-placeholder-secret"

PROJECT_ID = "int-c-e-project"
SNAPSHOT_ID = f"snap_{PROJECT_ID}"
GLOBAL_NODE_ID = f"{SNAPSHOT_ID}:global"
GLOBAL_CONSTRAINT_TEXT = "Keep the lighthouse vow tied to the central conflict."


@dataclass(frozen=True)
class SeededStores:
    db_path: Path
    provisional_path: Path
    event_log_path: Path
    snapshot_id: str
    global_node_id: str


@dataclass(frozen=True)
class LivePlannerEndpoint:
    config: AppConfig
    endpoint: EndpointConfig
    has_real_api_key: bool


class NoToolRegistry:
    """Registry that proves the deterministic script never asks for optional tools."""

    def permitted(self, level: str, tool_name: str) -> bool:  # noqa: ARG002
        raise AssertionError("deterministic INT-C-E path should not request tools")

    def call(self, *args: Any, **kwargs: Any) -> dict:  # noqa: ARG002
        raise AssertionError("deterministic INT-C-E path should not execute tools")


def _run(coro):
    return asyncio.run(coro)


def _endpoint_env_name(endpoint_name: str) -> str:
    return f"{endpoint_name.upper()}_API_KEY"


def _load_config_with_placeholders(monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    """Load config.yaml through the real loader without requiring live credentials."""

    for name in ENDPOINT_NAMES:
        monkeypatch.setenv(_endpoint_env_name(name), PLACEHOLDER_SECRET)
    return load_config(CONFIG_PATH)


def _load_live_planner_config(monkeypatch: pytest.MonkeyPatch) -> LivePlannerEndpoint:
    """Reuse the 05.T2 live convention: opt out/skip, never fake a backend."""

    if os.environ.get(SKIP_ENV, "").strip() not in ("", "0", "false", "False"):
        pytest.skip(f"{SKIP_ENV} set - live planner endpoint tests opted out")

    has_real = {
        name: bool(os.environ.get(_endpoint_env_name(name))) for name in ENDPOINT_NAMES
    }
    for name in ENDPOINT_NAMES:
        if not has_real[name]:
            monkeypatch.setenv(_endpoint_env_name(name), PLACEHOLDER_SECRET)

    try:
        config = load_config(CONFIG_PATH)
    except Exception as exc:  # noqa: BLE001 - config load failure means live path unavailable
        pytest.skip(f"config.yaml could not be loaded for a live planner call: {exc}")

    endpoint = getattr(config.endpoints, "planner", None)
    if endpoint is None:
        pytest.skip("no planner endpoint is configured in config.yaml")
    if not has_real["planner"]:
        pytest.skip("no real PLANNER_API_KEY is set for the live planner endpoint")

    return LivePlannerEndpoint(
        config=config,
        endpoint=endpoint,
        has_real_api_key=has_real["planner"],
    )


def _seed_temp_stores(tmp_path: Path, config: AppConfig) -> SeededStores:
    """Seed real temp M02/07.00 stores through their public helpers."""

    data_dir = tmp_path / "data"
    db_path = data_dir / "fictionwriter.db"
    provisional_path = data_dir / "provisional_claims.db"
    event_log_path = data_dir / "events.jsonl"

    init_db(db_path)
    init_provisional_store(provisional_path)
    init_event_log(config, event_log_path)

    claim_id = upsert_claim(
        provisional_path,
        claim_id="claim-premise",
        source_ref="premise",
        subject_id="project",
        claim_text="Synthetic premise context: the lighthouse vow frames the pressure.",
        confidence=config.context.coreference_high_confidence,
    )
    assert get_claim(provisional_path, claim_id)["claim_text"].startswith(
        "Synthetic premise context"
    )

    create_planning_snapshot(
        db_path,
        snapshot_id=SNAPSHOT_ID,
        project_id=PROJECT_ID,
        mode=config.planning.execution_mode,
    )
    upsert_planning_node(
        db_path,
        node_id=GLOBAL_NODE_ID,
        snapshot_id=SNAPSHOT_ID,
        level="global",
        status="planning",
        title="Global story target",
        summary="Synthetic global target for INT-C-E.",
    )
    insert_planning_annotation(
        db_path,
        annotation_id="ann-global-vow",
        snapshot_id=SNAPSHOT_ID,
        target_node_id=GLOBAL_NODE_ID,
        target_level="global",
        note_type="constraint",
        scope="this_node",
        priority="hard",
        text=GLOBAL_CONSTRAINT_TEXT,
    )

    return SeededStores(
        db_path=db_path,
        provisional_path=provisional_path,
        event_log_path=event_log_path,
        snapshot_id=SNAPSHOT_ID,
        global_node_id=GLOBAL_NODE_ID,
    )


def _state(config: AppConfig, stores: SeededStores) -> dict[str, Any]:
    state = make_initial_state(
        PROJECT_ID,
        FSM_Pointer(arc_id="", chapter_id="", scene_id="", beat_index=0),
        planning_snapshot_id=stores.snapshot_id,
        planning_execution_mode=config.planning.execution_mode,
        approval_mode=config.planning.approval_mode,
    )
    state["app_config"] = config
    state["config_path"] = CONFIG_PATH
    state["sqlite_db_path"] = stores.db_path
    state["planning_db_path"] = stores.db_path
    state["provisional_store_path"] = stores.provisional_path
    state["event_log_path"] = stores.event_log_path
    state["project_metadata"] = {
        "genre": "synthetic integration",
        "target_word_count": config.runtime.word_count_target,
        "premise_seed": "A keeper must honor a lighthouse vow before the harbor fails.",
    }
    state["world_rules"] = ["The harbor light is the visible measure of public trust."]
    return state


def _global_base_context(config: AppConfig) -> dict[str, Any]:
    return {
        "genre": "synthetic integration",
        "target_word_count": config.runtime.word_count_target,
        "premise_seed": "A keeper must honor a lighthouse vow before the harbor fails.",
        "world_rules": ["The harbor light is the visible measure of public trust."],
        "existing_arcs": [],
    }


def _valid_global_plan(config: AppConfig) -> dict[str, Any]:
    target = config.runtime.word_count_target
    first_allocation = target // 2
    return {
        "premise": "A keeper must honor a lighthouse vow before the harbor fails.",
        "central_conflict": "The keeper must choose between protecting a secret vow and saving the harbor.",
        "ending_target": "The harbor is saved when the keeper reveals the vow's true cost.",
        "arcs": [
            {
                "arc_id": "arc-vow",
                "title": "The Vow Tightens",
                "function": "establish the vow, the harbor danger, and the keeper's first costly choice",
                "word_allocation": first_allocation,
            },
            {
                "arc_id": "arc-harbor",
                "title": "The Harbor Answers",
                "function": "resolve the public danger through a distinct revelation and sacrifice",
                "word_allocation": target - first_allocation,
            },
        ],
        "promises": [
            {
                "id": "promise-vow",
                "promise": "the lighthouse vow hides a public consequence",
                "payoff": "the final act reveals the consequence when the keeper saves the harbor",
            }
        ],
    }


def _planning_purpose_by_level(stores: SeededStores, level: str) -> list[dict]:
    return [
        json.loads(node["purpose"])
        for node in get_planning_nodes(stores.db_path, stores.snapshot_id, level=level)
        if node.get("purpose")
    ]


def _assert_global_plan_persisted(stores: SeededStores, state: dict[str, Any]) -> None:
    snapshot = get_planning_snapshot(stores.db_path, stores.snapshot_id)
    assert snapshot is not None
    assert snapshot["active_revision_id"] == state["active_planning_revision_id"]

    purposes = _planning_purpose_by_level(stores, "global")
    assert len(purposes) == 1
    persisted = purposes[0]
    assert persisted["central_conflict"]
    assert persisted["arcs"]
    assert persisted["promises"]

    revisions = get_revisions_for_snapshot(stores.db_path, stores.snapshot_id)
    assert len(revisions) == 1
    assert revisions[0]["revision_id"] == state["active_planning_revision_id"]

    events = list(iter_events(stores.event_log_path))
    assert len(events) == 1
    event = events[0]
    assert event["event_type"] == "planning_commit"
    assert event["level"] == "global"
    assert event["snapshot_id"] == stores.snapshot_id
    assert event["revision_id"] == state["active_planning_revision_id"]
    assert event["validation_passes"] is True

    assert state["planning_snapshot_id"] == stores.snapshot_id
    assert state["active_planning_revision_id"]
    assert state["planning_block_reason"] is None


@contextmanager
def _io_log_capture(tmp_path: Path):
    """Capture records emitted by the real LLM I/O logger for secret-leak assertions."""

    capture_path = tmp_path / "llm_io_capture.log"
    logger = llm_io_logger.get_llm_io_logger()
    handler = logging.FileHandler(capture_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    try:
        yield capture_path
    finally:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _assert_secret_not_logged(capture_path: Path, endpoint: EndpointConfig) -> None:
    log_text = capture_path.read_text(encoding="utf-8") if capture_path.exists() else ""
    assert endpoint.api_key
    assert endpoint.api_key not in log_text


def test_deterministic_planner_wiring_persists_validated_plan_and_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _load_config_with_placeholders(monkeypatch)
    stores = _seed_temp_stores(tmp_path, config)
    state = _state(config, stores)

    def _no_network(*args: Any, **kwargs: Any):  # noqa: ARG001
        raise AssertionError("deterministic INT-C-E path must not call the network")

    monkeypatch.setattr(httpx, "AsyncClient", _no_network)

    constraints = compile_planning_constraints(
        stores.snapshot_id,
        stores.global_node_id,
        db_path=stores.db_path,
    )
    assert constraints["needs_clarification"] is False
    assert [c["text"] for c in constraints["hard_annotations"]] == [GLOBAL_CONSTRAINT_TEXT]

    endpoint = config.endpoints.planner
    assert endpoint.model_name
    assert endpoint.grammar_constraint_strategy in {"gbnf", "json_mode", "json_schema"}

    scripted_actions = [
        PlannerAction(action_type="continue_deliberation", rationale="Check the vow first."),
        PlannerAction(
            action_type="finalize_plan",
            final_plan=_valid_global_plan(config),
            self_check={"schema_valid": False},
        ),
    ]
    structured_calls: list[dict[str, Any]] = []

    async def _scripted_structured(messages, endpoint_arg, *, schema_model, validate_retry_cap):
        structured_calls.append(
            {
                "messages": messages,
                "endpoint": endpoint_arg,
                "schema_model": schema_model,
                "validate_retry_cap": validate_retry_cap,
            }
        )
        return scripted_actions.pop(0)

    decider = make_planner_decider(
        "node_plan_global",
        _global_base_context(config),
        config,
        call_structured=_scripted_structured,
    )

    result_state = _run(
        node_plan_global(state, decider=decider, registry=NoToolRegistry())
    )

    assert len(structured_calls) == 2
    assert structured_calls[0]["endpoint"] is endpoint
    assert structured_calls[0]["schema_model"] is PlannerAction
    assert structured_calls[0]["validate_retry_cap"] == config.runtime.model_validate_retry_cap
    rendered_prompt = structured_calls[0]["messages"][0]["content"]
    assert "<planner_task level=\"global\">" in rendered_prompt
    assert "A keeper must honor a lighthouse vow" in rendered_prompt
    assert GLOBAL_CONSTRAINT_TEXT in rendered_prompt

    action_trace = [
        record["action_type"]
        for record in result_state["planner_deliberation_trace"]
        if "action_type" in record
    ]
    assert action_trace == ["continue_deliberation", "finalize_plan"]

    _assert_global_plan_persisted(stores, result_state)


@pytest.mark.integration
def test_live_planner_decider_persists_or_skips_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    live = _load_live_planner_config(monkeypatch)
    config = live.config
    stores = _seed_temp_stores(tmp_path, config)
    state = _state(config, stores)
    parsed_actions: list[PlannerAction] = []

    with _io_log_capture(tmp_path) as io_log_capture:

        async def _fast_structured(messages, endpoint, *, schema_model, validate_retry_cap):
            try:
                return await call_llm_structured(
                    messages,
                    endpoint,
                    schema_model=schema_model,
                    validate_retry_cap=validate_retry_cap,
                    stream=False,
                    temperature=0.0,
                    max_tokens=512,
                    timeout_seconds=config.runtime.inference_timeout_seconds,
                    max_attempts=1,
                    retry_delays=(),
                )
            except UnsupportedGrammarStrategyError as exc:
                _assert_secret_not_logged(io_log_capture, live.endpoint)
                pytest.skip(
                    "planner endpoint does not support a structured-output strategy "
                    f"usable by call_llm_structured: {exc}"
                )

        real_decider = make_planner_decider(
            "node_plan_global",
            _global_base_context(config),
            config,
            call_structured=_fast_structured,
        )

        async def _recording_live_decider(loop_state):
            try:
                action = await real_decider(loop_state)
            except PlannerDeciderError as exc:
                _assert_secret_not_logged(io_log_capture, live.endpoint)
                pytest.skip(
                    "configured planner endpoint could not return a schema-valid "
                    f"PlannerAction: {exc}"
                )
            parsed_actions.append(action)
            return action

        try:
            result_state = _run(
                node_plan_global(
                    state,
                    decider=_recording_live_decider,
                    registry=PlanningToolRegistry(stores.db_path),
                )
            )
        except (LLMCallError, httpx.HTTPError, OSError) as exc:
            _assert_secret_not_logged(io_log_capture, live.endpoint)
            pytest.skip(
                "configured planner endpoint could not complete the live planner call "
                f"({live.endpoint.base_url}, model={live.endpoint.model_name}): {exc}"
            )

        assert parsed_actions
        assert all(isinstance(action, PlannerAction) for action in parsed_actions)
        _assert_global_plan_persisted(stores, result_state)
        _assert_secret_not_logged(io_log_capture, live.endpoint)
