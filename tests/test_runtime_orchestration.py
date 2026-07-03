"""Module: M14 (Configuration, Startup & Observability) + M01 (Coordinator)
Deterministic, no-network tests for the runtime-resource container and the
generation manager's vertical-slice run orchestration.

Endpoint secrets are synthetic env values; no live model endpoint is required —
the planner nodes receive an injected deterministic decider, so nothing here
can touch the network.
"""

from __future__ import annotations

import asyncio

import pytest

import core.runtime as runtime
from core.generation_manager import GenerationManager
from core.runtime import get_resources, init_resources, reset_resources_for_tests
from memory.sqlite_db import get_planning_nodes, get_revisions_for_snapshot

_ENDPOINT_ENV_KEYS = (
    "PLANNER_API_KEY",
    "DRAFTER_API_KEY",
    "CRITIC_API_KEY",
    "PAD_TRANSLATOR_API_KEY",
    "CRAFT_CONSULTANT_API_KEY",
)

_SNAPSHOT_ID = "snap_vertical_slice"  # derived from the default project id

# Minimal valid product payload: the premise is required at the manager level.
_START_PAYLOAD = {"premise": "A keeper must honor a lighthouse vow before the harbor fails."}


@pytest.fixture
def fake_secrets(monkeypatch):
    """Synthetic endpoint secrets so strict config load needs no real .env."""
    for key in _ENDPOINT_ENV_KEYS:
        monkeypatch.setenv(key, "synthetic-test-secret")


@pytest.fixture
def resources(tmp_path, fake_secrets):
    """An isolated RuntimeResources container rooted in a temp tree."""
    return reset_resources_for_tests(tmp_path)


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=60))


def test_runtime_initializes_with_temp_data_directory(resources, tmp_path):
    assert resources.data_dir == (tmp_path / "data").resolve()
    sqlite_path = resources.stores["sqlite"].path
    assert sqlite_path.is_file()
    assert sqlite_path.read_bytes().startswith(b"SQLite format 3\x00")
    assert resources.stores["event_log"].path.exists()
    assert resources.stores["provisional"].path.is_file()

    # Unbuilt stores are exposed as named, labelled stubs — never silently real.
    stub_names = {name for name, h in resources.stores.items() if h.kind == "stub"}
    assert stub_names == {"graphiti", "raptor", "chroma", "style", "snapshots"}

    # Idempotent access: the container is a process-wide singleton.
    assert get_resources() is resources
    assert init_resources() is resources

    # The bus snapshot is seeded with the run modes from config.
    snapshot = resources.event_bus.snapshot()
    assert snapshot["planning_execution_mode"] == resources.config.planning.execution_mode
    assert snapshot["approval_mode"] == resources.config.planning.approval_mode


def test_legacy_init_resources_config_path_is_preserved(tmp_path, monkeypatch):
    """`init_resources(config)` keeps the 02.05/INT-A·B artifact contract."""
    monkeypatch.setattr(runtime, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(runtime, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(runtime, "SQLITE_DB_PATH", tmp_path / "data" / "fictionwriter.db")
    monkeypatch.setattr(runtime, "GRAPHITI_DB_PATH", tmp_path / "data" / "graphiti.db")
    monkeypatch.setattr(runtime, "CHROMA_STORE_DIR", tmp_path / "data" / "chroma")
    monkeypatch.setattr(runtime, "STYLE_STORE_DIR", tmp_path / "data" / "styles")
    monkeypatch.setattr(runtime, "SNAPSHOT_DIR", tmp_path / "data" / "snapshots")
    monkeypatch.setattr(runtime, "EVENT_LOG_PATH", tmp_path / "data" / "events.jsonl")

    result = init_resources(object())  # synthetic config, as the legacy tests use
    assert result is None
    assert (tmp_path / "data" / "fictionwriter.db").is_file()


def test_generation_manager_deterministic_run_completes(resources):
    async def scenario():
        manager = GenerationManager(resources)
        resources.generation_manager = manager

        started = await manager.start(dict(_START_PAYLOAD))
        assert started["ok"] is True
        assert started["status"] == "starting"

        await manager.join()
        status = manager.status()
        assert status["status"] in {"completed", "blocked"}
        # Default config has approval_mode="off", so the run completes.
        assert status["status"] == "completed"
        assert status["last_error"] is None

        # Real planning persisted through the real nodes/stores: at least
        # global + arc + chapters + scene + beat revisions.
        db_path = resources.stores["sqlite"].path
        revisions = get_revisions_for_snapshot(db_path, _SNAPSHOT_ID)
        assert len(revisions) >= 5
        levels = {n["level"] for n in get_planning_nodes(db_path, _SNAPSHOT_ID)}
        assert {"global", "arc", "chapter", "scene", "beat"} <= levels

        final_state = manager.final_state
        assert final_state is not None
        assert final_state["macro_outline_ready"] is True
        assert final_state["planning_block_reason"] is None
        # Approval blocking is never conflated with the escalation/pause flags.
        assert final_state["pause_requested"] is False
        assert final_state["hard_stop_asserted"] is False

        snapshot = resources.event_bus.snapshot()
        assert snapshot["status"] == "completed"
        assert snapshot["running"] is False

    _run(scenario())


def test_duplicate_start_is_rejected_clearly(resources):
    async def scenario():
        manager = GenerationManager(resources)
        resources.generation_manager = manager

        first = await manager.start(dict(_START_PAYLOAD))
        assert first["ok"] is True
        duplicate = await manager.start(dict(_START_PAYLOAD))
        assert duplicate["ok"] is False
        assert "already active" in duplicate["message"]

        # Clean up so the loop closes with no dangling task.
        await manager.stop()
        assert manager.status()["status"] == "stopped"

    _run(scenario())


def test_stop_cancels_a_running_task(resources):
    async def scenario():
        manager = GenerationManager(resources)
        resources.generation_manager = manager

        await manager.start(dict(_START_PAYLOAD))
        result = await manager.stop()
        assert result["ok"] is True
        assert manager.status()["status"] == "stopped"
        assert manager.status()["running"] is False
        # The cancelled task was awaited, not abandoned.
        assert manager._task.done()

        stop_again = await manager.stop()
        assert stop_again["ok"] is False

    _run(scenario())


def test_invalid_start_payload_is_rejected(resources):
    async def scenario():
        manager = GenerationManager(resources)
        resources.generation_manager = manager

        bad_mode = await manager.start({**_START_PAYLOAD, "llm_mode": "bogus"})
        assert bad_mode["ok"] is False and "llm_mode" in bad_mode["message"]

        bad_key = await manager.start({**_START_PAYLOAD, "config_path": "/etc/passwd"})
        assert bad_key["ok"] is False and "Unknown start field" in bad_key["message"]

        no_premise = await manager.start({})
        assert no_premise["ok"] is False and "premise" in no_premise["message"]

        assert manager.status()["running"] is False

    _run(scenario())


def test_missing_live_llm_endpoint_does_not_break_deterministic_path(resources):
    """The synthetic secrets point at no reachable endpoint; the deterministic
    decider seam means the run never dials one."""

    async def scenario():
        manager = GenerationManager(resources)
        resources.generation_manager = manager
        await manager.start({**_START_PAYLOAD, "llm_mode": "deterministic"})
        await manager.join()
        status = manager.status()
        assert status["status"] == "completed"
        assert status["last_error"] is None

    _run(scenario())
