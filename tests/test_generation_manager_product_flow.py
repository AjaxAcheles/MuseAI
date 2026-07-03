"""Module: M01 (Coordinator & State Machine)
Product-flow tests for the generation manager: the macro-outline approval gate
blocks, `approve_plan()` resumes into scene/beat planning, stop cancels a
parked run, and the premise contract holds. Deterministic, temp stores,
zero network.
"""

from __future__ import annotations

import asyncio

import pytest

from core.generation_manager import GenerationManager
from core.runtime import reset_resources_for_tests
from memory.sqlite_db import get_planning_nodes, get_planning_snapshot

_ENDPOINT_ENV_KEYS = (
    "PLANNER_API_KEY",
    "DRAFTER_API_KEY",
    "CRITIC_API_KEY",
    "PAD_TRANSLATOR_API_KEY",
    "CRAFT_CONSULTANT_API_KEY",
)

_SNAPSHOT_ID = "snap_vertical_slice"
_PREMISE = "An archivist must smuggle a banned atlas out of a burning library."

_APPROVAL_PAYLOAD = {
    "premise": _PREMISE,
    "planning_execution_mode": "macro_outline_before_draft",
    "approval_mode": "macro_outline",
}


@pytest.fixture
def resources(tmp_path, monkeypatch):
    for key in _ENDPOINT_ENV_KEYS:
        monkeypatch.setenv(key, "synthetic-test-secret")
    return reset_resources_for_tests(tmp_path)


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=60))


async def _wait_until(predicate, timeout=20.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached within timeout")


def test_approval_gate_blocks_then_approve_resumes_to_completion(resources):
    async def scenario():
        manager = GenerationManager(resources)
        resources.generation_manager = manager
        db_path = resources.stores["sqlite"].path

        collected: list[dict] = []

        async def collect():
            async for event in resources.event_bus.subscribe():
                collected.append(event)

        collector = asyncio.create_task(collect())

        started = await manager.start(dict(_APPROVAL_PAYLOAD))
        assert started["ok"] is True

        await _wait_until(lambda: manager.status()["awaiting_approval"])
        status = manager.status()
        assert status["status"] == "blocked"
        assert status["active"] is True  # task parked at the gate, not finished

        # At the gate: macro outline persisted, snapshot presented as draft,
        # and no scene/beat planned yet.
        levels = {n["level"] for n in get_planning_nodes(db_path, _SNAPSHOT_ID)}
        assert {"global", "arc", "chapter"} <= levels
        assert "scene" not in levels and "beat" not in levels
        assert get_planning_snapshot(db_path, _SNAPSHOT_ID)["status"] == "draft"

        approved = await manager.approve_plan()
        assert approved["ok"] is True

        await manager.join()
        assert manager.status()["status"] == "completed"

        final_state = manager.final_state
        assert final_state["macro_outline_approved"] is True
        assert final_state["awaiting_planning_approval"] is False
        assert final_state["planning_block_reason"] is None
        # Approval gate never conflates with the escalation/pause flags.
        assert final_state["pause_requested"] is False
        assert final_state["hard_stop_asserted"] is False

        snapshot = get_planning_snapshot(db_path, _SNAPSHOT_ID)
        assert snapshot["status"] == "approved"
        assert snapshot["approved_at"]

        levels = {n["level"] for n in get_planning_nodes(db_path, _SNAPSHOT_ID)}
        assert {"scene", "beat"} <= levels

        await asyncio.sleep(0.05)  # let the collector drain its queue
        event_types = [event["event"] for event in collected]
        assert "planning_blocked" in event_types
        approvals = [e["data"] for e in collected if e["event"] == "approval_state"]
        assert any(d["awaiting_approval"] for d in approvals)
        assert any(d.get("macro_outline_approved") for d in approvals)

        collector.cancel()

    _run(scenario())


def test_approve_rejected_when_not_awaiting(resources):
    async def scenario():
        manager = GenerationManager(resources)
        resources.generation_manager = manager

        refused = await manager.approve_plan()
        assert refused["ok"] is False

        await manager.start({"premise": _PREMISE})  # approval_mode off
        await manager.join()
        assert manager.status()["status"] == "completed"

        after = await manager.approve_plan()
        assert after["ok"] is False

    _run(scenario())


def test_stop_while_awaiting_approval_cancels_cleanly(resources):
    async def scenario():
        manager = GenerationManager(resources)
        resources.generation_manager = manager

        await manager.start(dict(_APPROVAL_PAYLOAD))
        await _wait_until(lambda: manager.status()["awaiting_approval"])

        stopped = await manager.stop()
        assert stopped["ok"] is True
        assert manager.status()["status"] == "stopped"
        assert manager.status()["awaiting_approval"] is False
        assert manager._task.done()

    _run(scenario())


def test_duplicate_start_rejected_while_parked_at_gate(resources):
    async def scenario():
        manager = GenerationManager(resources)
        resources.generation_manager = manager

        await manager.start(dict(_APPROVAL_PAYLOAD))
        await _wait_until(lambda: manager.status()["awaiting_approval"])

        duplicate = await manager.start(dict(_APPROVAL_PAYLOAD))
        assert duplicate["ok"] is False
        assert "already active" in duplicate["message"]

        await manager.stop()

    _run(scenario())


def test_premise_required_at_manager_level(resources):
    async def scenario():
        manager = GenerationManager(resources)
        resources.generation_manager = manager
        refused = await manager.start({"genre": "mystery"})
        assert refused["ok"] is False
        assert "premise" in refused["message"]
        assert manager.status()["status"] == "idle"

    _run(scenario())
