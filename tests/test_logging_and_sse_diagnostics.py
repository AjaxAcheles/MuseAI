"""Module: M14 (Configuration, Startup & Observability) + M17 (Web UI)
Diagnostics tests: the structured JSONL app log, the /healthz and
/api/logs/recent debug endpoints, and the SSE envelope contract
(event / timestamp / run_id / data.message). Deterministic, temp stores,
zero network; no secrets or full premises may appear in logs.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app import create_app
from core.generation_manager import GenerationManager
from core.runtime import get_resources, reset_resources_for_tests

_ENDPOINT_ENV_KEYS = (
    "PLANNER_API_KEY",
    "DRAFTER_API_KEY",
    "CRITIC_API_KEY",
    "PAD_TRANSLATOR_API_KEY",
    "CRAFT_CONSULTANT_API_KEY",
)

# Deliberately long so log truncation is observable.
_LONG_PREMISE = (
    "A lighthouse keeper inherits a ledger of debts owed to the sea and must "
    "repay them one storm at a time before the town learns what was borrowed, "
    "because every unpaid debt surfaces as a wreck with her name on it."
)


@pytest.fixture
def resources(tmp_path, monkeypatch):
    for key in _ENDPOINT_ENV_KEYS:
        monkeypatch.setenv(key, "synthetic-test-secret")
    return reset_resources_for_tests(tmp_path)


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=60))


def _log_entries(resources):
    log_path = resources.logs_dir / "app.jsonl"
    assert log_path.is_file()
    entries = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))  # every line must parse as JSON
    return entries


def test_app_log_is_jsonl_with_run_lifecycle_events(resources):
    async def scenario():
        manager = GenerationManager(resources)
        resources.generation_manager = manager
        await manager.start({"premise": _LONG_PREMISE})
        await manager.join()
        assert manager.status()["status"] == "completed"

    _run(scenario())

    entries = _log_entries(resources)
    events = [entry["event"] for entry in entries]
    for expected in (
        "runtime_init",
        "run_start",
        "run_status",
        "node_start",
        "node_end",
        "pointer_update",
        "snapshot_update",
        "run_completed",
    ):
        assert expected in events, f"missing app-log event {expected!r}"
    assert all("event" in entry and "ts" in entry for entry in entries)

    node_starts = [e for e in entries if e["event"] == "node_start"]
    assert {e["node"] for e in node_starts} >= {
        "node_plan_global",
        "node_plan_arc",
        "node_plan_chapter",
        "node_plan_scene",
        "node_plan_beat",
    }


def test_log_never_contains_full_premise_or_secrets(resources):
    async def scenario():
        manager = GenerationManager(resources)
        resources.generation_manager = manager
        await manager.start({"premise": _LONG_PREMISE})
        await manager.join()

    _run(scenario())

    raw = (resources.logs_dir / "app.jsonl").read_text(encoding="utf-8")
    assert _LONG_PREMISE not in raw  # only the truncated form may be logged
    assert _LONG_PREMISE[:100] not in raw
    assert "synthetic-test-secret" not in raw

    run_starts = [e for e in _log_entries(resources) if e["event"] == "run_start"]
    assert run_starts
    logged_premise = run_starts[-1]["payload"]["premise"]
    assert len(logged_premise) <= 80


def test_healthz_reports_runtime_run_and_store_kinds(resources):
    async def scenario():
        app = create_app()
        async with app.test_app() as test_app:
            client = test_app.test_client()
            response = await client.get("/healthz")
            assert response.status_code == 200
            payload = await response.get_json()
            assert payload["ok"] is True
            assert payload["runtime_initialized"] is True
            assert payload["run"]["status"] == "idle"
            assert payload["stores"]["sqlite"] == "real"
            assert payload["stores"]["graphiti"] == "stub"

    _run(scenario())


def test_api_logs_recent_returns_tail(resources):
    async def scenario():
        app = create_app()
        async with app.test_app() as test_app:
            client = test_app.test_client()
            # Generate some route_call entries, then read the tail.
            await client.get("/healthz")
            await client.get("/api/status")
            response = await client.get("/api/logs/recent?limit=5")
            payload = await response.get_json()
            assert payload["ok"] is True
            assert payload["limit"] == 5
            assert 0 < len(payload["entries"]) <= 5
            assert all("event" in entry for entry in payload["entries"])
            assert any(entry["event"] == "route_call" for entry in payload["entries"])

    _run(scenario())


def test_sse_envelope_has_type_timestamp_run_id_and_message(resources):
    async def scenario():
        manager = GenerationManager(resources)
        resources.generation_manager = manager

        collected: list[dict] = []

        async def collect():
            async for event in resources.event_bus.subscribe():
                collected.append(event)

        collector = asyncio.create_task(collect())
        await asyncio.sleep(0)  # let the subscriber attach before the run

        await manager.start({"premise": _LONG_PREMISE})
        await manager.join()
        await asyncio.sleep(0.05)  # drain the queue
        collector.cancel()

        run_id = manager.status()["run_id"]
        assert run_id

        run_events = [e for e in collected if e.get("run_id") == run_id]
        assert run_events, "no events carried the run id"
        types = {e["event"] for e in run_events}
        assert {"status", "phase_change", "pointer_update", "planning_node",
                "planning_snapshot", "done"} <= types
        for event in run_events:
            assert event["event"]
            assert event["timestamp"].endswith("Z")
            assert isinstance(event["data"], dict)
            assert "message" in event["data"], f"{event['event']} event lacks a message"

    _run(scenario())
