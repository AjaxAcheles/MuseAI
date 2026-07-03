"""Module: M17 (Web UI & Real-time Observer Surface)
Tests for the /plan read surface: normalized timeline data from the REAL
persisted planning store (never frontend-invented), readable node detail with
a raw debug block, raw debug dump, and approval refusal outside the gate.
Deterministic, temp stores, zero network.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app import create_app
from core.runtime import get_resources, reset_resources_for_tests

_ENDPOINT_ENV_KEYS = (
    "PLANNER_API_KEY",
    "DRAFTER_API_KEY",
    "CRITIC_API_KEY",
    "PAD_TRANSLATOR_API_KEY",
    "CRAFT_CONSULTANT_API_KEY",
)

_PREMISE = "A tide-warden must relight a drowned beacon before the fleet returns."


@pytest.fixture
def isolated_resources(tmp_path, monkeypatch):
    for key in _ENDPOINT_ENV_KEYS:
        monkeypatch.setenv(key, "synthetic-test-secret")
    return reset_resources_for_tests(tmp_path)


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=60))


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


async def _run_to_completion(client):
    started = await client.post("/control/start", json={"premise": _PREMISE})
    assert (await started.get_json())["ok"] is True
    manager = get_resources().generation_manager
    await manager.join()
    assert manager.status()["status"] == "completed"


def test_plan_snapshot_empty_store_is_honest(isolated_resources):
    async def scenario():
        app = create_app()
        async with app.test_app() as test_app:
            client = test_app.test_client()
            response = await client.get("/plan/snapshot")
            payload = await response.get_json()
            assert payload["ok"] is True
            assert payload["snapshot_id"] is None
            assert payload["tree"] is None
            assert "start a run" in payload["message"].lower()

    _run(scenario())


def test_plan_snapshot_returns_readable_timeline(isolated_resources):
    async def scenario():
        app = create_app()
        async with app.test_app() as test_app:
            client = test_app.test_client()
            await _run_to_completion(client)

            response = await client.get("/plan/snapshot")
            payload = await response.get_json()
            assert payload["ok"] is True
            assert payload["snapshot_id"] == "snap_vertical_slice"
            assert payload["revision_count"] >= 5

            tree = payload["tree"]
            assert tree["level"] == "global"
            assert tree["title"]  # readable, not an id-only blob
            arcs = tree["children"]
            assert arcs and all(node["level"] == "arc" for node in arcs)
            chapters = [child for arc in arcs for child in arc["children"]]
            assert chapters and all(node["level"] == "chapter" for node in chapters)
            assert all(node["status"] == "planned" for node in chapters)

            runtime = payload["runtime"]
            assert len(runtime["scenes"]) == 1
            assert len(runtime["beats"]) == 1
            assert runtime["current_scene"]
            assert runtime["current_beat"]

            # Normalized view: no raw plan-JSON blobs leak into the timeline.
            assert "purpose" not in set(_walk_keys(payload))
            assert "raw" not in set(_walk_keys(payload))

    _run(scenario())


def test_plan_node_detail_readable_and_raw(isolated_resources):
    async def scenario():
        app = create_app()
        async with app.test_app() as test_app:
            client = test_app.test_client()
            await _run_to_completion(client)

            node_id = "snap_vertical_slice:global"
            response = await client.get(f"/plan/node/{node_id}")
            payload = await response.get_json()
            assert payload["ok"] is True
            node = payload["node"]
            assert node["level"] == "global"
            field_names = [field["name"] for field in node["fields"]]
            assert "premise" in field_names
            assert isinstance(node["raw"], dict) and node["raw"].get("arcs")

            missing = await client.get("/plan/node/snap_vertical_slice:nope")
            assert missing.status_code == 404
            missing_payload = await missing.get_json()
            assert missing_payload["ok"] is False

    _run(scenario())


def test_plan_debug_raw_returns_store_rows(isolated_resources):
    async def scenario():
        app = create_app()
        async with app.test_app() as test_app:
            client = test_app.test_client()
            await _run_to_completion(client)

            response = await client.get("/plan/debug/raw")
            payload = await response.get_json()
            assert payload["ok"] is True
            assert payload["snapshot"]["snapshot_id"] == "snap_vertical_slice"
            assert payload["nodes"] and payload["revisions"]
            # Raw endpoint IS the raw view: purpose blobs belong here.
            assert any(json.loads(row["purpose"] or "{}") for row in payload["nodes"])

    _run(scenario())


def test_plan_approve_rejected_when_nothing_is_blocked(isolated_resources):
    async def scenario():
        app = create_app()
        async with app.test_app() as test_app:
            client = test_app.test_client()

            idle = await client.post("/plan/approve")
            idle_payload = await idle.get_json()
            assert idle_payload["ok"] is False
            assert "approval" in idle_payload["message"].lower()

            await _run_to_completion(client)
            done = await client.post("/plan/approve")
            done_payload = await done.get_json()
            assert done_payload["ok"] is False

    _run(scenario())
