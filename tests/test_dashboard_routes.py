"""Module: M17 (Web UI & Real-time Observer Surface)
Route-level tests for the vertical-slice Quart app: dashboard render, status
JSON, SSE stream headers, and the control endpoints' consistent JSON shape.

No network, no live model: the app runs against an isolated temp resource
container with synthetic endpoint secrets.
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


@pytest.fixture
def isolated_resources(tmp_path, monkeypatch):
    for key in _ENDPOINT_ENV_KEYS:
        monkeypatch.setenv(key, "synthetic-test-secret")
    return reset_resources_for_tests(tmp_path)


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=60))


def test_app_factory_creates_quart_app(isolated_resources):
    app = create_app()
    assert app.name == "app"
    rules = {rule.rule for rule in app.url_map.iter_rules()}
    assert {"/", "/dashboard", "/events", "/api/status", "/control/start",
            "/control/pause", "/control/resume", "/control/stop"} <= rules


def test_root_redirects_to_dashboard(isolated_resources):
    async def scenario():
        app = create_app()
        async with app.test_app() as test_app:
            client = test_app.test_client()
            response = await client.get("/")
            assert response.status_code in (301, 302, 307, 308)
            assert response.headers["Location"].endswith("/dashboard")

    _run(scenario())


def test_dashboard_renders_expected_anchors(isolated_resources):
    async def scenario():
        app = create_app()
        async with app.test_app() as test_app:
            client = test_app.test_client()
            response = await client.get("/dashboard")
            assert response.status_code == 200
            body = (await response.get_data()).decode("utf-8")
            for anchor in (
                'id="status-ribbon"',
                'id="run-status"',
                'id="project-setup-card"',
                'id="planning-timeline-card"',
                'id="approval-card"',
                'id="block-card"',
                'id="event-stream-card"',
                'id="btn-start"',
                'id="btn-stop"',
            ):
                assert anchor in body
            # Stub stores are labelled honestly on the page.
            assert "badge-stub" in body

    _run(scenario())


def test_api_status_returns_json(isolated_resources):
    async def scenario():
        app = create_app()
        async with app.test_app() as test_app:
            client = test_app.test_client()
            response = await client.get("/api/status")
            assert response.status_code == 200
            payload = await response.get_json()
            assert payload["ok"] is True
            assert payload["manager"]["status"] == "idle"
            assert "snapshot" in payload
            assert payload["stores"]["graphiti"]["kind"] == "stub"

    _run(scenario())


def test_events_returns_sse_content_type_and_snapshot_first(isolated_resources):
    async def scenario():
        app = create_app()
        async with app.test_app() as test_app:
            client = test_app.test_client()
            # The stream never ends, so use the streaming connection API
            # rather than awaiting a complete response.
            async with client.request("/events", method="GET") as connection:
                first = await asyncio.wait_for(connection.receive(), timeout=5)
                assert connection.status_code == 200
                assert connection.headers["Content-Type"].startswith(
                    "text/event-stream"
                )
                assert connection.headers["Cache-Control"] == "no-cache"

                text = first.decode("utf-8")
                assert text.startswith("event: status\ndata: ")
                payload = json.loads(text.split("data: ", 1)[1].strip())
                assert payload["data"]["status"] == "idle"

                await connection.disconnect()

            # The disconnected client's subscriber queue is cleaned up.
            bus = get_resources().event_bus
            for _ in range(100):
                if bus.subscriber_count() == 0:
                    break
                await asyncio.sleep(0.01)
            assert bus.subscriber_count() == 0

    _run(scenario())


def test_control_start_and_stop_return_consistent_json(isolated_resources):
    async def scenario():
        app = create_app()
        async with app.test_app() as test_app:
            client = test_app.test_client()

            start = await client.post(
                "/control/start",
                json={"premise": "A courier must deliver a letter across a siege."},
            )
            start_body = await start.get_json()
            assert set(start_body) >= {"ok", "status", "message"}
            assert start_body["ok"] is True

            stop = await client.post("/control/stop")
            stop_body = await stop.get_json()
            assert set(stop_body) >= {"ok", "status", "message"}
            # The deterministic run is fast: by the time the stop request
            # lands it may have finished, in which case stop is refused —
            # both outcomes use the same consistent shape.
            if stop_body["ok"]:
                assert stop_body["status"] == "stopped"
            else:
                assert stop_body["status"] in {"completed", "blocked", "error"}

            manager = get_resources().generation_manager
            await manager.join()

            # A stop with nothing running is refused with the same shape.
            stop_again = await client.post("/control/stop")
            again_body = await stop_again.get_json()
            assert set(again_body) >= {"ok", "status", "message"}
            assert again_body["ok"] is False

    _run(scenario())


def test_control_pause_requires_active_run(isolated_resources):
    async def scenario():
        app = create_app()
        async with app.test_app() as test_app:
            client = test_app.test_client()
            pause = await client.post("/control/pause")
            body = await pause.get_json()
            assert body["ok"] is False
            assert set(body) >= {"ok", "status", "message"}

    _run(scenario())
