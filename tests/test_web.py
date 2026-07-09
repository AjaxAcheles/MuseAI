"""Tests for the MuseAI v1 Quart web surface."""

from __future__ import annotations

import asyncio
from typing import Any, cast

from museai.core.stream_bus import bus
from museai.memory.db import connect_db, upsert_arc, upsert_project
from museai.web import app as app_module
from museai.web.app import create_app
from museai.web.routes.dashboard import stream


class MockManager:
    def __init__(self, status: str = "idle") -> None:
        self.status = status
        self.state = None
        self.started: list[str] = []
        self.reviews: list[tuple[str, str | None]] = []

    async def start(self, project_id: str) -> None:
        self.started.append(project_id)
        self.status = "running"

    def pause(self) -> None:
        self.status = "paused"

    async def resume(self) -> None:
        self.status = "running"

    def stop(self) -> None:
        self.status = "stopped"

    async def resolve_review(self, decision: str, edited_text: str | None = None) -> None:
        self.reviews.append((decision, edited_text))
        self.status = "running"


async def _started_app(config, tmp_path):
    config_path = tmp_path / "config.yaml"
    app = create_app(config_path=config_path, test_config=config)
    test_app = app.test_app()
    await test_app.__aenter__()
    return app, test_app


async def _close_started_app(test_app) -> None:
    await test_app.__aexit__(None, None, None)


def _seed_project(config) -> None:
    conn = connect_db(config.db_path)
    try:
        with conn:
            upsert_project(
                conn,
                id=config.project_id,
                genre="mystery",
                premise="A precise premise.",
                word_count_target=3000,
            )
            upsert_arc(
                conn,
                id=f"{config.project_id}-arc-1",
                project_id=config.project_id,
                ordering=0,
                description="A first arc.",
                status="active",
            )
    finally:
        conn.close()


async def test_dashboard_renders(config_factory, tmp_path):
    app, test_app = await _started_app(config_factory(), tmp_path)
    try:
        response = await app.test_client().get("/")
        assert response.status_code == 200
        assert "MuseAI Dashboard" in await response.get_data(as_text=True)
    finally:
        await _close_started_app(test_app)


async def test_stream_yields_hydration_then_published_event(config_factory, tmp_path):
    app, test_app = await _started_app(config_factory(), tmp_path)
    bus.last_snapshot.clear()
    await bus.publish("run_status", {"status": "idle"})
    try:
        async with app.test_request_context("/stream"):
            response = await stream()
            body = cast(Any, response.response).__aiter__()
            first = await asyncio.wait_for(anext(body), timeout=1)
            assert "event: hydration" in first
            assert '"run_status": {"status": "idle"}' in first

            await bus.publish("token", {"text": "A"})
            second = await asyncio.wait_for(anext(body), timeout=1)
            assert "event: token" in second
            assert '"text": "A"' in second
            await body.aclose()
    finally:
        await _close_started_app(test_app)
        bus.last_snapshot.clear()


async def test_generate_rejects_when_unseeded(config_factory, tmp_path):
    app, test_app = await _started_app(config_factory(), tmp_path)
    app_module.manager = MockManager()
    try:
        response = await app.test_client().post("/generate")
        assert response.status_code == 400
        body = await response.get_json()
        assert body["ok"] is False
        assert "No project is seeded" in body["error"]
    finally:
        await _close_started_app(test_app)


async def test_generate_starts_when_seeded_with_mocked_manager(config_factory, tmp_path):
    config = config_factory()
    app, test_app = await _started_app(config, tmp_path)
    manager = MockManager()
    app_module.manager = manager
    _seed_project(config)
    try:
        response = await app.test_client().post("/generate")
        assert response.status_code == 200
        body = await response.get_json()
        assert body["ok"] is True
        assert body["status"] == "running"
        assert manager.started == [config.project_id]
    finally:
        await _close_started_app(test_app)


async def test_settings_save_rejects_unknown_key(config_factory, tmp_path):
    config = config_factory()
    app, test_app = await _started_app(config, tmp_path)
    payload = config.model_dump()
    payload["unknown"] = True
    try:
        response = await app.test_client().post("/settings/save", json=payload)
        assert response.status_code == 400
        body = await response.get_json()
        assert body["ok"] is False
        assert "Extra inputs are not permitted" in body["error"]
    finally:
        await _close_started_app(test_app)


async def test_control_review_forwards_decision_to_manager(config_factory, tmp_path):
    app, test_app = await _started_app(config_factory(), tmp_path)
    manager = MockManager(status="review")
    app_module.manager = manager
    try:
        response = await app.test_client().post(
            "/control/review",
            json={"decision": "accept", "edited_text": "Edited draft."},
        )
        assert response.status_code == 200
        assert manager.reviews == [("accept", "Edited draft.")]
    finally:
        await _close_started_app(test_app)