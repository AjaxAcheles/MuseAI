"""Module: M17 (Web UI & Real-time Observer Surface)
Product-surface tests for the VS-02 planning workbench: the project setup
form exists, the premise is required by the backend, and per-run mode
overrides are validated. Deterministic, temp stores, zero network.
"""

from __future__ import annotations

import asyncio

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

_PREMISE = "A cartographer must chart a coastline that keeps moving."


@pytest.fixture
def isolated_resources(tmp_path, monkeypatch):
    for key in _ENDPOINT_ENV_KEYS:
        monkeypatch.setenv(key, "synthetic-test-secret")
    return reset_resources_for_tests(tmp_path)


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=60))


def test_dashboard_has_project_setup_form(isolated_resources):
    async def scenario():
        app = create_app()
        async with app.test_app() as test_app:
            client = test_app.test_client()
            response = await client.get("/dashboard")
            assert response.status_code == 200
            body = (await response.get_data()).decode("utf-8")
            for anchor in (
                'id="inp-title"',
                'id="inp-premise"',
                'id="inp-genre"',
                'id="inp-words"',
                'id="sel-exec-mode"',
                'id="sel-approval-mode"',
                'id="planning-timeline-card"',
                'id="node-detail-card"',
                'id="approval-card"',
                'id="btn-approve"',
                'id="event-filters"',
                'data-filter="error"',
            ):
                assert anchor in body
            # Mode vocabularies are offered, never invented.
            assert 'value="macro_outline_before_draft"' in body
            assert 'value="rolling"' in body
            assert 'value="macro_outline"' in body

    _run(scenario())


def test_start_requires_premise(isolated_resources):
    async def scenario():
        app = create_app()
        async with app.test_app() as test_app:
            client = test_app.test_client()

            empty = await client.post("/control/start", json={})
            body = await empty.get_json()
            assert body["ok"] is False
            assert "premise" in body["message"]

            blank = await client.post("/control/start", json={"premise": "   "})
            blank_body = await blank.get_json()
            assert blank_body["ok"] is False
            assert "premise" in blank_body["message"]

            manager = get_resources().generation_manager
            assert manager.status()["running"] is False

    _run(scenario())


def test_valid_start_begins_a_run(isolated_resources):
    async def scenario():
        app = create_app()
        async with app.test_app() as test_app:
            client = test_app.test_client()
            started = await client.post(
                "/control/start",
                json={
                    "premise": _PREMISE,
                    "title": "The Moving Coast",
                    "genre": "fantasy",
                    "target_word_count": 60000,
                    "planning_execution_mode": "macro_outline_before_draft",
                    "approval_mode": "off",
                },
            )
            body = await started.get_json()
            assert body["ok"] is True

            manager = get_resources().generation_manager
            await manager.join()
            assert manager.status()["status"] == "completed"
            metadata = manager.final_state["project_metadata"]
            assert metadata["premise_seed"] == _PREMISE
            assert metadata["title"] == "The Moving Coast"
            assert metadata["target_word_count"] == 60000

    _run(scenario())


def test_invalid_mode_combination_is_rejected(isolated_resources):
    async def scenario():
        app = create_app()
        async with app.test_app() as test_app:
            client = test_app.test_client()
            response = await client.post(
                "/control/start",
                json={
                    "premise": _PREMISE,
                    "planning_execution_mode": "rolling",
                    "approval_mode": "macro_outline",
                },
            )
            body = await response.get_json()
            assert body["ok"] is False
            assert "macro_outline" in body["message"]

            bad_words = await client.post(
                "/control/start",
                json={"premise": _PREMISE, "target_word_count": -5},
            )
            bad_words_body = await bad_words.get_json()
            assert bad_words_body["ok"] is False
            assert "target_word_count" in bad_words_body["message"]

    _run(scenario())
