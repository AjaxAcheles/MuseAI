"""Tests for the MuseAI v1 Quart web surface."""

from __future__ import annotations

import asyncio
from typing import Any, cast

from museai.core.config import load_config
from museai.core.stream_bus import bus
from museai.llm.client import LLMCallError, TOOL_SUPPORT_DIAGNOSIS
from museai.memory.db import connect_db, upsert_arc, upsert_project
from museai.web import app as app_module
from museai.web.app import create_app
from museai.web.routes import settings as settings_route
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
        body = await response.get_data(as_text=True)
        assert "MuseAI Dashboard" in body
        # An unseeded dashboard must offer a way to seed and must not imply it can generate.
        assert "No seed loaded" in body
        assert "Load Seed" in body
        assert "role=\"progressbar\"" in body
    finally:
        await _close_started_app(test_app)


async def test_seed_page_presents_default_as_short_test_seed(config_factory, tmp_path):
    app, test_app = await _started_app(config_factory(), tmp_path)
    try:
        response = await app.test_client().get("/seed")
        assert response.status_code == 200
        body = await response.get_data(as_text=True)
        assert "Reset to test seed" in body
        assert "one compact arc" in body
        assert "rising action, climax, and resolution" in body
        assert "sets no" in body
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


async def test_settings_save_preserves_comments_and_removes_stale_keys(
    config_factory, tmp_path
):
    import yaml

    config = config_factory()
    config = config.model_copy(
        update={"generation": config.generation.model_copy(update={"emotion_words": ["fear", "anger"]})}
    )
    config_path = tmp_path / "config.yaml"
    on_disk = yaml.safe_dump(config.model_dump(), sort_keys=False)
    on_disk = "# This operational note must survive Save.\n" + on_disk
    on_disk = on_disk.replace(
        "word_count_target: 1500",
        "word_count_target: 1500 # This changed-value comment must survive Save.",
    )
    on_disk = on_disk.replace(
        "  emotion_words:\n  - fear\n  - anger",
        "  emotion_words:\n  - fear\n  # A category heading inside the list.\n  - anger",
    )
    on_disk = on_disk.replace(
        "  retry_backoff_seconds:\n  - 5.0\n  - 15.0",
        "  retry_backoff_seconds: [5.0, 15.0]",
    )
    config_path.write_text(on_disk + "legacy_setting: remove me\n", encoding="utf-8")

    app, test_app = await _started_app(config, tmp_path)
    payload = config.model_dump()
    payload["generation"]["word_count_target"] = 2400
    try:
        response = await app.test_client().post("/settings/save", json=payload)
        assert response.status_code == 200

        persisted = config_path.read_text(encoding="utf-8")
        assert "# This operational note must survive Save." in persisted
        assert "# This changed-value comment must survive Save." in persisted
        assert "# A category heading inside the list." in persisted
        assert "retry_backoff_seconds: [5.0, 15.0]" in persisted
        assert "max_output_tokens: null" in persisted
        assert "legacy_setting" not in persisted
        assert load_config(config_path).generation.word_count_target == 2400
    finally:
        await _close_started_app(test_app)


async def test_settings_save_creates_loadable_config_when_missing(config_factory, tmp_path):
    config = config_factory()
    config_path = tmp_path / "config.yaml"
    app, test_app = await _started_app(config, tmp_path)
    try:
        assert not config_path.exists()
        response = await app.test_client().post("/settings/save", json=config.model_dump())
        assert response.status_code == 200
        assert load_config(config_path) == config
    finally:
        await _close_started_app(test_app)


async def test_settings_renders_agent_reasoning_effort_controls(config_factory, tmp_path):
    app, test_app = await _started_app(config_factory(), tmp_path)
    try:
        response = await app.test_client().get("/settings")
        assert response.status_code == 200
        html = await response.get_data(as_text=True)
        for role in ("chapter_planner", "beat_planner", "drafter", "reviser", "critic"):
            assert f'id="setting-agent-{role}-reasoning-effort"' in html
    finally:
        await _close_started_app(test_app)


async def test_settings_save_persists_agent_reasoning_effort(config_factory, tmp_path):
    config = config_factory()
    app, test_app = await _started_app(config, tmp_path)
    config_path = tmp_path / "config.yaml"
    payload = config.model_dump()
    payload["agents"] = {"critic": {"reasoning_effort": "none"}}
    try:
        response = await app.test_client().post("/settings/save", json=payload)
        assert response.status_code == 200
        assert load_config(config_path).agents["critic"].reasoning_effort == "none"
    finally:
        await _close_started_app(test_app)


async def test_settings_save_keeps_agent_overrides_sparse(config_factory, tmp_path):
    import yaml

    config = config_factory()
    config_path = tmp_path / "config.yaml"
    on_disk = config.model_dump()
    on_disk["agents"] = {"chapter_planner": {"temperature": 0.4}}
    config_path.write_text(yaml.safe_dump(on_disk, sort_keys=False), encoding="utf-8")

    app, test_app = await _started_app(config, tmp_path)
    payload = config.model_dump()
    payload["agents"] = {"chapter_planner": {"temperature": 0.4}}
    try:
        response = await app.test_client().post("/settings/save", json=payload)
        assert response.status_code == 200

        persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert persisted["agents"]["chapter_planner"] == {"temperature": 0.4}
        assert "base_url" not in persisted["agents"]["chapter_planner"]
    finally:
        await _close_started_app(test_app)


async def test_settings_save_clears_agent_override(config_factory, tmp_path):
    import yaml

    config = config_factory(agents={"critic": {"reasoning_effort": "none"}})
    config_path = tmp_path / "config.yaml"
    on_disk = config.model_dump()
    on_disk["agents"] = {"critic": {"reasoning_effort": "none"}}
    config_path.write_text(yaml.safe_dump(on_disk, sort_keys=False), encoding="utf-8")

    app, test_app = await _started_app(config, tmp_path)
    payload = config.model_dump()
    payload["agents"] = {"critic": {"reasoning_effort": None}}
    try:
        response = await app.test_client().post("/settings/save", json=payload)
        assert response.status_code == 200

        persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert "reasoning_effort" not in persisted["agents"]["critic"]
        assert load_config(config_path).agents["critic"].reasoning_effort is None
    finally:
        await _close_started_app(test_app)


async def test_settings_save_keeps_empty_agent_max_output_tokens_null(
    config_factory, tmp_path
):
    config = config_factory()
    app, test_app = await _started_app(config, tmp_path)
    config_path = tmp_path / "config.yaml"
    payload = config.model_dump()
    # This is the JSON representation the nullable browser control sends when blank.
    payload["agents"] = {"drafter": {"max_output_tokens": None}}
    try:
        response = await app.test_client().post("/settings/save", json=payload)
        assert response.status_code == 200
        assert load_config(config_path).agents["drafter"].max_output_tokens is None
    finally:
        await _close_started_app(test_app)


async def test_settings_save_rejects_agent_cap_above_reservation_with_window(
    config_factory, tmp_path
):
    config = config_factory()
    app, test_app = await _started_app(config, tmp_path)
    config_path = tmp_path / "config.yaml"
    payload = config.model_dump()
    payload["endpoint"].update({"context_window": 4096, "output_reservation": 1024})
    payload["agents"] = {
        "critic": {"max_output_tokens": 1200, "output_reservation": 1000}
    }
    try:
        response = await app.test_client().post("/settings/save", json=payload)
        assert response.status_code == 400
        body = await response.get_json()
        assert "max_output_tokens" in body["error"]
        assert "output_reservation" in body["error"]
        assert not config_path.exists()
    finally:
        await _close_started_app(test_app)


async def test_settings_save_round_trips_research_mode_and_narrative_person(
    config_factory, tmp_path
):
    config = config_factory()
    app, test_app = await _started_app(config, tmp_path)
    config_path = tmp_path / "config.yaml"
    payload = config.model_dump()
    payload["generation"].update({"research_mode": True, "narrative_person": "third"})
    try:
        response = await app.test_client().post("/settings/save", json=payload)
        assert response.status_code == 200
        reloaded = load_config(config_path)
        assert reloaded.generation.research_mode is True
        assert reloaded.generation.narrative_person == "third"

        payload = reloaded.model_dump()
        payload["generation"]["narrative_person"] = None
        response = await app.test_client().post("/settings/save", json=payload)
        assert response.status_code == 200
        assert load_config(config_path).generation.narrative_person is None
    finally:
        await _close_started_app(test_app)


async def test_settings_render_omits_api_key_and_old_hints(config_factory, tmp_path):
    config = config_factory()
    app, test_app = await _started_app(config, tmp_path)
    try:
        response = await app.test_client().get("/settings")
        html = await response.get_data(as_text=True)
        assert config.endpoint.api_key not in html
        assert "Failed audits before parking for human review." not in html
        assert "Critic tool-use loop ceiling." not in html
        assert "Exceeding it sends the draft to revision." not in html
    finally:
        await _close_started_app(test_app)


async def test_endpoint_probe_reports_tool_support_rejection(
    config_factory, tmp_path, monkeypatch
):
    app, test_app = await _started_app(config_factory(), tmp_path)
    calls: list[dict[str, Any]] = []

    async def fake_call_llm(*args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("tools"):
            raise LLMCallError(TOOL_SUPPORT_DIAGNOSIS)
        return type("Response", (), {"text": "ok", "model_name": "test-model"})()

    monkeypatch.setattr(settings_route, "call_llm", fake_call_llm)
    try:
        response = await app.test_client().post("/settings/test_endpoint")
        body = await response.get_json()
        assert body == {"ok": False, "error": TOOL_SUPPORT_DIAGNOSIS}
        assert len(calls) == 2
        assert calls[1]["tools"]
    finally:
        await _close_started_app(test_app)


async def test_endpoint_probe_succeeds_when_plain_and_tool_calls_pass(
    config_factory, tmp_path, monkeypatch
):
    app, test_app = await _started_app(config_factory(), tmp_path)
    calls: list[dict[str, Any]] = []

    async def fake_call_llm(*args, **kwargs):
        calls.append(kwargs)
        return type("Response", (), {"text": "ok", "model_name": "test-model"})()

    monkeypatch.setattr(settings_route, "call_llm", fake_call_llm)
    try:
        response = await app.test_client().post("/settings/test_endpoint")
        assert await response.get_json() == {"ok": True, "model": "test-model"}
        assert len(calls) == 2
        assert "tools" not in calls[0]
        assert calls[1]["tools"]
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

async def test_seed_submit_error_preserves_user_text(config_factory, tmp_path):
    """A rejected seed must re-render with the submitted text, not erase it."""
    app, test_app = await _started_app(config_factory(), tmp_path)
    try:
        bad_seed = '{"project": {"id": "p1"}, "arcs": []}'
        response = await app.test_client().post(
            "/seed/submit", form={"seed_json": bad_seed}
        )
        assert response.status_code == 400
        body = await response.get_data(as_text=True)
        assert "seed.arcs must be a non-empty list" in body
        assert "&#34;p1&#34;" in body or '"p1"' in body  # submitted text survives
    finally:
        await _close_started_app(test_app)


async def test_settings_save_preserves_env_reference_on_disk(
    config_factory, tmp_path, monkeypatch
):
    """Saving with a blank key must keep the raw ${VAR} reference in config.yaml,
    never the resolved secret."""
    import yaml

    monkeypatch.setenv("TEST_MUSEAI_KEY", "sekrit-value")
    config = config_factory()
    config_path = tmp_path / "config.yaml"

    on_disk = config.model_dump()
    on_disk["endpoint"]["api_key"] = "${TEST_MUSEAI_KEY}"
    config_path.write_text(yaml.safe_dump(on_disk, sort_keys=False), encoding="utf-8")

    app = create_app(config_path=config_path, test_config=config)
    test_app = app.test_app()
    await test_app.__aenter__()
    try:
        response = await app.test_client().get("/settings")
        html = await response.get_data(as_text=True)
        assert "sekrit-value" not in html
        assert "${TEST_MUSEAI_KEY}" not in html

        payload = config.model_dump()
        payload["endpoint"]["api_key"] = ""  # "keep the current key"
        response = await app.test_client().post("/settings/save", json=payload)
        assert response.status_code == 200

        persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert persisted["endpoint"]["api_key"] == "${TEST_MUSEAI_KEY}"
        assert "sekrit-value" not in config_path.read_text(encoding="utf-8")
        assert "test-key" not in config_path.read_text(encoding="utf-8")
    finally:
        await _close_started_app(test_app)


async def test_status_reports_project_word_target(config_factory, tmp_path):
    config = config_factory()
    app, test_app = await _started_app(config, tmp_path)
    app_module.manager = MockManager()
    conn = connect_db(config.db_path)
    try:
        with conn:
            upsert_project(
                conn,
                id=config.project_id,
                genre="mystery",
                premise="A premise.",
                word_count_target=5000,
            )
    finally:
        conn.close()
    try:
        response = await app.test_client().get("/status")
        body = await response.get_json()
        assert body["ok"] is True
        assert body["word_target"] == 5000
    finally:
        await _close_started_app(test_app)


async def test_status_reports_no_word_target_before_seed(config_factory, tmp_path):
    app, test_app = await _started_app(config_factory(), tmp_path)
    app_module.manager = MockManager()
    try:
        response = await app.test_client().get("/status")
        body = await response.get_json()
        assert body["ok"] is True
        assert body["word_target"] is None
    finally:
        await _close_started_app(test_app)


async def test_stream_bus_drops_oldest_when_subscriber_stalls():
    from museai.core.stream_bus import _MAX_QUEUE_EVENTS, StreamBus

    stalled_bus = StreamBus()
    queue = stalled_bus.subscribe()
    try:
        overflow = 5
        for i in range(_MAX_QUEUE_EVENTS + overflow):
            await stalled_bus.publish("token", {"i": i})

        assert queue.qsize() == _MAX_QUEUE_EVENTS
        first = queue.get_nowait()
        assert first["data"]["i"] == overflow  # the oldest were dropped
    finally:
        stalled_bus.unsubscribe(queue)
