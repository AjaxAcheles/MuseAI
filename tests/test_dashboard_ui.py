"""Dashboard surface: seed gating, committed-story isolation, and the outline rail.

The rule these tests defend: the Committed Story panel is fed by `/committed`, and `/committed`
returns committed beats only. A draft that never passed review must not be reachable from it.
"""

from __future__ import annotations

from museai.web import app as app_module

from conftest import MockManager, add_beat, seed_project


async def test_dashboard_renders(config_factory, web_app):
    app = await web_app(config_factory())
    response = await app.test_client().get("/")
    assert response.status_code == 200
    body = await response.get_data(as_text=True)
    assert "Committed Story" in body
    assert "Live Activity" in body
    assert "Story progress" in body


async def test_generate_is_disabled_in_markup_before_seed(config_factory, web_app):
    """The button ships disabled; only a `/status` with `seed_loaded` re-enables it."""
    app = await web_app(config_factory())
    body = await (await app.test_client().get("/dashboard")).get_data(as_text=True)
    assert 'id="generate-button"' in body
    assert "disabled" in body.split('id="generate-button"')[1].split(">")[0]


async def test_status_reports_no_seed(config_factory, web_app):
    app = await web_app(config_factory())
    body = await (await app.test_client().get("/status")).get_json()
    assert body["seed_loaded"] is False
    assert body["can_generate"] is False
    assert body["project"] is None
    assert body["counts"] == {"arcs": 0, "threads": 0, "characters": 0}
    assert body["last_commit"] is None


async def test_status_reports_seed_and_allows_generate(config_factory, web_app):
    config = config_factory()
    app = await web_app(config)
    seed_project(config)

    body = await (await app.test_client().get("/status")).get_json()
    assert body["seed_loaded"] is True
    assert body["can_generate"] is True
    assert body["project"]["id"] == config.project_id
    assert body["counts"]["arcs"] == 1
    assert body["word_target"] == 3000


async def test_running_manager_cannot_generate(config_factory, web_app, monkeypatch):
    config = config_factory()
    app = await web_app(config)
    seed_project(config)
    monkeypatch.setattr(app_module, "manager", MockManager(status="running"))

    body = await (await app.test_client().get("/status")).get_json()
    assert body["seed_loaded"] is True
    assert body["can_generate"] is False


async def test_generate_rejects_before_seed_and_starts_after(config_factory, web_app, monkeypatch):
    config = config_factory()
    app = await web_app(config)
    mock = MockManager()
    monkeypatch.setattr(app_module, "manager", mock)
    client = app.test_client()

    rejected = await client.post("/generate")
    assert rejected.status_code == 400
    assert "No project is seeded" in (await rejected.get_json())["error"]
    assert mock.started == []

    seed_project(config)
    accepted = await client.post("/generate")
    assert accepted.status_code == 200
    assert mock.started == [config.project_id]


async def test_committed_returns_only_completed_beats(config_factory, web_app):
    """A planned beat that already has prose is a draft, not manuscript."""
    config = config_factory()
    app = await web_app(config)
    seed_project(config)

    add_beat(config, beat_id="beat-1", ordering=0, status="completed", prose="Committed prose.", word_count=2)
    add_beat(config, beat_id="beat-2", ordering=1, status="planned", prose="Discarded draft text.", word_count=3)

    body = await (await app.test_client().get("/committed")).get_json()
    beat_ids = [beat["beat_id"] for beat in body["beats"]]
    assert beat_ids == ["beat-1"]

    serialized = str(body)
    assert "Committed prose." in serialized
    assert "Discarded draft text." not in serialized


async def test_committed_word_total_comes_from_backend(config_factory, web_app):
    config = config_factory()
    app = await web_app(config)
    seed_project(config)
    add_beat(config, beat_id="beat-1", ordering=0, status="completed", prose="Two words.", word_count=17)

    body = await (await app.test_client().get("/committed")).get_json()
    assert body["project_word_total"] == 17

    status = await (await app.test_client().get("/status")).get_json()
    assert status["project_word_total"] == 17
    assert status["last_commit"]["beat_id"] == "beat-1"
    assert status["last_commit"]["word_count"] == 17


async def test_committed_is_empty_without_a_seed(config_factory, web_app):
    app = await web_app(config_factory())
    body = await (await app.test_client().get("/committed")).get_json()
    assert body["beats"] == []
    assert body["project_word_total"] == 0


async def test_outline_returns_arcs_without_prose(config_factory, web_app):
    config = config_factory()
    app = await web_app(config)
    seed_project(config)
    add_beat(config, beat_id="beat-1", ordering=0, status="completed", prose="Secret prose.", word_count=2)

    body = await (await app.test_client().get("/outline")).get_json()
    assert body["ok"] is True
    assert len(body["arcs"]) == 1

    arc = body["arcs"][0]
    assert arc["status"] == "active"
    assert len(arc["chapters"]) == 1
    assert arc["chapters"][0]["beats"][0]["status"] == "completed"
    # The rail shows position, never text.
    assert "Secret prose." not in str(body)


async def test_outline_shows_arcs_before_planning(config_factory, web_app):
    """A freshly seeded project has arcs and no chapters. The rail must render exactly that."""
    config = config_factory()
    app = await web_app(config)
    seed_project(config)

    body = await (await app.test_client().get("/outline")).get_json()
    assert len(body["arcs"]) == 1
    assert body["arcs"][0]["chapters"] == []


async def test_outline_is_empty_without_a_seed(config_factory, web_app):
    app = await web_app(config_factory())
    body = await (await app.test_client().get("/outline")).get_json()
    assert body["arcs"] == []
    assert body["pointer"] is None
