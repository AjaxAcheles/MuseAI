"""The Seed & Plan workspace: `/setup`, `/seed`, and seed submission."""

from __future__ import annotations

import json
from pathlib import Path

from museai.memory.db import connect_db


def _example_seed() -> dict:
    return json.loads(Path("seeds/example.json").read_text(encoding="utf-8"))


async def test_setup_renders(config_factory, web_app):
    app = await web_app(config_factory())
    response = await app.test_client().get("/setup")
    assert response.status_code == 200
    body = await response.get_data(as_text=True)
    assert "Seed &amp; Plan" in body


async def test_seed_is_an_alias_of_setup(config_factory, web_app):
    """`/seed` predates `/setup`; both must serve the same workspace."""
    app = await web_app(config_factory())
    client = app.test_client()
    setup = await (await client.get("/setup")).get_data(as_text=True)
    seed = await (await client.get("/seed")).get_data(as_text=True)
    assert setup == seed


async def test_workspace_offers_all_three_modes(config_factory, web_app):
    app = await web_app(config_factory())
    body = await (await app.test_client().get("/setup")).get_data(as_text=True)
    assert "JSON Editor" in body
    assert "Plain Text" in body
    assert "Preview Timeline" in body
    assert "Reset to test seed" in body


async def test_seed_page_prefills_the_example_seed(config_factory, web_app):
    app = await web_app(config_factory())
    body = await (await app.test_client().get("/setup")).get_data(as_text=True)
    seed = _example_seed()
    assert seed["project"]["id"] in body
    assert seed["characters"][0]["name"] in body


async def test_valid_seed_loads_and_flips_status(config_factory, web_app):
    config = config_factory()
    app = await web_app(config)
    client = app.test_client()

    before = await (await client.get("/status")).get_json()
    assert before["seed_loaded"] is False
    assert before["can_generate"] is False

    seed = _example_seed()
    seed["project"]["id"] = config.project_id
    response = await client.post("/seed/submit", json=seed)
    assert response.status_code == 200
    assert (await response.get_json())["project_id"] == config.project_id

    after = await (await client.get("/status")).get_json()
    assert after["seed_loaded"] is True
    assert after["can_generate"] is True
    assert after["counts"] == {"arcs": 1, "threads": 2, "characters": 2}


async def test_loaded_seed_populates_the_database(config_factory, web_app):
    config = config_factory()
    app = await web_app(config)
    seed = _example_seed()
    seed["project"]["id"] = config.project_id
    await app.test_client().post("/seed/submit", json=seed)

    conn = connect_db(config.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM Arcs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM Threads").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM CharacterEmotions").fetchone()[0] == 2
    finally:
        conn.close()


async def test_seed_with_new_project_id_rewrites_config_yaml(
    config_factory, web_app, tmp_path, monkeypatch
):
    """Loading a seed must point config.yaml (and the runtime) at that project.

    A stale project_id is exactly what made a finished run export an empty
    manuscript under the previous project's name.
    """
    import yaml

    monkeypatch.setenv("MUSEAI_API_KEY", "secret-from-env")
    config = config_factory()
    app = await web_app(config)

    # web_app registers tmp_path/config.yaml as MUSEAI_CONFIG_PATH but never
    # writes it; the sync path needs the real file, with an env-ref api_key.
    cfg_path = tmp_path / "config.yaml"
    on_disk = config.model_dump()
    on_disk["endpoint"]["api_key"] = "${MUSEAI_API_KEY}"
    cfg_path.write_text(yaml.safe_dump(on_disk, sort_keys=False), encoding="utf-8")

    seed = _example_seed()
    assert seed["project"]["id"] != config.project_id
    response = await app.test_client().post("/seed/submit", json=seed)
    assert response.status_code == 200

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert raw["project_id"] == seed["project"]["id"]
    # The ${VAR} reference survives the rewrite; the secret never lands on disk.
    assert raw["endpoint"]["api_key"] == "${MUSEAI_API_KEY}"

    status = await (await app.test_client().get("/status")).get_json()
    assert status["project"]["id"] == seed["project"]["id"]
    assert status["seed_loaded"] is True


async def test_seed_sync_failure_is_reported_not_silent(config_factory, web_app):
    """No config.yaml on disk: the seed loads but the sync failure surfaces."""
    app = await web_app(config_factory())
    seed = _example_seed()  # its project id differs from the configured one
    response = await app.test_client().post("/seed/submit", json=seed)
    assert response.status_code == 400
    assert "config file not found" in (await response.get_json())["error"]


async def test_seed_without_arcs_is_rejected(config_factory, web_app):
    app = await web_app(config_factory())
    response = await app.test_client().post("/seed/submit", json={"project": {"id": "p1"}, "arcs": []})
    assert response.status_code == 400
    assert "seed.arcs must be a non-empty list" in (await response.get_json())["error"]


async def test_seed_without_project_id_is_rejected(config_factory, web_app):
    app = await web_app(config_factory())
    response = await app.test_client().post("/seed/submit", json={"arcs": [{"description": "An arc."}]})
    assert response.status_code == 400
    assert "seed.project.id is required" in (await response.get_json())["error"]


async def test_out_of_range_pad_is_rejected(config_factory, web_app):
    """The loader's PAD bounds are enforced server-side, not just in the browser."""
    app = await web_app(config_factory())
    seed = {
        "project": {"id": "p1"},
        "arcs": [{"description": "An arc."}],
        "characters": [{"name": "Ada", "pad": {"pleasure": 5.0, "arousal": 0.0, "dominance": 0.0}}],
    }
    response = await app.test_client().post("/seed/submit", json=seed)
    assert response.status_code == 400
    assert "between -1 and 1" in (await response.get_json())["error"]


async def test_malformed_form_submission_preserves_user_text(config_factory, web_app):
    """A form post that fails validation must return the text the user typed, not the example."""
    app = await web_app(config_factory())
    response = await app.test_client().post(
        "/seed/submit", form={"seed_json": '{"project": {"id": "p1"}, "arcs": []}'}
    )
    assert response.status_code == 400
    body = await response.get_data(as_text=True)
    assert "seed.arcs must be a non-empty list" in body
    assert "&#34;p1&#34;" in body or '"p1"' in body


async def test_malformed_json_form_submission_reports_the_parse_error(config_factory, web_app):
    app = await web_app(config_factory())
    response = await app.test_client().post("/seed/submit", form={"seed_json": "{not json"})
    assert response.status_code == 400
    assert "Malformed seed JSON" in (await response.get_data(as_text=True))
