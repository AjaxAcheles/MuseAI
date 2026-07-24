"""Read-only database inspection routes."""

from __future__ import annotations

import json
from pathlib import Path

from museai.core.events import append_event
from museai.seed.loader import load_seed

from conftest import add_beat, seed_project


def _example_seed() -> dict:
    return json.loads(Path("seeds/example.json").read_text(encoding="utf-8"))


async def test_database_page_renders(config_factory, web_app):
    app = await web_app(config_factory())
    response = await app.test_client().get("/database")
    assert response.status_code == 200
    body = await response.get_data(as_text=True)
    # Every real table is offered; nothing that does not exist is.
    for table in ("Projects", "Arcs", "Chapters", "Beats", "Threads", "Characters", "CommitIntent"):
        assert table in body


async def test_unknown_record_type_is_rejected(config_factory, web_app):
    app = await web_app(config_factory())
    response = await app.test_client().get("/database/records?type=NotATable")
    assert response.status_code == 400
    body = await response.get_json()
    assert body["ok"] is False
    assert "Unknown record type" in body["error"]


async def test_missing_record_type_is_rejected(config_factory, web_app):
    app = await web_app(config_factory())
    response = await app.test_client().get("/database/records")
    assert response.status_code == 400


async def test_sql_injection_attempt_is_rejected_not_executed(config_factory, web_app):
    app = await web_app(config_factory())
    response = await app.test_client().get("/database/records?type=Projects;DROP TABLE Projects")
    assert response.status_code == 400


async def test_seeded_records_appear(config_factory, web_app):
    """Loading the bundled example seed makes its rows visible through the route."""
    config = config_factory()
    app = await web_app(config)
    seed = _example_seed()
    seed["project"]["id"] = config.project_id
    load_seed(seed, config)

    client = app.test_client()

    arcs = await (await client.get("/database/records?type=Arcs")).get_json()
    assert arcs["ok"] is True
    assert arcs["total"] == 1

    characters = await (await client.get("/database/records?type=Characters")).get_json()
    assert characters["total"] == 2
    # Derived from the seed file: a seed rewrite must not break this test.
    assert {record["name"] for record in characters["records"]} == {
        character["name"] for character in seed["characters"]
    }

    emotions = await (await client.get("/database/records?type=CharacterEmotions")).get_json()
    assert emotions["total"] == 2


async def test_record_search_filters_rows(config_factory, web_app):
    config = config_factory()
    app = await web_app(config)
    seed = _example_seed()
    seed["project"]["id"] = config.project_id
    load_seed(seed, config)

    client = app.test_client()
    wanted = seed["characters"][0]["name"]
    query = wanted.split()[-1].lower()  # surname: unique to one row
    hit = await (await client.get(f"/database/records?type=Characters&q={query}")).get_json()
    assert hit["total"] == 1
    assert hit["records"][0]["name"] == wanted

    miss = await (await client.get("/database/records?type=Characters&q=nobody")).get_json()
    assert miss["total"] == 0


async def test_beats_records_are_readable(config_factory, web_app):
    config = config_factory()
    app = await web_app(config)
    seed_project(config)
    add_beat(config, beat_id="beat-1", ordering=0, status="completed", prose="Prose.", word_count=1)

    body = await (await app.test_client().get("/database/records?type=Beats")).get_json()
    assert body["total"] == 1
    assert body["records"][0]["id"] == "beat-1"


async def test_event_log_tail_reads_the_jsonl(config_factory, web_app):
    config = config_factory()
    app = await web_app(config)
    append_event(config.event_log_path, {"type": "beat_commit", "beat_id": "beat-1", "word_count": 12})
    append_event(config.event_log_path, {"type": "beat_commit", "beat_id": "beat-2", "word_count": 15})

    body = await (await app.test_client().get("/database/event-log?limit=1")).get_json()
    assert body["total"] == 2
    assert body["returned"] == 1
    # Tail: the newest event, not the oldest.
    assert body["events"][0]["beat_id"] == "beat-2"


async def test_event_log_is_empty_before_any_commit(config_factory, web_app):
    app = await web_app(config_factory())
    body = await (await app.test_client().get("/database/event-log")).get_json()
    assert body["ok"] is True
    assert body["events"] == []
