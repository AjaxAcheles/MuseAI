"""Log tailing (with redaction) and manuscript export routes."""

from __future__ import annotations

from museai.core import logging_setup
from museai.web.routes.logs import redact

from conftest import add_beat, seed_project


# --------------------------------------------------------------------------- logs


async def test_logs_page_renders(config_factory, web_app):
    app = await web_app(config_factory())
    response = await app.test_client().get("/logs")
    assert response.status_code == 200
    body = await response.get_data(as_text=True)
    assert "fsm.log" in body
    assert "llm_io.log" in body


async def test_unknown_log_source_is_rejected(config_factory, web_app):
    app = await web_app(config_factory())
    response = await app.test_client().get("/logs/tail?source=/etc/passwd")
    assert response.status_code == 400
    assert "Unknown log source" in (await response.get_json())["error"]


async def test_missing_log_file_is_not_an_error(config_factory, web_app, tmp_path, monkeypatch):
    monkeypatch.setattr(logging_setup, "LOG_DIR", tmp_path / "logs")
    app = await web_app(config_factory())
    body = await (await app.test_client().get("/logs/tail?source=fsm")).get_json()
    assert body["ok"] is True
    assert body["exists"] is False
    assert body["lines"] == []


async def test_log_tail_returns_the_last_lines(config_factory, web_app, tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "fsm.log").write_text("line one\nline two\nline three\n", encoding="utf-8")
    monkeypatch.setattr(logging_setup, "LOG_DIR", log_dir)

    app = await web_app(config_factory())
    body = await (await app.test_client().get("/logs/tail?source=fsm&limit=2")).get_json()
    assert body["lines"] == ["line two", "line three"]


async def test_log_tail_redacts_secrets(config_factory, web_app, tmp_path, monkeypatch):
    """A planted credential must never reach the browser."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "llm_io.log").write_text(
        "\n".join(
            [
                "POST /v1/chat Authorization: Bearer sk-abcdef1234567890",
                'request api_key="sk-secretsecret123"',
                "resolved key sekrit-value-goes-here in flight",
                "ordinary prose line with no secret",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(logging_setup, "LOG_DIR", log_dir)

    config = config_factory()
    config.endpoint.api_key = "sekrit-value-goes-here"
    app = await web_app(config)

    body = await (await app.test_client().get("/logs/tail?source=llm_io")).get_json()
    joined = "\n".join(body["lines"])

    assert "sk-abcdef1234567890" not in joined
    assert "sk-secretsecret123" not in joined
    assert "sekrit-value-goes-here" not in joined
    assert "[REDACTED]" in joined
    # Redaction must not eat ordinary content.
    assert "ordinary prose line with no secret" in joined


def test_redact_leaves_innocent_text_alone():
    assert redact("node=commit beat_id=beat-1 word_count=412") == "node=commit beat_id=beat-1 word_count=412"


def test_redact_handles_several_key_spellings():
    assert "hunter2" not in redact('api-key: hunter2secret')
    assert "hunter2" not in redact('"api_key": "hunter2secret"')
    assert "topsecret" not in redact("authorization=Bearer topsecretvalue")


# ------------------------------------------------------------------------ exports


async def test_exports_page_renders_without_a_manuscript(config_factory, web_app):
    app = await web_app(config_factory())
    response = await app.test_client().get("/exports")
    assert response.status_code == 200
    body = await response.get_data(as_text=True)
    assert "No manuscript has been exported yet" in body


async def test_export_rejects_when_nothing_is_committed(config_factory, web_app):
    config = config_factory()
    app = await web_app(config)
    seed_project(config)

    response = await app.test_client().post("/exports/manuscript")
    assert response.status_code == 400
    assert "No committed story to export yet." in (await response.get_json())["error"]


async def test_export_writes_committed_prose(config_factory, web_app, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # export_manuscript writes to ./data/output
    config = config_factory()
    app = await web_app(config)
    seed_project(config)
    add_beat(config, beat_id="beat-1", ordering=0, status="completed", prose="The lamp held.", word_count=3)

    response = await app.test_client().post("/exports/manuscript")
    assert response.status_code == 200
    body = await response.get_json()
    assert body["ok"] is True
    assert body["word_count"] == 3

    written = (tmp_path / "data" / "output" / f"{config.project_id}.md").read_text(encoding="utf-8")
    assert "The lamp held." in written


async def test_export_excludes_uncommitted_drafts(config_factory, web_app, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = config_factory()
    app = await web_app(config)
    seed_project(config)
    add_beat(config, beat_id="beat-1", ordering=0, status="completed", prose="Committed line.", word_count=2)
    add_beat(config, beat_id="beat-2", ordering=1, status="planned", prose="Draft line.", word_count=2)

    await app.test_client().post("/exports/manuscript")
    written = (tmp_path / "data" / "output" / f"{config.project_id}.md").read_text(encoding="utf-8")
    assert "Committed line." in written
    assert "Draft line." not in written


async def test_download_404s_before_an_export(config_factory, web_app, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = await web_app(config_factory())
    response = await app.test_client().get("/exports/download")
    assert response.status_code == 404
