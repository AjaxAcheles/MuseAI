"""Log tailing (with redaction) and manuscript export routes."""

from __future__ import annotations

import logging

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


async def test_export_follows_the_seeded_project_not_a_stale_config(
    config_factory, web_app, tmp_path, monkeypatch
):
    """A config.yaml pointing at a project the DB no longer has must not
    produce an empty manuscript under the stale name."""
    from museai.memory.db import connect_db, upsert_arc, upsert_project

    monkeypatch.chdir(tmp_path)
    config = config_factory(project_id="stale-ghost")
    app = await web_app(config)

    # The DB holds a different, real project — the stale-config scenario.
    conn = connect_db(config.db_path)
    with conn:
        upsert_project(conn, id="real-project", genre="mystery", premise="p", word_count_target=100)
        upsert_arc(conn, id="real-project-arc-1", project_id="real-project",
                   ordering=0, description="arc", status="active")
        from museai.memory.db import upsert_beat, upsert_chapter
        upsert_chapter(conn, id="real-project-ch-1", arc_id="real-project-arc-1",
                       ordering=0, description="ch", status="active")
        upsert_beat(conn, id="real-beat", chapter_id="real-project-ch-1", ordering=0,
                    prose="The real prose.", word_count=3, status="completed")
    conn.close()

    response = await app.test_client().post("/exports/manuscript")
    assert response.status_code == 200
    body = await response.get_json()
    assert body["word_count"] == 3
    assert body["path"].endswith("real-project.md")

    written = (tmp_path / "data" / "output" / "real-project.md").read_text(encoding="utf-8")
    assert "The real prose." in written
    assert not (tmp_path / "data" / "output" / "stale-ghost.md").exists()


def test_export_numbers_chapters_across_arcs_with_epigraphs(
    config_factory, tmp_path, monkeypatch
):
    """Two arcs: arc headings, continuous chapter numbers, italic epigraphs."""
    from museai.core.runtime import init_resources
    from museai.fsm.export import export_manuscript
    from museai.memory.db import connect_db, upsert_arc, upsert_beat, upsert_chapter

    monkeypatch.chdir(tmp_path)
    config = config_factory()
    init_resources(config)
    seed_project(config)
    conn = connect_db(config.db_path)
    with conn:
        upsert_arc(conn, id="arc-2", project_id=config.project_id,
                   ordering=1, description="second arc", status="active")
        for arc_id, ordering, chapter_id, desc, beat, prose in [
            ("test-project-arc-1", 0, "ch-1", "The setup.", "b1", "Arc one prose."),
            ("arc-2", 1, "ch-2", "The payoff.", "b2", "Arc two prose."),
        ]:
            upsert_chapter(conn, id=chapter_id, arc_id=arc_id, ordering=ordering,
                           description=desc, status="completed")
            upsert_beat(conn, id=beat, chapter_id=chapter_id, ordering=0,
                        prose=prose, word_count=3, status="completed")
    conn.close()

    path = export_manuscript(config)
    text = path.read_text(encoding="utf-8")

    assert "# Arc 1" in text
    assert "# Arc 2" in text
    # Numbering runs continuously across arcs — no "Chapter 1" restart.
    assert "## Chapter 1" in text
    assert "## Chapter 2" in text
    assert text.count("## Chapter 1") == 1
    # Planner descriptions ride under the heading as italic epigraphs.
    assert "*The setup.*" in text
    assert "*The payoff.*" in text
    assert text.index("# Arc 1") < text.index("## Chapter 1") < text.index("# Arc 2") < text.index("## Chapter 2")


async def test_download_404s_before_an_export(config_factory, web_app, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = await web_app(config_factory())
    response = await app.test_client().get("/exports/download")
    assert response.status_code == 404


def test_log_node_event_defaults_to_info(caplog):
    with caplog.at_level(logging.DEBUG, logger="museai"):
        logging_setup.log_node_event("audit", event="audited", failures=0)
    record = caplog.records[-1]
    assert record.levelno == logging.INFO
    assert "node=audit event=audited failures=0" in record.getMessage()


def test_log_node_event_honours_an_explicit_level(caplog):
    with caplog.at_level(logging.DEBUG, logger="museai"):
        logging_setup.log_node_event("manager", level=logging.ERROR, event="run_failed")
    record = caplog.records[-1]
    assert record.levelno == logging.ERROR
    assert "node=manager event=run_failed" in record.getMessage()


def test_a_level_is_never_mistaken_for_a_log_field(caplog):
    """`level` is keyword-only and consumed, not printed as `level=40`."""
    with caplog.at_level(logging.DEBUG, logger="museai"):
        logging_setup.log_node_event("critics", level=logging.WARNING, event="health")
    message = caplog.records[-1].getMessage()
    assert "level=" not in message
