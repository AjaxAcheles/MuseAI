"""Tests for the seed loader."""

from __future__ import annotations

import json
from pathlib import Path

from museai.core.runtime import init_resources
from museai.memory import db
from museai.seed.loader import load_seed

EXAMPLE_SEED = Path(__file__).resolve().parent.parent / "seeds" / "example.json"


def test_example_seed_populates_every_table(config_factory):
    config = config_factory()
    init_resources(config)

    seed = json.loads(EXAMPLE_SEED.read_text(encoding="utf-8"))
    counts = load_seed(seed, config)

    assert counts["projects"] == 1
    assert counts["arcs"] == 2
    assert counts["threads"] == 3
    assert counts["characters"] == 2
    assert counts["character_emotions"] == 2

    conn = db.connect_db(config.db_path)
    pid = seed["project"]["id"]

    project = db.get_project(conn, pid)
    assert project is not None
    assert project["word_count_target"] == 40000

    arcs = db.get_arcs(conn, pid)
    assert [a["status"] for a in arcs] == ["active", "planned"]

    threads = db.get_open_threads(conn, pid)
    assert len(threads) == 3

    characters = db.get_characters(conn, pid)
    assert len(characters) == 2
    for character in characters:
        emo = db.get_character_emotions(conn, character["id"])
        assert emo is not None
        assert -1.0 <= emo["pleasure"] <= 1.0
    conn.close()


def test_load_seed_is_idempotent(config_factory):
    config = config_factory()
    init_resources(config)
    seed = json.loads(EXAMPLE_SEED.read_text(encoding="utf-8"))

    load_seed(seed, config)
    load_seed(seed, config)  # second load must not duplicate rows

    conn = db.connect_db(config.db_path)
    n_arcs = conn.execute("SELECT COUNT(*) AS n FROM Arcs").fetchone()["n"]
    n_chars = conn.execute("SELECT COUNT(*) AS n FROM Characters").fetchone()["n"]
    conn.close()
    assert n_arcs == 2
    assert n_chars == 2
