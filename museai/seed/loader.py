"""Seed loading for MuseAI v1.

``load_seed`` writes a project seed (project, arcs, threads, characters + their
initial PAD state) into SQLite via idempotent upserts, so re-loading the same
seed is safe.

Expected seed JSON shape::

    {"project": {"id": "p1", "genre": "...", "premise": "...",
                 "word_count_target": 40000},
     "arcs": [{"description": "..."}],
     "threads": [{"description": "...", "priority_score": 0.8, "status": "open"}],
     "characters": [{"name": "...", "description": "...",
                     "pad": {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0}}]}

The first arc is set ``active``; the rest ``planned``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from museai.core.config import AppConfig, load_config
from museai.memory.db import (
    connect_db,
    upsert_arc,
    upsert_character,
    upsert_character_emotions,
    upsert_project,
    upsert_thread,
)
from museai.core.runtime import init_resources

_DEFAULT_PAD = {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0}


def load_seed(seed: dict, config: AppConfig) -> dict[str, int]:
    """Write a seed into SQLite. Returns a per-table row count."""
    project = seed["project"]
    project_id = project["id"]

    counts = {
        "projects": 0,
        "arcs": 0,
        "threads": 0,
        "characters": 0,
        "character_emotions": 0,
    }

    conn = connect_db(config.db_path)
    try:
        with conn:
            upsert_project(
                conn,
                id=project_id,
                genre=project.get("genre"),
                premise=project.get("premise"),
                word_count_target=project.get("word_count_target"),
            )
            counts["projects"] += 1

            for i, arc in enumerate(seed.get("arcs", [])):
                arc_id = arc.get("id", f"{project_id}-arc-{i + 1}")
                upsert_arc(
                    conn,
                    id=arc_id,
                    project_id=project_id,
                    ordering=i,
                    description=arc["description"],
                    status="active" if i == 0 else "planned",
                )
                counts["arcs"] += 1

            for i, thread in enumerate(seed.get("threads", [])):
                thread_id = thread.get("id", f"{project_id}-thread-{i + 1}")
                upsert_thread(
                    conn,
                    id=thread_id,
                    project_id=project_id,
                    description=thread["description"],
                    status=thread.get("status", "open"),
                    priority_score=thread.get("priority_score"),
                )
                counts["threads"] += 1

            for i, character in enumerate(seed.get("characters", [])):
                char_id = character.get("id", f"{project_id}-char-{i + 1}")
                upsert_character(
                    conn,
                    id=char_id,
                    project_id=project_id,
                    name=character["name"],
                    description=character.get("description"),
                )
                counts["characters"] += 1

                pad = character.get("pad", _DEFAULT_PAD)
                upsert_character_emotions(
                    conn,
                    character_id=char_id,
                    pleasure=pad.get("pleasure", 0.0),
                    arousal=pad.get("arousal", 0.0),
                    dominance=pad.get("dominance", 0.0),
                )
                counts["character_emotions"] += 1
    finally:
        conn.close()

    return counts


def _main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m museai.seed.loader <path.json>", file=sys.stderr)
        return 2

    seed_path = Path(argv[0])
    if not seed_path.is_file():
        print(f"seed file not found: {seed_path}", file=sys.stderr)
        return 1

    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    config = load_config()
    init_resources(config)
    counts = load_seed(seed, config)

    print(f"Loaded seed from {seed_path} into {config.db_path}:")
    for table, count in counts.items():
        print(f"  {table}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
