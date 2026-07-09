"""MuseAI v1 command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from museai.core.config import load_config
from museai.core.runtime import init_resources
from museai.core.stream_bus import bus
from museai.fsm.manager import GenerationManager
from museai.seed.loader import load_seed
from museai.web.app import create_app


async def _run_headless(seed_path: Path) -> int:
    config = load_config()
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    project_id = seed["project"]["id"]
    config = config.model_copy(update={"project_id": project_id})

    init_resources(config)
    load_seed(seed, config)

    manager = GenerationManager(config)
    await manager.start(project_id)
    status = await manager.wait()

    if status == "done":
        ready = bus.last_snapshot.get("manuscript_ready") or {}
        path = ready.get("path")
        if path:
            print(f"Manuscript ready: {path}")
        else:
            print("Generation completed, but no manuscript path was published.")
            return 1
        return 0

    if status == "review":
        print(
            "Generation parked for human review. Resolve it through the review "
            "interface; headless mode does not auto-accept drafts."
        )
        return 2

    print(f"Generation stopped with status: {status}")
    return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MuseAI v1")
    parser.add_argument("--headless", action="store_true", help="run generation without the web UI")
    parser.add_argument("--seed", type=Path, help="seed JSON path for --headless")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.headless:
        if args.seed is None:
            raise SystemExit("--headless requires --seed <path>")
        return asyncio.run(_run_headless(args.seed))
    config = load_config()
    app = create_app()
    app.run(host=config.host, port=config.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
