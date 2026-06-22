"""Module: M17 (Web UI server entry) - hosts M01 background driver
Start the async Quart server shell so import wiring can be proven before routes exist.
STUB:
"""

from __future__ import annotations

import asyncio

from hypercorn.asyncio import serve
from hypercorn.config import Config
from quart import Quart

from core.generation_manager import GenerationManager


def create_app() -> Quart:
    """Create the Quart application without wiring routes yet."""
    app = Quart(__name__)
    app.generation_manager = GenerationManager()
    return app


async def _shutdown_immediately() -> None:
    """Allow the server task to start, then request a clean shutdown."""
    await asyncio.sleep(0)


async def main() -> None:
    """Run the minimal async server shell."""
    config = Config()
    config.bind = ["127.0.0.1:0"]
    config.use_reloader = False
    config.accesslog = None
    config.errorlog = None

    await serve(create_app(), config, shutdown_trigger=_shutdown_immediately)


if __name__ == "__main__":
    asyncio.run(main())
