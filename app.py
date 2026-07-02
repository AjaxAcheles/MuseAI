"""Module: M17 (Web UI server entry) - hosts M01 background driver

Start the async Quart server and wire the dashboard + control routes onto a shared
:class:`~core.generation_manager.GenerationManager` (which owns the SSE stream bus),
so a story can be driven end to end from the browser at ``http://127.0.0.1:8888``
(host/port overridable via ``MUSEAI_HOST`` / ``MUSEAI_PORT``).
"""

from __future__ import annotations

import asyncio
import os

from hypercorn.asyncio import serve
from hypercorn.config import Config
from quart import Quart

from core.generation_manager import GenerationManager
from core.stream_bus import StreamBus
from routes.control import create_blueprint as create_control_blueprint
from routes.dashboard import create_blueprint as create_dashboard_blueprint


def create_app() -> Quart:
    """Create the Quart app with routes wired to a shared generation manager."""
    app = Quart(__name__)
    app.generation_manager = GenerationManager(StreamBus())
    app.register_blueprint(create_dashboard_blueprint())
    app.register_blueprint(create_control_blueprint(), url_prefix="/control")
    return app


async def main() -> None:
    """Run the dashboard server until interrupted.

    Host/port are overridable via ``MUSEAI_HOST`` / ``MUSEAI_PORT`` (default
    ``127.0.0.1:8888``).
    """
    host = os.environ.get("MUSEAI_HOST", "127.0.0.1")
    port = os.environ.get("MUSEAI_PORT", "8888")
    config = Config()
    config.bind = [f"{host}:{port}"]
    config.use_reloader = False

    await serve(create_app(), config)


if __name__ == "__main__":
    asyncio.run(main())
