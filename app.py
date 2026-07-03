"""Module: M17 (Web UI server entry) - hosts M01 background driver
Quart app factory + server entry point for the vertical-slice product shell.

``create_app()`` wires the dashboard and control blueprints; runtime resources
initialize in ``before_serving`` (never at import time), and the generation
manager is attached to the shared :class:`~core.runtime.RuntimeResources`
container so routes reach everything through ``get_resources()``. Long work
never runs in a request handler — it lives in the manager's background task.

Bind address/port come from ``MUSEAI_HOST``/``MUSEAI_PORT`` env vars (defaults
127.0.0.1:5000); nothing model- or provider-related is configured here.
"""

from __future__ import annotations

import asyncio
import logging
import os

from hypercorn.asyncio import serve
from hypercorn.config import Config
from quart import Quart, redirect, request, url_for

import core.runtime as runtime
from core.generation_manager import GenerationManager
from routes import control, dashboard, plan

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000


def create_app() -> Quart:
    """Create the Quart application with routes wired and lazy runtime init."""
    app = Quart(__name__)
    app.register_blueprint(dashboard.create_blueprint())
    app.register_blueprint(control.create_blueprint())
    app.register_blueprint(plan.create_blueprint())

    @app.before_serving
    async def _init_runtime() -> None:
        resources = runtime.init_resources()
        if resources.generation_manager is None:
            resources.generation_manager = GenerationManager(resources)
        resources.log("app_startup")

    @app.before_request
    async def _log_route_call() -> None:
        # Skip static assets and the long-lived SSE stream (it logs itself).
        if request.path.startswith("/static") or request.path == "/events":
            return
        runtime.get_resources().log(
            "route_call", method=request.method, path=request.path
        )

    @app.after_serving
    async def _shutdown_runtime() -> None:
        resources = runtime.get_resources()
        manager = resources.generation_manager
        # "active" includes a run parked at the approval gate — stop it too.
        if manager is not None and manager.status()["active"]:
            await manager.stop()
        resources.log("app_shutdown")

    @app.get("/")
    async def index():
        return redirect(url_for("dashboard.dashboard"))

    return app


async def main() -> None:
    """Run the Quart server until interrupted."""
    host = os.environ.get("MUSEAI_HOST", DEFAULT_HOST)
    port = int(os.environ.get("MUSEAI_PORT", DEFAULT_PORT))

    config = Config()
    config.bind = [f"{host}:{port}"]
    config.use_reloader = False

    logger.info("MuseAI vertical slice serving on http://%s:%s/dashboard", host, port)
    print(f"MuseAI vertical slice: http://{host}:{port}/dashboard")
    await serve(create_app(), config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
