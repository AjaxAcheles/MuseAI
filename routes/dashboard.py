"""Module: M17 (Web UI & Real-time Observer Surface)
Serve the vertical-slice dashboard page.

Renders the initial run snapshot server-side (from the stream bus and the
generation manager) so the page is meaningful before the SSE connection
attaches; ``static/js/main.js`` keeps it live afterwards.
"""

from __future__ import annotations

from typing import Any

from quart import Blueprint, render_template

from core.runtime import get_resources


def create_blueprint() -> Blueprint:
    """Create the dashboard blueprint (/dashboard)."""
    bp = Blueprint("dashboard", __name__)

    @bp.get("/dashboard")
    async def dashboard() -> Any:
        resources = get_resources()
        manager = resources.generation_manager
        manager_status = (
            manager.status()
            if manager is not None
            else {"status": "idle", "running": False, "message": "", "last_error": None}
        )
        stores = [
            {"name": handle.name, "kind": handle.kind, "note": handle.note}
            for handle in resources.stores.values()
        ]
        return await render_template(
            "dashboard.html",
            snapshot=resources.event_bus.snapshot(),
            manager_status=manager_status,
            stores=stores,
        )

    return bp
