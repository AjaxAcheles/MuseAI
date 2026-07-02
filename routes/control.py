"""Module: M17 (Web UI & Real-time Observer Surface)

Control endpoints for starting a run, polling its status, and dev reset.

``POST /control/start`` kicks off a background generation run via the app's
:class:`~core.generation_manager.GenerationManager` and returns its ``run_id`` (the
browser then subscribes to ``/events/<run_id>``). ``GET /control/status/<run_id>``
returns the run's status record. ``POST /control/reset`` wipes the local file-based
stores (development convenience) via ``core.runtime.reset_resources``.
"""

from __future__ import annotations

from pathlib import Path

from quart import Blueprint, current_app, jsonify, request

import core.runtime as runtime

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def create_blueprint() -> Blueprint:
    """Create the control blueprint (start / status / reset)."""
    bp = Blueprint("control", __name__)

    @bp.post("/start")
    async def start():
        payload = await request.get_json(silent=True) or {}
        metadata = {
            "premise_seed": payload.get("premise_seed") or payload.get("premise") or "",
            "genre": payload.get("genre") or "",
        }
        if payload.get("target_word_count"):
            metadata["target_word_count"] = int(payload["target_word_count"])
        if payload.get("beat_word_target"):
            metadata["beat_word_target"] = int(payload["beat_word_target"])
        run_id = current_app.generation_manager.start(metadata)
        return jsonify({"run_id": run_id})

    @bp.get("/status/<run_id>")
    async def status(run_id: str):
        record = current_app.generation_manager.status(run_id)
        if record is None:
            return jsonify({"error": "unknown run_id"}), 404
        return jsonify(record)

    @bp.post("/reset")
    async def reset():
        from core.config_loader import load_config

        runtime.reset_resources(load_config(CONFIG_PATH))
        return jsonify({"reset": True})

    return bp
