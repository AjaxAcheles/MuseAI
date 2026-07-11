"""Seed intake routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quart import Blueprint, current_app, jsonify, redirect, render_template, request, url_for

from museai.core.config import ConfigError, persist_project_id
from museai.core.logging_setup import get_fsm_logger
from museai.core.runtime import init_resources
from museai.seed.loader import load_seed
from museai.web.app import get_config, set_runtime

bp = Blueprint("seed", __name__)


def _sync_project_id(seed_project_id: str) -> None:
    """Point ``config.yaml`` and the runtime at the freshly seeded project.

    Without this, a stale ``project_id`` from a previous project makes the run
    export under the wrong name with zero committed beats.
    """
    if seed_project_id == get_config().project_id:
        return
    path = Path(current_app.config["MUSEAI_CONFIG_PATH"])
    reloaded = persist_project_id(seed_project_id, path)
    set_runtime(reloaded, init_resources(reloaded))
    get_fsm_logger().info(
        "seed_project_sync project_id=%s config=%s", seed_project_id, path
    )


def _validate_seed(seed: Any) -> dict[str, Any]:
    if not isinstance(seed, dict):
        raise ValueError("seed must be a JSON object")
    project = seed.get("project")
    if not isinstance(project, dict) or not isinstance(project.get("id"), str):
        raise ValueError("seed.project.id is required")
    if "arcs" not in seed or not isinstance(seed["arcs"], list) or not seed["arcs"]:
        raise ValueError("seed.arcs must be a non-empty list")
    for index, arc in enumerate(seed["arcs"], start=1):
        if not isinstance(arc, dict) or not isinstance(arc.get("description"), str):
            raise ValueError(f"seed.arcs[{index}].description is required")
    for collection in ("threads", "characters"):
        if collection in seed and not isinstance(seed[collection], list):
            raise ValueError(f"seed.{collection} must be a list")
    for index, thread in enumerate(seed.get("threads") or [], start=1):
        if not isinstance(thread, dict) or not isinstance(thread.get("description"), str):
            raise ValueError(f"seed.threads[{index}].description is required")
    for index, character in enumerate(seed.get("characters") or [], start=1):
        if not isinstance(character, dict) or not isinstance(character.get("name"), str):
            raise ValueError(f"seed.characters[{index}].name is required")
    return seed


async def _seed_payload() -> dict[str, Any]:
    if request.is_json:
        return _validate_seed(await request.get_json())
    form = await request.form
    raw = form.get("seed_json", "")
    if not raw.strip():
        raise ValueError("Paste seed JSON before submitting.")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed seed JSON: {exc}") from exc
    return _validate_seed(parsed)


def example_seed_text() -> str:
    """The bundled starter seed, or empty string when it is absent.

    Shared with the dashboard, whose Load Seed drawer prefills the same JSON.
    """
    example_path = Path("seeds/example.json")
    return example_path.read_text(encoding="utf-8") if example_path.is_file() else ""


@bp.get("/seed")
@bp.get("/setup")
async def seed():
    """The Seed & Plan workspace. ``/setup`` is an alias — same page, friendlier URL."""
    example = example_seed_text()
    return await render_template("seed.html", example_seed=example, seed_text=example)


@bp.post("/seed/submit")
async def submit():
    try:
        seed_doc = await _seed_payload()
        load_seed(seed_doc, get_config())
        _sync_project_id(seed_doc["project"]["id"])
    except (ConfigError, KeyError, TypeError, ValueError) as exc:
        if request.is_json:
            return jsonify({"ok": False, "error": str(exc)}), 400
        # Re-render with the submitted text intact so the user's edits survive.
        form = await request.form
        return (
            await render_template(
                "seed.html",
                error=str(exc),
                example_seed=example_seed_text(),
                seed_text=form.get("seed_json", ""),
            ),
            400,
        )
    if request.is_json:
        return jsonify({"ok": True, "project_id": seed_doc["project"]["id"]})
    return redirect(url_for("dashboard.dashboard"))
