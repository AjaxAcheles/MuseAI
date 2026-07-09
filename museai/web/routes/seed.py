"""Seed intake routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quart import Blueprint, jsonify, redirect, render_template, request, url_for

from museai.seed.loader import load_seed
from museai.web.app import get_config

bp = Blueprint("seed", __name__)


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


@bp.get("/seed")
async def seed():
    example_path = Path("seeds/example.json")
    example = example_path.read_text(encoding="utf-8") if example_path.is_file() else ""
    return await render_template("seed.html", example_seed=example)


@bp.post("/seed/submit")
async def submit():
    try:
        seed_doc = await _seed_payload()
        load_seed(seed_doc, get_config())
    except (KeyError, TypeError, ValueError) as exc:
        if request.is_json:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return await render_template("seed.html", error=str(exc), example_seed=""), 400
    if request.is_json:
        return jsonify({"ok": True, "project_id": seed_doc["project"]["id"]})
    return redirect(url_for("dashboard.dashboard"))
