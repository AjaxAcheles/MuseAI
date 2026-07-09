"""Run-control routes for the generation manager."""

from __future__ import annotations

from quart import Blueprint, jsonify, request

from museai.core.runtime import reset_resources
from museai.fsm.manager import GenerationManagerError
from museai.web.app import get_config, get_manager, set_runtime

bp = Blueprint("control", __name__, url_prefix="/control")


def _status_payload() -> dict[str, str | bool]:
    return {"ok": True, "status": get_manager().status}


@bp.post("/pause")
async def pause():
    get_manager().pause()
    return jsonify(_status_payload())


@bp.post("/resume")
async def resume():
    try:
        await get_manager().resume()
    except GenerationManagerError as exc:
        return jsonify({"ok": False, "error": str(exc), "status": get_manager().status}), 409
    return jsonify(_status_payload())


@bp.post("/stop")
async def stop():
    get_manager().stop()
    return jsonify(_status_payload())


@bp.post("/review")
async def review():
    body = await request.get_json(silent=True) or {}
    decision = body.get("decision")
    if decision not in {"accept", "regenerate"}:
        return jsonify({"ok": False, "error": "decision must be 'accept' or 'regenerate'"}), 400
    edited_text = body.get("edited_text")
    if edited_text is not None and not isinstance(edited_text, str):
        return jsonify({"ok": False, "error": "edited_text must be a string when provided"}), 400
    try:
        await get_manager().resolve_review(decision, edited_text=edited_text)
    except GenerationManagerError as exc:
        return jsonify({"ok": False, "error": str(exc), "status": get_manager().status}), 409
    return jsonify(_status_payload())


@bp.post("/reset")
async def reset():
    cfg = get_config()
    if not cfg.allow_reset:
        return jsonify({"ok": False, "error": "Reset is disabled by configuration."}), 403
    resources = reset_resources(cfg)
    set_runtime(cfg, resources)
    return jsonify({"ok": True, "status": get_manager().status})
