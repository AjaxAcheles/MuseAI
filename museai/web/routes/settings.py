"""Settings display, validation, persistence, and endpoint test routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from quart import Blueprint, current_app, jsonify, render_template, request

from museai.core.config import AppConfig, ConfigError, load_config
from museai.core.runtime import init_resources
from museai.llm.client import call_llm
from museai.web.app import get_config, set_runtime

bp = Blueprint("settings", __name__)


def _settings_view(cfg: AppConfig) -> dict[str, Any]:
    """The settings the UI may show. The API key is reported as present, never returned."""
    return {
        "endpoint": {
            "base_url": cfg.endpoint.base_url,
            "model_name": cfg.endpoint.model_name,
            "api_key_present": bool(cfg.endpoint.api_key),
            "tokenizer_family": cfg.endpoint.tokenizer_family,
            "request_timeout": cfg.endpoint.request_timeout,
            "temperature": cfg.endpoint.temperature,
        },
        "generation": cfg.generation.model_dump(),
        "runtime": {
            "log_level": cfg.log_level,
            "web_search_timeout": cfg.web_search_timeout,
            "allow_reset": cfg.allow_reset,
        },
    }


def _coerce_scalar(value: str) -> Any:
    text = value.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return value


async def _payload() -> dict[str, Any]:
    if request.is_json:
        body = await request.get_json()
        if not isinstance(body, dict):
            raise ValueError("settings payload must be a JSON object")
        return body

    form = await request.form
    cfg = get_config().model_dump()
    for key, value in form.items():
        parts = key.split(".")
        target = cfg
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = _coerce_scalar(value)
    return cfg


@bp.get("/settings")
async def settings():
    cfg = get_config()
    safe_config = cfg.model_dump()
    safe_config["endpoint"]["api_key"] = ""
    return await render_template(
        "settings.html",
        settings=_settings_view(cfg),
        config_json=json.dumps(safe_config, ensure_ascii=False),
    )


def _raw_api_key_on_disk(path: Path) -> str | None:
    """Return the unresolved ``endpoint.api_key`` string from config.yaml.

    The in-memory config holds the *resolved* secret (a ``${VAR}`` reference is
    expanded at load). Persisting that resolved value would write the secret in
    plaintext and destroy the env reference, so saves must reuse the raw string.
    """
    if not path.is_file():
        return None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("endpoint"), dict):
        return None
    key = raw["endpoint"].get("api_key")
    return key if isinstance(key, str) else None


@bp.post("/settings/save")
async def save():
    path = Path(current_app.config["MUSEAI_CONFIG_PATH"])
    try:
        body = await _payload()
        keep_existing_key = (
            isinstance(body.get("endpoint"), dict) and body["endpoint"].get("api_key") == ""
        )
        if keep_existing_key:
            body["endpoint"]["api_key"] = get_config().endpoint.api_key
        validated = AppConfig(**body)
    except (ValidationError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    persisted = validated.model_dump()
    if keep_existing_key:
        raw_key = _raw_api_key_on_disk(path)
        if raw_key:
            persisted["endpoint"]["api_key"] = raw_key
    path.write_text(yaml.safe_dump(persisted, sort_keys=False), encoding="utf-8")
    try:
        reloaded = load_config(path)
    except ConfigError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    resources = init_resources(reloaded)
    set_runtime(reloaded, resources)
    return jsonify({"ok": True, "settings": _settings_view(reloaded)})


@bp.post("/settings/test_endpoint")
async def test_endpoint():
    cfg = get_config()
    try:
        response = await call_llm(
            cfg.endpoint,
            [{"role": "user", "content": "Reply with the single word: ok"}],
            agent="endpoint_test",
            max_tokens=8,
            retry_on_empty=True,
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)})
    if response.text.strip().casefold() != "ok":
        return jsonify(
            {
                "ok": False,
                "error": f"endpoint returned an unexpected probe reply: {response.text[:80]!r}",
            }
        )
    return jsonify({"ok": True, "model": response.model_name})
