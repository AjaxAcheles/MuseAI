"""Quart application factory for the MuseAI v1 web UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quart import Quart, jsonify, render_template, request

from museai.core.config import AppConfig, ConfigError, load_config
from museai.core.runtime import Resources, init_resources
from museai.fsm.manager import GenerationManager

manager: GenerationManager | None = None
config: AppConfig | None = None
resources: Resources | None = None


def get_config() -> AppConfig:
    """Return the loaded app config for request handlers."""
    if config is None:
        raise RuntimeError("configuration has not been loaded")
    return config


def get_manager() -> GenerationManager:
    """Return the single process-local generation manager."""
    if manager is None:
        raise RuntimeError("generation manager has not been initialized")
    return manager


def set_runtime(new_config: AppConfig, new_resources: Resources | None = None) -> None:
    """Replace process-local runtime handles after settings save or reset."""
    global config, resources, manager
    config = new_config
    resources = new_resources
    manager = GenerationManager(new_config)


def create_app(
    *,
    config_path: str | Path = "config.yaml",
    test_config: AppConfig | None = None,
) -> Quart:
    """Create the Quart app and register the v1 route surface."""
    app = Quart(
        __name__,
        static_folder="static",
        template_folder="templates",
    )
    app.config["MUSEAI_CONFIG_PATH"] = str(config_path)
    app.config["PROPAGATE_EXCEPTIONS"] = False

    @app.before_serving
    async def _startup() -> None:
        loaded = test_config or load_config(config_path)
        initialized = init_resources(loaded)
        set_runtime(loaded, initialized)

    @app.errorhandler(Exception)
    async def _clean_error(exc: Exception):
        status = getattr(exc, "code", 500)
        message = str(exc) or "Internal server error"
        if isinstance(exc, ConfigError):
            status = 500
            message = str(exc)

        wants_json = (
            request.path.startswith(
                (
                    "/control",
                    "/settings/save",
                    "/settings/test_endpoint",
                    "/database/",
                    "/logs/",
                    "/exports/",
                    "/chat/",
                )
            )
            or request.path in {"/generate", "/status", "/stream", "/seed/submit", "/committed", "/outline"}
            or request.accept_mimetypes.best == "application/json"
        )
        if wants_json:
            return jsonify({"ok": False, "error": message}), status
        return await render_template("error.html", status=status, message=message), status

    from museai.web.routes.chat import bp as chat_bp
    from museai.web.routes.control import bp as control_bp
    from museai.web.routes.dashboard import bp as dashboard_bp
    from museai.web.routes.database import bp as database_bp
    from museai.web.routes.exports import bp as exports_bp
    from museai.web.routes.logs import bp as logs_bp
    from museai.web.routes.seed import bp as seed_bp
    from museai.web.routes.settings import bp as settings_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(control_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(seed_bp)
    app.register_blueprint(database_bp)
    app.register_blueprint(logs_bp)
    app.register_blueprint(exports_bp)
    app.register_blueprint(chat_bp)

    if test_config is not None:
        set_runtime(test_config, None)

    return app
