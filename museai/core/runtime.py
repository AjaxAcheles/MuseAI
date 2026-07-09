"""Runtime bring-up for MuseAI v1.

``init_resources`` prepares durable storage (schema, event log) and runs crash
recovery, returning a small handle the rest of the app threads through.
``reset_resources`` wipes local state for a clean start — dev-only; the web layer
gates it before exposing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from museai.core.config import AppConfig
from museai.core.logging_setup import configure_logging, get_logger
from museai.memory.db import init_db
from museai.memory.reconcile import scan_and_recover

logger = get_logger("museai.core.runtime")


@dataclass
class Resources:
    """Handle to the app's durable resources."""

    config: AppConfig
    db_path: str
    event_log_path: str


def _ensure_event_log(path: str | Path) -> None:
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.touch()


def init_resources(config: AppConfig) -> Resources:
    """Initialize DB + event log, run recovery, and return a Resources handle."""
    configure_logging(config)
    init_db(config.db_path)
    _ensure_event_log(config.event_log_path)

    summary = scan_and_recover(config.db_path, config.event_log_path)
    if summary["pending_found"]:
        logger.info(
            "recovery: pending=%d recovered=%d cleared=%d",
            summary["pending_found"],
            len(summary["recovered"]),
            len(summary["cleared"]),
        )

    return Resources(
        config=config,
        db_path=config.db_path,
        event_log_path=config.event_log_path,
    )


def reset_resources(config: AppConfig) -> Resources:
    """Delete the DB and event log, then re-initialize. Dev-only."""
    for path in (config.db_path, config.event_log_path):
        p = Path(path)
        if p.exists():
            p.unlink()
    logger.info("reset: removed db and event log")
    return init_resources(config)
