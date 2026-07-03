"""Module: M14 (Configuration, Startup & Observability)
Structured JSON-lines application log for the product shell (server lifecycle,
route calls, run orchestration, planning progress, SSE diagnostics).

Mirrors the ``core/logger.py`` rotating JSON-lines pattern but writes a
separate ``app.jsonl`` trail: the FSM node-event log (``logs/fsm.log``) stays
reserved for node telemetry, and this file is the app-level debug record the
``/api/logs/recent`` endpoint reads back. Never log secrets or full prompts —
callers truncate user content before it reaches ``log_app_event``.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

# Rotation envelope: operational file-size housekeeping (same posture as
# core/logger.py), not a narrative-calibration threshold.
LOG_MAX_BYTES = 10_000_000
LOG_BACKUP_COUNT = 3

# Cap applied by callers to user-supplied content (e.g. the premise) before it
# enters a log line, so the trail never stores full creative prompts.
USER_CONTENT_LOG_CAP = 80


def truncate_user_content(value: str, cap: int = USER_CONTENT_LOG_CAP) -> str:
    """Truncate user-supplied text to ``cap`` chars for safe logging."""
    text = str(value)
    return text if len(text) <= cap else text[: cap - 1] + "…"


def get_app_logger(log_path: str | Path) -> logging.Logger:
    """Return the JSON-lines app logger writing to ``log_path`` (rotating).

    Logger identity is keyed by the resolved path, so one handler attaches per
    file even across repeated calls (never double-logs), and isolated test
    trees get their own logger. Propagation is off so records stay out of the
    root logger.
    """
    path = Path(log_path).resolve()
    logger = logging.getLogger(f"museai.app::{path}")
    if not logger.handlers:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_app_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit one structured app event as a single JSON line.

    Non-serializable field values degrade to ``str`` so a log call can never
    raise on payload shape.
    """
    record = {
        "ts": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "event": event,
        **fields,
    }
    logger.log(level, json.dumps(record, separators=(",", ":"), default=str))
