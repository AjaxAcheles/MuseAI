"""Module: M14 (Configuration, Startup & Observability)
Structured JSON logging for LangGraph nodes, written to a durable rotating file
trail under logs/. The SSE UI stream is live-display only; this file is the
durable debug record, kept separate from it. Log level is read from config.yaml.
"""

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from core.config_loader import load_config

# Durable node-event trail (separate from the live SSE display stream).
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "fsm.log"

# Rotation envelope per Data_Structures.md §3.1. This is operational plumbing
# (file-size housekeeping), not a narrative-calibration threshold; if it ever
# needs tuning, promote it to a `logging` rotation section in config.yaml.
LOG_MAX_BYTES = 10_000_000
LOG_BACKUP_COUNT = 5

# Path to the config whose log level governs the node loggers.
_CONFIG_PATH = "config.yaml"
_VALID_OUTCOMES = frozenset({"success", "failure", "escalated"})


def _resolve_log_level() -> int:
    """Return the numeric log level from strictly validated config.yaml."""
    level_name = load_config(_CONFIG_PATH).logging.log_level
    return logging.getLevelNamesMapping().get(level_name.upper(), logging.INFO)


def get_logger(node_name: str) -> logging.Logger:
    """Return a logger for ``node_name`` that writes JSON lines to the rotating
    durable trail under ``logs/``.

    The rotating file handler is attached only once per logger name: repeated
    calls return the already-configured logger without double-attaching, so a
    node never double-logs. Propagation is disabled so records do not also leak
    to the root logger / live stream.
    """
    logger = logging.getLogger(node_name)
    if not logger.handlers:
        LOG_DIR.mkdir(exist_ok=True)
        handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(_resolve_log_level())
        logger.propagate = False
    return logger


def log_node_event(
    logger: logging.Logger,
    fsm_pointer: dict,
    duration_ms: float,
    outcome: str,
    error: str | None = None,
) -> None:
    """Emit one structured FSM node event as a single JSON line."""
    if outcome not in _VALID_OUTCOMES:
        raise ValueError(f"Invalid node event outcome: {outcome!r}")

    logger.info(
        json.dumps(
            {
                "node_name": logger.name,
                "fsm_pointer": fsm_pointer,
                "duration_ms": duration_ms,
                "outcome": outcome,
                "error": error,
            },
            separators=(",", ":"),
        )
    )
