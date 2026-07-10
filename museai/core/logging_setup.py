"""Dual rotating-file logging for MuseAI v1.

Two dedicated logs are maintained:

* ``logs/fsm.log``     — FSM / node / event activity, and everything else.
* ``logs/llm_io.log``  — raw LLM request/response I/O.

The handler lives on the ``museai`` root, so *any* logger obtained through
:func:`get_logger` lands in ``fsm.log``. ``museai.llm_io`` is the one exception:
it owns its own handler and does not propagate, which keeps the I/O log
separate. Nothing under ``museai`` can log into a void.

Log level comes from config when configured; rotation sizes/backups are fixed
sensible defaults (config carries no speculative logging keys).
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from museai.core.config import AppConfig

LOG_DIR = Path("logs")
_MAX_BYTES = 5 * 1024 * 1024  # 5 MiB per file
_BACKUP_COUNT = 5

_APP_LOGGER_NAME = "museai"
_FSM_LOGGER_NAME = "museai.fsm"
_LLM_LOGGER_NAME = "museai.llm_io"

_LINE_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

# Guard so handlers are attached exactly once.
_configured = False


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _make_file_handler(filename: str) -> RotatingFileHandler:
    handler = RotatingFileHandler(
        LOG_DIR / filename,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(_LINE_FORMAT))
    return handler


def _dedicated_logger(name: str, filename: str) -> logging.Logger:
    """A non-propagating logger writing only to its own rotating file."""
    logger = logging.getLogger(name)
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        logger.addHandler(_make_file_handler(filename))
    logger.propagate = False
    return logger


def configure_logging(config: "AppConfig | None" = None) -> None:
    """Set up the two rotating loggers and the root level.

    Idempotent: safe to call more than once. Level is taken from ``config``
    when provided, otherwise defaults to ``INFO``.

    ``fsm.log``'s handler is attached to the ``museai`` root rather than to
    ``museai.fsm``, so a single handler owns the file — two rotating handlers on
    one path would race each other on rollover — and every application logger
    reaches it by propagation.
    """
    global _configured
    _ensure_log_dir()

    level_name = (config.log_level if config is not None else "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    app_logger = _dedicated_logger(_APP_LOGGER_NAME, "fsm.log")
    llm_logger = _dedicated_logger(_LLM_LOGGER_NAME, "llm_io.log")
    app_logger.setLevel(level)
    llm_logger.setLevel(level)

    # Propagates up to the app logger's handler; owns none of its own.
    logging.getLogger(_FSM_LOGGER_NAME).setLevel(level)
    _configured = True


def _auto_configure() -> None:
    if not _configured:
        configure_logging()


def get_logger(name: str) -> logging.Logger:
    """Return a general-purpose application logger under the ``museai`` tree."""
    _auto_configure()
    return logging.getLogger(name if name.startswith("museai") else f"museai.{name}")


def get_fsm_logger() -> logging.Logger:
    """Return the FSM/event logger (writes to ``logs/fsm.log``)."""
    _auto_configure()
    return logging.getLogger(_FSM_LOGGER_NAME)


def get_llm_logger() -> logging.Logger:
    """Return the LLM I/O logger (writes to ``logs/llm_io.log``)."""
    _auto_configure()
    return logging.getLogger(_LLM_LOGGER_NAME)


def _format_fields(fields: dict[str, object]) -> str:
    parts = []
    for key, value in fields.items():
        text = str(value).replace("\n", " ")
        if " " in text:
            text = f'"{text}"'
        parts.append(f"{key}={text}")
    return " ".join(parts)


def log_node_event(
    node_name: str, *, level: int = logging.INFO, **fields: object
) -> None:
    """Write one structured line describing a node event to ``logs/fsm.log``.

    ``level`` is keyword-only and defaults to INFO, which is what the overwhelming
    majority of node events are: a beat drafted, a chapter committed. Reserve
    WARNING for a run that is still going but is doing so on a degraded footing,
    and ERROR for a run that has stopped. Every beat emits a critic health line,
    so raising *that* to WARNING unconditionally would teach the reader to skim
    past warnings — which is exactly how two fatal crashes went unnoticed.
    """
    logger = get_fsm_logger()
    suffix = _format_fields(fields)
    if suffix:
        logger.log(level, "node=%s %s", node_name, suffix)
    else:
        logger.log(level, "node=%s", node_name)
