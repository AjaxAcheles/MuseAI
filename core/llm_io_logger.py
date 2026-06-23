"""Module: M14 (Configuration, Startup & Observability)
Dedicated JSON logging for inference-boundary request/response records.

This durable trail is separate from the FSM node-event log and records only one
entry per completed model call: the full request payload, the final assembled
response text, and wall-clock duration.
"""

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "llm_io.log"

# Rotation envelope per Data_Structures.md section 3.2. This is file-size
# housekeeping for the durable inference boundary trail.
LOG_MAX_BYTES = 50_000_000
LOG_BACKUP_COUNT = 3
LOGGER_NAME = "llm_io"


def get_llm_io_logger() -> logging.Logger:
    """Return the dedicated inference-I/O logger."""
    logger = logging.getLogger(LOGGER_NAME)
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
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    return logger


def log_llm_call(
    logger: logging.Logger,
    request_payload: dict,
    response_text: str,
    duration_ms: float,
) -> None:
    """Emit one completed inference call record as a single JSON line."""
    logger.info(
        json.dumps(
            {
                "request": request_payload,
                "response": response_text,
                "duration_ms": duration_ms,
            },
            separators=(",", ":"),
        )
    )
