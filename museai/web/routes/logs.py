"""Read-only, redacted access to the rotating log files.

The LLM I/O log records real request bodies, and `config.yaml` resolves `${MUSEAI_API_KEY}` to a
live credential before it reaches the client. Every line served from here therefore passes through
:func:`redact` first. Redaction is applied to the *served* text only; the files on disk are never
rewritten, because the log is evidence.
"""

from __future__ import annotations

import re
from pathlib import Path

from quart import Blueprint, jsonify, render_template, request

from museai.core import logging_setup
from museai.web.app import get_config

bp = Blueprint("logs", __name__)

REDACTED = "[REDACTED]"

# The two files `logging_setup` maintains. Anything else is refused.
LOG_SOURCES: dict[str, str] = {"fsm": "fsm.log", "llm_io": "llm_io.log"}

_DEFAULT_LIMIT = 200
_MAX_LIMIT = 2000

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Authorization: Bearer sk-abc...
    re.compile(r"(?i)\b(authorization\s*[:=]\s*)(?:bearer\s+)?\S+"),
    # api_key=..., "api_key": "...", api-key: ...
    re.compile(r"""(?i)\b(api[_-]?key\s*["']?\s*[:=]\s*)["']?[^\s"',}]+"""),
    # Bare provider-style tokens anywhere in the line.
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),
)


def _configured_secret() -> str | None:
    """The resolved API key, if one is actually set to a non-trivial value."""
    try:
        key = get_config().endpoint.api_key
    except RuntimeError:  # config not loaded (should not happen inside a request)
        return None
    key = (key or "").strip()
    # Short values are placeholders like "none"/"x"; blanking them would corrupt prose lines.
    return key if len(key) >= 8 else None


def redact(text: str) -> str:
    """Strip credentials from a log line before it leaves the process."""
    secret = _configured_secret()
    if secret and secret in text:
        text = text.replace(secret, REDACTED)
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            text = pattern.sub(lambda m: f"{m.group(1)}{REDACTED}", text)
        else:
            text = pattern.sub(REDACTED, text)
    return text


def _clamp_limit(raw: str | None) -> int:
    try:
        value = int(raw) if raw is not None else _DEFAULT_LIMIT
    except (TypeError, ValueError):
        return _DEFAULT_LIMIT
    return max(1, min(value, _MAX_LIMIT))


def _tail(path: Path, limit: int) -> list[str]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [redact(line) for line in lines[-limit:]]


@bp.get("/logs")
async def logs():
    return await render_template("logs.html", sources=sorted(LOG_SOURCES))


@bp.get("/logs/tail")
async def tail():
    source = request.args.get("source", "fsm")
    if source not in LOG_SOURCES:
        known = ", ".join(sorted(LOG_SOURCES))
        return jsonify({"ok": False, "error": f"Unknown log source {source!r}. Known sources: {known}."}), 400

    limit = _clamp_limit(request.args.get("limit"))
    # Read LOG_DIR through the module so tests can point it at a temp directory.
    path = logging_setup.LOG_DIR / LOG_SOURCES[source]
    lines = _tail(path, limit)
    return jsonify({"ok": True, "source": source, "exists": path.is_file(), "returned": len(lines), "lines": lines})
