"""Append-only JSONL transcript of every LLM call.

The View Chat page shows the deliberations of every agent — each prompt sent to
the model and each thinking/response that came back. Live traffic reaches the
browser through the stream bus; this module is the durable half, so a reload
replays the conversation instead of starting blank.

Two record shapes, paired by ``id``::

    {"event": "chat_start", "id": ..., "ts": ..., "agent": ..., "model": ...,
     "messages": [{"role": ..., "content": ...}, ...]}
    {"event": "chat_end", "id": ..., "ts": ..., "agent": ..., "ok": true,
     "thinking": ..., "text": ..., "tokens_in": ..., "tokens_out": ...,
     "finish_reason": ..., "error": null}

Unlike ``events.py`` this log is a viewing aid, not a recovery source: writes
are best-effort (no fsync) and a write failure must never fail the LLM call it
was describing. Until :func:`configure` runs, :func:`record` is a no-op, which
keeps unit tests that call the LLM client from writing files.
"""

from __future__ import annotations

import json
from pathlib import Path

from museai.core.logging_setup import get_fsm_logger

_path: Path | None = None


def default_path(event_log_path: str | Path) -> Path:
    """The transcript's canonical home: beside the event log, as ``chat.jsonl``."""
    return Path(event_log_path).with_name("chat.jsonl")


def configure(path: str | Path | None) -> None:
    """Set (or with ``None``, unset) the transcript file this process writes."""
    global _path
    _path = Path(path) if path is not None else None


def transcript_path() -> Path | None:
    """Where the transcript is being written, or ``None`` if unconfigured."""
    return _path


def record(entry: dict) -> None:
    """Append one record. Best-effort: a disk fault is logged, never raised."""
    if _path is None:
        return
    try:
        if _path.parent and not _path.parent.exists():
            _path.parent.mkdir(parents=True, exist_ok=True)
        with open(_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        get_fsm_logger().warning("chat_transcript_write_failed error=%s", exc)


def replay(path: str | Path, limit: int) -> list[dict]:
    """The last ``limit`` records in append order.

    Mirrors ``replay_events``: a missing file yields nothing and a torn or
    malformed line is skipped rather than aborting the replay.
    """
    p = Path(path)
    if not p.exists() or limit < 1:
        return []
    records: list[dict] = []
    with open(p, "r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                get_fsm_logger().warning(
                    "chat_transcript_line_skipped line=%d reason=malformed_json",
                    line_number,
                )
                continue
            if not isinstance(entry, dict):
                get_fsm_logger().warning(
                    "chat_transcript_line_skipped line=%d reason=not_an_object",
                    line_number,
                )
                continue
            records.append(entry)
    return records[-limit:]
