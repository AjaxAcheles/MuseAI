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


# How much of the transcript's tail to read on the first try. The View Chat page
# asks for the last couple of hundred records out of a file that grows for the
# whole run; reading all of it to discard nearly all of it put the cost of every
# page load on the same event loop the FSM runs on, and that cost rose for the
# rest of the session. One block covers a few hundred records comfortably.
_TAIL_BLOCK_BYTES = 65_536


def _read_tail(path: Path, window: int) -> tuple[bytes, int, bool]:
    """The last ``window`` bytes of a file, its start offset, and whether that
    offset is the beginning of the file."""
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        start = max(0, size - window)
        fh.seek(start)
        return fh.read(), start, start == 0


def _parse_tail(data: bytes, base: int) -> list[dict]:
    """Parse newline-delimited records out of a tail block.

    ``base`` is the block's byte offset in the file, so a skipped line can be
    reported by where it actually lives — more use than a line number in a
    multi-megabyte transcript, and the only locator a tail read still has.
    """
    records: list[dict] = []
    offset = base
    for raw in data.split(b"\n"):
        line_offset = offset
        offset += len(raw) + 1
        line = raw.strip()
        if not line:
            continue
        try:
            entry = json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            get_fsm_logger().warning(
                "chat_transcript_line_skipped offset=%d reason=malformed_json",
                line_offset,
            )
            continue
        if not isinstance(entry, dict):
            get_fsm_logger().warning(
                "chat_transcript_line_skipped offset=%d reason=not_an_object",
                line_offset,
            )
            continue
        records.append(entry)
    return records


def replay(path: str | Path, limit: int) -> list[dict]:
    """The last ``limit`` records in append order.

    Mirrors ``replay_events``: a missing file yields nothing and a torn or
    malformed line is skipped rather than aborting the replay. Read from the
    tail, so the work scales with ``limit`` and not with the transcript.
    """
    p = Path(path)
    if not p.exists() or limit < 1:
        return []

    window = _TAIL_BLOCK_BYTES
    while True:
        data, start, from_start = _read_tail(p, window)
        if from_start:
            block, base = data, 0
        else:
            # The window almost certainly opened mid-record. Drop that fragment
            # rather than reporting it as a torn line — nothing is wrong with it
            # except where we started reading.
            cut = data.find(b"\n")
            block, base = (b"", start) if cut < 0 else (data[cut + 1 :], start + cut + 1)
        records = _parse_tail(block, base)
        # Skipped lines mean a window holding `limit` lines can still yield
        # fewer than `limit` records, so grow until it does or the file runs out.
        if from_start or len(records) >= limit:
            return records[-limit:]
        # Grow to what this file's own records say is needed rather than
        # doubling blindly. A record here carries a whole reasoning trace, so
        # they run from a few hundred bytes to tens of kilobytes; a fixed block
        # can be an order of magnitude out, and each blind double re-reads and
        # re-parses everything it already had. Doubling stays the floor so a
        # wild underestimate still converges.
        window = max(
            window * 2,
            (len(block) // len(records)) * limit * 2 if records else 0,
        )
