"""Module: M02 (Persistent Memory Stores & Interfaces)
Append-only JSONL event-log writer for the immutable transaction ledger.

The event log is a line-delimited JSON file (``.jsonl``): exactly one
self-contained JSON object per line, written strictly in sequence and never
rewritten. It is the audit/reconciliation trail and the forward-replay source
for idempotent crash recovery.

Design rule preserved here (Data_Structures §2.4, Master_Blueprint §3.4):
character PAD state snapshots live *inside* a ``beat_commit`` event payload,
never as standalone events, so ``_apply_event`` can restore prose state and
emotional state atomically from a single record. This module therefore exposes
no helper that writes PAD state as its own event — PAD is carried as the
``pad_states`` field of a ``beat_commit`` payload by the caller.

Scope: append-side writer plus forward/tail readers live here. Crash replay,
branch restore, snapshot lookup, event application (``_apply_event`` belongs to
later Graphiti/branch work), and commit orchestration land in later increments.
"""

import json
import os
from collections import deque
from collections.abc import Iterator, Mapping
from pathlib import Path

try:
    import fcntl  # POSIX advisory file locking (Linux/WSL2/macOS)
except ImportError:  # pragma: no cover - native Windows has no fcntl
    fcntl = None


def init_event_log(config, log_path: str | Path) -> None:
    """No-op initializer that only ensures the append-only JSONL artifact exists.

    Creates the parent directory and touches the file. Never truncates an
    existing log.
    """
    del config
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def write_event(log_path: str | Path, payload: Mapping) -> None:
    """Append exactly one JSON object, as a single line, to the event log.

    The log is strictly append-only: the parent directory is created if missing
    and the file is opened in append mode, so an existing log is never truncated
    or rewritten. The whole record (encoded object plus trailing newline) is
    written under an exclusive advisory lock, flushed, and ``fsync``ed before the
    lock is released, so concurrent writers can never interleave a partial line
    into the ledger and a record that returns is durable on disk — the crash
    recovery this log exists to serve can therefore always parse every line. On a
    platform without ``fcntl`` (native Windows), the lock is skipped but the
    single flushed write still holds under a single writer.

    ``payload`` must be a JSON-serializable mapping — for example a
    ``beat_commit`` event carrying its nested ``pad_states`` snapshot. PAD state
    is never written as its own event; it travels inside the ``beat_commit``
    payload so prose and emotional state replay atomically from one record.

    Raises:
        TypeError: if ``payload`` is not a mapping.
        ValueError: if ``payload`` is not JSON-serializable.
    """
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"write_event payload must be a mapping, got {type(payload).__name__}"
        )
    try:
        line = json.dumps(dict(payload))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "write_event payload is not JSON-serializable; "
            "events must encode to a single JSON object"
        ) from exc

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = (line + "\n").encode("utf-8")
    with open(path, "ab") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(record)
            f.flush()
            os.fsync(f.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def iter_events(log_path: str | Path) -> Iterator[dict]:
    """Yield each event in the log as a parsed JSON object, in file order.

    A missing log file yields nothing rather than raising — an absent ledger is
    an empty history, not an error. Whitespace-only lines (e.g. a trailing blank
    line) are tolerated and skipped. A non-blank line that does not parse as JSON,
    or that parses to something other than a JSON object (e.g. a bare number,
    string, or array from a truncated/forged line), is malformed: it raises
    ``ValueError`` naming the 1-based line number. Every event is a mapping, so a
    non-dict line would otherwise slip through and blow up a downstream
    ``event["type"]`` mid-replay.

    Raises:
        ValueError: if a non-blank line is not a valid JSON object.
    """
    path = Path(log_path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line_number, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"{path}: malformed JSON on line {line_number}: {exc}"
                ) from exc
            if not isinstance(event, dict):
                raise ValueError(
                    f"{path}: line {line_number} is a JSON {type(event).__name__}, "
                    "not an object; the event log holds exactly one object per line"
                )
            yield event


def tail_events(log_path: str | Path, limit: int) -> list[dict]:
    """Return the last ``limit`` events, in original chronological order.

    A non-positive ``limit`` returns an empty list. Reads forward through
    :func:`iter_events`, retaining only the trailing window, so a missing log
    yields ``[]`` and malformed JSON still fails clearly with its line number.
    """
    if limit <= 0:
        return []
    return list(deque(iter_events(log_path), maxlen=limit))


def append_event(*args, **kwargs):
    """Append an event to the transaction log.

    STUB: superseded by :func:`write_event`, which is the implemented
    append-only entry point. Retained as an honest placeholder; not yet wired.
    """
    raise NotImplementedError("STUB: use write_event")
