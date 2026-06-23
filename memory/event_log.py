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
from collections import deque
from collections.abc import Iterator, Mapping
from pathlib import Path


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
    or rewritten. The encoded object plus a trailing newline is emitted in one
    write so a record is always a single complete line.

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
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def iter_events(log_path: str | Path) -> Iterator[dict]:
    """Yield each event in the log as a parsed JSON object, in file order.

    A missing log file yields nothing rather than raising — an absent ledger is
    an empty history, not an error. Whitespace-only lines (e.g. a trailing blank
    line) are tolerated and skipped. A non-blank line that does not parse as JSON
    is malformed: it raises ``ValueError`` naming the 1-based line number.

    Raises:
        ValueError: if a non-blank line is not valid JSON.
    """
    path = Path(log_path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line_number, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            try:
                yield json.loads(raw)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"{path}: malformed JSON on line {line_number}: {exc}"
                ) from exc


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
