"""Append-only event log for MuseAI v1.

Events are newline-delimited JSON, one object per line, never rewritten. The log
is the source of truth for crash recovery: the reconciler replays it to finish
or discard any commit that was interrupted.

The primary payload is the ``beat_commit`` event, emitted when a beat's prose is
committed. Its shape::

    {
        "type": "beat_commit",
        "beat_id": "<beat id>",
        "fsm_pointer": {"arc_id": ..., "chapter_id": ..., "beat_index": ...},
        "prose_delta": "<the beat's committed prose>",
        "thread_updates": [
            {"id": "<thread id>", "status": "progressing", "priority_score": 0.7}
        ],
        "pad_states": [
            {"character_id": "<id>", "pleasure": 0.1, "arousal": -0.2,
             "dominance": 0.0, "updated_at": "<iso8601>"}
        ],
        "word_count": 612
    }
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

from museai.core.logging_setup import get_fsm_logger


def append_event(path: str | Path, event: dict) -> None:
    """Append one event as a JSON line and fsync to durable storage.

    Logged after the fsync, so a line in ``fsm.log`` means the event is durable.
    """
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    get_fsm_logger().info(
        "event_appended type=%s beat_id=%s word_count=%s bytes=%d",
        event.get("type"),
        event.get("beat_id"),
        event.get("word_count"),
        len(line),
    )


def replay_events(path: str | Path) -> Iterator[dict]:
    """Yield events in append order. A missing log yields nothing."""
    p = Path(path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
