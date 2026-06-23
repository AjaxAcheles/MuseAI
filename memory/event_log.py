"""Module: M02 (Persistent Memory Stores)
Write append-only JSONL transaction events for crash recovery and replay.
STUB: append/replay behavior lands with the event-log store module.
"""

from pathlib import Path


def init_event_log(config, log_path: str | Path) -> None:
    """No-op initializer until the event-log store module exists.

    It only ensures the append-only JSONL artifact exists.
    """
    del config
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def append_event(*args, **kwargs):
    """Append an event to the transaction log."""
    raise NotImplementedError("STUB")
