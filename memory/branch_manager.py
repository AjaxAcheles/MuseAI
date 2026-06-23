"""Module: M10 (Commit, Consolidation & Crash Immunity)
Restore branches from snapshots and replay JSONL events after crashes.
STUB: branch restore and snapshot creation land with the commit/crash module.
"""

from pathlib import Path


def init_snapshot_store(config, snapshot_path: str | Path) -> None:
    """No-op initializer until snapshot management exists.

    It only ensures the snapshot archive directory exists.
    """
    del config
    Path(snapshot_path).mkdir(parents=True, exist_ok=True)


class BranchManager:
    """Branch and snapshot manager."""

    def restore(self, *args, **kwargs):
        """Restore a branch snapshot."""
        raise NotImplementedError("STUB")
