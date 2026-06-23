"""Module: M02 (Persistent Memory Stores)
Manage mathematical voice baselines and L2-norm evolution limits.
STUB: style profile persistence lands with the style-store module.
"""

from pathlib import Path


def init_style_stores(config, store_path: str | Path) -> None:
    """No-op initializer until the style-store module exists.

    It only ensures the directory for future style JSON artifacts exists.
    """
    del config
    Path(store_path).mkdir(parents=True, exist_ok=True)


class StyleStore:
    """Style baseline store."""

    def update(self, *args, **kwargs):
        """Update style state."""
        raise NotImplementedError("STUB")
