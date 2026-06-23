"""Module: M02 (Persistent Memory Stores)
Manage hierarchical chapter and arc summaries persisted to SQLite.
STUB: summary persistence lands with the RAPTOR memory module.
"""


def init_raptor_store(config) -> None:
    """No-op initializer until the RAPTOR store module exists."""
    del config


class RaptorStore:
    """RAPTOR summary store."""

    def summarize(self, *args, **kwargs):
        """Summarize committed text."""
        raise NotImplementedError("STUB")
