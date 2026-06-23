"""Module: M02 (Persistent Memory Stores)
Provide associative flavor context through an HNSW vector index.
STUB: vector-index initialization lands with the Chroma memory module.
"""

from pathlib import Path


def init_chroma_store(config, store_path: str | Path) -> None:
    """No-op initializer until the Chroma store module exists.

    It only ensures the vector-store directory exists.
    """
    del config
    Path(store_path).mkdir(parents=True, exist_ok=True)


class ChromaClient:
    """Vector store client."""

    def query(self, *args, **kwargs):
        """Query associative context."""
        raise NotImplementedError("STUB")
