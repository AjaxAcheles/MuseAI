"""Module: M02 (Persistent Memory Stores)
Provide bounded temporal graph traversal through FalkorDB Lite and Graphiti.
STUB: Graphiti client initialization lands with the graph memory module.
"""

from pathlib import Path


def init_graphiti_store(config, graph_path: str | Path) -> None:
    """No-op initializer until the Graphiti store module exists.

    It only ensures the FalkorDB Lite artifact directory exists.
    """
    del config
    Path(graph_path).mkdir(parents=True, exist_ok=True)


class GraphitiClient:
    """Temporal graph client."""

    def query(self, *args, **kwargs):
        """Query the temporal graph."""
        raise NotImplementedError("STUB")


def _apply_event(*args, **kwargs):
    """Intentional early no-op for graph writes during crash-recovery scaffolding."""
    return None
