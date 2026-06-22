"""Module: M02 (Persistent Memory Stores)
Provide bounded temporal graph traversal through FalkorDB Lite and Graphiti.
STUB:
"""


class GraphitiClient:
    """Temporal graph client."""

    def query(self, *args, **kwargs):
        """Query the temporal graph."""
        raise NotImplementedError("STUB")


def _apply_event(*args, **kwargs):
    """Intentional early no-op for graph writes during crash-recovery scaffolding."""
    return None
