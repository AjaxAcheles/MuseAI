"""Read-only, project-scoped database access for agent tools.

Every DB-backed tool opens its own short-lived connection through here. The
connection is pinned read-only with ``PRAGMA query_only``: these tools execute
inside a loop a *model* drives, and nothing a model asks for may write to the
manuscript store. A tool that tried would get an ``OperationalError``, which the
agent loop hands back to the model as an error string.

The scope is the active project — ``config.project_id``, which seed loading
keeps in sync with the run (see ``museai/web/routes/seed.py``).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from museai.fsm.nodes.deps import get_node_config
from museai.memory.db import connect_db


@contextmanager
def project_connection() -> Iterator[tuple[sqlite3.Connection, str]]:
    """Yield ``(connection, project_id)`` for the active project, read-only."""
    config = get_node_config()
    conn = connect_db(config.db_path)
    try:
        conn.execute("PRAGMA query_only = ON")
        yield conn, config.project_id
    finally:
        conn.close()
