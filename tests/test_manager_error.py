"""A node exception must park the manager in 'error', never a stuck 'running'."""

from __future__ import annotations

from museai.core.runtime import init_resources
from museai.core.stream_bus import bus
from museai.fsm import manager as manager_module
from museai.fsm.manager import GenerationManager
from museai.memory.db import connect_db, upsert_arc, upsert_project


class _BoomGraph:
    async def astream(self, state):
        raise RuntimeError("endpoint exploded")
        yield  # pragma: no cover - makes this an async generator


def _seed(config) -> None:
    conn = connect_db(config.db_path)
    try:
        with conn:
            upsert_project(conn, id=config.project_id, genre="g", premise="p",
                           word_count_target=1000)
            upsert_arc(conn, id="arc-1", project_id=config.project_id,
                       ordering=0, description="An arc.", status="active")
    finally:
        conn.close()


async def test_node_exception_sets_error_status(config_factory, monkeypatch):
    config = config_factory()
    init_resources(config)
    _seed(config)
    monkeypatch.setattr(
        manager_module, "build_graph", lambda cfg, entry_point="plan_chapter": _BoomGraph()
    )
    bus.last_snapshot.pop("run_status", None)

    manager = GenerationManager(config)
    await manager.start(config.project_id)
    status = await manager.wait()

    assert status == "error"
    published = bus.last_snapshot.get("run_status", {})
    assert published.get("status") == "error"
    assert "endpoint exploded" in published.get("error", "")
    bus.last_snapshot.pop("run_status", None)
