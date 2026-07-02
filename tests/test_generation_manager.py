"""Module: M01 (Coordinator & State Machine)
Wiring test for the linear generation driver (`core/generation_manager.py`). The node
functions, config load, and resource init are replaced with deterministic fakes so no
model server, config file, or real stores are needed — this verifies the DRIVER's
sequencing, the beat loop, status accounting, and the stream-bus event surface.
"""

import asyncio
from types import SimpleNamespace

import core.config_loader as config_loader
import core.generation_manager as gm_mod
import core.runtime as runtime
from core.generation_manager import GenerationManager
from core.stream_bus import StreamBus


class _RecordingBus(StreamBus):
    def __init__(self):
        super().__init__()
        self.events = []

    def publish(self, run_id, event):
        self.events.append(event)
        super().publish(run_id, event)


async def _fake_planner(state):
    return {}


async def _fake_structure(state):
    order = ["b1", "b2", "b3"]
    plans = {b: {"scene_id": "s1", "beat_index": i} for i, b in enumerate(order)}
    pointer = state["fsm_pointer"].model_copy(
        update={"beat_id": "b1", "scene_id": "s1", "beat_index": 0}
    )
    return {"beat_order": order, "beat_plan_by_id": plans, "fsm_pointer": pointer}


async def _fake_assemble(state):
    return {}


async def _fake_draft(state, *, on_token=None):
    if on_token is not None:
        on_token("word ")
    return {"current_draft_text": "word "}


async def _fake_commit(state):
    order = state["beat_order"]
    position = order.index(state["fsm_pointer"].beat_id)
    word_count = state.get("committed_word_count", 0) + 1
    delta = {"committed_word_count": word_count, "last_committed_prose": "word "}
    if position + 1 >= len(order):
        delta["generation_complete"] = True
    else:
        delta["generation_complete"] = False
        delta["fsm_pointer"] = state["fsm_pointer"].model_copy(
            update={"beat_id": order[position + 1]}
        )
    return delta


def _patch(monkeypatch):
    monkeypatch.setattr(gm_mod, "node_plan_global", _fake_planner)
    monkeypatch.setattr(gm_mod, "node_plan_arc", _fake_planner)
    monkeypatch.setattr(gm_mod, "node_plan_structure", _fake_structure)
    monkeypatch.setattr(gm_mod, "node_assemble_context", _fake_assemble)
    monkeypatch.setattr(gm_mod, "node_draft_prose", _fake_draft)
    monkeypatch.setattr(gm_mod, "node_commit_transaction", _fake_commit)
    monkeypatch.setattr(config_loader, "load_config", lambda *_a, **_k: SimpleNamespace())
    monkeypatch.setattr(runtime, "init_resources", lambda *_a, **_k: None)


def test_driver_runs_all_beats_and_reports_complete(monkeypatch):
    _patch(monkeypatch)
    bus = _RecordingBus()

    async def scenario():
        manager = GenerationManager(bus)
        run_id = manager.start({"premise_seed": "x", "target_word_count": 750})
        await manager._tasks[run_id]
        return run_id, manager.status(run_id)

    run_id, status = asyncio.run(scenario())

    assert status["status"] == "complete"
    assert status["beats_committed"] == 3  # all three planned beats drafted + committed
    assert status["word_count"] == 3
    assert status["error"] is None

    types_seen = [e["type"] for e in bus.events]
    assert types_seen.count("token") == 3  # one streamed token per beat
    assert types_seen.count("beat_committed") == 3
    assert "structure" in types_seen
    assert types_seen[-1] == "complete"


def test_driver_reports_error_when_a_node_raises(monkeypatch):
    _patch(monkeypatch)

    async def boom(state):
        raise RuntimeError("planner exploded")

    monkeypatch.setattr(gm_mod, "node_plan_global", boom)
    bus = _RecordingBus()

    async def scenario():
        manager = GenerationManager(bus)
        run_id = manager.start({})
        await manager._tasks[run_id]
        return manager.status(run_id)

    status = asyncio.run(scenario())
    assert status["status"] == "error"
    assert "planner exploded" in status["error"]
    assert bus.events[-1]["type"] == "error"
