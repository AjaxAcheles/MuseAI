"""Module: M06 (Drafting & Generation) x M03 (Context Assembly & Budgeting)

INT-D-F checkpoint: proves the real M03 context assembly feeds the real M06
drafting node end to end. A deterministic wiring path (always runs) seeds real
relational + planning + provisional stores under ``tmp_path``, runs the real
``node_assemble_context`` over them, then feeds the resulting
``active_context_package`` to the real ``node_draft_prose`` with a fake streaming
seam. A live path (skips cleanly when no drafter endpoint/credential is
configured) runs the same seeded pipeline against the node's own real default
seam and checks structure only, never exact wording.

This checkpoint ends at ``current_draft_text`` — no auditing (M07) or commit
(M10) behavior is exercised here.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from core.config_loader import load_config
from fsm.nodes.node_assemble_context import node_assemble_context
from fsm.nodes.node_draft_prose import node_draft_prose
from fsm.state import FSM_Pointer
from llm.call_llm import LLMCallError
from memory import provisional_store, sqlite_db

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
ENDPOINT_NAMES = ("planner", "drafter", "critic", "pad_translator", "craft_consultant")
_SKIP_ENV = "MUSEAI_SKIP_LIVE_LLM"

SNAPSHOT_ID = "snap-int-d-f"
BEAT_ID = "beat-1"

_BEAT_PLAN = {
    "immediate_objective": "INT_D_F_OBJECTIVE: Elena crosses to check the cracked door glass.",
    "physical_constraints": [
        "INT_D_F_CONSTRAINT: the front door glass has cracked since the delivery.",
    ],
    "entry_condition": "the shop has just closed for the night",
    "exit_condition": "Elena confirms the door is intact and turns back inside",
    "asserted_facts": ["the storm knocked the sign loose two nights ago"],
    "behavioral_constraint": (
        "INT_D_F_PAD: watchful and unhurried, noticing small physical details before "
        "any feeling"
    ),
}


def seed_relational_store(db_path: str | Path) -> None:
    """Seed one arc -> chapter -> scene, with one committed beat, into a real hub.

    Planner-owned parent rows (Arcs/Chapters/Scenes) have no public write helper
    yet, so they're seeded directly (FK-ordered); the committed beat goes through
    the real 04.02 write helper.
    """
    sqlite_db.init_db(db_path)
    conn = sqlite_db.connect_db(db_path)
    try:
        conn.execute(
            "INSERT INTO Arcs (id, description, status) VALUES (?, ?, ?)",
            ("arc-1", "the harbor arc", "active"),
        )
        conn.execute(
            "INSERT INTO Chapters (id, arc_id, description, status) VALUES (?, ?, ?, ?)",
            ("chapter-1", "arc-1", "closing time", "active"),
        )
        conn.execute(
            "INSERT INTO Scenes "
            "(id, chapter_id, description, word_budget, ordering, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("scene-1", "chapter-1", "the shop after hours", 600, 0, "active"),
        )
        conn.commit()
    finally:
        conn.close()

    sqlite_db.upsert_beat_commit(
        db_path,
        beat_id="beat-0",
        scene_id="scene-1",
        beat_index=0,
        prose="INT_D_F_MARKER: Elena locked the register and killed the lights.",
        status="completed",
    )


def seed_beat_plan(
    db_path: str | Path, *, snapshot_id: str, beat_id: str, plan: dict[str, Any]
) -> None:
    """Seed one planned-but-undrafted beat via the real two-write persist sequence.

    Mirrors ``node_plan_beat``'s own persist function: ``upsert_beat_plan`` first
    (structural ``Beats`` row + minimal PAD detail), then ``upsert_planning_node``
    again on the same ``node_id`` to upgrade ``purpose`` to the full plan JSON that
    ``node_draft_prose`` actually reads (entry/exit conditions, asserted facts, the
    full behavioral_constraint string). ``parent_node_id=None`` skips the
    arc/chapter/scene PlanningNode chain, which nothing here reads.
    """
    sqlite_db.create_planning_snapshot(
        db_path, snapshot_id=snapshot_id, project_id="proj-int-d-f", mode="rolling"
    )
    node_id = f"{snapshot_id}:beat:{beat_id}"
    sqlite_db.upsert_beat_plan(
        db_path,
        beat_id=beat_id,
        scene_id="scene-1",
        beat_index=1,
        status="planned",
        snapshot_id=snapshot_id,
        node_id=node_id,
        parent_node_id=None,
        ordering=1,
        title=plan["immediate_objective"],
        pad_constraint=plan["behavioral_constraint"],
        immediate_objective=plan["immediate_objective"],
        physical_constraints=json.dumps(plan["physical_constraints"]),
    )
    sqlite_db.upsert_planning_node(
        db_path,
        node_id=node_id,
        snapshot_id=snapshot_id,
        level="beat",
        status="planned",
        parent_id=None,
        ordering=1,
        title=plan["immediate_objective"],
        summary=plan["immediate_objective"],
        purpose=json.dumps(plan),
    )


def load_real_config(monkeypatch: pytest.MonkeyPatch):
    """Load the real config.yaml, injecting placeholder endpoint secrets.

    Secrets are required by the loader but never live in config.yaml; a
    placeholder per endpoint lets validation pass without a real backend.
    """
    for name in ENDPOINT_NAMES:
        monkeypatch.setenv(f"{name.upper()}_API_KEY", "placeholder-not-used")
    return load_config(CONFIG_PATH)


def guard_no_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any inference HTTP or call_llm invocation a hard failure."""

    def _explode(*args: Any, **kwargs: Any):
        raise AssertionError("the deterministic path must not make an inference call")

    monkeypatch.setattr(httpx, "AsyncClient", _explode)
    import llm.call_llm as call_llm_module

    monkeypatch.setattr(call_llm_module, "call_llm", _explode, raising=False)
    monkeypatch.setattr(call_llm_module, "call_llm_structured", _explode, raising=False)


def _chunk_seam(chunks: list[str], *, capture: dict[str, Any] | None = None):
    """Fake call seam yielding ``chunks`` in order, optionally capturing ``messages``."""

    async def _seam(messages, *, on_token, timeout_seconds):
        if capture is not None:
            capture["messages"] = messages
        collected: list[str] = []
        for chunk in chunks:
            await on_token(chunk)
            collected.append(chunk)
        return "".join(collected)

    return _seam


def _make_state(
    config: Any, *, db_path: Path, provisional_path: Path
) -> dict[str, Any]:
    return {
        "fsm_pointer": FSM_Pointer(
            arc_id="arc-1",
            chapter_id="chapter-1",
            scene_id="scene-1",
            beat_index=1,
            beat_id=BEAT_ID,
        ),
        "app_config": config,
        "sqlite_db_path": str(db_path),
        "provisional_store_path": str(provisional_path),
        "planning_snapshot_id": SNAPSHOT_ID,
        "streaming_buffer": "",
        "current_draft_text": "",
    }


def test_real_context_assembly_feeds_real_drafting_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    db_path = tmp_path / "fictionwriter.db"
    provisional_path = tmp_path / "provisional_claims.db"
    seed_relational_store(db_path)
    provisional_store.init_provisional_store(provisional_path)
    seed_beat_plan(db_path, snapshot_id=SNAPSHOT_ID, beat_id=BEAT_ID, plan=_BEAT_PLAN)

    guard_no_inference(monkeypatch)
    config = load_real_config(monkeypatch)

    state = _make_state(config, db_path=db_path, provisional_path=provisional_path)
    state = asyncio.run(node_assemble_context(state))
    package = state["active_context_package"]

    # --- M03 deferred-store degradation is unchanged by this checkpoint ------
    assert package["meta"]["temporal"]["available"] is False
    assert "not yet implemented" in package["meta"]["temporal"]["reason"]
    assert package["meta"]["flavour"]["available"] is False
    assert "not yet implemented" in package["meta"]["flavour"]["reason"]
    assert package["meta"]["relational"]["available"] is True
    assert package["meta"]["pruned_layers"] == []

    before_bytes = db_path.read_bytes()

    capture: dict[str, Any] = {}
    seam = _chunk_seam(["Once ", "upon ", "a time."], capture=capture)
    state = asyncio.run(node_draft_prose(state, call_seam=seam))

    prompt = capture["messages"][0]["content"]
    # relational layer content (the prior committed beat's prose) reached the prompt
    assert "INT_D_F_MARKER" in prompt
    # the beat's plan fields (from the real PlanningNode) reached the prompt
    assert "INT_D_F_OBJECTIVE" in prompt
    assert "INT_D_F_CONSTRAINT" in prompt
    assert "INT_D_F_PAD" in prompt

    assert state["current_draft_text"] == "Once upon a time."
    # drafting is state-only: no store gained a row
    assert db_path.read_bytes() == before_bytes


@pytest.fixture
def live_drafter_config(monkeypatch: pytest.MonkeyPatch):
    """Load the real config for a live drafter call, or skip with a clear reason."""
    if os.environ.get(_SKIP_ENV, "").strip() not in ("", "0", "false", "False"):
        pytest.skip(f"{_SKIP_ENV} set — live drafter endpoint test opted out")

    has_real_key = bool(os.environ.get("DRAFTER_API_KEY"))
    for name in ENDPOINT_NAMES:
        if not os.environ.get(f"{name.upper()}_API_KEY"):
            monkeypatch.setenv(f"{name.upper()}_API_KEY", "placeholder-not-used")

    try:
        config = load_config(CONFIG_PATH)
    except Exception as exc:  # noqa: BLE001 - any load failure means "cannot test live"
        pytest.skip(f"config.yaml could not be loaded for a live call: {exc}")

    if not has_real_key:
        pytest.skip("no real DRAFTER_API_KEY configured — live drafter path unavailable")
    return config


@pytest.fixture
def io_log_capture(tmp_path: Path):
    """Attach a temporary capture handler to the real ``llm_io`` logger.

    Mirrors ``tests/test_llm_config_integration.py``'s fixture: the production
    logger writes to a fixed ``logs/llm_io.log``, so an extra ``FileHandler`` on
    the same singleton logger captures exactly the records emitted while active.
    """
    import logging

    import core.llm_io_logger as llm_io_logger

    capture_path = tmp_path / "llm_io_capture.log"
    logger = llm_io_logger.get_llm_io_logger()
    handler = logging.FileHandler(capture_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    try:
        yield capture_path
    finally:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def test_live_drafter_over_real_context_or_skip(
    live_drafter_config: Any, io_log_capture: Path, tmp_path: Path
) -> None:
    config = live_drafter_config
    db_path = tmp_path / "fictionwriter.db"
    provisional_path = tmp_path / "provisional_claims.db"
    seed_relational_store(db_path)
    provisional_store.init_provisional_store(provisional_path)
    seed_beat_plan(db_path, snapshot_id=SNAPSHOT_ID, beat_id=BEAT_ID, plan=_BEAT_PLAN)

    state = _make_state(config, db_path=db_path, provisional_path=provisional_path)
    state = asyncio.run(node_assemble_context(state))

    chunks_received: list[str] = []

    async def _publisher(chunk: str) -> None:
        chunks_received.append(chunk)

    endpoint = config.endpoints.drafter
    try:
        state = asyncio.run(node_draft_prose(state, publisher=_publisher))
    except (LLMCallError, httpx.HTTPError, OSError) as exc:
        pytest.skip(
            f"drafter endpoint ({endpoint.base_url}, model={endpoint.model_name}) "
            f"could not complete a call: {exc}"
        )

    assert state["current_draft_text"].strip() != ""
    assert chunks_received, "expected at least one streamed chunk from the live endpoint"

    log_text = io_log_capture.read_text(encoding="utf-8")
    assert log_text.strip() != "", "expected an llm_io record for the drafter call"
    assert endpoint.model_name in log_text
    assert endpoint.api_key not in log_text
