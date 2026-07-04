"""Module: M06 (Drafting & Generation)
Deterministic, no-network tests for node_draft_prose (fsm/nodes/node_draft_prose.py).

Every test injects its own call_seam/publisher fakes; no network call, no real
model backend, and no writes under a real ``data/`` tree. The only store touched
is a throwaway ``tmp_path`` SQLite file seeded with one beat PlanningNode, so the
node's own beat-plan read has something real to resolve.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from jinja2 import UndefinedError

import fsm.nodes.node_draft_prose as node_draft_prose_mod
from core.antislop import detect_slop as _real_detect_slop
from core.antislop import resolve_slop as _real_resolve_slop
from fsm.nodes.node_draft_prose import node_draft_prose
from fsm.state import FSM_Pointer
from memory.sqlite_db import create_planning_snapshot, init_db, upsert_planning_node
from prompts.prompt_loader import PromptLoader

NODE_NAME = "node_draft_prose"
SNAPSHOT_ID = "snap-test-draft"
BEAT_ID = "beat-1"

# Matches config.yaml's endpoint keys — only needed by the failure-path test, where
# the node's own real get_logger()/log_node_event() call loads config.yaml for real.
_ENDPOINT_NAMES = ("planner", "drafter", "critic", "pad_translator", "craft_consultant")

_BEAT_PLAN = {
    "immediate_objective": "Elena notices the shop door hasn't latched since the storm.",
    "physical_constraints": [
        "Elena is behind the register; Marcus is by the west shelf, six feet away.",
        "The front door glass is cracked from last week's delivery truck.",
    ],
    "entry_condition": "the shop has just closed for the night",
    "exit_condition": "Elena crosses to the door and throws the bolt",
    "asserted_facts": ["the storm knocked the sign loose two nights ago"],
    "behavioral_constraint": (
        "watchful and a little tired: short declarative thoughts, notice small "
        "physical details before naming any feeling"
    ),
}


def _seed_beat_plan(db_path: Path, *, snapshot_id: str, beat_id: str, plan: dict) -> None:
    node_id = f"{snapshot_id}:beat:{beat_id}"
    create_planning_snapshot(
        db_path, snapshot_id=snapshot_id, project_id="proj-test", mode="micro"
    )
    upsert_planning_node(
        db_path,
        node_id=node_id,
        snapshot_id=snapshot_id,
        level="beat",
        status="planned",
        parent_id=None,
        ordering=0,
        title=plan.get("immediate_objective", ""),
        summary=plan.get("immediate_objective", ""),
        purpose=json.dumps(plan),
    )


def _config(timeout_seconds: float = 30.0) -> SimpleNamespace:
    return SimpleNamespace(
        endpoints=SimpleNamespace(
            drafter=SimpleNamespace(tokenizer_family="char_heuristic", model_name="test-drafter")
        ),
        runtime=SimpleNamespace(inference_timeout_seconds=timeout_seconds),
    )


def _pointer(beat_id: str = BEAT_ID) -> FSM_Pointer:
    return FSM_Pointer(
        arc_id="arc-1", chapter_id="chapter-1", scene_id="scene-1", beat_index=0, beat_id=beat_id
    )


def _package(*, word_budget: int = 400, committed_beats: list[dict] | None = None) -> dict:
    return {
        "relational": {
            "current_scene": {"id": "scene-1", "word_budget": word_budget},
            "committed_beats": committed_beats or [],
        },
        "coreference_candidates": [],
        "summaries": {},
        "flavour": [],
        "temporal": {},
        "macro_constraints": {},
    }


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


def _failing_seam(chunks: list[str], *, fail_after: int):
    """Fake call seam that raises after streaming ``fail_after`` chunks."""

    async def _seam(messages, *, on_token, timeout_seconds):
        for i, chunk in enumerate(chunks):
            await on_token(chunk)
            if i + 1 == fail_after:
                raise RuntimeError("transport exploded mid-stream")
        return "".join(chunks)

    return _seam


def _recording_publisher(state: dict, chunks_sink: list[str], buffer_sink: list[str]):
    """Publisher recording each chunk plus the accumulator's value at that instant."""

    async def _publisher(chunk: str) -> None:
        chunks_sink.append(chunk)
        buffer_sink.append(state.get("streaming_buffer", ""))

    return _publisher


class _RaisingIfCalled:
    """Callable that fails the test if invoked — proves a seam was skipped."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("call_seam must not be invoked")


def _full_render_context() -> dict[str, Any]:
    """Every documented context variable node_draft_prose.xml.j2 requires."""
    return {
        "relational_facts": {"fact": "value"},
        "epistemic_beliefs": [],
        "summary_context": {},
        "flavour_passages": [],
        "temporal_context": {},
        "macro_constraints": {},
        "beat_immediate_objective": "objective",
        "beat_physical_constraints": [],
        "active_continuity": {"entry_condition": "", "exit_condition": "", "continuity_facts": []},
        "pad_behavioral_constraint": "constraint",
        "prior_prose_tail": "",
        "scene_entry_state": "",
        "target_word_count": 100,
    }


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "planning.sqlite3"
    init_db(path)
    _seed_beat_plan(path, snapshot_id=SNAPSHOT_ID, beat_id=BEAT_ID, plan=_BEAT_PLAN)
    return path


@pytest.fixture
def base_state(db_path: Path) -> dict[str, Any]:
    return {
        "fsm_pointer": _pointer(),
        "active_context_package": _package(),
        "app_config": _config(),
        "sqlite_db_path": str(db_path),
        "planning_snapshot_id": SNAPSHOT_ID,
        "streaming_buffer": "",
        "current_draft_text": "",
    }


def test_render_contract_includes_beat_plan_fields(base_state: dict[str, Any]) -> None:
    capture: dict[str, Any] = {}
    seam = _chunk_seam(["Once upon a time."], capture=capture)

    result = asyncio.run(node_draft_prose(base_state, call_seam=seam))

    prompt = capture["messages"][0]["content"]
    assert _BEAT_PLAN["immediate_objective"] in prompt
    assert _BEAT_PLAN["physical_constraints"][0] in prompt
    assert _BEAT_PLAN["behavioral_constraint"] in prompt
    assert result is base_state


def test_render_contract_raises_on_missing_documented_variable() -> None:
    ctx = _full_render_context()
    del ctx["target_word_count"]
    with pytest.raises(UndefinedError):
        PromptLoader().render(NODE_NAME, ctx)


def test_streaming_accumulates_and_publishes_chunks_in_order(
    base_state: dict[str, Any],
) -> None:
    chunks = ["Once ", "upon ", "a time."]
    seam = _chunk_seam(chunks)
    received: list[str] = []
    buffers: list[str] = []
    publisher = _recording_publisher(base_state, received, buffers)

    result = asyncio.run(node_draft_prose(base_state, call_seam=seam, publisher=publisher))

    assert received == chunks
    assert buffers == ["Once ", "Once upon ", "Once upon a time."]
    assert result["current_draft_text"] == "Once upon a time."
    assert result["streaming_buffer"] == ""


def test_streaming_without_publisher_still_completes(base_state: dict[str, Any]) -> None:
    chunks = ["Once ", "upon ", "a time."]
    seam = _chunk_seam(chunks)

    result = asyncio.run(node_draft_prose(base_state, call_seam=seam, publisher=None))

    assert result["current_draft_text"] == "Once upon a time."
    assert result["streaming_buffer"] == ""


def test_antislop_handoff_invokes_detect_then_resolve(
    base_state: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: dict[str, Any] = {}

    def _spy_detect(text):
        recorded["detect_arg"] = text
        result = _real_detect_slop(text)
        recorded["detect_return"] = result
        return result

    def _spy_resolve(text, findings=None):
        recorded["resolve_args"] = (text, findings)
        result = _real_resolve_slop(text, findings)
        recorded["resolve_return"] = result
        return result

    monkeypatch.setattr(node_draft_prose_mod, "detect_slop", _spy_detect)
    monkeypatch.setattr(node_draft_prose_mod, "resolve_slop", _spy_resolve)

    seam = _chunk_seam(["Once upon a time."])
    result = asyncio.run(node_draft_prose(base_state, call_seam=seam))

    assert recorded["detect_arg"] == "Once upon a time."
    assert recorded["resolve_args"] == ("Once upon a time.", recorded["detect_return"])
    assert result["current_draft_text"] == recorded["resolve_return"]


def test_call_seam_failure_surfaces_and_leaves_draft_untouched(
    base_state: dict[str, Any], db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in _ENDPOINT_NAMES:
        monkeypatch.setenv(f"{name.upper()}_API_KEY", "placeholder-secret")

    base_state["current_draft_text"] = "PRIOR_DRAFT_UNCHANGED"
    before_bytes = db_path.read_bytes()
    before_files = sorted(p.name for p in db_path.parent.iterdir())

    seam = _failing_seam(["Once ", "upon "], fail_after=1)

    with pytest.raises(RuntimeError):
        asyncio.run(node_draft_prose(base_state, call_seam=seam))

    assert base_state["current_draft_text"] == "PRIOR_DRAFT_UNCHANGED"
    assert db_path.read_bytes() == before_bytes
    assert sorted(p.name for p in db_path.parent.iterdir()) == before_files


def test_publisher_failure_does_not_lose_the_draft(base_state: dict[str, Any]) -> None:
    chunks = ["Once ", "upon ", "a time."]
    seam = _chunk_seam(chunks)

    async def _raising_publisher(chunk: str) -> None:
        raise RuntimeError("SSE bus exploded")

    result = asyncio.run(
        node_draft_prose(base_state, call_seam=seam, publisher=_raising_publisher)
    )

    assert result["current_draft_text"] == "Once upon a time."
    assert result["streaming_buffer"] == ""


def test_missing_context_package_skips_the_seam(base_state: dict[str, Any]) -> None:
    base_state["active_context_package"] = None

    result = asyncio.run(node_draft_prose(base_state, call_seam=_RaisingIfCalled()))

    assert result is base_state
    assert result["current_draft_text"] == ""
    assert result["streaming_buffer"] == ""
