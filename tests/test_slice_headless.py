"""Headless end-to-end checkpoint for the v1 engine."""

from __future__ import annotations

import json
from pathlib import Path

from museai.core.events import replay_events
from museai.core.runtime import init_resources
from museai.fsm.export import committed_word_count, export_manuscript
from museai.fsm.manager import GenerationManager
from museai.fsm.nodes import critics as critics_module
from museai.fsm.nodes import draft_prose as draft_prose_module
from museai.fsm.nodes import plan_beat as plan_beat_module
from museai.fsm.nodes import plan_chapter as plan_chapter_module
from museai.fsm.nodes.deps import set_node_config
from museai.memory import db
from museai.seed.loader import load_seed


SEED_PATH = Path(__file__).resolve().parent.parent / "seeds" / "example.json"
PROJECT_ID = "lantern-keeper"

CHAPTERS_JSON = """[
  {"ordering": 1, "description": "Mara catalogs the impossible letters.",
   "obligations": ["Mara dates the earliest letter."]}
]"""

BEATS_JSON = """[
  {"ordering": 1, "intent": "The letter arrives in the day's post.",
   "entry_state": "A routine morning.", "exit_state": "Mara holds her own handwriting.",
   "word_target": 120, "focal_character_id": "lantern-keeper-char-1",
   "target_pad": {"pleasure": -0.7, "arousal": 0.8, "dominance": -0.5}},
  {"ordering": 2, "intent": "Mara locks the letter away.",
   "entry_state": "Mara holds the letter.", "exit_state": "The letter is hidden.",
   "word_target": 120, "focal_character_id": "lantern-keeper-char-1",
   "target_pad": {"pleasure": -0.2, "arousal": -0.3, "dominance": 0.4}}
]"""


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tool_calls: list[dict] = []
        self.tokens_out = len(text.split())
        self.finish_reason = "stop"


def _patch_clean_headless_endpoint(monkeypatch):
    drafts = [
        "Mara reads the first impossible letter.",
        "Mara locks the letter in the brass drawer.",
        "Idris finds the second letter waiting on the quay.",
        "Mara and Idris compare the postmarks before dusk.",
    ]

    async def fake_chapter_llm(endpoint, messages, **kwargs):
        return _Response(CHAPTERS_JSON)

    async def fake_beat_llm(endpoint, messages, **kwargs):
        return _Response(BEATS_JSON)

    async def fake_draft_llm(endpoint, messages, *, stream=False, on_token=None, **kw):
        draft = drafts.pop(0)
        await on_token(draft)
        return _Response(draft)

    async def fake_critic_loop(endpoint, messages, tools, tool_impls, max_iterations, on_event=None, **kwargs):
        return _Response("[]")

    monkeypatch.setattr(plan_chapter_module, "call_llm", fake_chapter_llm)
    monkeypatch.setattr(plan_beat_module, "call_llm", fake_beat_llm)
    monkeypatch.setattr(draft_prose_module, "call_llm", fake_draft_llm)
    monkeypatch.setattr(critics_module, "run_agent_loop", fake_critic_loop)


def _manuscript_prose_word_count(text: str) -> int:
    prose_lines = [line for line in text.splitlines() if line.strip() and not line.startswith("## ")]
    return len("\n".join(prose_lines).split())


async def test_headless_engine_runs_to_export_with_committed_prose_only(
    config_factory, monkeypatch
):
    config = config_factory(project_id=PROJECT_ID)
    init_resources(config)
    load_seed(json.loads(SEED_PATH.read_text(encoding="utf-8")), config)
    set_node_config(config)
    _patch_clean_headless_endpoint(monkeypatch)

    manager = GenerationManager(config)
    await manager.start(PROJECT_ID)
    assert await manager.wait() == "done"

    conn = db.connect_db(config.db_path)
    completed = conn.execute(
        "SELECT * FROM Beats WHERE status='completed' ORDER BY id ASC"
    ).fetchall()
    assert len(completed) > 1
    committed_total = committed_word_count(config)
    assert committed_total == sum(beat["word_count"] for beat in completed)
    conn.close()

    events = list(replay_events(config.event_log_path))
    beat_commits = [event for event in events if event.get("type") == "beat_commit"]
    assert len(beat_commits) == len(completed)

    manuscript_path = export_manuscript(config)
    manuscript = manuscript_path.read_text(encoding="utf-8")
    assert manuscript.strip()
    assert "## Chapter" in manuscript
    assert _manuscript_prose_word_count(manuscript) == committed_total
