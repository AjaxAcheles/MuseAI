"""Tests for the compiled LangGraph and background manager happy path."""

from __future__ import annotations

from museai.core.runtime import init_resources
from museai.fsm.manager import GenerationManager
from museai.fsm.nodes import critics as critics_module
from museai.fsm.nodes import draft_prose as draft_prose_module
from museai.fsm.nodes import plan_beat as plan_beat_module
from museai.fsm.nodes import plan_chapter as plan_chapter_module
from museai.fsm.nodes.deps import set_node_config
from museai.fsm.graph import build_graph
from museai.memory import db
from museai.seed.loader import load_seed


PROJECT_ID = "graph-project"

CHAPTERS_JSON = """[
  {"ordering": 1, "description": "Mara reads the first two letters.",
   "obligations": ["The letters are impossible."]}
]"""

BEATS_JSON = """[
  {"ordering": 1, "intent": "The first letter arrives.",
   "entry_state": "Routine watch.", "exit_state": "Mara sees her own hand.",
   "word_target": 100, "focal_character_id": "graph-project-char-1",
   "target_pad": {"pleasure": -0.5, "arousal": 0.6, "dominance": -0.2}},
  {"ordering": 2, "intent": "The second letter confirms the pattern.",
   "entry_state": "Mara doubts the first letter.", "exit_state": "Mara accepts the pattern.",
   "word_target": 100, "focal_character_id": "graph-project-char-1",
   "target_pad": {"pleasure": -0.2, "arousal": 0.4, "dominance": 0.1}}
]"""


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tool_calls: list[dict] = []
        self.tokens_out = len(text.split())
        self.finish_reason = "stop"


def _seed(config) -> None:
    init_resources(config)
    load_seed(
        {
            "project": {
                "id": PROJECT_ID,
                "genre": "mystery",
                "premise": "Letters arrive in impossible order.",
                "word_count_target": 10_000,
            },
            "arcs": [{"description": "Mara discovers the impossible letters."}],
            "threads": [{"description": "Who writes the letters?", "priority_score": 0.9}],
            "characters": [
                {
                    "name": "Mara",
                    "description": "A guarded keeper.",
                    "pad": {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0},
                }
            ],
        },
        config,
    )
    set_node_config(config)


def _patch_clean_endpoint(monkeypatch):
    drafts = ["First committed beat.", "Second committed beat."]

    async def fake_chapter_llm(endpoint, messages, **kwargs):
        return _Response(CHAPTERS_JSON)

    async def fake_beat_llm(endpoint, messages, **kwargs):
        return _Response(BEATS_JSON)

    async def fake_draft_llm(endpoint, messages, *, stream=False, on_token=None, **kw):
        draft = drafts.pop(0)
        for token in [draft]:
            await on_token(token)
        return _Response(draft)

    async def fake_critic_loop(endpoint, messages, tools, tool_impls, max_iterations, on_event=None, **kwargs):
        return _Response("[]")

    monkeypatch.setattr(plan_chapter_module, "call_llm", fake_chapter_llm)
    monkeypatch.setattr(plan_beat_module, "call_llm", fake_beat_llm)
    monkeypatch.setattr(draft_prose_module, "call_llm", fake_draft_llm)
    monkeypatch.setattr(critics_module, "run_agent_loop", fake_critic_loop)


def test_compiled_graph_builds(config_factory):
    config = config_factory(project_id=PROJECT_ID)
    graph = build_graph(config)
    assert graph is not None


async def test_manager_start_commits_and_router_loops_to_next_beat(config_factory, monkeypatch):
    config = config_factory(project_id=PROJECT_ID)
    _seed(config)
    _patch_clean_endpoint(monkeypatch)

    manager = GenerationManager(config)
    await manager.start(PROJECT_ID)
    assert await manager.wait() == "done"

    conn = db.connect_db(config.db_path)
    beats = conn.execute("SELECT * FROM Beats ORDER BY ordering ASC").fetchall()
    assert [beat["status"] for beat in beats] == ["completed", "completed"]
    assert [beat["prose"] for beat in beats] == ["First committed beat.", "Second committed beat."]
    conn.close()
