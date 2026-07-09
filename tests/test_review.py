"""Tests for the interactive review safe-boundary and manager resumption."""

from __future__ import annotations

from museai.core.runtime import init_resources
from museai.core.stream_bus import bus
from museai.fsm.manager import GenerationManager
from museai.fsm.nodes import critics as critics_module
from museai.fsm.nodes import draft_prose as draft_prose_module
from museai.fsm.nodes import plan_beat as plan_beat_module
from museai.fsm.nodes import plan_chapter as plan_chapter_module
from museai.fsm.nodes import revise as revise_module
from museai.fsm.nodes.deps import set_node_config
from museai.memory import db
from museai.seed.loader import load_seed


PROJECT_ID = "review-project"
CHAPTERS_JSON = """[
  {"ordering": 1, "description": "Mara confronts one impossible letter.",
   "obligations": ["The letter matters."]}
]"""
BEATS_JSON = """[
  {"ordering": 1, "intent": "The letter arrives.",
   "entry_state": "Routine watch.", "exit_state": "Mara must decide.",
   "word_target": 100, "focal_character_id": "review-project-char-1",
   "target_pad": {"pleasure": -0.5, "arousal": 0.6, "dominance": -0.2}}
]"""
FAILURE_JSON = """[
  {"error_code": "CONTRADICTS_CHARACTER", "offending_text": "draft",
   "suggested_fix": "Change the draft.", "critic_source": "continuity_critic"}
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
                "premise": "A letter tests Mara.",
                "word_count_target": 10_000,
            },
            "arcs": [{"description": "Mara faces one impossible choice."}],
            "threads": [{"description": "Who writes the letter?", "priority_score": 0.9}],
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
    bus.last_snapshot.clear()


def _patch_review_endpoint(monkeypatch, *, critic_responses: list[str], drafts: list[str]):
    scripted_critics = list(critic_responses)
    scripted_drafts = list(drafts)

    async def fake_chapter_llm(endpoint, messages, **kwargs):
        return _Response(CHAPTERS_JSON)

    async def fake_beat_llm(endpoint, messages, **kwargs):
        return _Response(BEATS_JSON)

    async def fake_draft_llm(endpoint, messages, *, stream=False, on_token=None, **kw):
        draft = scripted_drafts.pop(0)
        await on_token(draft)
        return _Response(draft)

    async def fake_critic_loop(endpoint, messages, tools, tool_impls, max_iterations, on_event=None):
        return _Response(scripted_critics.pop(0))

    async def fake_revise_llm(endpoint, messages, **kwargs):
        return _Response("draft still needing review")

    monkeypatch.setattr(plan_chapter_module, "call_llm", fake_chapter_llm)
    monkeypatch.setattr(plan_beat_module, "call_llm", fake_beat_llm)
    monkeypatch.setattr(draft_prose_module, "call_llm", fake_draft_llm)
    monkeypatch.setattr(critics_module, "run_agent_loop", fake_critic_loop)
    monkeypatch.setattr(revise_module, "call_llm", fake_revise_llm)


async def test_persistent_failures_park_at_review(config_factory, monkeypatch):
    config = config_factory(project_id=PROJECT_ID, revision_retry_cap=1)
    _seed(config)
    _patch_review_endpoint(
        monkeypatch,
        critic_responses=[FAILURE_JSON, FAILURE_JSON],
        drafts=["draft needing review"],
    )

    manager = GenerationManager(config)
    await manager.start(PROJECT_ID)
    assert await manager.wait() == "review"
    assert manager.state is not None
    assert manager.state["review_requested"] is True
    assert bus.last_snapshot["review_needed"]["best_seen_draft"] == "draft needing review"


async def test_resolve_review_accept_commits_best_seen_draft(config_factory, monkeypatch):
    config = config_factory(project_id=PROJECT_ID, revision_retry_cap=1)
    _seed(config)
    _patch_review_endpoint(
        monkeypatch,
        critic_responses=[FAILURE_JSON, FAILURE_JSON],
        drafts=["draft needing review"],
    )

    manager = GenerationManager(config)
    await manager.start(PROJECT_ID)
    assert await manager.wait() == "review"

    await manager.resolve_review("accept")
    assert await manager.wait() == "done"

    conn = db.connect_db(config.db_path)
    beat = conn.execute("SELECT * FROM Beats").fetchone()
    assert beat["status"] == "completed"
    assert beat["prose"] == "draft needing review"
    conn.close()


async def test_resolve_review_regenerate_reenters_drafting_with_retry_reset(
    config_factory, monkeypatch
):
    config = config_factory(project_id=PROJECT_ID, revision_retry_cap=1)
    _seed(config)
    _patch_review_endpoint(
        monkeypatch,
        critic_responses=[FAILURE_JSON, FAILURE_JSON, "[]"],
        drafts=["draft needing review", "clean regenerated draft"],
    )

    manager = GenerationManager(config)
    await manager.start(PROJECT_ID)
    assert await manager.wait() == "review"

    await manager.resolve_review("regenerate")
    assert manager.state is not None
    assert manager.state["retry_count"] == 0
    assert await manager.wait() == "done"

    conn = db.connect_db(config.db_path)
    beat = conn.execute("SELECT * FROM Beats").fetchone()
    assert beat["status"] == "completed"
    assert beat["prose"] == "clean regenerated draft"
    conn.close()
