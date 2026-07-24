"""End-to-end slice: seed → plan → draft → audit → critics → revise → mode_selector.

Every node and router is real, and every write lands in a real SQLite file. Only
the endpoint is faked: the planners get canned JSON, the drafter a canned
stream, the critic a canned findings array, the reviser canned prose.

This is the isolation check that the quality loop closes. Two runs:

* the critic finds one problem, `revise` fixes it, the critic comes back clean,
  and the router reaches ``commit``;
* the critic keeps finding the problem, the retry budget runs out, and the
  router reaches ``review`` — without anything having been discarded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from museai.core.runtime import init_resources
from museai.fsm.nodes import critics as critics_module
from museai.fsm.nodes import draft_prose as draft_prose_module
from museai.fsm.nodes import plan_beat as plan_beat_module
from museai.fsm.nodes import plan_chapter as plan_chapter_module
from museai.fsm.nodes import revise as revise_module
from museai.fsm.nodes.assemble_context import assemble_context
from museai.fsm.nodes.audit import audit
from museai.fsm.nodes.critics import adversarial_critics
from museai.fsm.nodes.deps import set_node_config
from museai.fsm.nodes.draft_prose import draft_prose
from museai.fsm.nodes.plan_beat import plan_beat
from museai.fsm.nodes.plan_chapter import plan_chapter
from museai.fsm.nodes.revise import revise_prose
from museai.fsm.routers.mode_selector import COMMIT, REVIEW, mode_selector
from museai.fsm.state import FSM_Pointer, accumulate_or_reset, make_initial_state
from museai.seed.loader import load_seed

from conftest import patch_planner_llm

SEED_PATH = Path(__file__).resolve().parent.parent / "seeds" / "example.json"
ARC_ID = "the-borrowed-ladder-arc-1"
RETRY_CAP = 2

CHAPTERS_JSON = """```json
[
  {"ordering": 1, "description": "Mara catalogs the impossible letters.",
   "obligations": ["Mara dates the earliest letter."]}
]
```"""

BEATS_JSON = """```json
[
  {"ordering": 1, "intent": "The letter arrives in the day's post.",
   "entry_state": "A routine morning.", "exit_state": "Mara holds her own handwriting.",
   "focal_character_id": "the-borrowed-ladder-nell-ardery",
   "target_pad": {"pleasure": -0.7, "arousal": 0.8, "dominance": -0.5}}
]
```"""

# Active voice throughout, so the programmatic audit stays quiet and the
# continuity critic is the only thing driving the loop.
PROSE_TOKENS = ["The post came up the path. ", "Mara lied about the letter. ", "She slept."]
DRAFT = "".join(PROSE_TOKENS)
OFFENDING = "Mara lied about the letter."
REPAIRED = "Mara said nothing about the letter."

ONE_FAILURE = """```json
[
  {
    "error_code": "CONTRADICTS_CHARACTER",
    "offending_text": "Mara lied about the letter.",
    "suggested_fix": "Mara never lies; have her say nothing.",
    "critic_source": "continuity_critic"
  }
]
```"""

CLEAN = "[]"


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tool_calls: list[dict] = []
        self.tokens_out = len(text.split())
        self.finish_reason = "stop"


def apply(state: dict, delta: dict) -> dict:
    """Merge a node's delta into state the way LangGraph's reducers would."""
    for key, value in delta.items():
        if key == "critic_failures":
            state[key] = accumulate_or_reset(state[key], value)
        else:
            state[key] = value
    return state


@pytest.fixture
def project(config_factory):
    config = config_factory(project_id="the-borrowed-ladder", revision_retry_cap=RETRY_CAP)
    init_resources(config)
    load_seed(json.loads(SEED_PATH.read_text(encoding="utf-8")), config)
    set_node_config(config)
    return config


@pytest.fixture
def endpoint(monkeypatch):
    """Fake the four call sites; the critic's responses are scripted per test."""

    def _install(critic_responses: list[str]):
        scripted = list(critic_responses)

        async def fake_chapter_llm(endpoint, messages, **kwargs):
            return _Response(CHAPTERS_JSON)

        async def fake_beat_llm(endpoint, messages, **kwargs):
            return _Response(BEATS_JSON)

        async def fake_draft_loop(
            endpoint, messages, tools, tool_impls, max_iterations, *, on_token=None, **kw
        ):
            for token in PROSE_TOKENS:
                await on_token(token)
            return _Response(DRAFT)

        async def fake_critic_loop(endpoint, messages, tools, tool_impls, max_iterations, on_event=None, **kwargs):
            return _Response(scripted.pop(0))

        async def fake_revise_loop(endpoint, messages, tools, tool_impls, max_iterations, **kwargs):
            return _Response(REPAIRED)

        patch_planner_llm(monkeypatch, chapter=fake_chapter_llm, beat=fake_beat_llm)
        monkeypatch.setattr(draft_prose_module, "run_agent_loop", fake_draft_loop)
        monkeypatch.setattr(critics_module, "run_agent_loop", fake_critic_loop)
        monkeypatch.setattr(revise_module, "run_agent_loop", fake_revise_loop)

    return _install


async def drafted_state(project):
    """Walk the plan → draft slice and return the state the audit will see."""
    state = make_initial_state(
        "the-borrowed-ladder", FSM_Pointer(arc_id=ARC_ID, chapter_id="", beat_index=0)
    )
    apply(state, await plan_chapter(state))
    apply(state, await plan_beat(state))
    apply(state, await assemble_context(state))
    apply(state, await draft_prose(state))
    return state


async def test_the_quality_loop_reaches_commit_once_the_failure_is_fixed(project, endpoint):
    endpoint([ONE_FAILURE, CLEAN])

    state = await drafted_state(project)
    assert state["current_draft_text"] == DRAFT

    # First pass: clean audit, one continuity failure, so the router revises.
    apply(state, await audit(state))
    assert state["critic_failures"] == []

    apply(state, await adversarial_critics(state))
    assert len(state["critic_failures"]) == 1
    assert state["best_seen_failure_count"] == 1
    assert mode_selector(state) == "revise"

    # The reviser locates the offending span and splices the repair in.
    apply(state, await revise_prose(state))
    assert OFFENDING not in state["current_draft_text"]
    assert REPAIRED in state["current_draft_text"]
    assert state["current_draft_text"].startswith("The post came up the path.")
    assert state["retry_count"] == 1
    assert state["critic_failures"] == []

    # Second pass over the repaired draft: both checks are clean.
    apply(state, await audit(state))
    apply(state, await adversarial_critics(state))
    assert state["critic_failures"] == []

    # The repaired draft is the best one seen, at zero failures.
    assert state["best_seen_failure_count"] == 0
    assert state["best_seen_draft"] == state["current_draft_text"]

    assert mode_selector(state) == COMMIT


async def test_a_failure_that_survives_the_cap_reaches_review(project, endpoint):
    # The critic finds the same problem on every pass, revision never fixes it.
    endpoint([ONE_FAILURE] * (RETRY_CAP + 1))

    state = await drafted_state(project)

    for expected_retry in range(RETRY_CAP):
        apply(state, await audit(state))
        apply(state, await adversarial_critics(state))
        assert len(state["critic_failures"]) == 1
        assert mode_selector(state) == "revise"

        apply(state, await revise_prose(state))
        assert state["retry_count"] == expected_retry + 1

    # The budget is spent and the failure is still there.
    apply(state, await audit(state))
    apply(state, await adversarial_critics(state))
    assert state["retry_count"] == RETRY_CAP
    assert len(state["critic_failures"]) == 1

    assert mode_selector(state) == REVIEW

    # Routing to review decides nothing: the draft survives untouched, and the
    # best draft seen is still on hand for the human.
    assert state["current_draft_text"]
    assert state["best_seen_draft"] is not None
    assert state["best_seen_failure_count"] == 1
    assert state["review_requested"] is False
