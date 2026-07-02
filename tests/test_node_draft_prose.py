"""Module: M06 (Drafting & Generation)
Synthetic tests for `node_draft_prose` using its injectable ``loader``/``call``
seams, so no model server, config file, or network is touched. Verifies that
streamed tokens are forwarded to ``on_token`` and accumulated into the returned
draft, and that the configured inference timeout is passed through to the boundary.
"""

import asyncio
from types import SimpleNamespace

from fsm.nodes.node_draft_prose import node_draft_prose
from fsm.state import FSM_Pointer


class _FakeLoader:
    def __init__(self):
        self.rendered_with = None

    def render(self, node_name, context):
        self.rendered_with = (node_name, context)
        return f"PROMPT for {context.get('beat_objective')}"


def _fake_config():
    # Only the fields node_draft_prose actually reads.
    return SimpleNamespace(
        endpoints=SimpleNamespace(drafter=SimpleNamespace(base_url="x", model_name="m")),
        runtime=SimpleNamespace(inference_timeout_seconds=42),
    )


def _state():
    return {
        "app_config": _fake_config(),
        "fsm_pointer": FSM_Pointer(
            arc_id="a1", chapter_id="c1", scene_id="s1", beat_index=0, beat_id="b1"
        ),
        "project_metadata": {"genre": "noir", "premise_seed": "a city that forgets"},
        "beat_plan_by_id": {"b1": {"objective": "Open on rain.", "scene_id": "s1", "beat_index": 0}},
        "active_context_package": {"relational": {"facts": ["Nadia is the detective"]}},
        "last_committed_prose": "",
    }


def test_draft_streams_tokens_and_returns_accumulated_text():
    captured = {"timeout": None, "messages": None}
    tokens = ["The ", "rain ", "fell."]

    async def fake_call(messages, endpoint, *, stream, on_token, timeout_seconds):
        captured["timeout"] = timeout_seconds
        captured["messages"] = messages
        assert stream is True
        for t in tokens:
            await on_token(t)
        return SimpleNamespace(text="The rain fell.")

    seen = []
    loader = _FakeLoader()
    delta = asyncio.run(
        node_draft_prose(_state(), on_token=lambda t: seen.append(t), loader=loader, call=fake_call)
    )

    assert seen == tokens  # every token forwarded to the on_token sink
    assert delta["current_draft_text"] == "The rain fell."
    assert captured["timeout"] == 42  # config timeout threaded to the boundary
    # The beat objective reached the prompt context.
    assert loader.rendered_with[0] == "node_draft_prose"
    assert loader.rendered_with[1]["beat_objective"] == "Open on rain."


def test_draft_falls_back_to_accumulated_tokens_when_response_text_empty():
    async def fake_call(messages, endpoint, *, stream, on_token, timeout_seconds):
        await on_token("a")
        await on_token("b")
        return SimpleNamespace(text="")  # boundary returned no aggregate text

    delta = asyncio.run(node_draft_prose(_state(), loader=_FakeLoader(), call=fake_call))
    assert delta["current_draft_text"] == "ab"
