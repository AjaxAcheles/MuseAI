"""Per-agent inference configuration.

Each agent role may override endpoint fields through ``agents:`` in config;
anything unset inherits from the shared ``endpoint``. These tests prove the
merge, the fallback, boot-time validation of role names, and that the nodes
actually route their calls through their own role's endpoint.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from museai.core.config import AGENT_ROLES, EndpointConfig
from museai.fsm.nodes import critics as critics_module
from museai.fsm.nodes.critics import adversarial_critics
from museai.fsm.nodes.deps import set_node_config
from museai.fsm.nodes.draft_prose import draft_prose
from museai.fsm.state import FSM_Pointer, make_initial_state
from museai.fsm.tools import loop as loop_module

PACKAGE = {
    "beat": {
        "id": "arc-1-c01-b01",
        "ordering": 1,
        "intent": "Mara finds the letter.",
        "entry_state": "A routine morning.",
        "exit_state": "Mara is holding her own handwriting.",
        "focal_character_id": "char-mara",
    },
    "pad_constraint": "Energy with nowhere to go.",
    "chapter": {"id": "arc-1-c01", "description": "Mara catalogs the letters.",
                "obligations": []},
    "threads": [],
    "characters": [],
    "recent_prose": [],
    "budget": {"budget": 8000, "tokens_before": 400, "tokens": 400,
               "dropped_prose_passages": 0, "dropped_threads": 0,
               "over_budget": False},
}


def _state():
    return make_initial_state(
        "test-project",
        FSM_Pointer(arc_id="arc-1", chapter_id="arc-1-c01", beat_index=0),
        active_context_package=PACKAGE,
        current_draft_text="A draft.",
    )


class TestEndpointFor:
    def test_every_role_falls_back_to_the_shared_endpoint(self, config_factory):
        """A config with no agents section behaves exactly as before."""
        config = config_factory()
        for role in AGENT_ROLES:
            assert config.endpoint_for(role) is config.endpoint

    def test_sparse_overrides_merge_onto_the_shared_endpoint(self, config_factory):
        config = config_factory(
            agents={"critic": {"model_name": "critic-model", "temperature": 0.2}}
        )
        critic = config.endpoint_for("critic")
        assert critic.model_name == "critic-model"
        assert critic.temperature == 0.2
        # Everything unset is inherited, not blanked.
        assert critic.base_url == config.endpoint.base_url
        assert critic.api_key == config.endpoint.api_key
        assert critic.tokenizer_family == config.endpoint.tokenizer_family
        # Roles without an override keep the shared endpoint.
        assert config.endpoint_for("drafter") is config.endpoint

    def test_reasoning_effort_override_wins_and_other_agents_inherit(self, config_factory):
        config = config_factory(
            endpoint=config_factory().endpoint.model_copy(
                update={"reasoning_effort": "medium"}
            ),
            agents={"critic": {"reasoning_effort": "none"}},
        )
        assert config.endpoint_for("critic").reasoning_effort == "none"
        assert config.endpoint_for("drafter").reasoning_effort == "medium"

    def test_invalid_endpoint_reasoning_effort_is_fatal_at_boot(self):
        with pytest.raises(ValidationError, match="off"):
            EndpointConfig(
                base_url="https://example.invalid/v1",
                api_key="k",
                model_name="m",
                tokenizer_family="char_heuristic",
                reasoning_effort="off",
            )

    def test_invalid_agent_reasoning_effort_is_fatal_at_boot(self, config_factory):
        with pytest.raises(ValidationError, match="off"):
            config_factory(agents={"critic": {"reasoning_effort": "off"}})

    def test_an_unknown_agent_role_is_fatal_at_boot(self, config_factory):
        with pytest.raises(ValidationError, match="proofreader"):
            config_factory(agents={"proofreader": {"temperature": 0.1}})

    def test_an_unknown_override_field_is_fatal_at_boot(self, config_factory):
        with pytest.raises(ValidationError):
            config_factory(agents={"drafter": {"max_tokens": 100}})


class TestNodeRouting:
    async def test_the_drafter_calls_its_own_endpoint(self, config_factory, monkeypatch):
        config = config_factory(agents={"drafter": {"model_name": "drafter-model"}})
        set_node_config(config)
        captured = {}

        async def fake_call_llm(endpoint, messages, *, stream=False, on_token=None, **kwargs):
            captured["endpoint"] = endpoint
            await on_token("prose")

            class _R:
                text = "prose"
                tokens_out = 1
                finish_reason = "stop"
                tool_calls: list = []

            return _R()

        monkeypatch.setattr(loop_module, "call_llm", fake_call_llm)

        await draft_prose(_state())
        assert captured["endpoint"].model_name == "drafter-model"

    async def test_the_critic_calls_its_own_endpoint(self, config_factory, monkeypatch):
        config = config_factory(agents={"critic": {"model_name": "critic-model"}})
        set_node_config(config)
        captured = {}

        async def fake_loop(endpoint, messages, tools, tool_impls, max_iterations,
                            on_event=None, **kwargs):
            captured["endpoint"] = endpoint

            class _R:
                text = "[]"
                finish_reason = "stop"
                tool_calls: list = []

            return _R()

        monkeypatch.setattr(critics_module, "run_agent_loop", fake_loop)

        await adversarial_critics(_state())
        assert captured["endpoint"].model_name == "critic-model"
        # The drafter would still have used the shared endpoint.
        assert config.endpoint_for("drafter") is config.endpoint
