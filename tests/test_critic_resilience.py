"""The critic must survive a model that cannot follow its schema.

On 2026-07-10 a live run died because `openbmb/minicpm5` answered the continuity
critic with the *schema template itself*: placeholder values, `offarming_text`
for `offending_text`, and no `critic_source`. `FailureObject` forbids extra keys,
so parsing raised and the exception killed a run holding two planned chapters and
a finished 336-word draft.

`PRODUCTION_FAILURE` below is that exact reply, byte for byte. Everything here
exists so it can never take a run down again.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from museai.core.config import EndpointConfig
from museai.core.stream_bus import bus
from museai.fsm.nodes import critics as critics_module
from museai.fsm.nodes.critics import adversarial_critics
from museai.fsm.nodes.deps import set_node_config
from museai.fsm.state import FSM_Pointer, make_initial_state
from museai.fsm.tools.loop import AgentLoopError
from museai.llm.client import LLMCallError
from museai.llm.structured import StructuredOutputError, parse_failure_objects

# The reply that killed the run, recovered verbatim from logs/llm_io.log.
PRODUCTION_FAILURE = (
    '```json\n'
    '[\n'
    '  {\n'
    '    "error_code": "CONTRADICTS_PRIOR_PROSE",\n'
    '    "offarming_text": "The exact sentence or phrase from the draft.",\n'
    '    "suggested_fix": "A concrete rewrite or correction. The text about Chloe '
    'feeling caught between loyalty and understanding is consistent with her character '
    'description and does not contradict established facts or context items."\n'
    '  }\n'
    ']\n'
    '```'
)

CORRECTED = """```json
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

COMMITTED_PROSE = "Mara had already lowered the borrowed ladder."
BORROWED_LADDER_PROSE = (
    "Nell stepped onto the grass with the ladder and set it down among the weeds "
    "near the base of Ida's shed. The aluminium was light in her hands but felt "
    "heavier now. She glanced at the third rung: not much more than a hairline "
    "split."
)
SANITIZED_FIX = (
    "Rewrite the offending text in fresh prose that carries this beat forward. "
    "Do not reuse wording from prose already committed earlier in the manuscript."
)

COMMITTED_PROSE_FINDING = """[
  {
    "error_code": "CONTRADICTS_PRIOR_PROSE",
    "offending_text": "Mara had already lowered the borrowed ladder.",
    "suggested_fix": "Keep the ladder where it was.",
    "critic_source": "continuity_critic"
  }
]"""

MIXED_FINDINGS = """[
  {
    "error_code": "CONTRADICTS_CHARACTER",
    "offending_text": "Mara lied about the letter.",
    "suggested_fix": "Have Mara say nothing.",
    "critic_source": "continuity_critic"
  },
  {
    "error_code": "CONTRADICTS_PRIOR_PROSE",
    "offending_text": "Mara had already lowered the borrowed ladder.",
    "suggested_fix": "Keep the ladder where it was.",
    "critic_source": "continuity_critic"
  }
]"""

DRAFT = "The sun stood at noon. Mara lied about the letter."

PACKAGE = {
    "beat": {
        "id": "arc-1-c01-b01", "ordering": 1,
        "intent": "Mara finds the letter.",
        "entry_state": "A routine morning.",
        "exit_state": "Mara is holding her own handwriting.",
    },
    "pad_constraint": "Energy with nowhere to go.",
    "chapter": {"id": "arc-1-c01", "description": "Mara catalogs the letters.", "obligations": []},
    "threads": [],
    "characters": [
        {"id": "char-mara", "name": "Mara", "description": "She never lies.",
         "pad": {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0}}
    ],
    "recent_prose": [],
    "budget": {"budget": 8000, "tokens": 400, "tokens_before": 400,
               "dropped_prose_passages": 0, "dropped_threads": 0, "over_budget": False},
}


class _Response:
    def __init__(self, text: str, *, finish_reason: str = "stop") -> None:
        self.text = text
        self.tool_calls: list = []
        self.finish_reason = finish_reason


def state_with(**overrides):
    return make_initial_state(
        "test-project",
        FSM_Pointer(arc_id="arc-1", chapter_id="arc-1-c01", beat_index=0),
        active_context_package=PACKAGE,
        current_draft_text=DRAFT,
        **overrides,
    )


@pytest.fixture
def configure(config_factory):
    """Install a node config with explicit retry/degrade budgets."""

    def _configure(**overrides):
        set_node_config(config_factory(**overrides))

    return _configure


@pytest.fixture
def scripted_loop(monkeypatch):
    """Replace `run_agent_loop` with a scripted sequence of critic replies.

    The last reply repeats once the script runs out, so a test can say "it never
    gets it right" without counting attempts.
    """
    calls: list[list[dict]] = []

    def _install(*replies: str):
        async def fake_loop(endpoint, messages, tools, tool_impls, max_iterations, on_event=None, **kwargs):
            calls.append(list(messages))
            index = min(len(calls) - 1, len(replies) - 1)
            return _Response(replies[index])

        monkeypatch.setattr(critics_module, "run_agent_loop", fake_loop)
        return calls

    return _install


@pytest.fixture
def health_events(monkeypatch):
    captured: list[dict] = []
    original = bus.publish

    async def spy(event_type, data):
        if event_type == "critic_health":
            captured.append(data)
        await original(event_type, data)

    monkeypatch.setattr(bus, "publish", spy)
    yield captured
    bus.last_snapshot.clear()


# ------------------------------------------------------- the production payload


def test_the_production_payload_still_fails_strict_parsing():
    """The regression's raw material. If this ever parses, the fixture is stale."""
    with pytest.raises(StructuredOutputError) as excinfo:
        parse_failure_objects(PRODUCTION_FAILURE)
    message = str(excinfo.value)
    assert "offending_text" in message  # the error names the field, for the re-prompt
    assert "offarming_text" in message


def test_lenient_mode_drops_the_production_payload_rather_than_faking_it():
    """`offarming_text` is ignored, which leaves no `offending_text` — so drop it."""
    assert parse_failure_objects(PRODUCTION_FAILURE, lenient=True) == []


async def test_the_production_payload_no_longer_kills_the_run(configure, scripted_loop, health_events):
    configure(critic_parse_retries=2, critic_degrade_threshold=3)
    scripted_loop(PRODUCTION_FAILURE)

    delta = await adversarial_critics(state_with())  # must not raise

    assert delta["critic_parse_failure_streak"] == 1
    assert "critic_failures" not in delta
    assert health_events[-1]["degraded"] is False


async def test_a_truncated_verdict_salvages_complete_findings(
    configure, monkeypatch, health_events, caplog
):
    configure(critic_parse_retries=0, critic_degrade_threshold=1)

    async def truncated(*args, **kwargs):
        return _Response(
            """[
            {"error_code": "CONTRADICTS_PRIOR_PROSE", "offending_text": "The sun stood at noon.", "suggested_fix": "Keep the time of day consistent."},
            {"error_code": "CONTRADICTS_CHARACTER", "offending_text": "Mara lied about the letter.", "suggested_fix": "Have Mara stay silent."},
            {"error_code": "CONTRADICTS_PRIOR_PROSE", "offending_text": "The sun
            """,
            finish_reason="length",
        )

    monkeypatch.setattr(critics_module, "run_agent_loop", truncated)
    with caplog.at_level(logging.INFO, logger="museai"):
        delta = await adversarial_critics(state_with())

    assert [finding.offending_text for finding in delta["critic_failures"]] == [
        "The sun stood at noon.",
        "Mara lied about the letter.",
    ]
    assert delta["critic_parse_failure_streak"] == 0
    assert health_events[-1]["lenient_used"] is True
    salvaged = [
        record for record in caplog.records
        if "event=truncated_verdict_salvaged" in record.getMessage()
    ]
    assert len(salvaged) == 1
    assert "findings=2" in salvaged[0].getMessage()


async def test_a_truncated_verdict_keeps_the_per_code_cap(configure, monkeypatch):
    configure(
        critic_parse_retries=0,
        critic_degrade_threshold=3,
        critic_max_findings_per_code=2,
    )
    same_code = """{
        "error_code": "CONTRADICTS_CHARACTER",
        "offending_text": "Mara lied about the letter.",
        "suggested_fix": "Have Mara stay silent."
    }"""

    async def truncated(*args, **kwargs):
        return _Response(
            "[" + ",".join([same_code, same_code, same_code])
            + ",{\"error_code\": \"CONTRADICTS_CHARACTER\"",
            finish_reason="length",
        )

    monkeypatch.setattr(critics_module, "run_agent_loop", truncated)
    delta = await adversarial_critics(state_with())

    assert len(delta["critic_failures"]) == 2


async def test_a_truncated_verdict_drops_unlocatable_findings(configure, monkeypatch, caplog):
    configure(critic_parse_retries=0, critic_degrade_threshold=3)

    async def truncated(*args, **kwargs):
        return _Response(
            """[
            {"error_code": "CONTRADICTS_PRIOR_PROSE", "offending_text": "A sentence from another beat.", "suggested_fix": "Do not borrow it."},
            {"error_code": "CONTRADICTS_CHARACTER", "offending_text": "Mara lied about the letter.", "suggested_fix": "Have Mara stay silent."},
            {"error_code": "CONTRADICTS_PRIOR_PROSE", "offending_text": "The sun
            """,
            finish_reason="length",
        )

    monkeypatch.setattr(critics_module, "run_agent_loop", truncated)
    with caplog.at_level(logging.WARNING, logger="museai"):
        delta = await adversarial_critics(state_with())

    assert [finding.offending_text for finding in delta["critic_failures"]] == [
        "Mara lied about the letter."
    ]
    assert any("event=finding_discarded" in r.getMessage() for r in caplog.records)


# ----------------------------------------------------------------- re-prompting


async def test_a_corrected_retry_succeeds_and_resets_the_streak(configure, scripted_loop):
    configure(critic_parse_retries=2, critic_degrade_threshold=3)
    calls = scripted_loop(PRODUCTION_FAILURE, CORRECTED)

    delta = await adversarial_critics(state_with(critic_parse_failure_streak=2))

    assert len(calls) == 2
    assert len(delta["critic_failures"]) == 1
    assert delta["critic_failures"][0].offending_text == "Mara lied about the letter."
    assert delta["critic_parse_failure_streak"] == 0


async def test_the_retry_shows_the_model_its_reply_and_the_validation_error(configure, scripted_loop):
    configure(critic_parse_retries=1, critic_degrade_threshold=3)
    calls = scripted_loop(PRODUCTION_FAILURE, CLEAN)

    await adversarial_critics(state_with())

    correction = calls[1]
    assert correction[-2]["role"] == "assistant"
    assert correction[-2]["content"] == PRODUCTION_FAILURE
    assert correction[-1]["role"] == "user"
    assert "offending_text" in correction[-1]["content"]  # the actual pydantic error
    assert "Do not return the example values" in correction[-1]["content"]


async def test_a_truncation_retry_gets_a_short_ask_not_operator_advice(
    configure, monkeypatch
):
    """B3 in the 2026-07-25 postmortem: a truncated reply (the model spent its
    whole grant on reasoning and produced no text) used to be re-prompted with
    an empty assistant turn plus the server-administration paragraph from
    `response_truncation_remedy` — advice about OLLAMA_CONTEXT_LENGTH aimed at
    an operator, not a continuity critic. The retry must instead be short,
    rebuilt from the original prompt rather than grown, and tool-free so it
    can't burn its budget on another tool round-trip."""
    configure(critic_parse_retries=1, critic_degrade_threshold=3)

    calls: list[dict] = []

    async def fake_loop(
        endpoint, messages, tools, tool_impls, max_iterations, on_event=None, **kwargs
    ):
        calls.append({"messages": list(messages), "tools": tools, "tool_impls": tool_impls})
        if len(calls) == 1:
            return _Response("", finish_reason="length")
        return _Response(CORRECTED)

    monkeypatch.setattr(critics_module, "run_agent_loop", fake_loop)

    delta = await adversarial_critics(state_with())

    assert len(calls) == 2
    retry = calls[1]
    assert retry["tools"] == []
    assert retry["tool_impls"] == {}
    assert not any(
        m["role"] == "assistant" and m["content"] == "" for m in retry["messages"]
    )
    joined = " ".join(str(m.get("content", "")) for m in retry["messages"])
    assert "OLLAMA_CONTEXT_LENGTH" not in joined
    assert len(delta["critic_failures"]) == 1


async def test_empty_truncation_retries_and_keeps_its_historic_diagnosis(
    configure, monkeypatch, caplog
):
    configure(critic_parse_retries=1, critic_degrade_threshold=3)
    calls: list[list[dict]] = []

    async def fake_loop(
        endpoint, messages, tools, tool_impls, max_iterations, on_event=None, **kwargs
    ):
        calls.append(list(messages))
        if len(calls) == 1:
            return _Response("", finish_reason="length")
        return _Response(CORRECTED)

    monkeypatch.setattr(critics_module, "run_agent_loop", fake_loop)
    with caplog.at_level(logging.WARNING, logger="museai"):
        delta = await adversarial_critics(state_with())

    assert len(calls) == 2
    retry = calls[1][-1]["content"]
    assert "the whole reply went to reasoning" in retry
    assert any("event=empty_reply" in r.getMessage() for r in caplog.records)
    assert len(delta["critic_failures"]) == 1


async def test_a_truncated_text_retry_asks_for_fewer_shorter_findings(
    configure, monkeypatch
):
    configure(critic_parse_retries=1, critic_degrade_threshold=3)
    calls: list[list[dict]] = []

    async def fake_loop(
        endpoint, messages, tools, tool_impls, max_iterations, on_event=None, **kwargs
    ):
        calls.append(list(messages))
        if len(calls) == 1:
            return _Response("The verdict was cut off before its JSON began", finish_reason="length")
        return _Response(CORRECTED)

    monkeypatch.setattr(critics_module, "run_agent_loop", fake_loop)
    delta = await adversarial_critics(state_with())

    retry = calls[1][-1]["content"]
    assert "ran past the reply budget" in retry
    assert "fewer findings" in retry
    assert "suggested_fix brief" in retry
    assert "or []" not in retry
    assert len(delta["critic_failures"]) == 1


async def test_a_critic_truncation_error_names_its_effective_agent_cap(
    configure, monkeypatch, health_events
):
    configure(critic_parse_retries=0, critic_degrade_threshold=3)

    async def truncated(*args, **kwargs):
        return _Response("A cut-off verdict with no recoverable JSON", finish_reason="length")

    monkeypatch.setattr(critics_module, "run_agent_loop", truncated)
    await adversarial_critics(state_with())

    assert "agents.critic.max_output_tokens" in health_events[-1]["error"]
    assert "endpoint.max_output_tokens" not in health_events[-1]["error"]


async def test_a_prose_clean_verdict_is_read_as_clean_without_retries(
    configure, scripted_loop, caplog
):
    """The critic said the draft is fine, just not in JSON. Burning re-prompts
    and climbing toward degraded mode over that loosens the gate for nothing."""
    configure(critic_parse_retries=2, critic_degrade_threshold=3)
    calls = scripted_loop("No continuity issues found.")

    with caplog.at_level(logging.WARNING, logger="museai.fsm"):
        delta = await adversarial_critics(state_with(critic_parse_failure_streak=2))

    assert len(calls) == 1, "a clean verdict must not be re-prompted"
    assert delta["critic_parse_failure_streak"] == 0
    assert "critic_failures" not in delta  # clean — audit's failures survive
    # The contract break is still visible to the operator.
    assert any("critic_clean_prose" in r.getMessage() for r in caplog.records)


async def test_a_hedged_prose_verdict_still_burns_retries(configure, scripted_loop):
    """"Clean, but…" is not clean. The narrow reading must not widen."""
    configure(critic_parse_retries=1, critic_degrade_threshold=3)
    calls = scripted_loop("Mostly clean, but the ending might contradict chapter 2")

    delta = await adversarial_critics(state_with())

    assert len(calls) == 2  # first attempt plus the one configured retry
    assert delta["critic_parse_failure_streak"] == 1


async def test_retries_are_bounded_by_the_config_key(configure, scripted_loop):
    configure(critic_parse_retries=2, critic_degrade_threshold=99)
    calls = scripted_loop(PRODUCTION_FAILURE)

    await adversarial_critics(state_with())

    assert len(calls) == 3, "one initial attempt plus two retries"


async def test_zero_retries_is_legal(configure, scripted_loop):
    configure(critic_parse_retries=0, critic_degrade_threshold=99)
    calls = scripted_loop(PRODUCTION_FAILURE)

    delta = await adversarial_critics(state_with())

    assert len(calls) == 1
    assert delta["critic_parse_failure_streak"] == 1


async def test_a_clean_critic_never_retries(configure, scripted_loop):
    configure(critic_parse_retries=2, critic_degrade_threshold=3)
    calls = scripted_loop(CLEAN)

    delta = await adversarial_critics(state_with())

    assert len(calls) == 1
    assert delta["critic_parse_failure_streak"] == 0


async def test_a_locatable_finding_is_kept(configure, scripted_loop):
    configure(critic_parse_retries=0, critic_degrade_threshold=3)
    scripted_loop(CORRECTED)

    delta = await adversarial_critics(state_with())

    assert [failure.offending_text for failure in delta["critic_failures"]] == [
        "Mara lied about the letter."
    ]
    assert delta["critic_parse_failure_streak"] == 0


async def test_borrowed_critic_fix_is_sanitized_but_its_finding_survives(
    configure, scripted_loop, caplog
):
    configure(critic_parse_retries=0, critic_degrade_threshold=3)
    scripted_loop(
        json.dumps([
            {
                "error_code": "CONTRADICTS_CHARACTER",
                "offending_text": "Mara lied about the letter.",
                "suggested_fix": BORROWED_LADDER_PROSE,
                "critic_source": "continuity_critic",
            }
        ])
    )
    package = {**PACKAGE, "recent_prose": [BORROWED_LADDER_PROSE]}

    with caplog.at_level(logging.WARNING, logger="museai"):
        state = state_with()
        state["active_context_package"] = package
        delta = await adversarial_critics(state)

    finding = delta["critic_failures"][0]
    assert finding.suggested_fix == SANITIZED_FIX
    assert (finding.error_code, finding.offending_text, finding.critic_source) == (
        "CONTRADICTS_CHARACTER", "Mara lied about the letter.", "continuity_critic"
    )
    sanitized = [r for r in caplog.records if "event=fix_sanitized" in r.getMessage()]
    assert len(sanitized) == 1
    assert "borrowed_words=" in sanitized[0].getMessage()


async def test_short_borrowed_critic_fix_is_untouched(configure, scripted_loop):
    configure(critic_parse_retries=0, critic_degrade_threshold=3)
    suggested_fix = "Check the third rung before she climbs."
    scripted_loop(
        json.dumps([
            {
                "error_code": "CONTRADICTS_CHARACTER",
                "offending_text": "Mara lied about the letter.",
                "suggested_fix": suggested_fix,
                "critic_source": "continuity_critic",
            }
        ])
    )
    package = {**PACKAGE, "recent_prose": [BORROWED_LADDER_PROSE]}

    state = state_with()
    state["active_context_package"] = package
    delta = await adversarial_critics(state)

    assert delta["critic_failures"][0].suggested_fix == suggested_fix


async def test_unlocatable_finding_is_not_sanitized(configure, scripted_loop, caplog):
    configure(critic_parse_retries=0, critic_degrade_threshold=3)
    scripted_loop(
        json.dumps([
            {
                "error_code": "CONTRADICTS_PRIOR_PROSE",
                "offending_text": "This is not in the draft.",
                "suggested_fix": BORROWED_LADDER_PROSE,
                "critic_source": "continuity_critic",
            }
        ])
    )
    package = {**PACKAGE, "recent_prose": [BORROWED_LADDER_PROSE]}

    with caplog.at_level(logging.WARNING, logger="museai"):
        state = state_with()
        state["active_context_package"] = package
        await adversarial_critics(state)

    assert not any("event=fix_sanitized" in r.getMessage() for r in caplog.records)


async def test_a_finding_lifted_from_committed_prose_is_discarded(
    configure, scripted_loop, caplog
):
    configure(critic_parse_retries=0, critic_degrade_threshold=3)
    scripted_loop(COMMITTED_PROSE_FINDING)

    with caplog.at_level(logging.WARNING, logger="museai"):
        delta = await adversarial_critics(state_with())

    assert "critic_failures" not in delta
    assert delta["critic_parse_failure_streak"] == 1
    discarded = [r for r in caplog.records if "event=finding_discarded" in r.getMessage()]
    assert len(discarded) == 1
    assert COMMITTED_PROSE in discarded[0].getMessage()


async def test_all_discarded_findings_reprompt_without_a_clean_verdict(
    configure, scripted_loop
):
    configure(critic_parse_retries=1, critic_degrade_threshold=3)
    calls = scripted_loop(COMMITTED_PROSE_FINDING)

    delta = await adversarial_critics(state_with())

    assert len(calls) == 2
    assert "critic quoted text that is not in the draft" in calls[1][-1]["content"]
    assert "critic_failures" not in delta
    assert delta["critic_parse_failure_streak"] == 1


async def test_an_empty_array_remains_a_clean_verdict(configure, scripted_loop):
    configure(critic_parse_retries=0, critic_degrade_threshold=3)
    scripted_loop(CLEAN)

    delta = await adversarial_critics(state_with())

    assert "critic_failures" not in delta
    assert delta["critic_parse_failure_streak"] == 0


async def test_mixed_findings_keep_only_the_locatable_finding(
    configure, scripted_loop, caplog
):
    configure(critic_parse_retries=0, critic_degrade_threshold=3)
    scripted_loop(MIXED_FINDINGS)

    with caplog.at_level(logging.WARNING, logger="museai"):
        delta = await adversarial_critics(state_with())

    assert [failure.offending_text for failure in delta["critic_failures"]] == [
        "Mara lied about the letter."
    ]
    assert sum("event=finding_discarded" in r.getMessage() for r in caplog.records) == 1


# ---------------------------------------------------------------- the streak


async def test_a_call_failure_does_not_kill_the_run(configure, monkeypatch):
    """`run_agent_loop` raising `LLMCallError` (an exhausted empty-completion
    retry budget) must be survived exactly like an unreadable verdict, not
    propagate and end a multi-hour generation."""
    configure(critic_parse_retries=2, critic_degrade_threshold=3)

    async def always_fails(*args, **kwargs):
        raise LLMCallError("endpoint returned empty completions repeatedly")

    monkeypatch.setattr(critics_module, "run_agent_loop", always_fails)

    delta = await adversarial_critics(state_with())  # must not raise

    assert delta["critic_parse_failure_streak"] == 1
    assert "critic_failures" not in delta


async def test_a_call_failure_is_retried_then_can_still_succeed(configure, monkeypatch):
    configure(critic_parse_retries=1, critic_degrade_threshold=3)
    attempts = 0

    async def fails_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise LLMCallError("transient empty completion")
        return _Response(CLEAN)

    monkeypatch.setattr(critics_module, "run_agent_loop", fails_once)

    delta = await adversarial_critics(state_with())

    assert attempts == 2
    assert delta["critic_parse_failure_streak"] == 0
    assert "critic_failures" not in delta


async def test_a_stuck_tool_loop_does_not_kill_the_run(configure, monkeypatch):
    """`run_agent_loop` raises `AgentLoopError` when an endpoint keeps emitting
    tool calls after the schemas were withheld — the stuck-local-model case the
    strike-out reaches sooner than the iteration cap did. It must degrade like
    any other unreadable verdict, not end the generation."""
    configure(critic_parse_retries=0, critic_degrade_threshold=3)

    async def always_stuck(*args, **kwargs):
        raise AgentLoopError(
            "endpoint returned tool calls after tools were withheld at the loop limit"
        )

    monkeypatch.setattr(critics_module, "run_agent_loop", always_stuck)

    delta = await adversarial_critics(state_with())  # must not raise

    assert delta["critic_parse_failure_streak"] == 1
    assert "critic_failures" not in delta


async def test_a_retry_that_would_overflow_the_window_drops_the_carried_context(
    configure, monkeypatch, config_factory
):
    """Carrying tool results forward is an optimisation, not a contract. When the
    accumulated conversation no longer leaves the output reservation free, the
    retry restarts from the trimmed prompt rather than sending a request that
    will come back truncated."""
    # A window barely wider than the prompt: one carried tool result overflows it.
    set_node_config(
        config_factory(
            critic_parse_retries=1,
            critic_degrade_threshold=3,
            endpoint=EndpointConfig(
                base_url="http://127.0.0.1:1234/v1",
                api_key="test-key",
                model_name="test-model",
                tokenizer_family="char_heuristic",
                context_window=2048,
                output_reservation=512,
            ),
        )
    )
    calls: list[list[dict]] = []

    async def fake_loop(endpoint, messages, tools, tool_impls, max_iterations,
                        on_event=None, conversation_out=None, **kwargs):
        calls.append(list(messages))
        if len(calls) == 1 and conversation_out is not None:
            conversation_out[:] = [
                *messages,
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "web_search", "arguments": "{}"}},
                ]},
                {"role": "tool", "tool_call_id": "c1", "name": "web_search",
                 "content": "x" * 4000},
            ]
        return _Response(PRODUCTION_FAILURE if len(calls) == 1 else CLEAN)

    monkeypatch.setattr(critics_module, "run_agent_loop", fake_loop)

    await adversarial_critics(state_with())

    assert len(calls) == 2
    assert not any("x" * 4000 == m.get("content") for m in calls[1])
    # The correction itself is never dropped — that is the whole point of the retry.
    assert calls[1][-1]["role"] == "user"
    assert "Do not return the example values" in calls[1][-1]["content"]


async def test_a_retry_carries_forward_the_prior_tool_results(configure, monkeypatch):
    """A parse retry must not discard tool results the failed attempt already
    paid for: it should build on `conversation_out`, not restart from the
    original 2-message prompt."""
    configure(critic_parse_retries=1, critic_degrade_threshold=3)
    calls: list[list[dict]] = []

    async def fake_loop(endpoint, messages, tools, tool_impls, max_iterations,
                         on_event=None, conversation_out=None, **kwargs):
        calls.append(list(messages))
        if len(calls) == 1:
            if conversation_out is not None:
                conversation_out[:] = [
                    *messages,
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "c1", "type": "function",
                         "function": {"name": "web_search", "arguments": "{}"}},
                    ]},
                    {"role": "tool", "tool_call_id": "c1", "name": "web_search",
                     "content": "search result payload"},
                ]
            return _Response(PRODUCTION_FAILURE)
        return _Response(CLEAN)

    monkeypatch.setattr(critics_module, "run_agent_loop", fake_loop)

    await adversarial_critics(state_with())

    assert len(calls) == 2
    # The retry's prompt carries the tool result from the first attempt.
    assert any(m.get("content") == "search result payload" for m in calls[1])
    # ...and still ends with the model's bad reply plus the correction, on top
    # of that carried-forward conversation rather than the original messages.
    assert calls[1][-2]["role"] == "assistant"
    assert calls[1][-2]["content"] == PRODUCTION_FAILURE
    assert calls[1][-1]["role"] == "user"


# ------------------------------------------- no attempt repeats a known answer


async def _truncating_loop(monkeypatch, *, answers_on=None):
    """Install a loop that truncates every attempt, recording each prompt.

    ``answers_on`` optionally names a 1-based attempt that answers cleanly
    instead, so a test can show a retry still being *taken*.
    """
    calls: list[list[dict]] = []

    async def fake_loop(endpoint, messages, tools, tool_impls, max_iterations,
                        on_event=None, conversation_out=None, **kwargs):
        calls.append([dict(m) for m in messages])
        if answers_on is not None and len(calls) == answers_on:
            return _Response(CLEAN)
        return _Response("", finish_reason="length")

    monkeypatch.setattr(critics_module, "run_agent_loop", fake_loop)
    return calls


async def test_no_two_critic_attempts_send_the_same_prompt(configure, monkeypatch):
    """The regression that motivated all of this.

    Every truncation rebuilt `conversation` from the same two constants, so the
    second re-ask reproduced the first byte for byte. In the 2026-07-25 live run
    one prompt went out six times in a row, ~62 s each, and returned zero
    characters every time. Counting attempts never caught it — only comparing
    their content does."""
    configure(critic_parse_retries=2, critic_degrade_threshold=99)
    calls = await _truncating_loop(monkeypatch)

    await adversarial_critics(state_with())

    assert len(calls) == 3, "one attempt plus two retries"
    rendered = [
        tuple((m["role"], m.get("content")) for m in call) for call in calls
    ]
    assert len(set(rendered)) == len(rendered), "an attempt repeated a prompt verbatim"


async def test_the_second_truncation_escalates_the_ask(configure, monkeypatch):
    """Distinct is not enough on its own — the retry has to be asking for
    something, and each truncation should narrow the demand rather than restate
    it. It must still never offer "[]" as a way out of deliberating: laundering
    an unreadable verdict into a clean pass is what the streak exists to stop."""
    configure(critic_parse_retries=2, critic_degrade_threshold=99)
    calls = await _truncating_loop(monkeypatch)

    await adversarial_critics(state_with())

    first, second = calls[1][-1]["content"], calls[2][-1]["content"]
    assert first != second
    assert "Skip the reasoning" in first
    assert "Do not deliberate at all" in second
    assert "partial array" in second


async def test_a_retry_is_still_taken_after_a_truncation(configure, monkeypatch):
    """The guard must not turn into "never retry a truncation". A fresh sample
    answered roughly a third of the time in the live run; the point is to stop
    re-asking an *identical* question, not to stop asking."""
    configure(critic_parse_retries=2, critic_degrade_threshold=99)
    calls = await _truncating_loop(monkeypatch, answers_on=2)

    delta = await adversarial_critics(state_with())

    assert len(calls) == 2
    assert delta["critic_parse_failure_streak"] == 0


async def test_an_exhausted_wording_budget_stops_instead_of_repeating(
    configure, monkeypatch, caplog
):
    """With more retries configured than there are ways to phrase the re-ask,
    the wording runs out. Rather than put a question the endpoint has already
    answered back on the wire, the pass stops and reports itself unreadable —
    exactly as an exhausted retry budget does."""
    configure(critic_parse_retries=5, critic_degrade_threshold=99)
    calls = await _truncating_loop(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="museai.fsm"):
        delta = await adversarial_critics(state_with())

    assert len(calls) == 3, "stopped once the wording had nowhere left to go"
    assert any("retry_skipped_identical" in r.getMessage() for r in caplog.records)
    # Stopping early is not a clean verdict: the streak still climbs.
    assert delta["critic_parse_failure_streak"] == 1


# ------------------------------------------- evidence carried across passes


def _evidence_conversation(messages):
    return [
        *messages,
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "find_repetition", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "name": "find_repetition",
         "content": "no repetition found for the kettle passage"},
    ]


async def test_an_unreadable_pass_hands_its_tool_results_to_the_next_one(
    configure, monkeypatch
):
    """`retry_critic` routes back into this node with the draft untouched, so
    everything the failed pass looked up is still true. It used to be thrown
    away: one live beat re-searched the same kettle/door prose across four
    passes because each one restarted from the bare two-message prompt."""
    configure(critic_parse_retries=0, critic_degrade_threshold=99)

    async def fake_loop(endpoint, messages, tools, tool_impls, max_iterations,
                        on_event=None, conversation_out=None, **kwargs):
        if conversation_out is not None:
            conversation_out[:] = _evidence_conversation(messages)
        return _Response("", finish_reason="length")

    monkeypatch.setattr(critics_module, "run_agent_loop", fake_loop)

    delta = await adversarial_critics(state_with())

    carried = delta["critic_evidence"]
    assert carried is not None
    assert any(m.get("role") == "tool" for m in carried["conversation"])


async def test_the_next_pass_starts_from_the_carried_evidence(configure, monkeypatch):
    configure(critic_parse_retries=0, critic_degrade_threshold=99)
    calls: list[list[dict]] = []

    async def fake_loop(endpoint, messages, tools, tool_impls, max_iterations,
                        on_event=None, conversation_out=None, **kwargs):
        calls.append([dict(m) for m in messages])
        return _Response(CLEAN)

    monkeypatch.setattr(critics_module, "run_agent_loop", fake_loop)

    prior = {
        "draft_fingerprint": critics_module._fingerprint(DRAFT),
        "conversation": _evidence_conversation([{"role": "user", "content": "seed"}]),
    }
    await adversarial_critics(state_with(critic_evidence=prior))

    assert any(
        m.get("content") == "no repetition found for the kettle passage"
        for m in calls[0]
    ), "the pass re-derived what the last one had already looked up"


async def test_a_revised_draft_invalidates_the_carried_evidence(configure, monkeypatch):
    """The fingerprint is the safety catch. Evidence gathered against the draft
    revise just replaced describes prose that no longer exists, and reusing it
    would have the critic judging the new draft on the old one's lookups."""
    configure(critic_parse_retries=0, critic_degrade_threshold=99)
    calls: list[list[dict]] = []

    async def fake_loop(endpoint, messages, tools, tool_impls, max_iterations,
                        on_event=None, conversation_out=None, **kwargs):
        calls.append([dict(m) for m in messages])
        return _Response(CLEAN)

    monkeypatch.setattr(critics_module, "run_agent_loop", fake_loop)

    stale = {
        "draft_fingerprint": critics_module._fingerprint("a completely different draft"),
        "conversation": _evidence_conversation([{"role": "user", "content": "seed"}]),
    }
    await adversarial_critics(state_with(critic_evidence=stale))

    assert not any(
        m.get("content") == "no repetition found for the kettle passage"
        for m in calls[0]
    ), "stale evidence survived a draft change"


async def test_a_readable_verdict_carries_no_evidence_forward(configure, scripted_loop):
    """Only `retry_critic` comes back to this node with the same draft. A
    readable verdict routes on to commit, revise, or review, so holding the
    transcript would drag it through the rest of the run for nothing."""
    configure(critic_parse_retries=2, critic_degrade_threshold=3)
    scripted_loop(CLEAN)

    delta = await adversarial_critics(state_with())

    assert delta["critic_evidence"] is None


async def test_the_streak_accumulates_across_beats(configure, scripted_loop, health_events):
    """The streak spans beats: one bad beat is noise, three is a broken critic."""
    configure(critic_parse_retries=0, critic_degrade_threshold=3)
    scripted_loop(PRODUCTION_FAILURE)

    streak = 0
    for expected in (1, 2, 3):
        delta = await adversarial_critics(state_with(critic_parse_failure_streak=streak))
        streak = delta["critic_parse_failure_streak"]
        assert streak == expected

    assert [event["degraded"] for event in health_events] == [False, False, True]


async def test_one_readable_reply_clears_the_streak(configure, scripted_loop, health_events):
    configure(critic_parse_retries=0, critic_degrade_threshold=3)
    scripted_loop(CLEAN)

    delta = await adversarial_critics(state_with(critic_parse_failure_streak=5))

    assert delta["critic_parse_failure_streak"] == 0
    assert health_events[-1]["degraded"] is False


async def test_a_degraded_run_still_tries_strict_first(configure, scripted_loop, health_events):
    """Recovery must be possible: a well-formed reply clears a degraded run."""
    configure(critic_parse_retries=0, critic_degrade_threshold=2)
    scripted_loop(CORRECTED)

    delta = await adversarial_critics(state_with(critic_parse_failure_streak=7))

    assert delta["critic_parse_failure_streak"] == 0
    assert health_events[-1]["degraded"] is False
    assert len(delta["critic_failures"]) == 1


# ------------------------------------------------------------------- degrading


async def test_crossing_the_threshold_degrades_and_reports(configure, scripted_loop, health_events):
    configure(critic_parse_retries=0, critic_degrade_threshold=2)
    scripted_loop(PRODUCTION_FAILURE)

    delta = await adversarial_critics(state_with(critic_parse_failure_streak=1))

    assert delta["critic_parse_failure_streak"] == 2
    event = health_events[-1]
    assert event["degraded"] is True
    assert event["streak"] == 2
    assert event["threshold"] == 2
    # The production payload salvages to nothing, so lenient parsing found no findings.
    assert event["lenient_used"] is False
    assert "critic_failures" not in delta


async def test_degraded_mode_salvages_what_it_can(configure, scripted_loop, health_events):
    """An unknown key is ignored; the readable element survives."""
    salvageable = """```json
    [
      {
        "error_code": "CONTRADICTS_CHARACTER",
        "offending_text": "Mara lied about the letter.",
        "suggested_fix": "Mara never lies.",
        "severity": "high"
      }
    ]
    ```"""
    configure(critic_parse_retries=0, critic_degrade_threshold=1)
    scripted_loop(salvageable)

    delta = await adversarial_critics(state_with())

    assert health_events[-1]["degraded"] is True
    assert health_events[-1]["lenient_used"] is True
    assert len(delta["critic_failures"]) == 1
    assert delta["critic_failures"][0].critic_source == "continuity_critic"


async def test_degraded_mode_never_invents_an_empty_offending_text(configure, scripted_loop):
    """An empty needle makes `revise.locate` return None and rewrite the whole draft."""
    configure(critic_parse_retries=0, critic_degrade_threshold=1)
    scripted_loop(PRODUCTION_FAILURE)

    delta = await adversarial_critics(state_with())

    assert "critic_failures" not in delta


async def test_degraded_mode_does_not_salvage_unlocatable_findings(
    configure, scripted_loop, health_events
):
    """Findings absent from the draft remain unreadable at the degrade threshold."""
    configure(critic_parse_retries=0, critic_degrade_threshold=1)
    scripted_loop(COMMITTED_PROSE_FINDING)

    delta = await adversarial_critics(state_with())

    assert health_events[-1]["degraded"] is True
    assert health_events[-1]["lenient_used"] is False
    assert "critic_failures" not in delta


async def test_unparseable_prose_is_not_rescued_by_degrading(configure, scripted_loop, health_events):
    configure(critic_parse_retries=0, critic_degrade_threshold=1)
    scripted_loop("Honestly, the draft reads fine to me.")

    delta = await adversarial_critics(state_with())

    assert health_events[-1]["degraded"] is True
    assert health_events[-1]["lenient_used"] is False
    assert "critic_failures" not in delta


# ------------------------------------------------------- the no-progress baseline


async def test_progress_is_measured_against_the_draft_revise_was_handed(
    configure, scripted_loop
):
    """`last_cycle_improved` answers "did the last revise help", so its baseline
    is the count revise was handed — not `best_seen_failure_count`, which this
    same pass updates."""
    configure(critic_parse_retries=0, critic_degrade_threshold=3)
    scripted_loop(CORRECTED)  # one finding

    delta = await adversarial_critics(
        state_with(retry_count=1, pre_revise_failure_count=3, best_seen_failure_count=3)
    )

    assert delta["last_cycle_improved"] is True
    assert delta["best_seen_failure_count"] == 1


async def test_rescoring_the_same_draft_is_not_a_failed_revise_cycle(
    configure, monkeypatch
):
    """`retry_critic` routes back into this node with no revise in between. A
    baseline that moved on every pass would read the second reading of unchanged
    prose as a regression and send the beat to review with its whole revision
    budget untouched."""
    configure(critic_parse_retries=0, critic_degrade_threshold=3)
    replies = iter([PRODUCTION_FAILURE, CORRECTED])

    async def fake_loop(*args, **kwargs):
        return _Response(next(replies))

    monkeypatch.setattr(critics_module, "run_agent_loop", fake_loop)

    # The revise that produced this draft was handed 5 failures.
    state = state_with(retry_count=1, pre_revise_failure_count=5)

    # Pass A is unreadable: no findings recovered, so it scores 0 and takes over
    # `best_seen_failure_count`.
    first = await adversarial_critics(state)
    assert first["best_seen_failure_count"] == 0
    assert first["last_cycle_improved"] is True

    # Pass B re-reads the *same* draft, this time successfully, and finds one
    # real problem. Against `best_seen_failure_count` (now 0) that looks like a
    # regression; against the 5 the revise was handed it is plain progress.
    second = await adversarial_critics(
        state_with(
            retry_count=1,
            pre_revise_failure_count=5,
            best_seen_failure_count=first["best_seen_failure_count"],
            critic_parse_failure_streak=first["critic_parse_failure_streak"],
        )
    )
    assert len(second["critic_failures"]) == 1
    assert second["last_cycle_improved"] is True


async def test_a_revise_that_did_not_lower_the_count_reports_no_progress(
    configure, scripted_loop
):
    configure(critic_parse_retries=0, critic_degrade_threshold=3)
    scripted_loop(CORRECTED)  # one finding, same as before the revise

    delta = await adversarial_critics(
        state_with(retry_count=1, pre_revise_failure_count=1)
    )

    assert delta["last_cycle_improved"] is False


async def test_the_first_cycle_has_no_baseline_to_fail_against(configure, scripted_loop):
    configure(critic_parse_retries=0, critic_degrade_threshold=3)
    scripted_loop(CORRECTED)

    delta = await adversarial_critics(state_with())

    assert delta["last_cycle_improved"] is True


# ------------------------------------------------------------ audit interaction


async def test_programmatic_audit_failures_survive_a_broken_critic(configure, scripted_loop):
    """A dead critic must not launder away what `audit` already found."""
    from museai.fsm.state import FailureObject

    audit_failure = FailureObject(
        error_code="PASSIVE_VOICE_DENSITY",
        offending_text="The door was opened by Mara.",
        suggested_fix="Rewrite the passive clauses.",
        critic_source="programmatic_audit",
    )
    configure(critic_parse_retries=0, critic_degrade_threshold=3)
    scripted_loop(PRODUCTION_FAILURE)

    delta = await adversarial_critics(state_with(critic_failures=[audit_failure]))

    # The node omits `critic_failures` when it found none, so the reducer keeps audit's.
    assert "critic_failures" not in delta
    assert delta["best_seen_failure_count"] == 1


# ------------------------------------------------------ crash salvage (db side)


def test_reset_active_beats_only_touches_prose_less_active_beats(config_factory):
    """A beat with prose is committed or awaiting review. Neither is ours to undo."""
    from museai.memory.db import connect_db, init_db, reset_active_beats

    from conftest import add_beat, seed_project

    config = config_factory()
    init_db(config.db_path)
    seed_project(config)
    add_beat(config, beat_id="b-active", ordering=1, status="active", prose=None)
    add_beat(config, beat_id="b-done", ordering=2, status="completed", prose="Committed.", word_count=1)
    add_beat(config, beat_id="b-review", ordering=3, status="active", prose="Awaiting review.", word_count=2)
    add_beat(config, beat_id="b-planned", ordering=4, status="planned", prose=None)

    conn = connect_db(config.db_path)
    try:
        with conn:
            reset = reset_active_beats(conn, config.project_id)
        rows = {r["id"]: r["status"] for r in conn.execute("SELECT id, status FROM Beats")}
    finally:
        conn.close()

    assert reset == 1
    assert rows["b-active"] == "planned"
    assert rows["b-done"] == "completed"
    assert rows["b-review"] == "active"
    assert rows["b-planned"] == "planned"


def test_reset_active_beats_is_idempotent(config_factory):
    from museai.memory.db import connect_db, init_db, reset_active_beats

    from conftest import add_beat, seed_project

    config = config_factory()
    init_db(config.db_path)
    seed_project(config)
    add_beat(config, beat_id="b-active", ordering=1, status="active", prose=None)

    conn = connect_db(config.db_path)
    try:
        with conn:
            assert reset_active_beats(conn, config.project_id) == 1
        with conn:
            assert reset_active_beats(conn, config.project_id) == 0
    finally:
        conn.close()


# --------------------------------------------------- crash salvage (manager side)


def test_a_failed_run_writes_its_draft_and_frees_the_beat(config_factory):
    from museai.fsm.manager import GenerationManager
    from museai.memory.db import connect_db, init_db

    from conftest import add_beat, seed_project

    config = config_factory()  # config.draft_dir already points at a tmp_path
    init_db(config.db_path)
    seed_project(config)
    add_beat(config, beat_id="arc-1-c01-b01", ordering=1, status="active", prose=None)

    manager = GenerationManager(config)
    manager.state = make_initial_state(
        config.project_id,
        FSM_Pointer(arc_id="arc-1", chapter_id="arc-1-c01", beat_index=0),
        active_context_package=PACKAGE,
        current_draft_text="The lamp held against the fog.",
    )

    draft_path = manager._salvage_failed_run()

    assert draft_path is not None
    written = Path(draft_path).read_text(encoding="utf-8")
    assert written == "The lamp held against the fog."

    conn = connect_db(config.db_path)
    try:
        status = conn.execute("SELECT status FROM Beats WHERE id='arc-1-c01-b01'").fetchone()[0]
    finally:
        conn.close()
    assert status == "planned"


def test_salvage_prefers_the_best_seen_draft(config_factory):
    from museai.fsm.manager import GenerationManager
    from museai.memory.db import init_db

    from conftest import seed_project

    config = config_factory()
    init_db(config.db_path)
    seed_project(config)

    manager = GenerationManager(config)
    manager.state = make_initial_state(
        config.project_id,
        FSM_Pointer(arc_id="arc-1", chapter_id="arc-1-c01", beat_index=0),
        active_context_package=PACKAGE,
        current_draft_text="A worse later draft.",
        best_seen_draft="The best draft seen.",
    )

    draft_path = manager._salvage_failed_run()
    assert Path(draft_path).read_text(encoding="utf-8") == "The best draft seen."


def test_salvage_writes_nothing_when_there_is_no_draft(config_factory):
    from museai.fsm.manager import GenerationManager
    from museai.memory.db import init_db

    from conftest import seed_project

    config = config_factory()
    init_db(config.db_path)
    seed_project(config)

    manager = GenerationManager(config)
    manager.state = make_initial_state(
        config.project_id,
        FSM_Pointer(arc_id="arc-1", chapter_id="arc-1-c01", beat_index=0),
        active_context_package=PACKAGE,
    )

    assert manager._salvage_failed_run() is None
    assert not Path(config.draft_dir).exists()


def test_salvage_never_masks_the_original_failure(config_factory, monkeypatch):
    """A fault inside the handler must not replace the real exception's diagnosis."""
    from museai.fsm import manager as manager_module
    from museai.fsm.manager import GenerationManager
    from museai.memory.db import init_db

    from conftest import seed_project

    config = config_factory()
    init_db(config.db_path)
    seed_project(config)

    def boom(*args, **kwargs):
        raise OSError("disk is on fire")

    monkeypatch.setattr(manager_module, "reset_active_beats", boom)

    manager = GenerationManager(config)
    manager.state = make_initial_state(
        config.project_id,
        FSM_Pointer(arc_id="arc-1", chapter_id="arc-1-c01", beat_index=0),
        active_context_package=PACKAGE,
        current_draft_text="Salvage me.",
    )

    draft_path = manager._salvage_failed_run()  # must not raise
    assert Path(draft_path).read_text(encoding="utf-8") == "Salvage me."


async def test_a_healthy_critic_does_not_warn(configure, scripted_loop, caplog):
    """Health fires once per beat. Warning on every clean beat trains you to skim."""
    configure(critic_parse_retries=0, critic_degrade_threshold=2)
    scripted_loop("[]")

    with caplog.at_level(logging.DEBUG, logger="museai"):
        await adversarial_critics(state_with(critic_parse_failure_streak=0))

    health = [r for r in caplog.records if "event=health" in r.getMessage()]
    assert len(health) == 1
    assert health[0].levelno == logging.INFO


async def test_a_degraded_critic_warns(configure, scripted_loop, caplog):
    configure(critic_parse_retries=0, critic_degrade_threshold=2)
    scripted_loop(PRODUCTION_FAILURE)

    with caplog.at_level(logging.DEBUG, logger="museai"):
        await adversarial_critics(state_with(critic_parse_failure_streak=1))

    health = [r for r in caplog.records if "event=health" in r.getMessage()]
    assert len(health) == 1
    assert health[0].levelno == logging.WARNING

    failed = [r for r in caplog.records if "event=parse_failed" in r.getMessage()]
    assert failed and all(r.levelno == logging.WARNING for r in failed)
