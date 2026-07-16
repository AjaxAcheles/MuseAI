"""Tests for museai.fsm.tools.loop.

``call_llm`` is replaced in the loop's namespace with a scripted fake that
records the ``tools`` argument of every call, so the two properties that matter
are directly observable: the loop terminates, and the forced final call is made
without tools.
"""

from __future__ import annotations

import asyncio

import pytest

from museai.fsm.tools import loop as loop_module
from museai.fsm.tools.loop import AgentLoopError, run_agent_loop
from museai.llm.client import LLMResponse

ENDPOINT = object()
MESSAGES = [{"role": "user", "content": "When did the Perseids peak?"}]
TOOLS = [{"type": "function", "function": {"name": "web_search"}}]


def response(text: str = "", tool_calls: list[dict] | None = None) -> LLMResponse:
    return LLMResponse(
        text=text,
        raw={},
        model_name="test-model",
        endpoint_base_url="http://endpoint.invalid/v1",
        tokens_in=1,
        tokens_out=1,
        tool_calls=tool_calls or [],
        finish_reason="tool_calls" if tool_calls else "stop",
    )


def tool_call(name: str, arguments: str, call_id: str = "call-1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class ScriptedLLM:
    """Returns the next scripted response, recording each call's arguments."""

    def __init__(self, *responses: LLMResponse) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def __call__(self, endpoint, messages, **kwargs):
        self.calls.append({"messages": [dict(m) for m in messages], **kwargs})
        if not self._responses:
            raise AssertionError("the loop made more calls than the script allows")
        return self._responses.pop(0)

    @property
    def tools_per_call(self) -> list[object]:
        return [call.get("tools") for call in self.calls]


@pytest.fixture
def patched(monkeypatch):
    def _install(*responses: LLMResponse) -> ScriptedLLM:
        fake = ScriptedLLM(*responses)
        monkeypatch.setattr(loop_module, "call_llm", fake)
        return fake

    return _install


class TestToolExecution:
    async def test_a_tool_call_runs_the_impl_then_the_loop_continues(self, patched):
        fake = patched(
            response(tool_calls=[tool_call("web_search", '{"query": "perseids"}')]),
            response(text="They peak on August 12."),
        )
        seen: list[dict] = []

        def web_search(query, max_results=5):
            seen.append({"query": query, "max_results": max_results})
            return [{"title": "t", "url": "https://e.org/p", "snippet": "August 12"}]

        result = await run_agent_loop(
            ENDPOINT, MESSAGES, TOOLS, {"web_search": web_search}, max_iterations=4
        )

        assert result.text == "They peak on August 12."
        assert result.tool_calls == []
        assert seen == [{"query": "perseids", "max_results": 5}]
        assert len(fake.calls) == 2

        # The second call carries the assistant tool-call turn and its result.
        messages = fake.calls[1]["messages"]
        assert messages[0] == MESSAGES[0]
        assert messages[1]["role"] == "assistant"
        assert messages[1]["tool_calls"][0]["function"]["name"] == "web_search"
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "call-1"
        assert "August 12" in messages[2]["content"]

    async def test_on_token_reaches_every_model_turn(self, patched):
        """The drafter streams through the loop: on_token must pass through."""
        fake = patched(
            response(tool_calls=[tool_call("web_search", "{}")]),
            response(text="Prose after the lookup."),
        )

        async def on_token(token: str) -> None:  # pragma: no cover - identity only
            pass

        await run_agent_loop(
            ENDPOINT,
            MESSAGES,
            TOOLS,
            {"web_search": lambda **k: []},
            max_iterations=4,
            on_token=on_token,
        )

        assert [call.get("on_token") for call in fake.calls] == [on_token, on_token]

    async def test_a_response_without_tool_calls_returns_immediately(self, patched):
        fake = patched(response(text="No search needed."))

        result = await run_agent_loop(
            ENDPOINT, MESSAGES, TOOLS, {"web_search": lambda **k: []}, max_iterations=4
        )

        assert result.text == "No search needed."
        assert len(fake.calls) == 1

    async def test_an_async_tool_is_awaited(self, patched):
        patched(
            response(tool_calls=[tool_call("fetch", "{}")]),
            response(text="done"),
        )

        async def fetch():
            return {"ok": True}

        result = await run_agent_loop(
            ENDPOINT, MESSAGES, TOOLS, {"fetch": fetch}, max_iterations=3
        )
        assert result.text == "done"

    async def test_several_tool_calls_in_one_turn_each_get_a_message(self, patched):
        fake = patched(
            response(
                tool_calls=[
                    tool_call("web_search", '{"query": "a"}', "c1"),
                    tool_call("web_search", '{"query": "b"}', "c2"),
                ]
            ),
            response(text="both checked"),
        )

        await run_agent_loop(
            ENDPOINT,
            MESSAGES,
            TOOLS,
            {"web_search": lambda query, max_results=5: [query]},
            max_iterations=3,
        )

        tool_messages = [m for m in fake.calls[1]["messages"] if m["role"] == "tool"]
        assert [m["tool_call_id"] for m in tool_messages] == ["c1", "c2"]

    async def test_the_caller_message_list_is_not_mutated(self, patched):
        patched(
            response(tool_calls=[tool_call("web_search", "{}")]),
            response(text="ok"),
        )
        messages = [dict(MESSAGES[0])]

        await run_agent_loop(
            ENDPOINT, messages, TOOLS, {"web_search": lambda **k: []}, max_iterations=3
        )

        assert messages == [dict(MESSAGES[0])]


class TestToolFaults:
    """A bad tool is data the model reads, never an exception the loop raises."""

    async def test_an_unknown_tool_name_returns_an_error_result(self, patched):
        fake = patched(
            response(tool_calls=[tool_call("delete_everything", "{}")]),
            response(text="I will use another approach."),
        )

        result = await run_agent_loop(
            ENDPOINT, MESSAGES, TOOLS, {"web_search": lambda **k: []}, max_iterations=3
        )

        assert result.text == "I will use another approach."
        content = fake.calls[1]["messages"][2]["content"]
        assert "unknown tool 'delete_everything'" in content
        assert "web_search" in content

    async def test_a_raising_tool_returns_an_error_result(self, patched):
        fake = patched(
            response(tool_calls=[tool_call("web_search", '{"query": "x"}')]),
            response(text="recovered"),
        )

        def boom(query, max_results=5):
            raise RuntimeError("upstream is down")

        result = await run_agent_loop(
            ENDPOINT, MESSAGES, TOOLS, {"web_search": boom}, max_iterations=3
        )

        assert result.text == "recovered"
        content = fake.calls[1]["messages"][2]["content"]
        assert "RuntimeError: upstream is down" in content

    async def test_malformed_arguments_return_an_error_result(self, patched):
        fake = patched(
            response(tool_calls=[tool_call("web_search", "{not json")]),
            response(text="recovered"),
        )
        called = []

        await run_agent_loop(
            ENDPOINT,
            MESSAGES,
            TOOLS,
            {"web_search": lambda **k: called.append(k)},
            max_iterations=3,
        )

        assert called == []
        assert "not valid JSON" in fake.calls[1]["messages"][2]["content"]

    async def test_a_bad_keyword_argument_returns_an_error_result(self, patched):
        fake = patched(
            response(tool_calls=[tool_call("web_search", '{"nonexistent": 1}')]),
            response(text="recovered"),
        )

        await run_agent_loop(
            ENDPOINT,
            MESSAGES,
            TOOLS,
            {"web_search": lambda query, max_results=5: []},
            max_iterations=3,
        )

        assert "TypeError" in fake.calls[1]["messages"][2]["content"]

    async def test_a_malformed_tool_envelope_becomes_a_structured_error(self, patched):
        fake = patched(
            response(tool_calls=[{"id": "call-bad", "function": "web_search"}]),
            response(text="recovered"),
        )

        result = await run_agent_loop(
            ENDPOINT, MESSAGES, TOOLS, {"web_search": lambda **k: []}, max_iterations=3
        )

        assert result.text == "recovered"
        assert "bad_arguments" in fake.calls[1]["messages"][2]["content"]

    async def test_duplicate_argument_keys_do_not_reach_the_tool(self, patched):
        fake = patched(
            response(
                tool_calls=[tool_call("web_search", '{"query":"a","query":"b"}')]
            ),
            response(text="recovered"),
        )
        called = []

        await run_agent_loop(
            ENDPOINT,
            MESSAGES,
            TOOLS,
            {"web_search": lambda **kwargs: called.append(kwargs)},
            max_iterations=3,
        )

        assert called == []
        assert "duplicate JSON key" in fake.calls[1]["messages"][2]["content"]

    async def test_a_tool_timeout_is_returned_to_the_model(self, patched):
        fake = patched(
            response(tool_calls=[tool_call("web_search", "{}")]),
            response(text="continued without it"),
        )

        async def never_returns():
            await asyncio.Event().wait()

        result = await run_agent_loop(
            ENDPOINT,
            MESSAGES,
            TOOLS,
            {"web_search": never_returns},
            max_iterations=3,
            tool_timeout=0.01,
        )

        assert result.text == "continued without it"
        assert "tool_timeout" in fake.calls[1]["messages"][2]["content"]


class TestBound:
    async def test_max_iterations_forces_a_final_tool_free_call(self, patched):
        # The model asks for a tool on every single turn it is offered one.
        fake = patched(
            *[response(tool_calls=[tool_call("web_search", "{}")]) for _ in range(3)],
            response(text="Forced to answer."),
        )

        result = await run_agent_loop(
            ENDPOINT,
            MESSAGES,
            TOOLS,
            {"web_search": lambda **k: []},
            max_iterations=3,
        )

        assert result.text == "Forced to answer."
        assert result.tool_calls == []
        assert len(fake.calls) == 4
        # Three tool-enabled turns, then one with the schemas withheld entirely.
        assert fake.tools_per_call == [TOOLS, TOOLS, TOOLS, None]

    async def test_the_loop_stops_early_when_the_model_stops_asking(self, patched):
        fake = patched(
            response(tool_calls=[tool_call("web_search", "{}")]),
            response(text="Answered."),
        )

        await run_agent_loop(
            ENDPOINT, MESSAGES, TOOLS, {"web_search": lambda **k: []}, max_iterations=10
        )

        assert len(fake.calls) == 2
        assert fake.tools_per_call == [TOOLS, TOOLS]

    async def test_a_non_positive_budget_is_rejected(self, patched):
        patched()
        with pytest.raises(AgentLoopError):
            await run_agent_loop(ENDPOINT, MESSAGES, TOOLS, {}, max_iterations=0)

    async def test_tool_calls_from_the_forced_tool_free_turn_fail_loudly(self, patched):
        patched(
            response(tool_calls=[tool_call("web_search", "{}")]),
            response(tool_calls=[tool_call("web_search", "{}")]),
        )

        with pytest.raises(AgentLoopError, match="after tools were withheld"):
            await run_agent_loop(
                ENDPOINT,
                MESSAGES,
                TOOLS,
                {"web_search": lambda **k: []},
                max_iterations=1,
            )


class TestEvents:
    async def test_one_event_is_published_per_tool_call(self, patched):
        patched(
            response(tool_calls=[tool_call("web_search", '{"query": "perseids"}')]),
            response(text="done"),
        )
        events: list[dict] = []

        async def on_event(event):
            events.append(event)

        await run_agent_loop(
            ENDPOINT,
            MESSAGES,
            TOOLS,
            {"web_search": lambda query, max_results=5: [{"url": "https://e.org/p"}]},
            max_iterations=3,
            on_event=on_event,
        )

        assert len(events) == 1
        assert events[0]["tool"] == "web_search"
        assert events[0]["arguments"] == {"query": "perseids"}
        assert "https://e.org/p" in events[0]["result_preview"]

    async def test_a_sync_event_callback_works_too(self, patched):
        patched(
            response(tool_calls=[tool_call("web_search", "{}")]),
            response(text="done"),
        )
        events: list[dict] = []

        await run_agent_loop(
            ENDPOINT,
            MESSAGES,
            TOOLS,
            {"web_search": lambda **k: []},
            max_iterations=3,
            on_event=events.append,
        )

        assert [e["tool"] for e in events] == ["web_search"]

    async def test_result_previews_are_short(self, patched):
        patched(
            response(tool_calls=[tool_call("web_search", "{}")]),
            response(text="done"),
        )
        events: list[dict] = []

        await run_agent_loop(
            ENDPOINT,
            MESSAGES,
            TOOLS,
            {"web_search": lambda **k: "z" * 5000},
            max_iterations=3,
            on_event=events.append,
        )

        assert len(events[0]["result_preview"]) == 200


class TestStructuredErrors:
    async def test_a_fault_is_a_json_object_with_tool_type_and_message(self, patched):
        fake = patched(
            response(tool_calls=[tool_call("web_search", '{"query": "x"}')]),
            response(text="recovered"),
        )

        def boom(query, max_results=5):
            raise RuntimeError("upstream is down")

        await run_agent_loop(ENDPOINT, MESSAGES, TOOLS, {"web_search": boom}, max_iterations=3)

        import json as json_module

        payload = json_module.loads(fake.calls[1]["messages"][2]["content"])
        assert payload["error"]["tool"] == "web_search"
        assert payload["error"]["type"] == "tool_failure"
        assert "upstream is down" in payload["error"]["message"]


class TestCallCap:
    async def test_a_tool_at_its_cap_errors_instead_of_running(self, patched):
        fake = patched(
            response(
                tool_calls=[
                    tool_call("web_search", '{"query": "a"}', "c1"),
                    tool_call("web_search", '{"query": "b"}', "c2"),
                ]
            ),
            response(text="done"),
        )
        runs: list[str] = []

        await run_agent_loop(
            ENDPOINT,
            MESSAGES,
            TOOLS,
            {"web_search": lambda query, max_results=5: runs.append(query)},
            max_iterations=3,
            tool_call_cap=1,
        )

        assert runs == ["a"]
        second_result = fake.calls[1]["messages"][3]["content"]
        assert "call_cap_exceeded" in second_result

    async def test_the_cap_counts_across_turns(self, patched):
        fake = patched(
            *[response(tool_calls=[tool_call("web_search", "{}")]) for _ in range(3)],
            response(text="done"),
        )
        runs: list[int] = []

        await run_agent_loop(
            ENDPOINT,
            MESSAGES,
            TOOLS,
            {"web_search": lambda **k: runs.append(1)},
            max_iterations=3,
            tool_call_cap=2,
        )

        assert len(runs) == 2

    async def test_no_cap_means_unlimited(self, patched):
        patched(
            *[response(tool_calls=[tool_call("web_search", "{}")]) for _ in range(4)],
            response(text="done"),
        )
        runs: list[int] = []

        await run_agent_loop(
            ENDPOINT,
            MESSAGES,
            TOOLS,
            {"web_search": lambda **k: runs.append(1)},
            max_iterations=4,
        )

        assert len(runs) == 4

    async def test_malformed_calls_do_not_burn_the_budget(self, patched):
        """An unknown tool or bad arguments ran nothing; charging for them can
        blind an agent whose next calls would have been well-formed."""
        fake = patched(
            response(
                tool_calls=[
                    tool_call("nonexistent", "{}", "c1"),
                    tool_call("web_search", "not json", "c2"),
                ]
            ),
            response(tool_calls=[tool_call("web_search", '{"query": "real"}', "c3")]),
            response(text="done"),
        )
        runs: list[str] = []

        await run_agent_loop(
            ENDPOINT,
            MESSAGES,
            TOOLS,
            {"web_search": lambda query, max_results=5: runs.append(query)},
            max_iterations=3,
            tool_call_cap=1,
        )

        # The two error payloads left the budget intact for the real call.
        assert runs == ["real"]
        first_turn = fake.calls[1]["messages"]
        assert "unknown_tool" in first_turn[2]["content"]
        assert "bad_arguments" in first_turn[3]["content"]


class TestConversationOut:
    async def test_the_working_conversation_is_exported_without_the_final_reply(
        self, patched
    ):
        patched(
            response(tool_calls=[tool_call("web_search", '{"query": "a"}')]),
            response(text="the answer"),
        )
        conversation: list = []

        await run_agent_loop(
            ENDPOINT,
            MESSAGES,
            TOOLS,
            {"web_search": lambda **k: [{"title": "t"}]},
            max_iterations=3,
            conversation_out=conversation,
        )

        roles = [m["role"] for m in conversation]
        assert roles == ["user", "assistant", "tool"]
        assert conversation[1]["tool_calls"][0]["function"]["name"] == "web_search"
        # The final prose reply is the caller's to append (or replace).
        assert all("the answer" not in (m.get("content") or "") for m in conversation)

    async def test_it_is_filled_on_the_forced_final_call_too(self, patched):
        patched(
            response(tool_calls=[tool_call("web_search", "{}")]),
            response(tool_calls=[tool_call("web_search", "{}")]),
            response(text="forced"),
        )
        conversation: list = []

        await run_agent_loop(
            ENDPOINT,
            MESSAGES,
            TOOLS,
            {"web_search": lambda **k: []},
            max_iterations=2,
            conversation_out=conversation,
        )

        assert [m["role"] for m in conversation] == [
            "user", "assistant", "tool", "assistant", "tool",
        ]


class TestRetryOnEmptyOptIn:
    async def test_every_loop_call_opts_into_empty_retries(self, patched):
        fake = patched(
            response(tool_calls=[tool_call("web_search", "{}")]),
            response(tool_calls=[tool_call("web_search", "{}")]),
            response(text="forced"),
        )

        await run_agent_loop(
            ENDPOINT, MESSAGES, TOOLS, {"web_search": lambda **k: []}, max_iterations=2
        )

        # Both the tool-enabled turns and the forced tool-free turn always
        # need an answer, so all of them pass retry_on_empty.
        assert [call.get("retry_on_empty") for call in fake.calls] == [True, True, True]
