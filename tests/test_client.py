"""Tests for museai.llm.client.

Every test drives the client through an injected ``httpx.MockTransport``; no
test opens a socket.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from museai.core.config import EndpointConfig
from museai.core.logging_setup import get_llm_logger
from museai.llm.client import (
    DEFAULT_BACKOFF_SECONDS,
    MAX_ATTEMPTS,
    LLMCallError,
    LLMResponse,
    _build_request,
    call_llm,
    resolve_inference_url,
)

NO_BACKOFF = [0.0, 0.0]
MESSAGES = [{"role": "user", "content": "Write a chapter."}]


@pytest.fixture
def endpoint() -> EndpointConfig:
    return EndpointConfig(
        base_url="https://example.invalid/v1",
        api_key="secret-key-do-not-log",
        model_name="test-model",
        tokenizer_family="char_heuristic",
        request_timeout=5,
        temperature=0.3,
    )


def json_response(payload: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


def completion(content: str = "hello", **extra) -> dict:
    message = {"role": "assistant", "content": content}
    message.update(extra.pop("message", {}))
    return {"choices": [{"message": message, "finish_reason": "stop"}], **extra}


def sse(*chunks: dict, done: bool = True) -> bytes:
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def content_chunk(text: str, finish_reason: str | None = None) -> dict:
    return {"choices": [{"delta": {"content": text}, "finish_reason": finish_reason}]}


class TestResolveInferenceUrl:
    def test_appends_chat_completions_to_a_root(self):
        assert (
            resolve_inference_url("https://example.invalid/v1")
            == "https://example.invalid/v1/chat/completions"
        )

    def test_tolerates_a_trailing_slash(self):
        assert (
            resolve_inference_url("https://example.invalid/v1/")
            == "https://example.invalid/v1/chat/completions"
        )

    def test_bare_host_gets_the_path(self):
        assert (
            resolve_inference_url("https://example.invalid")
            == "https://example.invalid/chat/completions"
        )

    def test_full_completions_path_is_used_as_is(self):
        url = "https://example.invalid/v1/chat/completions"
        assert resolve_inference_url(url) == url

    def test_nonstandard_full_path_is_used_as_is(self):
        """A deployment-style path already naming completions is not rewritten."""
        url = "https://example.invalid/openai/deployments/x/chat/completions"
        assert resolve_inference_url(url) == url

    def test_empty_base_url_raises(self):
        with pytest.raises(ValueError, match="empty"):
            resolve_inference_url("   ")

    def test_relative_base_url_raises(self):
        with pytest.raises(ValueError, match="absolute"):
            resolve_inference_url("/v1")


class TestBuildRequest:
    def test_omits_optional_params_when_none(self, endpoint):
        _, _, body = _build_request(
            endpoint,
            MESSAGES,
            stream=False,
            temperature=None,
            max_tokens=None,
            tools=None,
            tool_choice=None,
            response_format=None,
            extra_body=None,
            extra_headers=None,
        )
        assert set(body) == {"model", "messages", "temperature", "stream"}
        assert body["temperature"] == 0.3  # endpoint default
        assert body["model"] == "test-model"

    def test_includes_optional_params_when_supplied(self, endpoint):
        tools = [{"type": "function", "function": {"name": "search"}}]
        _, _, body = _build_request(
            endpoint,
            MESSAGES,
            stream=True,
            temperature=0.9,
            max_tokens=128,
            tools=tools,
            tool_choice="auto",
            response_format={"type": "json_object"},
            extra_body={"top_p": 0.5},
            extra_headers=None,
        )
        assert body["temperature"] == 0.9
        assert body["max_tokens"] == 128
        assert body["tools"] == tools
        assert body["tool_choice"] == "auto"
        assert body["response_format"] == {"type": "json_object"}
        assert body["top_p"] == 0.5
        assert body["stream"] is True

    def test_authorization_header_carries_the_key(self, endpoint):
        _, headers, _ = _build_request(
            endpoint,
            MESSAGES,
            stream=False,
            temperature=None,
            max_tokens=None,
            tools=None,
            tool_choice=None,
            response_format=None,
            extra_body=None,
            extra_headers={"X-Trace": "abc"},
        )
        assert headers["Authorization"] == "Bearer secret-key-do-not-log"
        assert headers["X-Trace"] == "abc"


class TestNonStreaming:
    async def test_returns_text_and_token_counts(self, endpoint):
        transport = httpx.MockTransport(lambda r: json_response(completion("hello world")))
        result = await call_llm(endpoint, MESSAGES, transport=transport)

        assert isinstance(result, LLMResponse)
        assert result.text == "hello world"
        assert result.finish_reason == "stop"
        assert result.tool_calls == []
        assert result.model_name == "test-model"
        assert result.endpoint_base_url == "https://example.invalid/v1"
        assert result.tokens_out == 3  # ceil(11/4) under char_heuristic
        assert result.tokens_in > 0

    async def test_posts_to_the_resolved_url_with_the_key(self, endpoint):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return json_response(completion())

        await call_llm(endpoint, MESSAGES, transport=httpx.MockTransport(handler))

        assert str(seen[0].url) == "https://example.invalid/v1/chat/completions"
        assert seen[0].headers["authorization"] == "Bearer secret-key-do-not-log"
        assert json.loads(seen[0].content)["stream"] is False

    async def test_null_content_yields_empty_text(self, endpoint):
        payload = {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}
        transport = httpx.MockTransport(lambda r: json_response(payload))
        result = await call_llm(endpoint, MESSAGES, transport=transport)
        assert result.text == ""
        assert result.tokens_out == 0

    async def test_missing_choices_raises(self, endpoint):
        transport = httpx.MockTransport(lambda r: json_response({"choices": []}))
        with pytest.raises(LLMCallError, match="no choices"):
            await call_llm(endpoint, MESSAGES, transport=transport)

    async def test_non_json_body_raises(self, endpoint):
        transport = httpx.MockTransport(lambda r: httpx.Response(200, content=b"not json"))
        with pytest.raises(LLMCallError, match="not valid JSON"):
            await call_llm(endpoint, MESSAGES, transport=transport)


class TestToolCalls:
    async def test_tool_calls_are_parsed(self, endpoint):
        tool_call = {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "web_search", "arguments": '{"query": "moon"}'},
        }
        payload = {
            "choices": [
                {
                    "message": {"content": None, "tool_calls": [tool_call]},
                    "finish_reason": "tool_calls",
                }
            ]
        }
        transport = httpx.MockTransport(lambda r: json_response(payload))
        result = await call_llm(endpoint, MESSAGES, transport=transport)

        assert result.finish_reason == "tool_calls"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "web_search"
        assert json.loads(result.tool_calls[0]["function"]["arguments"]) == {"query": "moon"}

    async def test_tools_are_forwarded_on_the_wire(self, endpoint):
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return json_response(completion())

        tools = [{"type": "function", "function": {"name": "web_search"}}]
        await call_llm(
            endpoint,
            MESSAGES,
            tools=tools,
            tool_choice="auto",
            transport=httpx.MockTransport(handler),
        )
        assert seen[0]["tools"] == tools
        assert seen[0]["tool_choice"] == "auto"

    async def test_streamed_tool_call_fragments_are_assembled(self, endpoint):
        chunks = (
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {"name": "web_search", "arguments": ""},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"query":'}}
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": ' "moon"}'}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )
        transport = httpx.MockTransport(lambda r: httpx.Response(200, content=sse(*chunks)))
        result = await call_llm(endpoint, MESSAGES, stream=True, transport=transport)

        assert result.text == ""
        assert len(result.tool_calls) == 1
        call = result.tool_calls[0]
        assert call["id"] == "call_abc"
        assert call["function"]["name"] == "web_search"
        assert json.loads(call["function"]["arguments"]) == {"query": "moon"}
        assert result.finish_reason == "tool_calls"


class TestStreaming:
    async def test_assembles_text_from_sse_chunks(self, endpoint):
        payload = sse(
            content_chunk("hello"),
            content_chunk(" "),
            content_chunk("world", finish_reason="stop"),
        )
        transport = httpx.MockTransport(lambda r: httpx.Response(200, content=payload))
        result = await call_llm(endpoint, MESSAGES, stream=True, transport=transport)

        assert result.text == "hello world"
        assert result.finish_reason == "stop"
        assert result.tokens_out == 3
        assert result.raw["stream"] is True

    async def test_sync_on_token_receives_each_delta(self, endpoint):
        seen: list[str] = []
        payload = sse(content_chunk("a"), content_chunk("b"), content_chunk("c"))
        transport = httpx.MockTransport(lambda r: httpx.Response(200, content=payload))

        result = await call_llm(
            endpoint, MESSAGES, stream=True, on_token=seen.append, transport=transport
        )
        assert seen == ["a", "b", "c"]
        assert result.text == "abc"

    async def test_async_on_token_is_awaited(self, endpoint):
        seen: list[str] = []

        async def on_token(token: str) -> None:
            seen.append(token)

        payload = sse(content_chunk("x"), content_chunk("y"))
        transport = httpx.MockTransport(lambda r: httpx.Response(200, content=payload))

        result = await call_llm(
            endpoint, MESSAGES, stream=True, on_token=on_token, transport=transport
        )
        assert seen == ["x", "y"]
        assert result.text == "xy"

    async def test_nonstandard_delta_fields_are_ignored(self, endpoint):
        """Reasoning models stream a `reasoning` delta alongside empty content.

        Only `content` is prose. A reasoning delta must not reach `on_token`,
        must not be assembled into the text, and must not be counted as output.
        """
        seen: list[str] = []
        payload = sse(
            {"choices": [{"delta": {"content": "", "reasoning": "Thinking"}}]},
            {"choices": [{"delta": {"content": "", "reasoning": " harder"}}]},
            content_chunk("done", finish_reason="stop"),
        )
        transport = httpx.MockTransport(lambda r: httpx.Response(200, content=payload))

        result = await call_llm(
            endpoint, MESSAGES, stream=True, on_token=seen.append, transport=transport
        )
        assert seen == ["done"]
        assert result.text == "done"
        assert result.tokens_out == 1
        # The unrecognised field still survives in `raw` for the caller to inspect.
        assert result.raw["chunks"][0]["choices"][0]["delta"]["reasoning"] == "Thinking"

    async def test_blank_lines_and_done_are_handled(self, endpoint):
        body = (
            b"\n"
            b': keep-alive comment\n'
            b"\n"
            + f"data: {json.dumps(content_chunk('ok'))}\n".encode()
            + b"\n"
            b"data: [DONE]\n"
            b'data: {"never": "parsed"}\n'
        )
        transport = httpx.MockTransport(lambda r: httpx.Response(200, content=body))
        result = await call_llm(endpoint, MESSAGES, stream=True, transport=transport)
        assert result.text == "ok"

    async def test_malformed_stream_json_raises(self, endpoint):
        body = b"data: {not valid json}\n\n"
        transport = httpx.MockTransport(lambda r: httpx.Response(200, content=body))
        with pytest.raises(LLMCallError, match="malformed JSON in stream chunk"):
            await call_llm(
                endpoint, MESSAGES, stream=True, transport=transport, retry_backoff=NO_BACKOFF
            )

    async def test_streaming_4xx_raises_without_consuming_a_stream(self, endpoint):
        transport = httpx.MockTransport(
            lambda r: httpx.Response(400, content=b'{"error": "bad request"}')
        )
        with pytest.raises(LLMCallError, match="HTTP 400"):
            await call_llm(endpoint, MESSAGES, stream=True, transport=transport)


class TestRetry:
    def test_policy_constants(self):
        assert MAX_ATTEMPTS == 3
        assert DEFAULT_BACKOFF_SECONDS == (5.0, 15.0)

    async def test_4xx_raises_immediately_without_retrying(self, endpoint):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(429, content=b'{"error": "rate limited"}')

        with pytest.raises(LLMCallError, match="HTTP 429"):
            await call_llm(
                endpoint,
                MESSAGES,
                transport=httpx.MockTransport(handler),
                retry_backoff=NO_BACKOFF,
            )
        assert len(calls) == 1, "a 4xx must not be retried"

    async def test_5xx_is_retried_to_exhaustion(self, endpoint):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(503, content=b"upstream down")

        with pytest.raises(LLMCallError, match="3 attempts failed"):
            await call_llm(
                endpoint,
                MESSAGES,
                transport=httpx.MockTransport(handler),
                retry_backoff=NO_BACKOFF,
            )
        assert len(calls) == MAX_ATTEMPTS

    async def test_5xx_then_success_recovers(self, endpoint):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(500, content=b"boom")
            return json_response(completion("recovered"))

        result = await call_llm(
            endpoint,
            MESSAGES,
            transport=httpx.MockTransport(handler),
            retry_backoff=NO_BACKOFF,
        )
        assert result.text == "recovered"
        assert len(calls) == 2

    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ConnectError("refused"),
            httpx.ConnectTimeout("timed out"),
            httpx.ReadTimeout("slow"),
            httpx.RemoteProtocolError("dropped"),
        ],
    )
    async def test_transient_network_faults_are_retried(self, endpoint, exc):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if len(calls) < MAX_ATTEMPTS:
                raise exc
            return json_response(completion("ok"))

        result = await call_llm(
            endpoint,
            MESSAGES,
            transport=httpx.MockTransport(handler),
            retry_backoff=NO_BACKOFF,
        )
        assert result.text == "ok"
        assert len(calls) == MAX_ATTEMPTS

    async def test_backoff_delays_are_injectable(self, endpoint, monkeypatch):
        slept: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            slept.append(seconds)

        monkeypatch.setattr("museai.llm.client.asyncio.sleep", fake_sleep)
        transport = httpx.MockTransport(lambda r: httpx.Response(500, content=b"boom"))

        with pytest.raises(LLMCallError):
            await call_llm(
                endpoint, MESSAGES, transport=transport, retry_backoff=[0.01, 0.02]
            )
        assert slept == [0.01, 0.02], "two sleeps for three attempts"

    async def test_stream_fault_after_first_token_is_not_retried(self, endpoint):
        """Replaying a partially emitted stream would duplicate tokens."""
        calls = []
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            # A valid chunk, then the connection dies mid-stream.
            async def body():
                yield f"data: {json.dumps(content_chunk('partial'))}\n\n".encode()
                raise httpx.RemoteProtocolError("connection dropped")

            return httpx.Response(200, content=body())

        with pytest.raises(LLMCallError, match="after emitting tokens"):
            await call_llm(
                endpoint,
                MESSAGES,
                stream=True,
                on_token=seen.append,
                transport=httpx.MockTransport(handler),
                retry_backoff=NO_BACKOFF,
            )
        assert seen == ["partial"]
        assert len(calls) == 1


class TestLogging:
    @pytest.fixture
    def records(self):
        logger = get_llm_logger()
        captured: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(record)

        handler = Capture()
        logger.addHandler(handler)
        yield captured
        logger.removeHandler(handler)

    @staticmethod
    def payloads(records) -> list[dict]:
        return [json.loads(record.getMessage()) for record in records]

    async def test_a_call_writes_one_request_then_one_response(self, endpoint, records):
        transport = httpx.MockTransport(lambda r: json_response(completion("hi there")))
        await call_llm(endpoint, MESSAGES, transport=transport)

        request, response = self.payloads(records)
        assert [request["event"], response["event"]] == ["request", "response"]

        assert request["url"] == "https://example.invalid/v1/chat/completions"
        assert request["model"] == "test-model"
        assert request["stream"] is False
        assert request["attempt"] == 1
        assert request["messages"][0]["role"] == "user"

        assert response["attempt"] == 1
        assert response["text"] == "hi there"
        assert response["tokens_in"] > 0
        assert response["tokens_out"] == 2
        assert response["finish_reason"] == "stop"

        for record in records:
            assert "secret-key-do-not-log" not in record.getMessage()

    async def test_a_streamed_call_logs_once_per_message_not_per_token(
        self, endpoint, records
    ):
        chunks = [content_chunk("one"), content_chunk(" two"), content_chunk(" three")]
        transport = httpx.MockTransport(lambda r: httpx.Response(200, content=sse(*chunks)))

        seen: list[str] = []
        await call_llm(
            endpoint, MESSAGES, stream=True, on_token=seen.append, transport=transport
        )

        # Three tokens reached the caller; the log still holds exactly two records.
        assert seen == ["one", " two", " three"]
        request, response = self.payloads(records)
        assert request["event"] == "request" and request["stream"] is True
        assert response["event"] == "response"
        assert response["text"] == "one two three"

    async def test_each_retry_logs_its_own_request_and_error(self, endpoint, records):
        transport = httpx.MockTransport(lambda r: httpx.Response(500, content=b"boom"))
        with pytest.raises(LLMCallError):
            await call_llm(
                endpoint, MESSAGES, transport=transport, retry_backoff=NO_BACKOFF
            )

        payloads = self.payloads(records)
        requests = [p for p in payloads if p["event"] == "request"]
        errors = [p for p in payloads if p["event"] == "error"]

        assert len(requests) == MAX_ATTEMPTS
        assert [p["attempt"] for p in requests] == [1, 2, 3]
        # One error per failed attempt, plus the final give-up record.
        assert [p["retrying"] for p in errors] == [True, True, True, False]
        assert not [p for p in payloads if p["event"] == "response"]

        for record in records:
            assert "secret-key-do-not-log" not in record.getMessage()

    async def test_a_4xx_logs_one_request_and_one_error(self, endpoint, records):
        transport = httpx.MockTransport(lambda r: httpx.Response(400, content=b"nope"))
        with pytest.raises(LLMCallError):
            await call_llm(endpoint, MESSAGES, transport=transport)

        request, error = self.payloads(records)
        assert request["event"] == "request"
        assert error["event"] == "error"
        assert error["retrying"] is False
        assert "400" in error["error"]

    async def test_url_credentials_and_query_are_stripped(self, records):
        endpoint = EndpointConfig(
            base_url="https://user:pw@example.invalid/v1?api-key=leak",
            api_key="k",
            model_name="m",
            tokenizer_family="char_heuristic",
        )
        transport = httpx.MockTransport(lambda r: json_response(completion()))
        await call_llm(endpoint, MESSAGES, transport=transport)

        for record in records:
            logged = record.getMessage()
            assert "pw" not in logged
            assert "leak" not in logged
            assert (
                json.loads(logged)["url"]
                == "https://example.invalid/v1/chat/completions"
            )

    async def test_long_bodies_are_truncated_in_the_log(self, endpoint, records):
        long_text = "x" * 900
        transport = httpx.MockTransport(lambda r: json_response(completion(long_text)))
        result = await call_llm(
            endpoint, [{"role": "user", "content": "y" * 900}], transport=transport
        )

        request, response = self.payloads(records)
        assert "[+400 chars]" in request["messages"][0]["content"]
        assert "[+400 chars]" in response["text"]
        # Truncation is a log concern only; the caller still gets everything.
        assert result.text == long_text
