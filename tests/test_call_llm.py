"""Module: M04 (LLM Inference Boundary)
Synthetic tests for the normalized async LLM call boundary.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

import core.llm_io_logger as llm_io_logger
from llm.call_llm import (
    LLMCallError,
    LLMResponse,
    call_llm,
    resolve_inference_url,
)


@dataclass(frozen=True)
class SyntheticEndpoint:
    base_url: str = "https://endpoint.example.test/v1"
    api_key: str = "secret"
    model_name: str = "synthetic-model"
    tokenizer_family: str = "char_heuristic"
    supports_concurrent_critics: bool = False
    grammar_constraint_strategy: str = "none"


@pytest.fixture(autouse=True)
def isolated_llm_io_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    log_path = tmp_path / "logs" / "llm_io.log"
    logger_name = f"llm_io_{tmp_path.name}"
    monkeypatch.setattr(llm_io_logger, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(llm_io_logger, "LOG_FILE", log_path)
    monkeypatch.setattr(llm_io_logger, "LOGGER_NAME", logger_name)
    _clear_logger(logger_name)
    yield log_path
    _clear_logger(logger_name)


def _clear_logger(logger_name: str) -> None:
    logger = logging.getLogger(logger_name)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _log_records(log_path: Path) -> list[dict]:
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]


def test_resolve_inference_url_appends_only_for_roots() -> None:
    assert (
        resolve_inference_url("https://endpoint.example.test")
        == "https://endpoint.example.test/chat/completions"
    )
    assert (
        resolve_inference_url("https://endpoint.example.test/v1")
        == "https://endpoint.example.test/v1/chat/completions"
    )
    full = "https://endpoint.example.test/custom/infer"
    assert resolve_inference_url(full) == full


def test_call_llm_non_streaming_uses_endpoint_config_and_counts_tokens() -> None:
    asyncio.run(_call_llm_non_streaming_uses_endpoint_config_and_counts_tokens())


async def _call_llm_non_streaming_uses_endpoint_config_and_counts_tokens() -> None:
    endpoint = SyntheticEndpoint()
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hello there"}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await call_llm(
            [{"role": "user", "content": "hello"}],
            endpoint,
            stream=False,
            temperature=0.2,
            max_tokens=8,
            extra_body={"metadata": {"purpose": "test"}},
            extra_headers={"X-Test": "yes"},
            client=client,
        )

    assert isinstance(response, LLMResponse)
    assert response.text == "hello there"
    assert response.model_name == endpoint.model_name
    assert response.endpoint_base_url == endpoint.base_url
    assert response.tokens_in > 0
    assert response.tokens_out > 0
    assert response.streamed_chunks == []
    assert response.attempt_count == 1
    assert captured["url"] == "https://endpoint.example.test/v1/chat/completions"
    assert captured["body"] == {
        "model": "synthetic-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 8,
        "metadata": {"purpose": "test"},
    }
    assert captured["headers"]["authorization"] == "Bearer secret"
    assert captured["headers"]["x-test"] == "yes"


def test_call_llm_streaming_assembles_chunks_and_calls_sync_callback() -> None:
    asyncio.run(_call_llm_streaming_assembles_chunks_and_calls_sync_callback())


async def _call_llm_streaming_assembles_chunks_and_calls_sync_callback() -> None:
    endpoint = SyntheticEndpoint()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        events = [
            b"\n",
            b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n',
            b'data: {"choices":[{"delta":{}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        return httpx.Response(200, content=b"".join(events))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await call_llm(
            [{"role": "user", "content": "hello"}],
            endpoint,
            on_token=seen.append,
            client=client,
        )

    assert response.text == "hello"
    assert response.streamed_chunks == ["hel", "lo"]
    assert seen == ["hel", "lo"]
    assert response.raw["stream_events"]


def test_call_llm_streaming_accepts_async_callback() -> None:
    asyncio.run(_call_llm_streaming_accepts_async_callback())


async def _call_llm_streaming_accepts_async_callback() -> None:
    endpoint = SyntheticEndpoint()
    seen: list[str] = []

    async def callback(chunk: str) -> None:
        seen.append(chunk)

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            content=b'data: {"message":{"content":"ok"}}\n\ndata: [DONE]\n\n',
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await call_llm(
            [{"role": "user", "content": "hello"}],
            endpoint,
            on_token=callback,
            client=client,
        )

    assert response.text == "ok"
    assert seen == ["ok"]


def test_malformed_stream_json_raises_llm_call_error() -> None:
    asyncio.run(_malformed_stream_json_raises_llm_call_error())


async def _malformed_stream_json_raises_llm_call_error() -> None:
    endpoint = SyntheticEndpoint()

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, content=b"data: {not json}\n\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(LLMCallError, match="malformed stream JSON"):
            await call_llm(
                [{"role": "user", "content": "hello"}],
                endpoint,
                client=client,
            )


def test_transient_5xx_retries_then_logs_success(isolated_llm_io_log: Path) -> None:
    asyncio.run(_transient_5xx_retries_then_logs_success(isolated_llm_io_log))


async def _transient_5xx_retries_then_logs_success(log_path: Path) -> None:
    endpoint = SyntheticEndpoint(api_key="top-secret-api-key")
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, json={"error": "try later"}, request=request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await call_llm(
            [{"role": "user", "content": "hello"}],
            endpoint,
            stream=False,
            client=client,
            retry_delays=(0.0, 0.0),
            extra_body={"api_key": "body-secret", "safe": "kept"},
        )

    assert attempts == 2
    assert response.text == "ok"
    assert response.attempt_count == 2

    records = _log_records(log_path)
    assert len(records) == 1
    record = records[0]
    assert record["response"] == "ok"
    request_log = record["request"]
    assert request_log["attempt_count"] == 2
    assert request_log["tokens_in"] > 0
    assert request_log["tokens_out"] > 0
    assert request_log["request_body"]["api_key"] == "[REDACTED]"
    assert request_log["request_body"]["safe"] == "kept"
    assert "top-secret-api-key" not in json.dumps(record)


def test_4xx_failure_does_not_retry_and_logs_error(isolated_llm_io_log: Path) -> None:
    asyncio.run(_4xx_failure_does_not_retry_and_logs_error(isolated_llm_io_log))


async def _4xx_failure_does_not_retry_and_logs_error(log_path: Path) -> None:
    endpoint = SyntheticEndpoint()
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(401, json={"error": "bad auth"}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(LLMCallError, match="HTTP 401"):
            await call_llm(
                [{"role": "user", "content": "hello"}],
                endpoint,
                stream=False,
                client=client,
                retry_delays=(0.0, 0.0),
            )

    assert attempts == 1
    records = _log_records(log_path)
    assert len(records) == 1
    record = records[0]
    assert record["response"].startswith("ERROR: HTTP 401")
    assert record["request"]["attempt_count"] == 1
    assert record["request"]["tokens_in"] > 0
    assert record["request"]["tokens_out"] is None


def test_redaction_masks_secret_fields_but_keeps_max_tokens() -> None:
    from llm.call_llm import _REDACTED, _redact_secrets

    body = {
        "model": "synthetic-model",
        "max_tokens": 256,  # must NOT be redacted despite containing "token"
        "n_tokens": 12,
        "temperature": 0.7,
        "api_key": "sk-live-should-be-hidden",
        "authorization": "Bearer sk-live",
        "access_token": "tok-should-be-hidden",
        "client_secret": "hunter2",
        "nested": {"password": "hunter2", "keep_me": "visible"},
        "messages": [{"role": "user", "content": "hi"}],
    }
    redacted = _redact_secrets(body)

    # Observability preserved for the (very common) max_tokens knob and friends.
    assert redacted["max_tokens"] == 256
    assert redacted["n_tokens"] == 12
    assert redacted["temperature"] == 0.7
    assert redacted["model"] == "synthetic-model"
    assert redacted["messages"] == [{"role": "user", "content": "hi"}]

    # Every secret-bearing field is masked, including a nested one.
    for secret_key in ("api_key", "authorization", "access_token", "client_secret"):
        assert redacted[secret_key] == _REDACTED
    assert redacted["nested"]["password"] == _REDACTED
    assert redacted["nested"]["keep_me"] == "visible"

    # The original payload is untouched — redaction is non-destructive.
    assert body["api_key"] == "sk-live-should-be-hidden"
