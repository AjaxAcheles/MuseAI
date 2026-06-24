"""Module: M04 (LLM Inference Boundary)
Provider-neutral async inference boundary over configured endpoint objects.

The public call surface is endpoint/config routed. The default adapter speaks a
chat-completions-compatible JSON/SSE shape, while alternate endpoint formats can
be supplied through the adapter seam without changing callers.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx

import core.llm_io_logger as llm_io_logger
from llm.tokenizer import count_payload_tokens, count_tokens

Message = dict[str, str]
TokenCallback = Callable[[str], Any | Awaitable[Any]]
SleepCallback = Callable[[float], Awaitable[None]]

_DEFAULT_CHAT_COMPLETIONS_PATH = "/chat/completions"
_CHAT_COMPLETIONS_ROOT_PATHS = {"", "/", "/v1"}
_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_RETRY_DELAYS_SECONDS = (5.0, 15.0)
_TRANSIENT_HTTP_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    httpx.RemoteProtocolError,
)
_SECRET_KEY_MARKERS = ("api_key", "authorization", "password", "secret", "token")
_REDACTED = "[REDACTED]"


class LLMCallError(RuntimeError):
    """Raised when an inference call fails hard at the boundary."""

    def __init__(self, message: str, attempt_count: int = 0) -> None:
        super().__init__(message)
        self.attempt_count = attempt_count


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response returned by the inference boundary."""

    text: str
    raw: Any
    model_name: str
    endpoint_base_url: str
    tokens_in: int
    tokens_out: int
    streamed_chunks: list[str]
    attempt_count: int


class EndpointAdapter(Protocol):
    """Adapter contract for one endpoint wire format."""

    supports_streaming: bool
    protocol_name: str

    def build_request(
        self,
        *,
        messages: list[Message],
        endpoint: Any,
        model_name: str,
        stream: bool,
        temperature: float | None,
        max_tokens: int | None,
        response_format: Mapping[str, Any] | None,
        extra_body: Mapping[str, Any] | None,
        extra_headers: Mapping[str, str] | None,
    ) -> "PreparedLLMRequest":
        ...

    def extract_text(self, response_json: Mapping[str, Any]) -> str:
        ...

    def extract_stream_chunk(self, event_json: Mapping[str, Any]) -> str | None:
        ...


@dataclass(frozen=True)
class PreparedLLMRequest:
    """HTTP request assembled by an endpoint adapter."""

    url: str
    headers: dict[str, str]
    body: dict[str, Any]


@dataclass(frozen=True)
class ChatCompletionsAdapter:
    """Default adapter for chat-completions-compatible HTTP endpoints."""

    supports_streaming: bool = True
    protocol_name: str = "chat_completions_compatible"
    chat_completions_path: str = _DEFAULT_CHAT_COMPLETIONS_PATH

    def build_request(
        self,
        *,
        messages: list[Message],
        endpoint: Any,
        model_name: str,
        stream: bool,
        temperature: float | None,
        max_tokens: int | None,
        response_format: Mapping[str, Any] | None,
        extra_body: Mapping[str, Any] | None,
        extra_headers: Mapping[str, str] | None,
    ) -> PreparedLLMRequest:
        body: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": stream,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if response_format is not None:
            body["response_format"] = dict(response_format)
        if extra_body:
            body.update(dict(extra_body))

        headers = {"Content-Type": "application/json"}
        api_key = getattr(endpoint, "api_key", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(dict(extra_headers))

        return PreparedLLMRequest(
            url=resolve_inference_url(
                getattr(endpoint, "base_url"), self.chat_completions_path
            ),
            headers=headers,
            body=body,
        )

    def extract_text(self, response_json: Mapping[str, Any]) -> str:
        text = _extract_protocol_text(response_json)
        if text is None:
            raise LLMCallError(
                "non-streaming response did not contain a recognized text field"
            )
        return text

    def extract_stream_chunk(self, event_json: Mapping[str, Any]) -> str | None:
        return _extract_protocol_text(event_json)


def resolve_inference_url(
    base_url: str, chat_completions_path: str = _DEFAULT_CHAT_COMPLETIONS_PATH
) -> str:
    """Resolve an endpoint base URL to the default inference URL.

    Root-style URLs and the common versioned API root append the default
    chat-completions path. URLs that already include a non-root path are treated
    as full inference paths and returned unchanged.
    """
    if not base_url:
        raise ValueError("base_url is required")

    parts = urlsplit(base_url)
    path = parts.path.rstrip("/")
    if parts.path.rstrip("/").endswith(chat_completions_path.rstrip("/")):
        return base_url
    if parts.path in _CHAT_COMPLETIONS_ROOT_PATHS or path == "/v1":
        joined = f"{path}{chat_completions_path}" if path else chat_completions_path
        return urlunsplit((parts.scheme, parts.netloc, joined, parts.query, parts.fragment))
    return base_url


async def call_llm(
    messages: list[Message],
    endpoint: Any,
    *,
    stream: bool | None = None,
    on_token: TokenCallback | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    extra_headers: Mapping[str, str] | None = None,
    adapter: EndpointAdapter | None = None,
    model_name: str | None = None,
    client: httpx.AsyncClient | None = None,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    retry_delays: tuple[float, ...] = _DEFAULT_RETRY_DELAYS_SECONDS,
    sleep: SleepCallback = asyncio.sleep,
) -> LLMResponse:
    """Call a configured LLM endpoint through a normalized async boundary."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    started = perf_counter()
    adapter = adapter or ChatCompletionsAdapter()
    resolved_model_name = model_name if model_name is not None else getattr(endpoint, "model_name")
    should_stream = adapter.supports_streaming if stream is None else stream

    request = adapter.build_request(
        messages=messages,
        endpoint=endpoint,
        model_name=resolved_model_name,
        stream=should_stream,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        extra_body=extra_body,
        extra_headers=extra_headers,
    )

    logger = llm_io_logger.get_llm_io_logger()
    owns_client = client is None
    async_client = client or httpx.AsyncClient()
    tokenizer_family = getattr(endpoint, "tokenizer_family")
    tokens_in: int | None = None
    tokens_out: int | None = None
    attempt_count = 0
    response_text = ""
    error_summary: str | None = None
    try:
        tokens_in = count_payload_tokens(messages, tokenizer_family, resolved_model_name)
        if should_stream:
            (text, raw, chunks), attempt_count = await _call_with_retries(
                lambda: _call_streaming(async_client, request, adapter, on_token),
                max_attempts=max_attempts,
                retry_delays=retry_delays,
                sleep=sleep,
            )
        else:
            (text, raw), attempt_count = await _call_with_retries(
                lambda: _call_non_streaming(async_client, request, adapter),
                max_attempts=max_attempts,
                retry_delays=retry_delays,
                sleep=sleep,
            )
            chunks = []

        tokens_out = count_tokens(text, tokenizer_family, resolved_model_name)
        response_text = text
        return LLMResponse(
            text=text,
            raw=raw,
            model_name=resolved_model_name,
            endpoint_base_url=getattr(endpoint, "base_url"),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            streamed_chunks=chunks,
            attempt_count=attempt_count,
        )
    except LLMCallError as exc:
        attempt_count = max(attempt_count, exc.attempt_count)
        error_summary = str(exc)
        response_text = f"ERROR: {error_summary}"
        raise
    except Exception as exc:
        error = LLMCallError(f"inference boundary failed: {exc}")
        error_summary = str(error)
        response_text = f"ERROR: {error_summary}"
        raise error from exc
    finally:
        if owns_client:
            await async_client.aclose()
        duration_ms = (perf_counter() - started) * 1000
        llm_io_logger.log_llm_call(
            logger,
            _build_log_payload(
                endpoint=endpoint,
                request=request,
                adapter=adapter,
                model_name=resolved_model_name,
                stream=should_stream,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                attempt_count=attempt_count,
                error_summary=error_summary,
            ),
            response_text,
            duration_ms,
        )


async def _call_with_retries(
    operation: Callable[[], Awaitable[Any]],
    *,
    max_attempts: int,
    retry_delays: tuple[float, ...],
    sleep: SleepCallback,
) -> tuple[Any, int]:
    for attempt in range(1, max_attempts + 1):
        try:
            return await operation(), attempt
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            summary = _http_status_summary(exc.response)
            if not _is_transient_status(status):
                raise LLMCallError(summary, attempt_count=attempt) from exc
            if attempt == max_attempts:
                raise LLMCallError(
                    f"transient inference HTTP failure after {attempt} attempts: {summary}",
                    attempt_count=attempt,
                ) from exc
            await sleep(_retry_delay(attempt, retry_delays))
        except _TRANSIENT_HTTP_EXCEPTIONS as exc:
            if attempt == max_attempts:
                raise LLMCallError(
                    f"transient inference request failed after {attempt} attempts: {exc}",
                    attempt_count=attempt,
                ) from exc
            await sleep(_retry_delay(attempt, retry_delays))
        except LLMCallError as exc:
            if exc.attempt_count == 0:
                exc.attempt_count = attempt
            raise

    raise LLMCallError("inference request failed before an attempt could complete")


async def _call_non_streaming(
    client: httpx.AsyncClient, request: PreparedLLMRequest, adapter: EndpointAdapter
) -> tuple[str, Any]:
    try:
        response = await client.post(
            request.url,
            headers=request.headers,
            json=request.body,
        )
        response.raise_for_status()
        response_json = response.json()
    except ValueError as exc:
        raise LLMCallError(f"non-streaming response JSON was invalid: {exc}") from exc

    if not isinstance(response_json, Mapping):
        raise LLMCallError("non-streaming response JSON must be an object")
    return adapter.extract_text(response_json), response_json


async def _call_streaming(
    client: httpx.AsyncClient,
    request: PreparedLLMRequest,
    adapter: EndpointAdapter,
    on_token: TokenCallback | None,
) -> tuple[str, dict[str, Any], list[str]]:
    chunks: list[str] = []
    raw_events: list[Any] = []

    async with client.stream(
        "POST",
        request.url,
        headers=request.headers,
        json=request.body,
    ) as response:
        response.raise_for_status()
        async for event_json in _iter_sse_json(response):
            raw_events.append(event_json)
            chunk = adapter.extract_stream_chunk(event_json)
            if chunk is None:
                continue
            chunks.append(chunk)
            if on_token is not None:
                await _call_token_callback(on_token, chunk)

    return "".join(chunks), {"stream_events": raw_events}, chunks


async def _iter_sse_json(response: httpx.Response):
    async for line in response.aiter_lines():
        stripped = line.strip()
        if not stripped or stripped.startswith(":"):
            continue
        if not stripped.startswith("data:"):
            continue

        data = stripped.removeprefix("data:").strip()
        if data == "[DONE]":
            break
        try:
            event_json = json.loads(data)
        except json.JSONDecodeError as exc:
            raise LLMCallError(f"malformed stream JSON payload: {data[:120]}") from exc
        if not isinstance(event_json, Mapping):
            raise LLMCallError("stream JSON payload must be an object")
        yield event_json


async def _call_token_callback(callback: TokenCallback, chunk: str) -> None:
    result = callback(chunk)
    if inspect.isawaitable(result):
        await result


def _extract_protocol_text(payload: Mapping[str, Any]) -> str | None:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            delta = first.get("delta")
            if isinstance(delta, Mapping) and isinstance(delta.get("content"), str):
                return delta["content"]
            message = first.get("message")
            if isinstance(message, Mapping) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]

    message = payload.get("message")
    if isinstance(message, Mapping) and isinstance(message.get("content"), str):
        return message["content"]
    for key in ("content", "text", "response"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def _retry_delay(failed_attempt: int, retry_delays: tuple[float, ...]) -> float:
    if not retry_delays:
        return 0.0
    index = min(failed_attempt - 1, len(retry_delays) - 1)
    return retry_delays[index]


def _is_transient_status(status_code: int) -> bool:
    return 500 <= status_code <= 599


def _http_status_summary(response: httpx.Response) -> str:
    text = response.text[:200].replace("\n", " ")
    suffix = f": {text}" if text else ""
    return f"HTTP {response.status_code} from inference endpoint{suffix}"


def _build_log_payload(
    *,
    endpoint: Any,
    request: PreparedLLMRequest,
    adapter: EndpointAdapter,
    model_name: str,
    stream: bool,
    tokens_in: int | None,
    tokens_out: int | None,
    attempt_count: int,
    error_summary: str | None,
) -> dict[str, Any]:
    return {
        "endpoint_base_url": _sanitize_url(getattr(endpoint, "base_url")),
        "resolved_url": _sanitize_url(request.url),
        "model_name": model_name,
        "stream": stream,
        "adapter_protocol": getattr(adapter, "protocol_name", "configured_endpoint"),
        "request_body": _redact_secrets(request.body),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "attempt_count": attempt_count,
        "error": error_summary,
    }


def _sanitize_url(url: str) -> str:
    parts = urlsplit(url)
    hostname = parts.hostname or ""
    if parts.port is not None:
        hostname = f"{hostname}:{parts.port}"
    return urlunsplit((parts.scheme, hostname, parts.path, "", ""))


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_secret_key(key_text):
                redacted[key_text] = _REDACTED
            else:
                redacted[key_text] = _redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_secrets(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _is_secret_key(key: str) -> bool:
    lower = key.lower().replace("-", "_")
    return any(marker in lower for marker in _SECRET_KEY_MARKERS)
