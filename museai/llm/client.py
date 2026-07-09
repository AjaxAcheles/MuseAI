"""The single async inference boundary for MuseAI v1.

Every LLM call in the system goes through :func:`call_llm`. The boundary is
endpoint- and model-agnostic: no provider, host, port, or model name appears in
any conditional. Everything wire-format-specific lives behind the private
``_build_request`` / ``_parse_response`` / ``_parse_stream`` adapter seam, and
anything an unusual endpoint needs beyond that is passed through verbatim via
``extra_body`` / ``extra_headers``.

The wire format is OpenAI-compatible chat completions, which is what every
endpoint MuseAI targets speaks.

Behaviour worth knowing:

* **Retries** — three attempts, sleeping 5s then 15s. Only transient faults
  retry: connection/read timeouts, dropped connections, and 5xx. A 4xx is a
  request the endpoint will reject again, so it raises immediately.
* **Streaming** — token deltas are pushed to ``on_token`` (sync or async) as
  they arrive, and the full assembled text is still returned. Once a stream has
  emitted a token, a mid-stream fault does *not* retry: replaying the attempt
  would emit those tokens twice.
* **Secrets** — the API key arrives already resolved from the config layer.
  This module never reads the environment and never logs the key.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import httpx

from museai.core.config import EndpointConfig
from museai.core.logging_setup import get_llm_logger
from museai.llm.tokenizer import count_message_tokens, count_tokens

# The standard chat-completions path, appended to an endpoint root.
_CHAT_COMPLETIONS_PATH = "chat/completions"

# Retry policy. Three attempts means two sleeps.
MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS: tuple[float, ...] = (5.0, 15.0)

# Faults that mean "the endpoint was unreachable or gave up", not "the request
# was wrong". These are worth retrying; a 4xx never is.
_TRANSIENT_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
)

_SSE_DATA_PREFIX = "data:"
_SSE_TERMINATOR = "[DONE]"

# Logged message bodies are previews, not transcripts: llm_io.log is for
# debugging a call, and full prose bodies would swamp it.
_LOG_CONTENT_PREVIEW_CHARS = 500

TokenCallback = Callable[[str], Any | Awaitable[Any]]


@dataclass
class LLMResponse:
    """The result of one completed inference call."""

    text: str
    raw: dict
    model_name: str
    endpoint_base_url: str
    tokens_in: int
    tokens_out: int
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str | None = None


class LLMCallError(Exception):
    """A call that failed for good: retries exhausted, 4xx, or a bad response."""


class _TransientStatusError(Exception):
    """Internal: a 5xx worth retrying. Never escapes this module."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body}")
        self.status_code = status_code


# Everything the retry loop is willing to attempt again.
_RETRYABLE = _TRANSIENT_EXCEPTIONS + (_TransientStatusError,)


def resolve_inference_url(base_url: str) -> str:
    """Resolve an endpoint ``base_url`` to a full chat-completions URL.

    A URL that already names a completions path is used as-is; a bare endpoint
    root gets the standard chat-completions path appended. No host, port, or
    provider is special-cased — the only thing inspected is whether the path
    already terminates in a completions endpoint.
    """
    trimmed = base_url.strip()
    if not trimmed:
        raise ValueError("endpoint base_url is empty")

    parts = urlsplit(trimmed)
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"endpoint base_url is not an absolute URL: {base_url!r}")

    path = parts.path.rstrip("/")
    if path.rsplit("/", 1)[-1] == "completions":
        return urlunsplit(parts._replace(path=path))

    return urlunsplit(parts._replace(path=f"{path}/{_CHAT_COMPLETIONS_PATH}"))


def _strip_secrets(url: str) -> str:
    """Drop credentials and query parameters from a URL before logging it.

    Some endpoints carry keys in userinfo or a query string; neither belongs in
    a log file.
    """
    parts = urlsplit(url)
    netloc = parts.hostname or ""
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _safe_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Render messages for the log: roles kept, long bodies truncated."""
    safe: list[dict] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and len(content) > _LOG_CONTENT_PREVIEW_CHARS:
            content = f"{content[:_LOG_CONTENT_PREVIEW_CHARS]}…[+{len(content) - _LOG_CONTENT_PREVIEW_CHARS} chars]"
        entry = {"role": message.get("role"), "content": content}
        if message.get("tool_calls"):
            entry["tool_calls"] = message["tool_calls"]
        if message.get("name"):
            entry["name"] = message["name"]
        safe.append(entry)
    return safe


def _build_request(
    endpoint: EndpointConfig,
    messages: Sequence[Mapping[str, Any]],
    *,
    stream: bool,
    temperature: float | None,
    max_tokens: int | None,
    tools: Sequence[Mapping[str, Any]] | None,
    tool_choice: Any | None,
    response_format: Mapping[str, Any] | None,
    extra_body: Mapping[str, Any] | None,
    extra_headers: Mapping[str, str] | None,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Assemble the (url, headers, body) triple for one chat-completions call.

    This is the adapter seam. Optional parameters are omitted from the body
    entirely when ``None`` rather than sent as nulls, because some endpoints
    reject an explicit null where they would accept an absent key.
    """
    url = resolve_inference_url(endpoint.base_url)

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"
    if extra_headers:
        headers.update(extra_headers)

    body: dict[str, Any] = {
        "model": endpoint.model_name,
        "messages": [dict(m) for m in messages],
        "temperature": endpoint.temperature if temperature is None else temperature,
        "stream": stream,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if tools is not None:
        body["tools"] = [dict(t) for t in tools]
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if response_format is not None:
        body["response_format"] = dict(response_format)
    if extra_body:
        body.update(extra_body)

    return url, headers, body


def _normalise_tool_calls(raw_calls: Any) -> list[dict]:
    """Coerce a response's ``tool_calls`` to a plain list of dicts."""
    if not raw_calls:
        return []
    return [dict(call) for call in raw_calls]


def _parse_response(payload: Mapping[str, Any]) -> tuple[str, list[dict], str | None]:
    """Extract (text, tool_calls, finish_reason) from a non-streaming response."""
    choices = payload.get("choices")
    if not choices:
        raise LLMCallError(f"response contained no choices: {payload!r}")

    choice = choices[0]
    message = choice.get("message") or {}

    # A tool-calling reply legitimately has null content.
    text = message.get("content") or ""
    tool_calls = _normalise_tool_calls(message.get("tool_calls"))
    return text, tool_calls, choice.get("finish_reason")


def _accumulate_tool_call_deltas(
    accumulator: dict[int, dict], deltas: Sequence[Mapping[str, Any]]
) -> None:
    """Fold streamed tool-call fragments into ``accumulator``, keyed by index.

    Endpoints stream a tool call across chunks: the id and function name arrive
    first, then the JSON arguments in pieces. Text fields concatenate.
    """
    for delta in deltas:
        index = delta.get("index", 0)
        entry = accumulator.setdefault(
            index,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if delta.get("id"):
            entry["id"] = delta["id"]
        if delta.get("type"):
            entry["type"] = delta["type"]
        function = delta.get("function") or {}
        if function.get("name"):
            entry["function"]["name"] += function["name"]
        if function.get("arguments"):
            entry["function"]["arguments"] += function["arguments"]


async def _emit_token(on_token: TokenCallback | None, token: str) -> None:
    """Invoke a token callback that may be sync or async."""
    if on_token is None or not token:
        return
    result = on_token(token)
    if inspect.isawaitable(result):
        await result


async def _parse_stream(
    lines: Any,
    on_token: TokenCallback | None,
    emitted: list[bool],
) -> tuple[str, list[dict], str | None, dict]:
    """Consume an SSE line stream into (text, tool_calls, finish_reason, raw).

    ``emitted`` is a one-element flag the caller reads to decide whether a
    mid-stream fault may be retried: once a token has reached ``on_token``,
    replaying the attempt would duplicate it.
    """
    chunks: list[dict] = []
    pieces: list[str] = []
    tool_call_parts: dict[int, dict] = {}
    finish_reason: str | None = None

    async for line in lines:
        line = line.strip()
        if not line or not line.startswith(_SSE_DATA_PREFIX):
            continue

        data = line[len(_SSE_DATA_PREFIX) :].strip()
        if data == _SSE_TERMINATOR:
            break

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as exc:
            raise LLMCallError(f"malformed JSON in stream chunk: {data!r}") from exc

        chunks.append(chunk)
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                pieces.append(content)
                emitted[0] = True
                await _emit_token(on_token, content)
            if delta.get("tool_calls"):
                _accumulate_tool_call_deltas(tool_call_parts, delta["tool_calls"])
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    tool_calls = [tool_call_parts[i] for i in sorted(tool_call_parts)]
    return "".join(pieces), tool_calls, finish_reason, {"stream": True, "chunks": chunks}


def _raise_for_status(response: httpx.Response, body: str) -> None:
    """Turn a non-2xx status into the right kind of error.

    5xx becomes a retryable internal signal; 4xx is a hard failure, since the
    endpoint will reject the identical request again.
    """
    status = response.status_code
    if status < 400:
        return
    if status >= 500:
        raise _TransientStatusError(status, body[:500])
    raise LLMCallError(f"endpoint returned HTTP {status}: {body[:500]}")


async def _attempt_stream(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    on_token: TokenCallback | None,
    emitted: list[bool],
) -> tuple[str, list[dict], str | None, dict]:
    async with client.stream("POST", url, headers=headers, json=body) as response:
        if response.status_code >= 400:
            body_text = (await response.aread()).decode(errors="replace")
            _raise_for_status(response, body_text)
        return await _parse_stream(response.aiter_lines(), on_token, emitted)


async def _attempt_once(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
) -> tuple[str, list[dict], str | None, dict]:
    response = await client.post(url, headers=headers, json=body)
    if response.status_code >= 400:
        _raise_for_status(response, response.text)

    try:
        payload = response.json()
    except ValueError as exc:
        raise LLMCallError(f"response was not valid JSON: {response.text[:500]!r}") from exc

    text, tool_calls, finish_reason = _parse_response(payload)
    return text, tool_calls, finish_reason, payload


async def call_llm(
    endpoint: EndpointConfig,
    messages: Sequence[Mapping[str, Any]],
    *,
    stream: bool = False,
    on_token: TokenCallback | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
    tool_choice: Any | None = None,
    response_format: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    extra_headers: Mapping[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    retry_backoff: Sequence[float] | None = None,
) -> LLMResponse:
    """Call the configured endpoint once, retrying transient faults.

    ``transport`` and ``retry_backoff`` exist so tests can inject a mock
    transport and collapse the backoff sleeps; production callers leave both
    unset and get real HTTP with the 5s/15s policy.

    Raises ``LLMCallError`` on a 4xx, a malformed response, or once the retry
    budget is exhausted.
    """
    url, headers, body = _build_request(
        endpoint,
        messages,
        stream=stream,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_format,
        extra_body=extra_body,
        extra_headers=extra_headers,
    )

    backoff = tuple(retry_backoff) if retry_backoff is not None else DEFAULT_BACKOFF_SECONDS
    timeout = httpx.Timeout(float(endpoint.request_timeout))
    safe_url = _strip_secrets(url)
    attempt = 0
    last_error: Exception | None = None

    while attempt < MAX_ATTEMPTS:
        attempt += 1
        emitted = [False]
        _log_request(safe_url, endpoint, messages, stream, attempt, max_tokens)
        try:
            async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
                if stream:
                    text, tool_calls, finish_reason, raw = await _attempt_stream(
                        client, url, headers, body, on_token, emitted
                    )
                else:
                    text, tool_calls, finish_reason, raw = await _attempt_once(
                        client, url, headers, body
                    )
        except LLMCallError as exc:
            _log_error(safe_url, endpoint, stream, attempt, str(exc))
            raise
        except _RETRYABLE as exc:
            last_error = exc
            if emitted[0]:
                # Tokens already reached the caller; replaying would double them.
                message = f"stream failed after emitting tokens: {exc}"
                _log_error(safe_url, endpoint, stream, attempt, message)
                raise LLMCallError(message) from exc
            _log_error(
                safe_url, endpoint, stream, attempt, f"transient: {exc}", retrying=True
            )
            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(backoff[min(attempt - 1, len(backoff) - 1)])
            continue

        tokens_in = count_message_tokens(
            messages, endpoint.tokenizer_family, endpoint.model_name
        )
        tokens_out = count_tokens(text, endpoint.tokenizer_family, endpoint.model_name)
        _log_response(
            safe_url,
            endpoint,
            stream,
            attempt,
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tool_calls=len(tool_calls),
            finish_reason=finish_reason,
        )
        return LLMResponse(
            text=text,
            raw=raw,
            model_name=endpoint.model_name,
            endpoint_base_url=endpoint.base_url,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    summary = f"{MAX_ATTEMPTS} attempts failed, last error: {last_error}"
    _log_error(safe_url, endpoint, stream, attempt, summary)
    raise LLMCallError(summary) from last_error


def _emit(record: dict[str, Any], *, error: bool = False) -> None:
    """Write one JSON record to ``logs/llm_io.log``."""
    logger = get_llm_logger()
    line = json.dumps(record, ensure_ascii=False, default=str)
    logger.error(line) if error else logger.info(line)


def _truncate(text: str) -> str:
    if len(text) <= _LOG_CONTENT_PREVIEW_CHARS:
        return text
    overflow = len(text) - _LOG_CONTENT_PREVIEW_CHARS
    return f"{text[:_LOG_CONTENT_PREVIEW_CHARS]}…[+{overflow} chars]"


def _log_request(
    safe_url: str,
    endpoint: EndpointConfig,
    messages: Sequence[Mapping[str, Any]],
    stream: bool,
    attempt: int,
    max_tokens: int | None,
) -> None:
    """Record a request as it goes out, before the endpoint has answered.

    Written per attempt, so a retry logs a second request. The URL is stripped of
    credentials and the API key is never included.
    """
    _emit(
        {
            "event": "request",
            "url": safe_url,
            "model": endpoint.model_name,
            "stream": stream,
            "attempt": attempt,
            "max_tokens": max_tokens,
            "messages": _safe_messages(messages),
        }
    )


def _log_response(
    safe_url: str,
    endpoint: EndpointConfig,
    stream: bool,
    attempt: int,
    *,
    text: str,
    tokens_in: int,
    tokens_out: int,
    tool_calls: int,
    finish_reason: str | None,
) -> None:
    """Record one completed response.

    Exactly one record per returned message. A streamed call logs here once, on
    the assembled text — never per token, which would drown the log in fragments.
    """
    _emit(
        {
            "event": "response",
            "url": safe_url,
            "model": endpoint.model_name,
            "stream": stream,
            "attempt": attempt,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "text": _truncate(text),
        }
    )


def _log_error(
    safe_url: str,
    endpoint: EndpointConfig,
    stream: bool,
    attempt: int,
    error: str,
    *,
    retrying: bool = False,
) -> None:
    """Record a failed attempt, noting whether another one follows."""
    _emit(
        {
            "event": "error",
            "url": safe_url,
            "model": endpoint.model_name,
            "stream": stream,
            "attempt": attempt,
            "retrying": retrying,
            "error": error,
        },
        error=True,
    )
