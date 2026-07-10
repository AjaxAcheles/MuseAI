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
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import httpx

from museai.core import chat_log
from museai.core.config import EndpointConfig
from museai.core.logging_setup import get_llm_logger
from museai.core.stream_bus import bus
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
    # Chain-of-thought the model exposed, kept apart from `text` so no caller
    # ever mistakes deliberation for prose. Empty for non-reasoning models.
    thinking: str = ""


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


# OpenAI-compatible endpoints expose chain-of-thought two ways: a dedicated
# field beside `content` (DeepSeek/vLLM use `reasoning_content`, OpenRouter and
# some Ollama builds use `reasoning`), or a literal <think>...</think> block
# opening the content itself (Ollama serving qwen3 / r1 distills). Both are
# handled; a model that emits neither simply has empty thinking.
_REASONING_FIELDS = ("reasoning_content", "reasoning")
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _reasoning_text(container: Mapping[str, Any]) -> str:
    """The reasoning field of one message or delta, whatever it is called."""
    for name in _REASONING_FIELDS:
        value = container.get(name)
        if isinstance(value, str) and value:
            return value
    return ""


def _split_think_block(text: str) -> tuple[str, str]:
    """Split a complete content body into (thinking, response).

    Only a tag opening the body counts — a <think> mid-prose is prose. An
    unclosed block means the model spent its whole reply deliberating, so it
    is all thinking and the response is empty.
    """
    lead = len(text) - len(text.lstrip())
    stripped = text[lead:]
    if not stripped.startswith(_THINK_OPEN):
        return "", text
    rest = stripped[len(_THINK_OPEN) :]
    close = rest.find(_THINK_CLOSE)
    if close == -1:
        return rest.strip(), ""
    return rest[:close].strip(), rest[close + len(_THINK_CLOSE) :].lstrip("\n")


class _ThinkTagSplitter:
    """Route a streamed content sequence into thinking and response pieces.

    Stateful because the tags can arrive split across chunk boundaries: the
    splitter buffers until it can tell whether the stream opens with
    ``<think>``, and while inside a block it holds back a tag-sized tail in
    case ``</think>`` straddles two chunks. ``feed`` returns
    ``(kind, text)`` pieces; ``flush`` drains whatever a finished stream left
    buffered.
    """

    def __init__(self) -> None:
        self._mode = "start"  # start -> thinking? -> response
        self._buffer = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        pieces: list[tuple[str, str]] = []
        if self._mode == "response":
            if text:
                pieces.append(("response", text))
            return pieces

        self._buffer += text
        if self._mode == "start":
            candidate = self._buffer.lstrip()
            if _THINK_OPEN.startswith(candidate):
                return pieces  # still ambiguous; keep buffering
            if not candidate.startswith(_THINK_OPEN):
                self._mode = "response"
                pieces.append(("response", self._buffer))
                self._buffer = ""
                return pieces
            self._mode = "thinking"
            self._buffer = candidate[len(_THINK_OPEN) :]

        close = self._buffer.find(_THINK_CLOSE)
        if close != -1:
            thinking = self._buffer[:close]
            remainder = self._buffer[close + len(_THINK_CLOSE) :].lstrip("\n")
            self._mode = "response"
            self._buffer = ""
            if thinking:
                pieces.append(("thinking", thinking))
            if remainder:
                pieces.append(("response", remainder))
            return pieces

        # Hold back one tag-length of tail in case </think> is split.
        safe = len(self._buffer) - (len(_THINK_CLOSE) - 1)
        if safe > 0:
            pieces.append(("thinking", self._buffer[:safe]))
            self._buffer = self._buffer[safe:]
        return pieces

    def flush(self) -> list[tuple[str, str]]:
        buffered, self._buffer = self._buffer, ""
        if not buffered:
            return []
        # A stream that ended mid-block was all deliberation; one that ended
        # while the opening tag was still ambiguous was ordinary prose.
        kind = "thinking" if self._mode == "thinking" else "response"
        self._mode = "response"
        return [(kind, buffered)]


def _parse_response(
    payload: Mapping[str, Any],
) -> tuple[str, str, list[dict], str | None]:
    """Extract (text, thinking, tool_calls, finish_reason) from a response."""
    choices = payload.get("choices")
    if not choices:
        raise LLMCallError(f"response contained no choices: {payload!r}")

    choice = choices[0]
    message = choice.get("message") or {}

    # A tool-calling reply legitimately has null content.
    text = message.get("content") or ""
    thinking = _reasoning_text(message)
    tagged_thinking, text = _split_think_block(text)
    if tagged_thinking:
        thinking = f"{thinking}\n{tagged_thinking}".strip() if thinking else tagged_thinking
    tool_calls = _normalise_tool_calls(message.get("tool_calls"))
    return text, thinking, tool_calls, choice.get("finish_reason")


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


ChatCallback = Callable[[str, str], Awaitable[None]]


async def _parse_stream(
    lines: Any,
    on_token: TokenCallback | None,
    emitted: list[bool],
    on_chat: ChatCallback | None = None,
) -> tuple[str, str, list[dict], str | None, dict]:
    """Consume an SSE line stream into (text, thinking, tool_calls, finish_reason, raw).

    ``emitted`` is a one-element flag the caller reads to decide whether a
    mid-stream fault may be retried: once a piece has reached ``on_token`` or
    ``on_chat``, replaying the attempt would duplicate it. Reasoning — a
    ``reasoning_content``/``reasoning`` delta or a leading <think> block — is
    routed to thinking and never reaches ``on_token``, which receives prose only.
    """
    chunks: list[dict] = []
    pieces: list[str] = []
    thinking_pieces: list[str] = []
    splitter = _ThinkTagSplitter()
    tool_call_parts: dict[int, dict] = {}
    finish_reason: str | None = None

    async def _route(kind: str, text: str) -> None:
        emitted[0] = True
        if kind == "response":
            pieces.append(text)
            await _emit_token(on_token, text)
        else:
            thinking_pieces.append(text)
        if on_chat is not None:
            await on_chat(kind, text)

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
            reasoning = _reasoning_text(delta)
            if reasoning:
                await _route("thinking", reasoning)
            content = delta.get("content")
            if content:
                for kind, piece in splitter.feed(content):
                    await _route(kind, piece)
            if delta.get("tool_calls"):
                _accumulate_tool_call_deltas(tool_call_parts, delta["tool_calls"])
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    for kind, piece in splitter.flush():
        await _route(kind, piece)

    tool_calls = [tool_call_parts[i] for i in sorted(tool_call_parts)]
    return (
        "".join(pieces),
        "".join(thinking_pieces).strip(),
        tool_calls,
        finish_reason,
        {"stream": True, "chunks": chunks},
    )


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
    on_chat: ChatCallback | None,
) -> tuple[str, str, list[dict], str | None, dict]:
    async with client.stream("POST", url, headers=headers, json=body) as response:
        if response.status_code >= 400:
            body_text = (await response.aread()).decode(errors="replace")
            _raise_for_status(response, body_text)
        return await _parse_stream(response.aiter_lines(), on_token, emitted, on_chat)


async def _attempt_once(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
) -> tuple[str, str, list[dict], str | None, dict]:
    response = await client.post(url, headers=headers, json=body)
    if response.status_code >= 400:
        _raise_for_status(response, response.text)

    try:
        payload = response.json()
    except ValueError as exc:
        raise LLMCallError(f"response was not valid JSON: {response.text[:500]!r}") from exc

    text, thinking, tool_calls, finish_reason = _parse_response(payload)
    return text, thinking, tool_calls, finish_reason, payload


def _chat_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Full-fidelity copies of the outgoing messages for the chat transcript.

    Unlike ``_safe_messages`` nothing is truncated: the View Chat page's whole
    point is showing exactly what the model was sent. Secrets live in headers,
    never in message bodies, so there is nothing to strip here.
    """
    return [dict(m) for m in messages]


# Calls that have published ``chat_start`` but not yet ``chat_end``, with the
# thinking and response text accumulated so far. The transcript on disk only
# knows a call once it ends; this registry is what lets a View Chat page that
# loads mid-call show the tokens that streamed before it arrived. ``seq``
# counts the tokens published for the call so that page can also discard any
# it already received through the snapshot.
_LIVE_CALLS: dict[str, dict[str, Any]] = {}


def live_chat_calls() -> list[dict[str, Any]]:
    """Snapshot every in-flight call for the View Chat history endpoint."""
    return [dict(entry) for entry in _LIVE_CALLS.values()]


async def _chat_start(
    call_id: str, agent: str, endpoint: EndpointConfig, messages: Sequence[Mapping[str, Any]], stream: bool
) -> None:
    data = {
        "id": call_id,
        "ts": time.time(),
        "agent": agent,
        "model": endpoint.model_name,
        "stream": stream,
        "messages": _chat_messages(messages),
    }
    chat_log.record({"event": "chat_start", **data})
    _LIVE_CALLS[call_id] = {**data, "thinking": "", "text": "", "seq": 0}
    await bus.publish("chat_start", data)


async def _chat_end(
    call_id: str,
    agent: str,
    endpoint: EndpointConfig,
    *,
    ok: bool,
    thinking: str = "",
    text: str = "",
    tokens_in: int = 0,
    tokens_out: int = 0,
    tool_calls: int = 0,
    finish_reason: str | None = None,
    attempt: int = 1,
    error: str | None = None,
) -> None:
    data = {
        "id": call_id,
        "ts": time.time(),
        "agent": agent,
        "model": endpoint.model_name,
        "ok": ok,
        "thinking": thinking,
        "text": text,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "attempt": attempt,
        "error": error,
    }
    chat_log.record({"event": "chat_end", **data})
    _LIVE_CALLS.pop(call_id, None)
    await bus.publish("chat_end", data)


async def call_llm(
    endpoint: EndpointConfig,
    messages: Sequence[Mapping[str, Any]],
    *,
    agent: str = "system",
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

    ``agent`` names who is asking — drafter, beat_planner, critic — purely for
    the chat transcript and stream; it never reaches the wire. Every call
    publishes a ``chat_start``/``chat_end`` pair (with ``chat_token`` deltas
    while streaming) so the View Chat page shows all model traffic.

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

    call_id = uuid.uuid4().hex[:12]
    await _chat_start(call_id, agent, endpoint, messages, stream)

    async def on_chat(kind: str, text: str) -> None:
        live = _LIVE_CALLS.get(call_id)
        seq = 0
        if live is not None:
            live["seq"] += 1
            seq = live["seq"]
            live["thinking" if kind == "thinking" else "text"] += text
        await bus.publish(
            "chat_token",
            {"id": call_id, "agent": agent, "kind": kind, "text": text, "seq": seq},
        )

    while attempt < MAX_ATTEMPTS:
        attempt += 1
        emitted = [False]
        _log_request(safe_url, endpoint, messages, stream, attempt, max_tokens)
        try:
            async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
                if stream:
                    text, thinking, tool_calls, finish_reason, raw = await _attempt_stream(
                        client, url, headers, body, on_token, emitted, on_chat
                    )
                else:
                    text, thinking, tool_calls, finish_reason, raw = await _attempt_once(
                        client, url, headers, body
                    )
        except LLMCallError as exc:
            _log_error(safe_url, endpoint, stream, attempt, str(exc))
            await _chat_end(call_id, agent, endpoint, ok=False, attempt=attempt, error=str(exc))
            raise
        except _RETRYABLE as exc:
            last_error = exc
            if emitted[0]:
                # Tokens already reached the caller; replaying would double them.
                message = f"stream failed after emitting tokens: {exc}"
                _log_error(safe_url, endpoint, stream, attempt, message)
                await _chat_end(
                    call_id, agent, endpoint, ok=False, attempt=attempt, error=message
                )
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
        await _chat_end(
            call_id,
            agent,
            endpoint,
            ok=True,
            thinking=thinking,
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tool_calls=len(tool_calls),
            finish_reason=finish_reason,
            attempt=attempt,
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
            thinking=thinking,
        )

    summary = f"{MAX_ATTEMPTS} attempts failed, last error: {last_error}"
    _log_error(safe_url, endpoint, stream, attempt, summary)
    await _chat_end(call_id, agent, endpoint, ok=False, attempt=attempt, error=summary)
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
