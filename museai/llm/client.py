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
  retry: connection/read/write timeouts, dropped connections, 5xx, and the two
  4xx that mean "try later" (408, 429). Any other 4xx is a request the endpoint
  will reject again, so it raises immediately.
* **Streaming** — token deltas are pushed to ``on_token`` (sync or async) as
  they arrive, and the full assembled text is still returned. A mid-stream
  fault retries even after tokens have been emitted: a ``chat_restart`` event
  tells the live view to discard the partial, and the attempt's text is
  rebuilt from scratch. Callback consumers can use ``on_restart`` to discard
  tokens from the failed attempt too.
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

# Retry policy lives on the endpoint: endpoint.max_attempts (N attempts
# means N-1 sleeps) and endpoint.retry_backoff_seconds.

# Faults that mean "the endpoint was unreachable or gave up", not "the request
# was wrong". These are worth retrying; a 4xx (other than 408/429) never is.
_TRANSIENT_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.ReadError,
    httpx.WriteError,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)

_SSE_DATA_PREFIX = "data:"
_SSE_TERMINATOR = "[DONE]"

# Logged message bodies are previews, not transcripts: llm_io.log is for
# debugging a call, and full prose bodies would swamp it.
_LOG_CONTENT_PREVIEW_CHARS = 500

TokenCallback = Callable[[str], Any | Awaitable[Any]]
RestartCallback = Callable[[], Any | Awaitable[Any]]


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
    # The server's own counts, None when it reported none. `tokens_in`/`tokens_out`
    # above fall back to our tokenizer's estimate and so are always populated;
    # these stay None precisely so a caller can tell measurement from guess. A
    # truncation diagnosis needs that distinction — see `truncation_remedy`.
    served_prompt_tokens: int | None = None
    served_completion_tokens: int | None = None
    # The output-token cap this call actually ran under — `endpoint.max_output_tokens`
    # when set, otherwise the room-derived cap `call_llm` computes when only
    # `context_window` is set. A truncation diagnosis needs the cap that was
    # actually in force, not just the one the config named explicitly, or a role
    # relying on the derived cap gets misdiagnosed as a context-window overrun.
    effective_max_tokens: int | None = None


class LLMCallError(Exception):
    """A call that failed for good: retries exhausted, 4xx, or a bad response."""


class _TransientStatusError(Exception):
    """Internal: a status worth retrying (5xx/408/429). Never escapes this module."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body}")
        self.status_code = status_code


class _EmptyCompletionError(Exception):
    """Internal: a completion with no text and no tool calls, when the caller
    opted into ``retry_on_empty``. Never escapes this module."""


class _TransientStreamError(Exception):
    """A 200/SSE response ended malformed or before a terminal completion."""


# Everything the retry loop is willing to attempt again.
_RETRYABLE = _TRANSIENT_EXCEPTIONS + (
    _TransientStatusError,
    _EmptyCompletionError,
    _TransientStreamError,
)


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
    if stream and endpoint.stream_usage:
        # A streamed reply carries no usage block unless asked. Without it the
        # only token counts available are our own tokenizer's estimates, which
        # cannot contradict a wrong context_window — see _usage_counts.
        body["stream_options"] = {"include_usage": True}
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
    """Validate a response's ``tool_calls`` as OpenAI-style function calls."""
    if not raw_calls:
        return []
    if not isinstance(raw_calls, list):
        raise LLMCallError(
            f"message.tool_calls must be an array, got {type(raw_calls).__name__}"
        )
    calls: list[dict] = []
    seen_ids: set[str] = set()
    for index, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, Mapping):
            raise LLMCallError(
                f"message.tool_calls[{index}] must be an object, "
                f"got {type(raw_call).__name__}"
            )
        call = dict(raw_call)
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise LLMCallError(f"message.tool_calls[{index}].id must be a non-empty string")
        if call_id in seen_ids:
            raise LLMCallError(f"message.tool_calls contains duplicate id {call_id!r}")
        seen_ids.add(call_id)
        if call.get("type", "function") != "function":
            raise LLMCallError(
                f"message.tool_calls[{index}].type must be 'function'"
            )
        function = call.get("function")
        if not isinstance(function, Mapping):
            raise LLMCallError(
                f"message.tool_calls[{index}].function must be an object"
            )
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise LLMCallError(
                f"message.tool_calls[{index}].function.name must be a non-empty string"
            )
        arguments = function.get("arguments", "")
        if arguments is not None and not isinstance(arguments, (str, Mapping)):
            raise LLMCallError(
                f"message.tool_calls[{index}].function.arguments has unusable type "
                f"{type(arguments).__name__}"
            )
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return calls


_FINISH_REASONS = frozenset(
    {"stop", "length", "tool_calls", "content_filter", "function_call"}
)


def _normalise_finish_reason(raw: Any) -> str:
    if not isinstance(raw, str) or raw not in _FINISH_REASONS:
        raise LLMCallError(f"response had invalid finish_reason {raw!r}")
    return raw


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
    if not stripped.lower().startswith(_THINK_OPEN):
        return "", text
    rest = stripped[len(_THINK_OPEN) :]
    close = rest.lower().find(_THINK_CLOSE)
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
            lowered = candidate.lower()
            if _THINK_OPEN.startswith(lowered):
                return pieces  # still ambiguous; keep buffering
            if not lowered.startswith(_THINK_OPEN):
                self._mode = "response"
                pieces.append(("response", self._buffer))
                self._buffer = ""
                return pieces
            self._mode = "thinking"
            self._buffer = candidate[len(_THINK_OPEN) :]

        close = self._buffer.lower().find(_THINK_CLOSE)
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


def _usage_counts(raw: Mapping[str, Any]) -> tuple[int | None, int | None]:
    """The server's own ``(prompt_tokens, completion_tokens)``, if it reported them.

    These are the only authoritative token counts in the system. Everything else
    — ``context_token_budget``, the pruner's drop loop, the ``max_tokens``
    remainder — is computed from :mod:`museai.llm.tokenizer`, which for
    ``char_heuristic`` is a length/4 approximation and for ``tiktoken`` is the
    wrong BPE table whenever the endpoint is not an OpenAI model. Neither can
    detect that a server is serving a *smaller window than it was configured
    for*, because both describe the prompt we sent rather than the prompt the
    server accepted. The usage block does, which is what makes a truncation
    diagnosable instead of merely reportable.

    Returns ``(None, None)`` when the endpoint omits usage, so callers fall back
    to the estimate rather than treating a missing count as zero.

    Handles both response shapes: a non-streamed payload carries ``usage`` at the
    top level, while a streamed one is ``{"stream": True, "chunks": [...]}`` with
    usage in a trailing chunk that has an empty ``choices`` list.
    """

    def _read(block: Any) -> tuple[int | None, int | None]:
        if not isinstance(block, Mapping):
            return None, None
        prompt = block.get("prompt_tokens")
        completion = block.get("completion_tokens")
        # A non-integer (or bool, which int accepts) count is a broken endpoint,
        # not something to propagate into a budget calculation.
        if not isinstance(prompt, int) or isinstance(prompt, bool) or prompt < 0:
            prompt = None
        if (
            not isinstance(completion, int)
            or isinstance(completion, bool)
            or completion < 0
        ):
            completion = None
        return prompt, completion

    if raw.get("stream"):
        # Last writer wins: the usage chunk is emitted once, at the end.
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        for chunk in raw.get("chunks") or []:
            if not isinstance(chunk, Mapping):
                continue
            found_prompt, found_completion = _read(chunk.get("usage"))
            if found_prompt is not None:
                prompt_tokens = found_prompt
            if found_completion is not None:
                completion_tokens = found_completion
        return prompt_tokens, completion_tokens

    return _read(raw.get("usage"))


def _parse_response(
    payload: Mapping[str, Any],
) -> tuple[str, str, list[dict], str | None]:
    """Extract (text, thinking, tool_calls, finish_reason) from a response."""
    if not isinstance(payload, Mapping):
        raise LLMCallError(
            f"response body must be a JSON object, got {type(payload).__name__}"
        )
    if payload.get("error"):
        raise LLMCallError(f"endpoint returned an error object: {payload['error']!r}")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMCallError(f"response contained no choices: {payload!r}")

    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise LLMCallError(
            f"response choice 0 must be an object, got {type(choice).__name__}"
        )
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise LLMCallError("response choice 0 contained no message object")

    # A tool-calling reply legitimately has null content.
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise LLMCallError(
            f"message.content must be a string or null, got {type(content).__name__}"
        )
    text = content or ""
    thinking = _reasoning_text(message)
    tagged_thinking, text = _split_think_block(text)
    if tagged_thinking:
        thinking = f"{thinking}\n{tagged_thinking}".strip() if thinking else tagged_thinking
    tool_calls = _normalise_tool_calls(message.get("tool_calls"))
    finish_reason = _normalise_finish_reason(choice.get("finish_reason"))
    if finish_reason == "tool_calls" and not tool_calls:
        raise LLMCallError("finish_reason='tool_calls' but message.tool_calls was empty")
    return text, thinking, tool_calls, finish_reason


def _accumulate_tool_call_deltas(
    accumulator: dict[int, dict], deltas: Sequence[Mapping[str, Any]]
) -> None:
    """Fold streamed tool-call fragments into ``accumulator``, keyed by index.

    Endpoints stream a tool call across chunks: the id and function name arrive
    first, then the JSON arguments in pieces. Text fields concatenate.
    """
    if not isinstance(deltas, list):
        raise LLMCallError(
            f"delta.tool_calls must be an array, got {type(deltas).__name__}"
        )
    if len(deltas) > 1 and any(
        not isinstance(delta, Mapping) or "index" not in delta for delta in deltas
    ):
        raise LLMCallError("parallel streamed tool calls must carry distinct indices")
    for position, delta in enumerate(deltas):
        if not isinstance(delta, Mapping):
            raise LLMCallError(
                f"delta.tool_calls[{position}] must be an object, "
                f"got {type(delta).__name__}"
            )
        index = delta.get("index", 0)
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise LLMCallError(
                f"delta.tool_calls[{position}].index must be a non-negative integer"
            )
        entry = accumulator.setdefault(
            index,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if delta.get("id"):
            incoming_id = delta["id"]
            if not isinstance(incoming_id, str):
                raise LLMCallError(
                    f"delta.tool_calls[{position}].id must be a string"
                )
            if entry["id"] and entry["id"] != incoming_id:
                raise LLMCallError(
                    f"streamed tool-call index {index} was reused for multiple ids"
                )
            entry["id"] = incoming_id
        if delta.get("type"):
            if delta["type"] != "function":
                raise LLMCallError("streamed tool-call type must be 'function'")
            entry["type"] = delta["type"]
        function = delta.get("function") or {}
        if not isinstance(function, Mapping):
            raise LLMCallError(
                f"delta.tool_calls[{position}].function must be an object"
            )
        if function.get("name"):
            if not isinstance(function["name"], str):
                raise LLMCallError("streamed tool-call function.name must be a string")
            entry["function"]["name"] += function["name"]
        if function.get("arguments"):
            if not isinstance(function["arguments"], str):
                raise LLMCallError("streamed tool-call arguments must be a string")
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
    saw_done = False

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
            saw_done = True
            break

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as exc:
            raise _TransientStreamError(
                f"malformed JSON in stream chunk: {data!r}"
            ) from exc

        if not isinstance(chunk, Mapping):
            raise _TransientStreamError(
                f"stream chunk must be a JSON object, got {type(chunk).__name__}"
            )
        if chunk.get("error"):
            raise _TransientStreamError(
                f"endpoint streamed an error object: {chunk['error']!r}"
            )

        chunks.append(chunk)
        choices = chunk.get("choices") or []
        if not isinstance(choices, list):
            raise _TransientStreamError("stream chunk choices must be an array")
        for position, choice in enumerate(choices):
            if not isinstance(choice, Mapping):
                raise _TransientStreamError(
                    f"stream choice {position} must be an object"
                )
            delta = choice.get("delta") or {}
            if not isinstance(delta, Mapping):
                raise _TransientStreamError(
                    f"stream choice {position}.delta must be an object"
                )
            reasoning = _reasoning_text(delta)
            if reasoning:
                await _route("thinking", reasoning)
            content = delta.get("content")
            if content is not None and not isinstance(content, str):
                raise _TransientStreamError(
                    f"stream choice {position}.delta.content must be a string or null"
                )
            if content:
                for kind, piece in splitter.feed(content):
                    await _route(kind, piece)
            if delta.get("tool_calls"):
                try:
                    _accumulate_tool_call_deltas(tool_call_parts, delta["tool_calls"])
                except LLMCallError as exc:
                    raise _TransientStreamError(str(exc)) from exc
            if choice.get("finish_reason"):
                try:
                    finish_reason = _normalise_finish_reason(choice["finish_reason"])
                except LLMCallError as exc:
                    raise _TransientStreamError(str(exc)) from exc

    for kind, piece in splitter.flush():
        await _route(kind, piece)

    if finish_reason is None:
        terminal = "[DONE]" if saw_done else "clean EOF"
        raise _TransientStreamError(
            f"stream ended at {terminal} without a terminal finish_reason"
        )
    try:
        tool_calls = _normalise_tool_calls(
            [tool_call_parts[i] for i in sorted(tool_call_parts)]
        )
    except LLMCallError as exc:
        raise _TransientStreamError(str(exc)) from exc
    if finish_reason == "tool_calls" and not tool_calls:
        raise _TransientStreamError(
            "finish_reason='tool_calls' but the stream contained no tool calls"
        )
    return (
        "".join(pieces),
        "".join(thinking_pieces).strip(),
        tool_calls,
        finish_reason,
        {"stream": True, "chunks": chunks},
    )


# The two 4xx statuses that mean "try again later" rather than "the request is
# wrong": request timeout and rate limit. Everything else in the 4xx range is a
# request the endpoint will reject identically on a retry.
_TRANSIENT_STATUSES = (408, 429)


def _raise_for_status(response: httpx.Response, body: str) -> None:
    """Turn a non-2xx status into the right kind of error.

    5xx, 408, and 429 become a retryable internal signal; any other 4xx is a
    hard failure, since the endpoint will reject the identical request again.
    """
    status = response.status_code
    if status < 400:
        return
    if status >= 500 or status in _TRANSIENT_STATUSES:
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


def _tool_call_summaries(tool_calls: Sequence[Mapping[str, Any]] | None) -> list[dict]:
    """Name plus parsed arguments per requested tool call, for the transcript.

    Arguments arrive as a JSON string on the wire; the View Chat page wants the
    object. Unparseable text is kept verbatim rather than dropped.
    """
    summaries: list[dict] = []
    for call in tool_calls or []:
        function = call.get("function") or {}
        arguments: Any = function.get("arguments")
        if isinstance(arguments, str) and arguments.strip():
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, ValueError):
                pass
        summaries.append({"name": function.get("name") or "", "arguments": arguments})
    return summaries


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
    tool_calls: Sequence[Mapping[str, Any]] | None = None,
    finish_reason: str | None = None,
    attempt: int = 1,
    error: str | None = None,
) -> None:
    summaries = _tool_call_summaries(tool_calls)
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
        # Structured list for rendering; the count survives beside it for
        # consumers of the old integer field.
        "tool_calls": summaries,
        "tool_call_count": len(summaries),
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
    on_restart: RestartCallback | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
    tool_choice: Any | None = None,
    response_format: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    extra_headers: Mapping[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    retry_backoff: Sequence[float] | None = None,
    retry_on_empty: bool = False,
) -> LLMResponse:
    """Call the configured endpoint once, retrying transient faults.

    ``agent`` names who is asking — drafter, beat_planner, critic — purely for
    the chat transcript and stream; it never reaches the wire. Every call
    publishes a ``chat_start``/``chat_end`` pair (with ``chat_token`` deltas
    while streaming) so the View Chat page shows all model traffic.

    ``retry_on_empty`` treats a completion with no text and no tool calls as a
    transient fault. Off by default: a tool-calling reply legitimately has null
    content, and only callers that always need an answer (the agent loop, the
    planners' tool-free path) opt in.

    ``transport`` and ``retry_backoff`` exist so tests can inject a mock
    transport and collapse the backoff sleeps; production callers leave both
    unset and get real HTTP with the endpoint's configured retry policy.

    Raises ``LLMCallError`` on a hard 4xx, a malformed response, or once the
    retry budget is exhausted.
    """
    if max_tokens is None:
        max_tokens = endpoint.max_output_tokens
    # With a declared context window but no explicit output cap, reserve output
    # room so an endpoint that would otherwise run unbounded (or apply its own
    # small default) leaves the planned space. Cap at the room the window has
    # left after this prompt — not a fixed reservation, which would silently
    # strangle long prose whenever the window has ample headroom — but never
    # below output_reservation (the prompt was trimmed to keep at least that
    # much free).
    if max_tokens is None and endpoint.context_window is not None:
        prompt_tokens = count_message_tokens(
            messages, endpoint.tokenizer_family, endpoint.model_name
        )
        room = endpoint.context_window - prompt_tokens
        max_tokens = max(endpoint.output_reservation, room)
    # The endpoint's configured extra_body is the base; a per-call extra_body
    # (rare) overrides key-by-key. Merged here so every caller benefits without
    # threading the config through — e.g. Ollama's options.num_ctx.
    if endpoint.extra_body:
        merged_extra = dict(endpoint.extra_body)
        if extra_body:
            merged_extra.update(extra_body)
        extra_body = merged_extra
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

    max_attempts = endpoint.max_attempts
    backoff = (
        tuple(retry_backoff)
        if retry_backoff is not None
        else tuple(endpoint.retry_backoff_seconds)
    )
    # request_timeout governs connect/write/pool; the read timeout — the gap
    # between streamed chunks — is separate, because a reasoning endpoint can
    # legitimately pause between tokens far longer than a connect should take.
    timeout = httpx.Timeout(
        connect=float(endpoint.request_timeout),
        write=float(endpoint.request_timeout),
        pool=float(endpoint.request_timeout),
        read=float(endpoint.stream_read_timeout),
    )
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

    async def _note_transient(exc: Exception, attempt: int, emitted_tokens: bool) -> None:
        """Log a transient fault and, when a retry follows, prepare for it."""
        nonlocal last_error
        last_error = exc
        # type(exc).__name__ is load-bearing: str(httpx.ReadTimeout()) is "",
        # which once produced an error message with no error in it.
        description = f"{type(exc).__name__}: {exc}".rstrip(": ")
        retrying = attempt < max_attempts
        _log_error(
            safe_url, endpoint, stream, attempt,
            f"transient: {description}", retrying=retrying,
        )
        if not retrying:
            return
        if emitted_tokens:
            # Tokens already reached the live view; tell it to discard the
            # partial before the retry replays them. The caller is safe
            # regardless — each attempt assembles its text from scratch.
            live = _LIVE_CALLS.get(call_id)
            if live is not None:
                live["text"] = ""
                live["thinking"] = ""
                live["seq"] = 0
            await bus.publish("chat_restart", {"id": call_id, "agent": agent})
            if on_restart is not None:
                restart_result = on_restart()
                if inspect.isawaitable(restart_result):
                    await restart_result
        await asyncio.sleep(backoff[min(attempt - 1, len(backoff) - 1)])

    while attempt < max_attempts:
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
            await _note_transient(exc, attempt, emitted[0])
            continue
        except Exception as exc:  # shape/callback bugs must still close live state
            description = f"{type(exc).__name__}: {exc}".rstrip(": ")
            message = f"unexpected response-processing failure: {description}"
            _log_error(safe_url, endpoint, stream, attempt, message)
            await _chat_end(
                call_id, agent, endpoint, ok=False, attempt=attempt, error=message
            )
            raise LLMCallError(message) from exc

        # A reply that is empty *because* it was cut off is not a transient
        # fault — retrying truncates again, three times, and buries the cause.
        # It is returned as-is so the caller diagnoses the truncation and names
        # the knob. (A reasoning model can spend its whole budget thinking,
        # which lands here with empty text and finish_reason "length".)
        if (
            retry_on_empty
            and not text.strip()
            and not tool_calls
            and finish_reason != "length"
        ):
            await _note_transient(
                _EmptyCompletionError(
                    f"completion contained no text and no tool calls "
                    f"(finish_reason={finish_reason!r})"
                ),
                attempt,
                emitted[0],
            )
            continue

        # Prefer what the server counted over what we estimated. Beyond accuracy,
        # `tokens_out` from the estimate counts `text` alone, so a reasoning model
        # that spent its budget thinking reports near-zero output; the server's
        # completion_tokens includes that deliberation and shows where the budget
        # actually went.
        served_prompt_tokens, served_completion_tokens = _usage_counts(raw)
        tokens_in = (
            served_prompt_tokens
            if served_prompt_tokens is not None
            else count_message_tokens(
                messages, endpoint.tokenizer_family, endpoint.model_name
            )
        )
        tokens_out = (
            served_completion_tokens
            if served_completion_tokens is not None
            else count_tokens(text, endpoint.tokenizer_family, endpoint.model_name)
        )
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
            thinking_chars=len(thinking),
            served_completion_tokens=served_completion_tokens,
            max_tokens=max_tokens,
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
            tool_calls=tool_calls,
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
            served_prompt_tokens=served_prompt_tokens,
            served_completion_tokens=served_completion_tokens,
            effective_max_tokens=max_tokens,
        )

    summary = (
        f"{max_attempts} attempts failed, last error: "
        f"{type(last_error).__name__}: {last_error}".rstrip(": ")
    )
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
    thinking_chars: int = 0,
    served_completion_tokens: int | None = None,
    max_tokens: int | None = None,
) -> None:
    """Record one completed response.

    Exactly one record per returned message. A streamed call logs here once, on
    the assembled text — never per token, which would drown the log in fragments.

    ``thinking_chars``, ``served_completion_tokens``, and ``max_tokens`` exist so
    a reasoning model burning its whole output grant on deliberation (``text``
    empty, ``finish_reason == "length"``) is readable straight from this log
    instead of inferred from ``tokens_out`` happening to equal the configured
    cap. Before this field existed that inference was the only way to confirm it
    (see the 2026-07-25 postmortem's evidence caveat).
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
            "thinking_chars": thinking_chars,
            "served_completion_tokens": served_completion_tokens,
            "max_tokens": max_tokens,
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
