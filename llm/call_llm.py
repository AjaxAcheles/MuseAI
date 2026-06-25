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
from pydantic import BaseModel, ValidationError

import core.llm_io_logger as llm_io_logger
from llm.gbnf_compiler import json_schema_to_gbnf
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


class StructuredOutputError(ValueError):
    """Raised when model text cannot be turned into a schema-valid object.

    Surfaced to the caller (e.g. the critic loop) so a bounded
    ``model_validate_retry_cap`` ladder can decide whether to retry or hand a
    hard error to the FSM escalation path. These helpers never loop themselves.
    """


def extract_first_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` object found in ``text``.

    Markdown code fences are stripped first, then the text is scanned for the
    first top-level brace and walked to its matching close. Braces that appear
    inside quoted strings (and escaped quotes within them) are ignored, so an
    object whose string values contain ``{``/``}``/``"`` is returned intact.
    Returns ``None`` when no balanced object is present.
    """
    if not text:
        return None

    scanned = _strip_markdown_fences(text)
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(scanned):
        if start is None:
            if char == "{":
                start = index
                depth = 1
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return scanned[start : index + 1]

    return None


def validate_structured_text(text: str, schema_model: type[BaseModel]) -> BaseModel:
    """Parse ``text`` as JSON and validate it against ``schema_model``.

    Strict: the text must already be a single JSON document. Raises
    :class:`StructuredOutputError` with a clear message on either a JSON parse
    failure or a schema-validation failure.
    """
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise StructuredOutputError(
            f"structured output was not valid JSON: {exc}"
        ) from exc

    try:
        return schema_model.model_validate(parsed)
    except ValidationError as exc:
        raise StructuredOutputError(
            f"structured output failed {schema_model.__name__} schema validation: {exc}"
        ) from exc


def validate_with_salvage(text: str, schema_model: type[BaseModel]) -> BaseModel:
    """Validate ``text`` against ``schema_model`` with one lenient salvage pass.

    Order: (1) strict validation of the text as given; (2) on failure, strip
    wrappers and extract the first balanced JSON object; (3) one final strict
    validation of the extracted object. There is no retry loop here — at most
    two validation attempts run, so the call provably terminates. Persistent
    failure raises :class:`StructuredOutputError` for the caller's bounded
    retry/escalation ladder to handle.
    """
    try:
        return validate_structured_text(text, schema_model)
    except StructuredOutputError as first_error:
        candidate = extract_first_json_object(text)
        if candidate is None:
            raise StructuredOutputError(
                "structured output failed validation and no JSON object "
                "could be salvaged from the text"
            ) from first_error
        return validate_structured_text(candidate, schema_model)


def _strip_markdown_fences(text: str) -> str:
    """Strip a single wrapping triple-backtick fence (e.g. ```json ... ```)."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    newline = stripped.find("\n")
    stripped = stripped[newline + 1 :] if newline != -1 else stripped[3:]
    if stripped.rstrip().endswith("```"):
        stripped = stripped.rstrip()[:-3]
    return stripped


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


class UnsupportedGrammarStrategyError(ValueError):
    """Raised when an endpoint declares a grammar strategy the boundary cannot honor."""


async def call_llm_structured(
    messages: list[Message],
    endpoint: Any,
    *,
    schema_model: type[BaseModel],
    validate_retry_cap: int,
    stream: bool | None = None,
    on_token: TokenCallback | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    extra_body: Mapping[str, Any] | None = None,
    extra_headers: Mapping[str, str] | None = None,
    adapter: EndpointAdapter | None = None,
    model_name: str | None = None,
    client: httpx.AsyncClient | None = None,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    retry_delays: tuple[float, ...] = _DEFAULT_RETRY_DELAYS_SECONDS,
    sleep: SleepCallback = asyncio.sleep,
    call: Callable[..., Awaitable[LLMResponse]] = call_llm,
) -> BaseModel:
    """Call the configured endpoint and return a schema-validated Pydantic object.

    The structured-output contract (Data_Structures §1.3, node 8):

    1. Build the structured request option from the endpoint's
       ``grammar_constraint_strategy`` only — GBNF grammar in a generic request
       field, or chat-completions JSON mode — never a provider-named branch.
    2. Validate each response with ``schema_model.model_validate()``.
    3. Retry up to ``validate_retry_cap`` (sourced from
       ``config.runtime.model_validate_retry_cap`` — never a hidden constant).
    4. On cap exhaustion run :func:`validate_with_salvage` exactly once; if it
       recovers, return the object and log that salvage was used.
    5. Otherwise raise :class:`LLMCallError` so the FSM routes a hard failure.

    The loop is bounded by ``validate_retry_cap`` and every endpoint call is
    itself bounded by the transient-retry ladder, so the path always terminates.
    Validation failures are never swallowed — this never returns ``None``.
    """
    if validate_retry_cap < 1:
        raise ValueError("validate_retry_cap must be at least 1")

    strategy = getattr(endpoint, "grammar_constraint_strategy")
    response_format, structured_extra_body = _build_structured_options(
        strategy, schema_model, extra_body
    )

    last_text = ""
    for attempt in range(1, validate_retry_cap + 1):
        response = await call(
            messages,
            endpoint,
            stream=stream,
            on_token=on_token,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            extra_body=structured_extra_body,
            extra_headers=extra_headers,
            adapter=adapter,
            model_name=model_name,
            client=client,
            max_attempts=max_attempts,
            retry_delays=retry_delays,
            sleep=sleep,
        )
        last_text = response.text
        try:
            return validate_structured_text(response.text, schema_model)
        except StructuredOutputError:
            if attempt < validate_retry_cap:
                continue

    # Cap exhausted: one lenient salvage pass over the final response text.
    try:
        salvaged = validate_with_salvage(last_text, schema_model)
    except StructuredOutputError as exc:
        raise LLMCallError(
            f"structured output failed {schema_model.__name__} validation after "
            f"{validate_retry_cap} attempts and salvage: {exc}",
            attempt_count=validate_retry_cap,
        ) from exc

    _log_structured_salvage(endpoint, schema_model, validate_retry_cap)
    return salvaged


def _build_structured_options(
    strategy: str,
    schema_model: type[BaseModel],
    extra_body: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Translate an endpoint grammar strategy into generic request options.

    Returns ``(response_format, extra_body)`` for :func:`call_llm`. GBNF puts a
    compiled grammar string into a provider-neutral ``grammar`` request field;
    JSON mode uses the chat-completions ``response_format`` envelope. Pydantic
    validation guards the FSM even if an endpoint ignores the hint.
    """
    merged = dict(extra_body) if extra_body else {}
    if strategy == "gbnf":
        grammar = json_schema_to_gbnf(schema_model.model_json_schema())
        merged["grammar"] = grammar
        return None, merged
    if strategy == "json_mode":
        return {"type": "json_object"}, (merged or None)
    raise UnsupportedGrammarStrategyError(
        f"unsupported grammar_constraint_strategy: {strategy!r}"
    )


def _log_structured_salvage(
    endpoint: Any, schema_model: type[BaseModel], validate_retry_cap: int
) -> None:
    logger = llm_io_logger.get_llm_io_logger()
    logger.info(
        json.dumps(
            {
                "event": "structured_output_salvage_used",
                "endpoint_base_url": _sanitize_url(getattr(endpoint, "base_url", "")),
                "schema_model": schema_model.__name__,
                "validate_retry_cap": validate_retry_cap,
            }
        )
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
    try:
        text = response.text[:200].replace("\n", " ")
    except httpx.ResponseNotRead:
        # Streaming response body was never consumed before raise_for_status()
        # fired; accessing .text would require an awaited .aread() call which
        # is not available here. Return the status line without a body excerpt.
        text = ""
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
