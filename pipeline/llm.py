"""Provider abstraction for MuseAI.

Each pipeline stage talks to an `LLMBackend` instead of a specific SDK, so the
same pipeline runs against Claude or a local Ollama server.

Backends expose three operations:
  - parse(...)    -> a validated Pydantic object (structured stages)
  - complete(...) -> a plain string (the rolling-summary stage)
  - stream(...)   -> an async iterator of text chunks (drafting / revision)

`system_segments` (a list) lets each backend handle prompt caching its own way:
Anthropic marks the last segment with `cache_control` to reuse the cached
story-bible prefix; the Ollama backend just joins them.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import AsyncIterator, Protocol, Type, TypeVar

from pydantic import BaseModel, ValidationError

from config import ANTHROPIC, LOCAL, settings

log = logging.getLogger("museai.llm")

T = TypeVar("T", bound=BaseModel)

# The local server can abort a generation with a 5xx (runner crash/OOM, or a
# cancelled request). These are usually transient, so retry a few times with a
# short backoff before giving up.
_MAX_RETRIES = 3
_RETRY_BACKOFF = 1.5  # seconds, multiplied by attempt number


class LLMBackend(Protocol):
    async def parse(self, model: str, system: str, user: str, schema: Type[T], max_tokens: int) -> T: ...
    async def complete(self, model: str, system: str, user: str, max_tokens: int) -> str: ...
    def stream(self, model: str, system_segments: list[str], user: str, max_tokens: int) -> AsyncIterator[str]: ...


# --------------------------------------------------------------------------- #
# Anthropic (Claude)
# --------------------------------------------------------------------------- #

class AnthropicBackend:
    def __init__(self) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic()

    async def parse(self, model: str, system: str, user: str, schema: Type[T], max_tokens: int) -> T:
        resp = await self._client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        return resp.parsed_output

    async def complete(self, model: str, system: str, user: str, max_tokens: int) -> str:
        resp = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    async def stream(
        self, model: str, system_segments: list[str], user: str, max_tokens: int
    ) -> AsyncIterator[str]:
        blocks = [{"type": "text", "text": seg} for seg in system_segments]
        if blocks:
            # Cache the last (largest, most stable) segment — e.g. the story bible.
            blocks[-1]["cache_control"] = {"type": "ephemeral"}
        async with self._client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=blocks,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            async for text in stream.text_stream:
                yield text


# --------------------------------------------------------------------------- #
# Ollama (native /api/generate)
# --------------------------------------------------------------------------- #

class OllamaBackend:
    """Talks to an Ollama server via its native generate API.

    Unlike an OpenAI client, this posts directly to the exact URL configured in
    MUSEAI_LOCAL_BASE_URL (e.g. http://host:11434/api/generate) — nothing is
    appended to the path. Structured stages use Ollama's `format` field (a JSON
    schema, or "json") to constrain the reply.
    """

    def __init__(self) -> None:
        import httpx

        self._url = settings.local_base_url
        # Local generation can be slow; give it a generous read timeout.
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
        log.info("local (Ollama) backend → %s", self._url)

    def _friendly(self, exc: Exception, model: str) -> RuntimeError:
        """Turn the common local-server failures into actionable messages."""
        import httpx

        if isinstance(exc, httpx.ConnectError):
            return RuntimeError(
                f"Could not reach the Ollama server at {self._url} — "
                "is it running? (e.g. start Ollama, or check MUSEAI_LOCAL_BASE_URL)."
            )
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
            return RuntimeError(
                f"Model '{model}' was not found on the Ollama server at "
                f"{self._url} — pull it first (e.g. `ollama pull {model}`)."
            )
        return RuntimeError(f"Local model request failed ({type(exc).__name__}): {exc}")

    def _payload(self, model, system, user, max_tokens, *, stream, fmt=None) -> dict:
        payload: dict = {
            "model": model,
            "prompt": user,
            "system": system,
            "stream": stream,
            # Disable "thinking" mode. Reasoning models (e.g. qwen3.6) otherwise
            # emit their chain-of-thought into a separate `thinking` field and can
            # leave `response` empty; we only ever want the final answer here.
            "think": False,
            "options": {"num_predict": max_tokens},
        }
        if fmt is not None:
            payload["format"] = fmt
        return payload

    async def _generate(self, model, system, user, max_tokens, fmt=None) -> str:
        import httpx

        payload = self._payload(model, system, user, max_tokens, stream=False, fmt=fmt)
        # Transient failures (connection resets, read timeouts, 5xx aborts) are
        # retried; a 4xx (e.g. an unsupported `format`) is raised so parse() can
        # fall back to a looser mode.
        last: str = "unknown error"
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = await self._client.post(self._url, json=payload)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                if isinstance(exc, httpx.ConnectError) and attempt == _MAX_RETRIES:
                    raise self._friendly(exc, model) from exc
                last = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == 404:
                    raise self._friendly(
                        httpx.HTTPStatusError("404", request=resp.request, response=resp), model
                    )
                if 400 <= resp.status_code < 500:
                    # Client error (bad `format`, etc.) — not retryable; let
                    # parse() try the next mode. Include the body for context.
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}: {resp.text[:300]}",
                        request=resp.request,
                        response=resp,
                    )
                if resp.status_code >= 500:
                    # Server aborted generation — capture the body and retry.
                    last = f"HTTP {resp.status_code}: {resp.text[:300]}"
                else:
                    # Ollama /api/generate returns {"response": "...", "done": true}.
                    return resp.json().get("response", "")

            log.warning(
                "Ollama generate failed (attempt %d/%d): %s", attempt, _MAX_RETRIES, last
            )
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_BACKOFF * attempt)

        raise RuntimeError(
            f"The Ollama server at {self._url} aborted generation for model '{model}' "
            f"on all {_MAX_RETRIES} attempts (last error: {last}). This is a server-side "
            "failure — the model runner likely crashed or ran out of memory. Check the "
            "Ollama logs (`journalctl -u ollama` / server console) and free VRAM."
        )

    async def parse(self, model: str, system: str, user: str, schema: Type[T], max_tokens: int) -> T:
        import httpx

        strict = _strict_schema(schema)

        # Build a textual field description instead of dumping raw schema JSON.
        # Many local models (e.g. qwen3.6) get confused by the raw schema and
        # echo it back instead of generating data.
        _field_descriptions = _schema_to_text_hint(schema)

        schema_hint = (
            "\n\nRespond with ONLY a single JSON object matching this schema "
            "(no prose, no markdown fences).\n"
            "Expected fields:\n" + _field_descriptions
        )

        # Hardening: add a JSON-only directive to the system prompt so the model
        # is less likely to produce markdown fences or conversational filler.
        _json_only = (
            "\n\nReturn only a raw JSON object. Do NOT wrap it in markdown "
            "fences or include any other text before or after the JSON."
        )
        system = system + _json_only

        # Try Ollama `format` modes in decreasing strictness; servers/versions vary.
        # Schema-constrained decoding is tried first: with thinking disabled it
        # reliably produces a schema-valid object. json/plain are looser fallbacks
        # for servers that reject a full schema as `format`.
        attempts = [
            ("schema", strict),   # structured outputs: constrain to the JSON schema
            ("json", "json"),     # loose JSON mode
            ("plain", None),      # no constraint — lean on the schema hint
        ]
        last_content: str | None = None
        for mode, fmt in attempts:
            prompt = user if mode == "schema" else user + schema_hint
            log.debug("parse %s on %s: format=%s", schema.__name__, model, mode)
            try:
                content = await self._generate(model, system, prompt, max_tokens, fmt=fmt)
            except httpx.HTTPStatusError as exc:
                log.debug("server rejected format=%s (%s) — trying next", mode, exc)
                continue  # this server doesn't accept this format — try the next
            last_content = content
            extracted = _extract_json(content)
            if extracted is None:
                log.warning(
                    "parse %s: %s reply contained no valid JSON — attempting one repair pass",
                    schema.__name__, mode,
                )
                break
            try:
                return schema.model_validate_json(extracted)
            except (ValueError, ValidationError) as exc:
                log.warning(
                    "parse %s: %s reply did not validate (%s) — attempting one repair pass",
                    schema.__name__, mode, exc.__class__.__name__,
                )
                break  # got a reply, but it didn't validate — go to the repair pass

        if last_content is not None:
            repaired = await self._repair(model, system, schema, last_content, max_tokens)
            extracted = _extract_json(repaired)
            if extracted is None:
                raise RuntimeError(
                    f"Local model '{model}' returned invalid JSON for {schema.__name__} "
                    "even after a repair pass. Raw content received:\n\n"
                    + (repaired or "(empty response)")
                )
            try:
                return schema.model_validate_json(extracted)
            except (ValidationError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Local model '{model}' returned JSON that failed schema validation "
                    f"for {schema.__name__} after a repair pass. "
                    f"Validation error: {exc}\n\nRaw content:\n\n{repaired}"
                ) from exc

        raise RuntimeError(
            f"Local model '{model}' did not return usable JSON for {schema.__name__}. "
            "Try a more capable instruct model for the planning stages, or run them on Claude."
        )

    async def _repair(self, model, system, schema, bad, max_tokens) -> str:
        import httpx

        _field_descriptions = _schema_to_text_hint(schema)

        msg = (
            "Your previous reply was not valid JSON for the required schema.\n\n"
            "Expected fields:\n" + _field_descriptions + "\n\n"
            f"Your previous reply:\n{bad}\n\n"
            "Return ONLY the corrected JSON object, with no commentary or fences."
        )
        try:
            return await self._generate(model, system, msg, max_tokens, fmt="json")
        except httpx.HTTPStatusError:
            return await self._generate(model, system, msg, max_tokens, fmt=None)

    async def complete(self, model: str, system: str, user: str, max_tokens: int) -> str:
        return (await self._generate(model, system, user, max_tokens)).strip()

    async def stream(
        self, model: str, system_segments: list[str], user: str, max_tokens: int
    ) -> AsyncIterator[str]:
        import httpx

        system = "\n\n".join(system_segments)
        payload = self._payload(model, system, user, max_tokens, stream=True)
        # Retry transient failures, but only before the first token — once we've
        # yielded prose downstream, restarting would duplicate it.
        last: str = "unknown error"
        for attempt in range(1, _MAX_RETRIES + 1):
            started = False
            try:
                async with self._client.stream("POST", self._url, json=payload) as resp:
                    if resp.status_code == 404:
                        await resp.aread()
                        raise self._friendly(
                            httpx.HTTPStatusError("404", request=resp.request, response=resp), model
                        )
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode(errors="replace")[:300]
                        last = f"HTTP {resp.status_code}: {body}"
                        if resp.status_code < 500:  # not retryable
                            raise RuntimeError(f"Ollama rejected the request: {last}")
                        raise httpx.HTTPStatusError(last, request=resp.request, response=resp)
                    # Ollama streams newline-delimited JSON objects, each carrying a
                    # `response` fragment. (Thinking is disabled in _payload, so the
                    # model's reasoning never leaks into the draft.)
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        chunk = json.loads(line).get("response")
                        if chunk:
                            started = True
                            yield chunk
                return
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError) as exc:
                if started:
                    # Failed mid-stream — can't safely retry; surface it.
                    raise self._friendly(exc, model) from exc
                if isinstance(exc, httpx.HTTPStatusError):
                    last = str(exc)
                else:
                    last = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "Ollama stream failed before first token (attempt %d/%d): %s",
                    attempt, _MAX_RETRIES, last,
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(_RETRY_BACKOFF * attempt)

        raise RuntimeError(
            f"The Ollama server at {self._url} aborted streaming for model '{model}' "
            f"on all {_MAX_RETRIES} attempts (last error: {last}). This is a server-side "
            "failure — the model runner likely crashed or ran out of memory. Check the "
            "Ollama logs and free VRAM."
        )


# --------------------------------------------------------------------------- #
# Registry + helpers
# --------------------------------------------------------------------------- #

_backends: dict[str, object] = {}


def get_backend(provider: str):
    if provider not in _backends:
        _backends[provider] = AnthropicBackend() if provider == ANTHROPIC else OllamaBackend()
    return _backends[provider]


def _strict_schema(model: Type[BaseModel]) -> dict:
    """Pydantic JSON schema, hardened for OpenAI strict structured outputs:
    every object node gets additionalProperties:false and lists all properties
    as required (recurses into $defs and nested models)."""
    schema = model.model_json_schema()

    def walk(node) -> None:
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict):
                node["additionalProperties"] = False
                node["required"] = list(props.keys())
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)
    return schema


def _schema_to_text_hint(model: Type[BaseModel]) -> str:
    """Convert a Pydantic model's JSON schema into a human-readable field list
    suitable for prompting local models, avoiding raw JSON schema that can
    confuse them into echoing it back."""
    schema = model.model_json_schema()
    props = schema.get("properties", {})
    lines = []
    for name, info in props.items():
        typ = info.get("type", "string")
        desc = info.get("description", "")
        items = info.get("items", {})
        if items:
            item_type = items.get("type", "string")
            if item_type == "object":
                # Nested object array — list the sub-fields.
                sub_props = items.get("properties", {})
                sub_lines = []
                for sub_name, sub_info in sub_props.items():
                    sub_typ = sub_info.get("type", "string")
                    sub_desc = sub_info.get("description", "")
                    if sub_desc:
                        sub_lines.append(f"      - {sub_name} ({sub_typ}): {sub_desc}")
                    else:
                        sub_lines.append(f"      - {sub_name} ({sub_typ})")
                sub_hint = "\n" + "\n".join(sub_lines) if sub_lines else ""
                if desc:
                    lines.append(f"  - {name} (array of objects): {desc}{sub_hint}")
                else:
                    lines.append(f"  - {name} (array of objects){sub_hint}")
            else:
                typ = f"array of {item_type}s"
                if desc:
                    lines.append(f"  - {name} ({typ}): {desc}")
                else:
                    lines.append(f"  - {name} ({typ})")
        else:
            if desc:
                lines.append(f"  - {name} ({typ}): {desc}")
            else:
                lines.append(f"  - {name} ({typ})")
    return "\n".join(lines)


def _extract_json(text: str) -> str | None:
    """Pull the first balanced JSON object out of a model reply, tolerating code
    fences and surrounding prose (or <think> preambles).

    Returns ``None`` when no valid JSON could be found — callers **must** check
    for ``None`` rather than passing an empty string to ``model_validate_json``.
    """
    text = text.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text).strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                # Sanity-check: try parsing it.
                try:
                    json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return candidate
    return None
