"""Provider abstraction for MuseAI.

Each pipeline stage talks to an `LLMBackend` instead of a specific SDK, so the
same pipeline runs against Claude or any OpenAI-compatible local server (Ollama,
LM Studio, vLLM, llama.cpp).

Backends expose three operations:
  - parse(...)    -> a validated Pydantic object (structured stages)
  - complete(...) -> a plain string (the rolling-summary stage)
  - stream(...)   -> an async iterator of text chunks (drafting / revision)

`system_segments` (a list) lets each backend handle prompt caching its own way:
Anthropic marks the last segment with `cache_control` to reuse the cached
story-bible prefix; the OpenAI-compatible backend just joins them.
"""
from __future__ import annotations

import json
import re
from typing import AsyncIterator, Protocol, Type, TypeVar

from pydantic import BaseModel, ValidationError

from config import ANTHROPIC, LOCAL, settings

T = TypeVar("T", bound=BaseModel)


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
# OpenAI-compatible (Ollama / LM Studio / vLLM / llama.cpp)
# --------------------------------------------------------------------------- #

class OpenAICompatBackend:
    def __init__(self) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(base_url=settings.local_base_url, api_key=settings.local_api_key)

    async def _chat(self, model, system, user, max_tokens, response_format=None) -> str:
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = await self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    async def parse(self, model: str, system: str, user: str, schema: Type[T], max_tokens: int) -> T:
        import openai

        strict = _strict_schema(schema)
        schema_hint = (
            "\n\nRespond with ONLY a single JSON object matching this schema "
            "(no prose, no markdown fences):\n" + json.dumps(strict)
        )

        # Try response_format modes in decreasing strictness; local servers vary.
        attempts = [
            {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": strict, "strict": True}},
            {"type": "json_object"},
            None,
        ]
        last_content: str | None = None
        for rf in attempts:
            is_schema_mode = bool(rf) and rf.get("type") == "json_schema"
            prompt = user if is_schema_mode else user + schema_hint
            try:
                content = await self._chat(model, system, prompt, max_tokens, response_format=rf)
            except openai.BadRequestError:
                continue  # this server doesn't accept this response_format — try the next
            last_content = content
            try:
                return schema.model_validate_json(_extract_json(content))
            except (ValueError, ValidationError):
                break  # got a reply, but it didn't validate — go to the repair pass

        if last_content is not None:
            repaired = await self._repair(model, system, schema, strict, last_content, max_tokens)
            return schema.model_validate_json(_extract_json(repaired))

        raise RuntimeError(
            f"Local model '{model}' did not return JSON for {schema.__name__}. "
            "Try a more capable instruct model for the planning stages, or run them on Claude."
        )

    async def _repair(self, model, system, schema, strict, bad, max_tokens) -> str:
        msg = (
            "Your previous reply was not valid JSON for the required schema.\n\n"
            f"Schema:\n{json.dumps(strict)}\n\n"
            f"Your previous reply:\n{bad}\n\n"
            "Return ONLY the corrected JSON object, with no commentary or fences."
        )
        try:
            return await self._chat(model, system, msg, max_tokens, response_format={"type": "json_object"})
        except Exception:
            return await self._chat(model, system, msg, max_tokens, response_format=None)

    async def complete(self, model: str, system: str, user: str, max_tokens: int) -> str:
        return (await self._chat(model, system, user, max_tokens)).strip()

    async def stream(
        self, model: str, system_segments: list[str], user: str, max_tokens: int
    ) -> AsyncIterator[str]:
        system = "\n\n".join(system_segments)
        stream = await self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta


# --------------------------------------------------------------------------- #
# Registry + helpers
# --------------------------------------------------------------------------- #

_backends: dict[str, object] = {}


def get_backend(provider: str):
    if provider not in _backends:
        _backends[provider] = AnthropicBackend() if provider == ANTHROPIC else OpenAICompatBackend()
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


def _extract_json(text: str) -> str:
    """Pull the first balanced JSON object out of a model reply, tolerating code
    fences and surrounding prose (or <think> preambles)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text).strip()
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]
