"""Module: M14 (Configuration, Startup & Observability)
Integration tests for configured model API endpoints.

These tests read the real config.yaml, deduplicate unique (base_url,
model_name) pairs, detect the server API dialect dynamically, and send a tiny
real prompt. They are marked as integration because they require live servers.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from core.config_loader import AppConfig, EndpointConfig, load_config

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
DUMMY_API_KEY = "test-dummy-key"
ENDPOINT_NAMES = (
    "planner",
    "drafter",
    "critic",
    "pad_translator",
    "craft_consultant",
)
HELLO_MESSAGES = [{"role": "user", "content": 'Reply with only the word "ok".'}]
DIALECT_CACHE: dict[tuple[str, str], "ServerDialect"] = {}


@dataclass(frozen=True)
class EndpointCase:
    endpoint_name: str
    base_url: str
    model_name: str
    api_key: str
    has_real_api_key: bool

    @property
    def id(self) -> str:
        return f"{self.endpoint_name}:{self.base_url}:{self.model_name}"

    @property
    def auth_headers(self) -> dict[str, str]:
        if not self.has_real_api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}


@dataclass(frozen=True)
class ServerDialect:
    name: str
    chat_endpoint: str
    json_body: dict[str, Any]


class AuthRequiredWithoutSecret(Exception):
    """Server requires auth, but the matching endpoint env secret is absent."""


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "endpoint_case" not in metafunc.fixturenames:
        return
    cases = unique_endpoint_cases()
    metafunc.parametrize("endpoint_case", cases, ids=[case.id for case in cases])


def unique_endpoint_cases() -> list[EndpointCase]:
    config = load_config_with_dummy_secrets()
    seen: set[tuple[str, str]] = set()
    cases: list[EndpointCase] = []
    for name, endpoint in iterate_endpoints(config):
        key = (endpoint.base_url, endpoint.model_name)
        if key in seen:
            continue
        seen.add(key)
        env_name = endpoint_env_name(name)
        env_value = os.environ.get(env_name)
        cases.append(
            EndpointCase(
                endpoint_name=name,
                base_url=endpoint.base_url.rstrip("/"),
                model_name=endpoint.model_name,
                api_key=endpoint.api_key,
                has_real_api_key=bool(env_value),
            )
        )
    return cases


def load_config_with_dummy_secrets() -> AppConfig:
    """Load config.yaml while allowing local unauthenticated servers.

    load_config() correctly requires endpoint secrets. For local servers such as
    Ollama, these tests inject dummy secrets only for absent env vars, then omit
    auth headers when sending requests. If a remote server responds with an auth
    error while only a dummy secret is available, the affected case is skipped
    with a clear reason.
    """
    for name in ENDPOINT_NAMES:
        os.environ.setdefault(endpoint_env_name(name), DUMMY_API_KEY)
    return load_config(CONFIG_PATH)


def iterate_endpoints(config: AppConfig) -> list[tuple[str, EndpointConfig]]:
    return [(name, getattr(config.endpoints, name)) for name in ENDPOINT_NAMES]


def endpoint_env_name(endpoint_name: str) -> str:
    return f"{endpoint_name.upper()}_API_KEY"


async def with_client(coro):
    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    async with httpx.AsyncClient(limits=limits, timeout=30.0) as client:
        return await coro(client)


async def detect_dialect(
    client: httpx.AsyncClient, endpoint_case: EndpointCase
) -> ServerDialect:
    cache_key = (endpoint_case.base_url, endpoint_case.model_name)
    if cache_key in DIALECT_CACHE:
        return DIALECT_CACHE[cache_key]

    openai_body = {
        "model": endpoint_case.model_name,
        "messages": HELLO_MESSAGES,
        "max_tokens": 10,
        "temperature": 0.0,
    }
    openai_url = f"{endpoint_case.base_url}/v1/chat/completions"
    openai_response = await post_or_none(
        client, openai_url, openai_body, endpoint_case.auth_headers
    )
    if openai_response is not None:
        if openai_response.status_code == 200:
            dialect = ServerDialect("openai", openai_url, openai_body)
            DIALECT_CACHE[cache_key] = dialect
            return dialect
        if is_missing_auth(openai_response, endpoint_case):
            raise AuthRequiredWithoutSecret(openai_url)
        if openai_response.status_code != 404:
            openai_response.raise_for_status()

    ollama_chat_body = {
        "model": endpoint_case.model_name,
        "messages": HELLO_MESSAGES,
        "stream": False,
    }
    ollama_chat_url = f"{endpoint_case.base_url}/api/chat"
    ollama_chat_response = await post_or_none(
        client, ollama_chat_url, ollama_chat_body, {}
    )
    if ollama_chat_response is not None:
        if ollama_chat_response.status_code == 200:
            dialect = ServerDialect("ollama_chat", ollama_chat_url, ollama_chat_body)
            DIALECT_CACHE[cache_key] = dialect
            return dialect
        if ollama_chat_response.status_code != 404:
            ollama_chat_response.raise_for_status()

    ollama_generate_body = {
        "model": endpoint_case.model_name,
        "prompt": 'Reply with only the word "ok".',
        "stream": False,
    }
    ollama_generate_url = f"{endpoint_case.base_url}/api/generate"
    ollama_generate_response = await post_or_none(
        client, ollama_generate_url, ollama_generate_body, {}
    )
    if ollama_generate_response is not None:
        if ollama_generate_response.status_code == 200:
            dialect = ServerDialect(
                "ollama_generate", ollama_generate_url, ollama_generate_body
            )
            DIALECT_CACHE[cache_key] = dialect
            return dialect
        ollama_generate_response.raise_for_status()

    raise ConnectionError(
        f"Server at {endpoint_case.base_url} is unreachable or exposes no "
        "supported OpenAI-compatible or Ollama inference route."
    )


async def post_or_none(
    client: httpx.AsyncClient,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
) -> httpx.Response | None:
    try:
        return await client.post(url, json=body, headers=headers)
    except (httpx.ConnectError, httpx.TimeoutException):
        return None


def is_missing_auth(response: httpx.Response, endpoint_case: EndpointCase) -> bool:
    return response.status_code in {401, 403} and not endpoint_case.has_real_api_key


async def reachable(client: httpx.AsyncClient, endpoint_case: EndpointCase) -> bool:
    probes = (
        (f"{endpoint_case.base_url}/v1/models", endpoint_case.auth_headers),
        (f"{endpoint_case.base_url}/api/tags", {}),
        (endpoint_case.base_url, {}),
    )
    auth_blocked = False
    for url, headers in probes:
        try:
            response = await client.get(url, headers=headers, timeout=10.0)
        except (httpx.ConnectError, httpx.TimeoutException):
            continue
        if is_missing_auth(response, endpoint_case):
            auth_blocked = True
            continue
        if response.status_code < 500:
            return True
    if auth_blocked:
        raise AuthRequiredWithoutSecret(endpoint_case.base_url)
    return False


async def model_list_contains(
    client: httpx.AsyncClient, endpoint_case: EndpointCase
) -> bool | None:
    ollama_response = await get_or_none(client, f"{endpoint_case.base_url}/api/tags", {})
    if ollama_response is not None and ollama_response.status_code == 200:
        models = ollama_response.json().get("models", [])
        return endpoint_case.model_name in {model.get("name") for model in models}

    openai_response = await get_or_none(
        client, f"{endpoint_case.base_url}/v1/models", endpoint_case.auth_headers
    )
    if openai_response is None:
        return None
    if is_missing_auth(openai_response, endpoint_case):
        raise AuthRequiredWithoutSecret(f"{endpoint_case.base_url}/v1/models")
    if openai_response.status_code == 200:
        models = openai_response.json().get("data", [])
        return endpoint_case.model_name in {model.get("id") for model in models}
    if openai_response.status_code in {404, 405}:
        return None
    openai_response.raise_for_status()
    return None


async def get_or_none(
    client: httpx.AsyncClient, url: str, headers: dict[str, str]
) -> httpx.Response | None:
    try:
        return await client.get(url, headers=headers, timeout=10.0)
    except (httpx.ConnectError, httpx.TimeoutException):
        return None


def extract_generated_text(response_json: dict[str, Any]) -> str:
    if "choices" in response_json:
        return response_json["choices"][0]["message"]["content"]
    if "message" in response_json:
        return response_json["message"]["content"]
    if "response" in response_json:
        return response_json["response"]
    raise AssertionError(
        "Response used an unrecognized format. "
        f"Top-level keys: {sorted(response_json)}"
    )


@pytest.mark.integration
def test_server_reachable(endpoint_case: EndpointCase) -> None:
    async def run(client: httpx.AsyncClient) -> None:
        try:
            assert await reachable(client, endpoint_case), (
                f"Cannot connect to {endpoint_case.base_url}; is the server running?"
            )
        except AuthRequiredWithoutSecret:
            pytest.skip(
                f"{endpoint_case.endpoint_name} requires a real "
                f"{endpoint_env_name(endpoint_case.endpoint_name)} secret."
            )

    asyncio.run(with_client(run))


@pytest.mark.integration
def test_model_exists(endpoint_case: EndpointCase) -> None:
    async def run(client: httpx.AsyncClient) -> None:
        try:
            contains_model = await model_list_contains(client, endpoint_case)
        except AuthRequiredWithoutSecret:
            pytest.skip(
                f"{endpoint_case.endpoint_name} requires a real "
                f"{endpoint_env_name(endpoint_case.endpoint_name)} secret."
            )
        if contains_model is None:
            pytest.skip(
                f"Server at {endpoint_case.base_url} does not expose a model list."
            )
        assert contains_model, (
            f"Endpoint '{endpoint_case.endpoint_name}' expects model "
            f"'{endpoint_case.model_name}', but the server model list does not "
            "include it."
        )

    asyncio.run(with_client(run))


@pytest.mark.integration
def test_send_prompt_and_get_response(endpoint_case: EndpointCase) -> None:
    async def run(client: httpx.AsyncClient) -> None:
        try:
            dialect = await detect_dialect(client, endpoint_case)
        except AuthRequiredWithoutSecret:
            pytest.skip(
                f"{endpoint_case.endpoint_name} requires a real "
                f"{endpoint_env_name(endpoint_case.endpoint_name)} secret."
            )

        response = await client.post(
            dialect.chat_endpoint,
            json=dialect.json_body,
            headers=endpoint_case.auth_headers,
        )
        assert response.status_code == 200, (
            f"Endpoint '{endpoint_case.endpoint_name}' "
            f"({endpoint_case.base_url}, {endpoint_case.model_name}) returned "
            f"HTTP {response.status_code}: {response.text[:300]}"
        )
        content = extract_generated_text(response.json())
        assert content and content.strip(), (
            f"Endpoint '{endpoint_case.endpoint_name}' returned empty text."
        )

    asyncio.run(with_client(run))
