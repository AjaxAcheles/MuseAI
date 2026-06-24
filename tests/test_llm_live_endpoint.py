"""Module: M04 (LLM Inference Boundary)
Live, config-driven smoke tests for the real inference boundary.

These exercise the genuine endpoint/model described by ``config.yaml`` through
``call_llm()`` — no fake backend, no local server assumption, and no
provider/model special-casing. When no endpoint can complete a clean call
(unconfigured, unreachable, or auth-without-secret) the test skips with an
explicit reason rather than faking a response. Set ``MUSEAI_SKIP_LIVE_LLM=1``
to opt out entirely (CI / offline development).
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from core.config_loader import EndpointConfig, load_config
from llm.call_llm import (
    LLMCallError,
    UnsupportedGrammarStrategyError,
    call_llm,
    call_llm_structured,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# Config exposes these routing names; the default generation endpoint is
# preferred, falling back to the first configured endpoint.
_ENDPOINT_NAMES = ("planner", "drafter", "critic", "pad_translator", "craft_consultant")
_PREFERRED_ORDER = ("drafter", "planner", "critic", "pad_translator", "craft_consultant")

_SKIP_ENV = "MUSEAI_SKIP_LIVE_LLM"
# Placeholder injected ONLY for endpoints with no real secret, so config (which
# requires a secret) loads for local unauthenticated servers. Never logged.
_PLACEHOLDER_SECRET = "live-test-placeholder-secret"

_SMOKE_MESSAGES = [{"role": "user", "content": "Reply with one short sentence."}]


def _endpoint_env_name(endpoint_name: str) -> str:
    return f"{endpoint_name.upper()}_API_KEY"


@dataclass(frozen=True)
class SelectedEndpoint:
    name: str
    endpoint: EndpointConfig
    has_real_api_key: bool
    validate_retry_cap: int  # from config.runtime.model_validate_retry_cap


@pytest.fixture
def live_endpoint(monkeypatch: pytest.MonkeyPatch) -> SelectedEndpoint:
    """Load config.yaml and select a real endpoint, or skip with a clear reason."""
    if os.environ.get(_SKIP_ENV, "").strip() not in ("", "0", "false", "False"):
        pytest.skip(f"{_SKIP_ENV} set — live LLM endpoint tests opted out")

    # Capture which endpoints have a real secret before injecting placeholders.
    has_real = {
        name: bool(os.environ.get(_endpoint_env_name(name))) for name in _ENDPOINT_NAMES
    }
    for name in _ENDPOINT_NAMES:
        if not has_real[name]:
            monkeypatch.setenv(_endpoint_env_name(name), _PLACEHOLDER_SECRET)

    try:
        config = load_config(CONFIG_PATH)
    except Exception as exc:  # noqa: BLE001 — any load failure means "cannot test live"
        pytest.skip(f"config.yaml could not be loaded for a live call: {exc}")

    available = {
        name: getattr(config.endpoints, name)
        for name in _ENDPOINT_NAMES
        if getattr(config.endpoints, name, None) is not None
    }
    if not available:
        pytest.skip("no endpoint is configured in config.yaml")

    chosen = next((n for n in _PREFERRED_ORDER if n in available), next(iter(available)))
    return SelectedEndpoint(
        name=chosen,
        endpoint=available[chosen],
        has_real_api_key=has_real[chosen],
        validate_retry_cap=config.runtime.model_validate_retry_cap,
    )


def _run_or_skip(coro_factory, selected: SelectedEndpoint):
    """Run the live call, skipping (never failing) when the endpoint is unusable."""
    try:
        return asyncio.run(coro_factory())
    except (LLMCallError, httpx.HTTPError, OSError) as exc:
        # base_url + model only — never the secret.
        reason = (
            f"live endpoint {selected.name!r} "
            f"({selected.endpoint.base_url}, model={selected.endpoint.model_name}) "
            f"could not complete a call"
        )
        if not selected.has_real_api_key:
            reason += f"; no real {_endpoint_env_name(selected.name)} is set"
        pytest.skip(f"{reason}: {exc}")


@pytest.mark.integration
def test_live_plaintext_smoke(live_endpoint: SelectedEndpoint) -> None:
    response = _run_or_skip(
        lambda: call_llm(
            _SMOKE_MESSAGES,
            live_endpoint.endpoint,
            stream=False,
            temperature=0.0,
            max_tokens=64,
        ),
        live_endpoint,
    )

    assert isinstance(response.text, str)
    assert response.text.strip() != ""
    if hasattr(response, "model_name"):
        # Any model may be configured; assert identity, never the prose.
        assert response.model_name == live_endpoint.endpoint.model_name


@pytest.mark.integration
def test_live_streaming_callback(live_endpoint: SelectedEndpoint) -> None:
    chunks: list[str] = []
    response = _run_or_skip(
        lambda: call_llm(
            _SMOKE_MESSAGES,
            live_endpoint.endpoint,
            stream=True,
            on_token=chunks.append,
            temperature=0.0,
            max_tokens=64,
        ),
        live_endpoint,
    )

    assert isinstance(response.text, str)
    # Final text is non-empty unless the endpoint exposed no streamed chunks
    # (a compliant endpoint may buffer internally — that must not fail the test).
    assert response.text.strip() != "" or response.streamed_chunks == []
    # Either we observed chunks, or the response explicitly records none.
    assert len(chunks) > 0 or response.streamed_chunks == []
    # The on_token callback must see exactly what the response recorded.
    assert list(chunks) == list(response.streamed_chunks)


class _LiveAnswer(BaseModel):
    """Tiny provider-neutral schema for the live structured-output smoke test."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    ok: bool


_STRUCTURED_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Reply with ONLY a JSON object, no prose, with exactly these keys: "
            '"answer" (a short string) and "ok" (a boolean). '
            'Example: {"answer": "hello", "ok": true}'
        ),
    }
]


@pytest.mark.integration
def test_live_structured_output_smoke(live_endpoint: SelectedEndpoint) -> None:
    # Drive the real endpoint through the 05.04 structured path. The grammar
    # strategy is whatever the endpoint declares; an unsupported/absent
    # capability skips explicitly rather than faking a backend.
    strategy = live_endpoint.endpoint.grammar_constraint_strategy
    try:
        result = _run_or_skip(
            lambda: call_llm_structured(
                _STRUCTURED_MESSAGES,
                live_endpoint.endpoint,
                schema_model=_LiveAnswer,
                # Retry cap stays sourced from config — never inflated to coax
                # a weak prompt into validating.
                validate_retry_cap=live_endpoint.validate_retry_cap,
                stream=False,
                temperature=0.0,
                max_tokens=128,
            ),
            live_endpoint,
        )
    except UnsupportedGrammarStrategyError as exc:
        pytest.skip(
            f"endpoint {live_endpoint.name!r} declares "
            f"grammar_constraint_strategy={strategy!r}, which is not a supported "
            f"structured-output capability: {exc}"
        )

    # Model-agnostic: validate shape and non-empty fields, never exact content.
    assert isinstance(result, _LiveAnswer)
    assert isinstance(result.answer, str)
    assert result.answer.strip() != ""
    assert isinstance(result.ok, bool)
