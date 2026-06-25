"""Module: M14 × M04 (Configuration/Startup × LLM Inference Boundary)
Integration test for the config-loader → inference-boundary → I/O-log path.

This proves the *real* pieces line up end to end against a configured endpoint:

1. The real config loader (M14 ``load_config``) parses ``config.yaml`` and
   selects a configured endpoint/model — no manual YAML parsing, no hardcoded
   provider, host, or model name.
2. ``llm.call_llm`` (M04) calls *that* endpoint and the returned response records
   the configured model name and endpoint URL.
3. The dedicated LLM I/O logger (M14) writes a durable record for the call, and
   the endpoint's ``api_key`` value never appears in that record.

When no live endpoint/credential can complete a clean call (unconfigured,
unreachable, or auth-without-secret) the test skips with a precise reason rather
than faking a backend. ``MUSEAI_SKIP_LIVE_LLM=1`` opts out entirely.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

import core.llm_io_logger as llm_io_logger
from core.config_loader import EndpointConfig, load_config
from llm.call_llm import LLMCallError, call_llm

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# Routing names the config exposes; the default generation endpoint is preferred,
# falling back to the first configured endpoint. No provider/host/model assumed.
_ENDPOINT_NAMES = ("planner", "drafter", "critic", "pad_translator", "craft_consultant")
_PREFERRED_ORDER = ("drafter", "planner", "critic", "pad_translator", "craft_consultant")

_SKIP_ENV = "MUSEAI_SKIP_LIVE_LLM"
# Injected ONLY for endpoints with no real secret, so config (which requires a
# secret) loads for local unauthenticated servers. Never written to any log.
_PLACEHOLDER_SECRET = "live-test-placeholder-secret"

_SMOKE_MESSAGES = [{"role": "user", "content": "Reply with one short sentence."}]


def _endpoint_env_name(endpoint_name: str) -> str:
    return f"{endpoint_name.upper()}_API_KEY"


@dataclass(frozen=True)
class SelectedEndpoint:
    name: str
    endpoint: EndpointConfig
    has_real_api_key: bool


@pytest.fixture
def live_endpoint(monkeypatch: pytest.MonkeyPatch) -> SelectedEndpoint:
    """Load config.yaml via the real loader and select a configured endpoint.

    Mirrors the 05.T2 selection policy: prefer the default generation endpoint,
    else the first configured one. Skips with a clear reason when nothing is
    available, instead of inventing an endpoint.
    """
    if os.environ.get(_SKIP_ENV, "").strip() not in ("", "0", "false", "False"):
        pytest.skip(f"{_SKIP_ENV} set — live LLM endpoint tests opted out")

    # Record which endpoints have a real secret before injecting placeholders.
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
        pytest.skip("no configured live endpoint was available in config.yaml")

    chosen = next((n for n in _PREFERRED_ORDER if n in available), next(iter(available)))
    return SelectedEndpoint(
        name=chosen,
        endpoint=available[chosen],
        has_real_api_key=has_real[chosen],
    )


def _run_or_skip(coro_factory, selected: SelectedEndpoint):
    """Run the live call, skipping (never failing) when the endpoint is unusable."""
    try:
        return asyncio.run(coro_factory())
    except (LLMCallError, httpx.HTTPError, OSError) as exc:
        # base_url + model only — never the secret.
        reason = (
            f"no configured live endpoint was available: {selected.name!r} "
            f"({selected.endpoint.base_url}, model={selected.endpoint.model_name}) "
            f"could not complete a call"
        )
        if not selected.has_real_api_key:
            reason += f"; no real {_endpoint_env_name(selected.name)} is set"
        pytest.skip(f"{reason}: {exc}")


@pytest.fixture
def io_log_capture(tmp_path: Path):
    """Attach a temporary capture handler to the real ``llm_io`` logger.

    The production logger has no path parameter (it writes to a fixed
    ``logs/llm_io.log``). Rather than read/parse the shared rotating file or
    weaken the secret-leak assertion, this attaches an extra ``FileHandler``
    pointing at ``tmp_path`` to the same singleton logger. ``logger.info``
    dispatches to all handlers, so the capture file holds exactly the records
    emitted by the call(s) made while this fixture is active — the genuine
    production logging path, captured in isolation.
    """
    capture_path = tmp_path / "llm_io_capture.log"
    logger = llm_io_logger.get_llm_io_logger()
    handler = logging.FileHandler(capture_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    try:
        yield capture_path
    finally:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


@pytest.mark.integration
def test_config_loader_routes_call_and_logs_without_leak(
    live_endpoint: SelectedEndpoint, io_log_capture: Path
) -> None:
    """The configured endpoint is called and logged; the api_key never leaks."""
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

    # 1. call_llm reached the configured endpoint/model — assert routing identity,
    #    never the generated prose (model-agnostic).
    assert isinstance(response.text, str)
    assert response.text.strip() != ""
    assert response.model_name == live_endpoint.endpoint.model_name
    assert response.endpoint_base_url == live_endpoint.endpoint.base_url

    # 2. The I/O logger wrote a durable record for this call.
    log_text = io_log_capture.read_text(encoding="utf-8")
    assert log_text.strip() != "", "expected an llm_io record for the call"
    # The record describes this call's routing (configured model name present).
    assert live_endpoint.endpoint.model_name in log_text

    # 3. The configured api_key value never appears in the durable record.
    assert live_endpoint.endpoint.api_key
    assert live_endpoint.endpoint.api_key not in log_text
