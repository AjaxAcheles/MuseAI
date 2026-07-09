"""Shared test fixtures."""

from __future__ import annotations

import pytest

from museai.core.config import AppConfig, EndpointConfig, GenerationConfig


_GENERATION_DEFAULTS = dict(
    word_count_target=3000,
    beat_word_target=600,
    revision_retry_cap=3,
    max_agent_iterations=6,
    recent_prose_beats=4,
    context_token_budget=8000,
    passive_voice_threshold=0.25,
)


@pytest.fixture
def config_factory(tmp_path):
    """Build an AppConfig pointing at a temp DB and event log.

    Overrides naming a ``GenerationConfig`` field (``context_token_budget``, …)
    are applied there; everything else is applied to the top-level ``AppConfig``.
    """

    def _make(**overrides) -> AppConfig:
        generation = dict(_GENERATION_DEFAULTS)
        for key in list(overrides):
            if key in _GENERATION_DEFAULTS:
                generation[key] = overrides.pop(key)

        base = dict(
            project_id="test-project",
            endpoint=EndpointConfig(
                base_url="http://127.0.0.1:1234/v1",
                api_key="test-key",
                model_name="test-model",
                tokenizer_family="char_heuristic",
            ),
            generation=GenerationConfig(**generation),
            db_path=str(tmp_path / "museai.db"),
            event_log_path=str(tmp_path / "events.jsonl"),
        )
        base.update(overrides)
        return AppConfig(**base)

    return _make
