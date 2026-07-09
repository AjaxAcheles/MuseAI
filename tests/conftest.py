"""Shared test fixtures."""

from __future__ import annotations

import pytest

from museai.core.config import AppConfig, EndpointConfig, GenerationConfig


@pytest.fixture
def config_factory(tmp_path):
    """Build an AppConfig pointing at a temp DB and event log."""

    def _make(**overrides) -> AppConfig:
        base = dict(
            project_id="test-project",
            endpoint=EndpointConfig(
                base_url="http://127.0.0.1:1234/v1",
                api_key="test-key",
                model_name="test-model",
                tokenizer_family="char_heuristic",
            ),
            generation=GenerationConfig(
                word_count_target=3000,
                beat_word_target=600,
                revision_retry_cap=3,
                max_agent_iterations=6,
                recent_prose_beats=4,
            ),
            db_path=str(tmp_path / "museai.db"),
            event_log_path=str(tmp_path / "events.jsonl"),
        )
        base.update(overrides)
        return AppConfig(**base)

    return _make
