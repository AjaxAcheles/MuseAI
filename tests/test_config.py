"""Module: M14 (Configuration, Startup & Observability)
Synthetic tests for strict config loading.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from core.config_loader import load_config


ENDPOINT_ENV_NAMES = (
    "PLANNER_API_KEY",
    "DRAFTER_API_KEY",
    "CRITIC_API_KEY",
    "PAD_TRANSLATOR_API_KEY",
    "CRAFT_CONSULTANT_API_KEY",
)


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def valid_config_yaml() -> str:
    return """
endpoints:
  planner:
    base_url: "https://planner.example.test"
    model_name: "planner-synthetic"
    tokenizer_family: "tiktoken"
    supports_concurrent_critics: false
    grammar_constraint_strategy: "json_schema"
  drafter:
    base_url: "https://drafter.example.test"
    model_name: "drafter-synthetic"
    tokenizer_family: "tiktoken"
    supports_concurrent_critics: false
    grammar_constraint_strategy: "none"
  critic:
    base_url: "https://critic.example.test"
    model_name: "critic-synthetic"
    tokenizer_family: "tiktoken"
    supports_concurrent_critics: true
    grammar_constraint_strategy: "json_schema"
  pad_translator:
    base_url: "https://pad.example.test"
    model_name: "pad-synthetic"
    tokenizer_family: "tiktoken"
    supports_concurrent_critics: false
    grammar_constraint_strategy: "json_schema"
  craft_consultant:
    base_url: "https://craft.example.test"
    model_name: "craft-synthetic"
    tokenizer_family: "tiktoken"
    supports_concurrent_critics: false
    grammar_constraint_strategy: "json_schema"
thresholds:
  stylometric_drift_threshold: 0.12
  pad_ewma_alpha: 0.35
  coreference_confidence_floor: 0.65
  cluster_similarity_threshold: 0.65
  passive_voice_density: 0.25
  voice_evolution_l2_cap: 0.30
context:
  token_budget: 8000
  coreference_high_confidence: 0.85
  coreference_mid_confidence: 0.50
runtime:
  model_validate_retry_cap: 3
  headless_mode: false
  beats_per_scene_min: 3
  word_count_target: 80000
logging:
  log_level: "INFO"
"""


@pytest.fixture
def endpoint_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_name in ENDPOINT_ENV_NAMES:
        monkeypatch.setenv(env_name, f"{env_name.lower()}-secret")


def test_valid_synthetic_config_loads_typed_values(
    tmp_path: Path, endpoint_secrets: None
) -> None:
    del endpoint_secrets
    config = load_config(write_config(tmp_path, valid_config_yaml()))

    assert config.endpoints.planner.base_url == "https://planner.example.test"
    assert config.endpoints.planner.model_name == "planner-synthetic"
    assert config.endpoints.planner.supports_concurrent_critics is False
    assert config.endpoints.planner.api_key == "planner_api_key-secret"
    assert config.endpoints.critic.supports_concurrent_critics is True
    assert config.thresholds.stylometric_drift_threshold == 0.12
    assert config.thresholds.voice_evolution_l2_cap == 0.30
    assert config.context.token_budget == 8000
    assert config.context.coreference_high_confidence == 0.85
    assert config.context.coreference_mid_confidence == 0.50
    assert config.runtime.model_validate_retry_cap == 3
    assert config.logging.log_level == "INFO"


def test_unknown_key_raises_validation_error(
    tmp_path: Path, endpoint_secrets: None
) -> None:
    del endpoint_secrets
    bad_yaml = valid_config_yaml() + "\nbogus_top_level_key: true\n"

    with pytest.raises(ValidationError) as exc_info:
        load_config(write_config(tmp_path, bad_yaml))

    assert ("bogus_top_level_key",) in {
        tuple(error["loc"]) for error in exc_info.value.errors()
    }


def test_type_mismatch_raises_validation_error(
    tmp_path: Path, endpoint_secrets: None
) -> None:
    del endpoint_secrets
    bad_yaml = valid_config_yaml().replace(
        "stylometric_drift_threshold: 0.12",
        'stylometric_drift_threshold: "not-a-number"',
    )

    with pytest.raises(ValidationError) as exc_info:
        load_config(write_config(tmp_path, bad_yaml))

    assert ("thresholds", "stylometric_drift_threshold") in {
        tuple(error["loc"]) for error in exc_info.value.errors()
    }


def test_missing_endpoint_secret_raises_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for env_name in ENDPOINT_ENV_NAMES:
        monkeypatch.delenv(env_name, raising=False)

    with pytest.raises(ValidationError, match="api_key|secret|PLANNER_API_KEY"):
        load_config(write_config(tmp_path, valid_config_yaml()))
