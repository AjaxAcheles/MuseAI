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
planning:
  execution_mode: "macro_outline_before_draft"
  approval_mode: "off"
  planner_max_deliberation_loops: {global: 4, arc: 3, chapter: 3, scene: 2, beat: 2}
  planner_max_tool_calls_per_loop: {global: 8, arc: 6, chapter: 5, scene: 4, beat: 3}
  planner_required_checks:
    global: ["schema", "arc_coverage", "major_promise_payoff"]
    arc: ["schema", "escalation", "thread_distribution"]
    chapter: ["schema", "chapter_function", "pacing", "annotation_satisfaction"]
    scene: ["schema", "continuity", "scene_function", "entry_exit_state"]
    beat: ["schema", "draftability", "continuity", "pad_grounding"]
runtime:
  model_validate_retry_cap: 3
  headless_mode: false
  beats_per_scene_min: 3
  word_count_target: 80000
  inference_timeout_seconds: 120
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
    assert config.planning.execution_mode == "macro_outline_before_draft"
    assert config.planning.approval_mode == "off"
    assert config.planning.planner_max_deliberation_loops["global"] == 4
    assert config.planning.planner_max_tool_calls_per_loop["beat"] == 3
    assert config.planning.planner_required_checks["beat"] == [
        "schema",
        "draftability",
        "continuity",
        "pad_grounding",
    ]


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


def test_macro_outline_approval_requires_macro_execution_mode(
    tmp_path: Path, endpoint_secrets: None
) -> None:
    """approval_mode='macro_outline' is invalid unless execution_mode is macro-first."""
    del endpoint_secrets
    bad_yaml = valid_config_yaml().replace(
        'execution_mode: "macro_outline_before_draft"',
        'execution_mode: "rolling"',
    ).replace(
        'approval_mode: "off"',
        'approval_mode: "macro_outline"',
    )

    with pytest.raises(
        ValidationError, match="approval_mode.*macro_outline|execution_mode"
    ):
        load_config(write_config(tmp_path, bad_yaml))


def test_headless_with_approval_gate_raises(
    tmp_path: Path, endpoint_secrets: None
) -> None:
    """A headless run must never enable an approval gate it can never clear."""
    del endpoint_secrets
    # execution_mode stays macro-first (so the first rule passes), isolating the
    # headless/approval conflict as the failing rule.
    bad_yaml = valid_config_yaml().replace(
        "headless_mode: false",
        "headless_mode: true",
    ).replace(
        'approval_mode: "off"',
        'approval_mode: "macro_outline"',
    )

    with pytest.raises(ValidationError, match="headless"):
        load_config(write_config(tmp_path, bad_yaml))


def test_default_repo_config_still_loads(endpoint_secrets: None) -> None:
    """Regression guard: the real config.yaml loads now that planning is required."""
    del endpoint_secrets
    repo_config = Path(__file__).resolve().parents[1] / "config.yaml"

    config = load_config(repo_config)

    assert config.planning.execution_mode in {
        "rolling",
        "macro_outline_before_draft",
    }
    assert config.planning.approval_mode in {"off", "macro_outline"}


def test_unknown_required_check_name_fails_at_load(
    tmp_path: Path, endpoint_secrets: None
) -> None:
    del endpoint_secrets
    # A typo in a required-check name would otherwise load cleanly and only fail
    # mid-cascade (fail-closed) once a run is underway; it must be fatal at boot.
    bad_yaml = valid_config_yaml().replace('"escalation"', '"escalaton"')
    assert '"escalaton"' in bad_yaml  # guard: the replacement actually happened

    with pytest.raises(ValidationError) as exc_info:
        load_config(write_config(tmp_path, bad_yaml))

    assert "escalaton" in str(exc_info.value)


def test_no_drafting_is_an_accepted_required_check(
    tmp_path: Path, endpoint_secrets: None
) -> None:
    del endpoint_secrets
    # The prose-boundary guard is a registered validator, so wiring it into a
    # level's required checks (as config.yaml now does) loads cleanly.
    yaml_with_guard = valid_config_yaml().replace(
        'beat: ["schema", "draftability", "continuity", "pad_grounding"]',
        'beat: ["schema", "no_drafting", "draftability", "continuity", "pad_grounding"]',
    )
    config = load_config(write_config(tmp_path, yaml_with_guard))
    assert "no_drafting" in config.planning.planner_required_checks["beat"]
