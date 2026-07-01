"""Module: M14 (Configuration, Startup & Observability)
Parse config.yaml into a validated, strongly-typed AppConfig, applying
environment-sourced per-endpoint secrets.

Unknown keys and missing or empty endpoint secrets fail at load time. Every
numeric threshold is a calibratable proposed default read from config, never
hardcoded.
"""

import os
from pathlib import Path

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_validator,
    model_validator,
)

# The five planner cascade levels every per-level planning map must key over.
_PLANNER_LEVELS = frozenset({"global", "arc", "chapter", "scene", "beat"})

# Per-endpoint secrets are never stored in config.yaml. Each endpoint's API key
# is read from "{ENDPOINT_NAME_UPPER}_API_KEY" (mirrors .env.example).
_API_KEY_ENV_SUFFIX = "_API_KEY"


class EndpointConfig(BaseModel):
    """One LLM endpoint's routing surface and per-backend capability toggles."""

    model_config = ConfigDict(extra="forbid")

    base_url: str
    model_name: str
    tokenizer_family: str
    supports_concurrent_critics: bool
    grammar_constraint_strategy: str
    api_key: str  # populated from env at load time, not from config.yaml

    @field_validator("api_key")
    @classmethod
    def require_api_key(cls, value: str) -> str:
        """Reject missing or empty endpoint secrets at config-load time."""
        if not value:
            raise ValueError("endpoint api_key secret is required")
        return value


class EndpointsConfig(BaseModel):
    """The five named LLM endpoints the pipeline routes to."""

    model_config = ConfigDict(extra="forbid")

    planner: EndpointConfig
    drafter: EndpointConfig
    critic: EndpointConfig
    pad_translator: EndpointConfig
    craft_consultant: EndpointConfig


class ThresholdsConfig(BaseModel):
    """Every numeric proposed default; values are provisional and calibrated empirically."""

    model_config = ConfigDict(extra="forbid")

    stylometric_drift_threshold: float
    pad_ewma_alpha: float
    coreference_confidence_floor: float
    cluster_similarity_threshold: float
    passive_voice_density: float
    voice_evolution_l2_cap: float


class ContextConfig(BaseModel):
    """Context assembly sizing keys; defaults are provisional until calibrated."""

    model_config = ConfigDict(extra="forbid")

    token_budget: int
    coreference_high_confidence: float
    coreference_mid_confidence: float

    @field_validator("token_budget")
    @classmethod
    def require_positive_token_budget(cls, value: int) -> int:
        """Reject non-positive context package ceilings."""
        if value <= 0:
            raise ValueError("context token_budget must be positive")
        return value

    @field_validator("coreference_high_confidence", "coreference_mid_confidence")
    @classmethod
    def require_probability_band(cls, value: float) -> float:
        """Reject confidence bands outside the normalized probability range."""
        if not 0 <= value <= 1:
            raise ValueError("context coreference confidence bands must be between 0 and 1")
        return value

    def model_post_init(self, __context: object) -> None:
        """Require the high-confidence band to be at or above the mid band."""
        if self.coreference_high_confidence < self.coreference_mid_confidence:
            raise ValueError(
                "context coreference_high_confidence must be >= "
                "coreference_mid_confidence"
            )


class RuntimeConfig(BaseModel):
    """Global generation keys (not per-endpoint)."""

    # 'model_validate_retry_cap' keys off config.yaml; opt out of the 'model_'
    # protected namespace so the design-mandated name is usable as a field.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_validate_retry_cap: int
    headless_mode: bool
    beats_per_scene_min: int
    word_count_target: int
    inference_timeout_seconds: int


class PlanningConfig(BaseModel):
    """Five-level planner cascade: deliberation/tool-call caps, required validator
    checks, and the execution/approval mode. Caps and checks are proposed defaults
    read from config, never hardcoded in planner logic."""

    model_config = ConfigDict(extra="forbid")

    execution_mode: str
    approval_mode: str
    planner_max_deliberation_loops: dict[str, int]
    planner_max_tool_calls_per_loop: dict[str, int]
    planner_required_checks: dict[str, list[str]]
    # Deterministic-fallback (baseline) scaffold shape. Proposed defaults; the baseline
    # only runs when the planner loop yields no validated plan.
    baseline_act_count: int = 3
    baseline_word_weights: list[float] | None = None
    # When true, a validated plan gets a craft-consultant revision pass before persist; the
    # revision is re-validated and only kept if it still passes. Off until calibrated.
    creative_second_pass_enabled: bool = False

    @field_validator("baseline_act_count")
    @classmethod
    def require_positive_act_count(cls, value: int) -> int:
        """The fallback baseline must produce at least one arc."""
        if value < 1:
            raise ValueError("planning baseline_act_count must be >= 1")
        return value

    @field_validator("baseline_word_weights")
    @classmethod
    def require_non_negative_weights(cls, value: list[float] | None) -> list[float] | None:
        """Pacing weights, when given, must be non-negative and not all zero."""
        if value is None:
            return value
        if any(w < 0 for w in value):
            raise ValueError("planning baseline_word_weights must be non-negative")
        if sum(value) <= 0:
            raise ValueError("planning baseline_word_weights must not sum to zero")
        return value

    @field_validator("execution_mode")
    @classmethod
    def require_known_execution_mode(cls, value: str) -> str:
        """Reject any execution_mode outside the documented vocabulary."""
        allowed = {"rolling", "macro_outline_before_draft"}
        if value not in allowed:
            raise ValueError(
                f"planning execution_mode must be one of {sorted(allowed)}"
            )
        return value

    @field_validator("approval_mode")
    @classmethod
    def require_known_approval_mode(cls, value: str) -> str:
        """Reject any approval_mode outside the documented vocabulary."""
        allowed = {"off", "macro_outline"}
        if value not in allowed:
            raise ValueError(
                f"planning approval_mode must be one of {sorted(allowed)}"
            )
        return value

    @field_validator(
        "planner_max_deliberation_loops",
        "planner_max_tool_calls_per_loop",
        "planner_required_checks",
    )
    @classmethod
    def require_exact_planner_levels(cls, value: dict, info: ValidationInfo) -> dict:
        """Require each per-level map to key over exactly the five planner levels.

        A missing or extra level is a load-time error so the planner can never read
        an undefined level cap or check list.
        """
        keys = set(value)
        if keys != set(_PLANNER_LEVELS):
            raise ValueError(
                f"planning {info.field_name} must have exactly the five planner "
                f"levels {sorted(_PLANNER_LEVELS)}; got {sorted(keys)}"
            )
        return value

    @field_validator("planner_required_checks")
    @classmethod
    def require_known_check_names(cls, value: dict) -> dict:
        """Every configured required-check name must have a registered validator.

        Fail closed at boot rather than at first planner run: a typo like
        ``"escalaton"`` would otherwise load cleanly and only raise mid-cascade
        (validators.py's fail-closed guard) after the run is already underway. The
        registry is imported lazily so this low-level config module stays free of a
        static dependency on ``fsm``.
        """
        from fsm.planning_validators import REQUIRED_CHECK_REGISTRY

        known = set(REQUIRED_CHECK_REGISTRY)
        unknown = sorted(
            {name for names in value.values() for name in names if name not in known}
        )
        if unknown:
            raise ValueError(
                f"planning planner_required_checks names not registered as "
                f"validators: {unknown}; each required check must map to an "
                f"implementation in REQUIRED_CHECK_REGISTRY"
            )
        return value


class LoggingConfig(BaseModel):
    """Observability surface configuration."""

    model_config = ConfigDict(extra="forbid")

    log_level: str


class AppConfig(BaseModel):
    """Top-level validated configuration composing endpoints, thresholds, and global keys."""

    model_config = ConfigDict(extra="forbid")

    endpoints: EndpointsConfig
    thresholds: ThresholdsConfig
    context: ContextConfig
    planning: PlanningConfig
    runtime: RuntimeConfig
    logging: LoggingConfig

    @model_validator(mode="after")
    def _validate_planning_safety_rules(self) -> "AppConfig":
        """Reject unsafe planning/runtime combinations at load (§3a cross-field rules).

        These span two sub-models (``planning`` and ``runtime``), so they live here on
        the top-level config rather than on ``PlanningConfig`` alone. Both are safety
        rules: the message names the rule so a misconfiguration fails loudly at boot
        instead of stranding a run mid-flight.
        """
        if (
            self.planning.approval_mode == "macro_outline"
            and self.planning.execution_mode != "macro_outline_before_draft"
        ):
            raise ValueError(
                "planning.approval_mode == 'macro_outline' is valid only when "
                "planning.execution_mode == 'macro_outline_before_draft' (got "
                f"execution_mode='{self.planning.execution_mode}')"
            )
        if self.runtime.headless_mode and self.planning.approval_mode != "off":
            raise ValueError(
                "runtime.headless_mode is true but planning.approval_mode is "
                f"'{self.planning.approval_mode}': a headless run must never be parked "
                "at an approval gate it can never clear. Set approval_mode='off' or "
                "disable headless_mode."
            )
        return self


def load_config(path: str | os.PathLike) -> AppConfig:
    """Read the YAML at ``path``, apply env overrides for per-endpoint secrets,
    and return the validated top-level :class:`AppConfig`."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    endpoints = raw.get("endpoints") or {}
    for name, endpoint in endpoints.items():
        if not isinstance(endpoint, dict):
            continue
        env_key = f"{name.upper()}{_API_KEY_ENV_SUFFIX}"
        env_value = os.environ.get(env_key)
        if env_value is not None:
            endpoint["api_key"] = env_value

    return AppConfig.model_validate(raw)
