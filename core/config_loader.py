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
from pydantic import BaseModel, ConfigDict, field_validator

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


class RuntimeConfig(BaseModel):
    """Global generation keys (not per-endpoint)."""

    # 'model_validate_retry_cap' keys off config.yaml; opt out of the 'model_'
    # protected namespace so the design-mandated name is usable as a field.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_validate_retry_cap: int
    headless_mode: bool
    beats_per_scene_min: int
    word_count_target: int


class LoggingConfig(BaseModel):
    """Observability surface configuration."""

    model_config = ConfigDict(extra="forbid")

    log_level: str


class AppConfig(BaseModel):
    """Top-level validated configuration composing endpoints, thresholds, and global keys."""

    model_config = ConfigDict(extra="forbid")

    endpoints: EndpointsConfig
    thresholds: ThresholdsConfig
    runtime: RuntimeConfig
    logging: LoggingConfig


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
