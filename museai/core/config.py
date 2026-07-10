"""Strict configuration loading for MuseAI v1.

Config is loaded from ``config.yaml`` into Pydantic v2 models. Every model
forbids extra keys, so an unknown or mistyped key is fatal at boot. Thresholds,
caps, and endpoint details all live in config — nothing here is hardcoded.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    ValidationInfo,
    field_validator,
)

# Load .env once at import so ${VAR} references can be resolved.
load_dotenv()

_ENV_REF = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or is invalid."""


class EndpointConfig(BaseModel):
    """LLM endpoint description.

    Endpoint/model-agnostic on purpose: no provider, model, or port names appear
    in logic — everything the adapter needs is here.
    """

    model_config = ConfigDict(extra="forbid")

    base_url: str
    api_key: str
    model_name: str
    tokenizer_family: Literal["tiktoken", "char_heuristic"]
    request_timeout: int = 60
    temperature: float = 0.7

    @field_validator("api_key")
    @classmethod
    def _resolve_env_ref(cls, value: str) -> str:
        """Resolve a ``${VAR}`` api_key from the environment (via .env)."""
        match = _ENV_REF.match(value.strip())
        if not match:
            return value
        var = match.group(1)
        resolved = os.environ.get(var)
        if resolved is None or resolved == "":
            raise ValueError(
                f"api_key references environment variable ${{{var}}}, "
                f"but it is unset or empty"
            )
        return resolved


class GenerationConfig(BaseModel):
    """Generation shape: targets and caps for the draft loop."""

    model_config = ConfigDict(extra="forbid")

    word_count_target: int
    beat_word_target: int
    revision_retry_cap: int
    max_agent_iterations: int
    recent_prose_beats: int
    # Soft ceiling on the assembled drafting context. Over it, the context node
    # drops the lowest-priority material until the prompt fits.
    context_token_budget: int
    # Proportion of a beat's sentences that may be passive before the audit node
    # faults the draft. A proportion, not a count: 0.25 means a quarter.
    passive_voice_threshold: float
    # How many times the critic is re-prompted when its reply will not parse as
    # failure objects. Distinct from `revision_retry_cap`, which counts
    # draft->audit->revise cycles: this counts "say that again, correctly".
    critic_parse_retries: int
    # Consecutive beats whose critic output stayed unreadable after every retry
    # before the run degrades: validation loosens and the UI warns. A weak model
    # must not silently disable the continuity gate.
    critic_degrade_threshold: int
    # How many times a planner is re-prompted when its reply will not parse as a
    # JSON array. A planner cannot degrade the way the critic can — there is no
    # honest empty plan — so this budget, then a quote repair, then the run dies.
    planner_parse_retries: int

    @field_validator("passive_voice_threshold")
    @classmethod
    def _proportion(cls, value: float) -> float:
        """A threshold outside 0..1 silently disables the gate. Fail at boot."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"passive_voice_threshold is a proportion in 0..1, got {value}"
            )
        return value

    @field_validator("critic_parse_retries", "planner_parse_retries")
    @classmethod
    def _non_negative(cls, value: int, info: ValidationInfo) -> int:
        """0 is legal — never re-prompt — but a negative budget is a typo."""
        if value < 0:
            raise ValueError(f"{info.field_name} must be >= 0, got {value}")
        return value

    @field_validator("critic_degrade_threshold")
    @classmethod
    def _positive(cls, value: int) -> int:
        """A threshold of 0 would degrade before the first failure ever happened."""
        if value < 1:
            raise ValueError(f"critic_degrade_threshold must be >= 1, got {value}")
        return value


class AppConfig(BaseModel):
    """Top-level application configuration."""

    model_config = ConfigDict(extra="forbid")

    project_id: str
    endpoint: EndpointConfig
    generation: GenerationConfig

    log_level: str = "INFO"
    host: str = "127.0.0.1"
    port: int = 8000
    db_path: str = "data/museai.db"
    event_log_path: str = "data/events.jsonl"
    # Dev-only reset route guard. Production deployments should set this false.
    allow_reset: bool = True
    # Seconds the continuity critic's web_search tool waits on a search engine.
    web_search_timeout: int = 10


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load and validate configuration from a YAML file.

    Raises ``ConfigError`` on a missing file, invalid YAML, or any unknown key
    or type mismatch (extra keys are forbidden at every level).
    """
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise ConfigError(f"config file not found: {cfg_path}")

    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse YAML in {cfg_path}: {exc}") from exc

    if raw is None:
        raise ConfigError(f"config file is empty: {cfg_path}")
    if not isinstance(raw, dict):
        raise ConfigError(
            f"config root must be a mapping, got {type(raw).__name__}: {cfg_path}"
        )

    try:
        return AppConfig(**raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration in {cfg_path}:\n{exc}") from exc
