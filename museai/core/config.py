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


# The five LLM-powered agent roles. Every ``agents:`` key must be one of these.
AGENT_ROLES = ("chapter_planner", "beat_planner", "drafter", "reviser", "critic")


def _resolve_env_api_key(value: str) -> str:
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
    # Maximum gap between streamed chunks, separate from request_timeout (which
    # governs connect/write/pool). A reasoning endpoint legitimately pauses far
    # longer between tokens than it takes to open a connection.
    stream_read_timeout: int = 300
    temperature: float = 0.7
    # Sent as max_tokens when set. None omits the field entirely, preserving the
    # v1.02 wire-format finding: some endpoints bill hidden reasoning against
    # this budget. Truncation is detected via finish_reason either way.
    max_output_tokens: int | None = None

    @field_validator("api_key")
    @classmethod
    def _resolve_env_ref(cls, value: str) -> str:
        return _resolve_env_api_key(value)


class AgentEndpointOverride(BaseModel):
    """Sparse per-agent override of the shared endpoint.

    Every field is optional: only the fields set here differ for that agent,
    and everything left unset is inherited from the top-level ``endpoint``. A
    config with no ``agents:`` section behaves exactly as before.
    """

    model_config = ConfigDict(extra="forbid")

    base_url: str | None = None
    api_key: str | None = None
    model_name: str | None = None
    tokenizer_family: Literal["tiktoken", "char_heuristic"] | None = None
    request_timeout: int | None = None
    stream_read_timeout: int | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None

    @field_validator("api_key")
    @classmethod
    def _resolve_env_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _resolve_env_api_key(value)


class GenerationConfig(BaseModel):
    """Generation shape: targets and caps for the draft loop."""

    model_config = ConfigDict(extra="forbid")

    # Whole-manuscript stop signal. 0, empty, or absent means no word limit: the
    # run ends only when every outlined beat is committed. A seeded project's own
    # target takes precedence over this value.
    word_count_target: int | None = None
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
    # --- Repetition guard (audit) ------------------------------------------
    # Similarity at or above which two paragraphs count as "the same" prose.
    # A proportion: 0.9 means 90% of the difflib ratio.
    repetition_threshold: float
    # A duplicated run must reach this many consecutive near-duplicate sentences
    # (or a whole paragraph) before it is faulted. Short deliberate refrains stay
    # under the gate and are never touched.
    repetition_min_run: int
    # Phrases the author has declared may recur verbatim. The beat planner may
    # also register a refrain per beat (`intended_refrain`); both are exempt.
    repetition_allowlist: list[str] = []
    # --- Emotion-tell guard (audit) ----------------------------------------
    # Proportion of a beat's sentences that may name an emotion outright before
    # the audit faults the draft for telling rather than showing.
    emotion_word_threshold: float
    # The named-emotion vocabulary the guard counts. Data, not logic: overridable
    # in config, with a sensible default so a fresh config need not restate it.
    emotion_words: list[str] = [
        "panic", "terror", "horror", "dread", "rage", "fury", "despair",
        "anguish", "misery", "grief", "guilt", "shame", "spite", "hatred",
        "anxiety", "fear", "elation", "euphoria",
    ]
    # --- Style-tic guard (audit) --------------------------------------------
    # Proportion of a beat's sentences that may lean on a stock gesture or an
    # abstract emotional shorthand before the audit faults the draft. Separate
    # from the emotion gate so the two can be tuned independently.
    tic_phrase_threshold: float = 0.15
    # The stock-phrase vocabulary the guard counts. Multi-word phrases are
    # matched whole. Data, not logic: overridable in config, defaulted to the
    # tics observed in generated drafts so a fresh config need not restate them.
    tic_phrases: list[str] = [
        "trembling", "trembled", "deep breath", "shaky breath",
        "tears welled", "welled with tears", "eyes filled with tears",
        "traced the handwriting", "traced the letters", "traced the words",
        "heavy silence", "silence hung", "the weight of",
        "shared history", "legacy", "closure", "bittersweet",
    ]
    # --- Intensity arc (beat planner) --------------------------------------
    # How many times the beat planner is re-prompted for a varied emotional arc
    # when its plan comes back nearly all high-arousal. Bounded, then accepted.
    planner_intensity_retries: int
    # Absolute target-arousal above which a beat counts as "hot".
    intensity_hot_threshold: float
    # Fraction of a chapter's beats that may be hot before the plan is re-prompted.
    intensity_flat_fraction: float
    # --- Agent tools --------------------------------------------------------
    # Offers `web_search` to every agent when true. Off by default: agents
    # ground themselves in the story's own canon (seed, outline, threads,
    # committed manuscript); the public web is an explicit research mode, not a
    # default reflex.
    research_mode: bool = False
    # How many times any one tool may be called within a single agent loop.
    # Stops a model from spending its bounded iterations re-running the same
    # search instead of answering.
    tool_call_cap: int = 3
    # Wall-clock bound for one tool execution. The loop turns expiry into a
    # structured tool_timeout result instead of hanging the generation task.
    tool_timeout: float = 30.0

    @field_validator("word_count_target", mode="before")
    @classmethod
    def _optional_target(cls, value: object) -> object:
        """0, None, and "" all mean "no limit"; a negative target is a typo."""
        if value is None or value == 0 or (isinstance(value, str) and not value.strip()):
            return None
        if isinstance(value, int) and value < 0:
            raise ValueError(f"word_count_target must be >= 0 or empty, got {value}")
        return value

    @field_validator(
        "passive_voice_threshold",
        "repetition_threshold",
        "emotion_word_threshold",
        "tic_phrase_threshold",
        "intensity_hot_threshold",
        "intensity_flat_fraction",
    )
    @classmethod
    def _proportion(cls, value: float, info: ValidationInfo) -> float:
        """A threshold outside 0..1 silently disables the gate. Fail at boot."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{info.field_name} is a proportion in 0..1, got {value}"
            )
        return value

    @field_validator(
        "critic_parse_retries", "planner_parse_retries", "planner_intensity_retries"
    )
    @classmethod
    def _non_negative(cls, value: int, info: ValidationInfo) -> int:
        """0 is legal — never re-prompt — but a negative budget is a typo."""
        if value < 0:
            raise ValueError(f"{info.field_name} must be >= 0, got {value}")
        return value

    @field_validator("critic_degrade_threshold", "repetition_min_run", "tool_call_cap")
    @classmethod
    def _positive(cls, value: int, info: ValidationInfo) -> int:
        """A threshold of 0 would degrade before the first failure ever happened."""
        if value < 1:
            raise ValueError(f"{info.field_name} must be >= 1, got {value}")
        return value

    @field_validator("tool_timeout")
    @classmethod
    def _positive_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(f"tool_timeout must be > 0, got {value}")
        return value


class AppConfig(BaseModel):
    """Top-level application configuration."""

    model_config = ConfigDict(extra="forbid")

    project_id: str
    endpoint: EndpointConfig
    generation: GenerationConfig

    # Sparse per-agent inference overrides, keyed by agent role. An agent with
    # no entry uses ``endpoint`` unchanged, so existing single-endpoint configs
    # keep working without modification.
    agents: dict[str, AgentEndpointOverride] = {}

    log_level: str = "INFO"
    host: str = "127.0.0.1"
    port: int = 8000
    db_path: str = "data/museai.db"
    event_log_path: str = "data/events.jsonl"
    # Dev-only reset route guard. Production deployments should set this false.
    allow_reset: bool = True
    # Seconds the agents' web_search tool waits on a search engine.
    web_search_timeout: int = 10

    @field_validator("agents")
    @classmethod
    def _known_agent_roles(
        cls, value: dict[str, AgentEndpointOverride]
    ) -> dict[str, AgentEndpointOverride]:
        """A typo'd agent role must fail at boot, not silently use the default."""
        unknown = sorted(set(value) - set(AGENT_ROLES))
        if unknown:
            raise ValueError(
                f"unknown agent role(s) in agents: {unknown}; "
                f"known roles: {', '.join(AGENT_ROLES)}"
            )
        return value

    def endpoint_for(self, agent: str) -> EndpointConfig:
        """The inference endpoint one agent role actually uses.

        The shared ``endpoint`` with that agent's sparse overrides applied.
        An agent with no override — or a role this config never mentions —
        gets the shared endpoint itself.
        """
        override = self.agents.get(agent)
        if override is None:
            return self.endpoint
        merged = self.endpoint.model_dump()
        merged.update(
            {
                key: value
                for key, value in override.model_dump().items()
                if value is not None
            }
        )
        return EndpointConfig(**merged)


def persist_project_id(project_id: str, path: str | Path = "config.yaml") -> AppConfig:
    """Rewrite ``project_id`` in the config file and return the reloaded config.

    Edits the raw YAML document rather than dumping a resolved ``AppConfig`` so
    unresolved ``${VAR}`` references (the API key) survive the write.
    """
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise ConfigError(f"config file not found: {cfg_path}")
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse YAML in {cfg_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(
            f"config root must be a mapping, got {type(raw).__name__}: {cfg_path}"
        )
    raw["project_id"] = project_id
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_config(cfg_path)


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
