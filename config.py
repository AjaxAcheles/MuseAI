"""Central configuration for MuseAI.

All knobs live here so the pipeline, app, and storage layers share one source of
truth. Values come from the environment (loaded from a local .env if present).

MuseAI can run each pipeline stage group against a different provider:
  - "anthropic" — Claude via the Anthropic SDK (needs ANTHROPIC_API_KEY)
  - "local"     — any OpenAI-compatible server (Ollama, LM Studio, vLLM,
                  llama.cpp) at MUSEAI_LOCAL_BASE_URL
Planning (premise/bible/outline/summary) and drafting (scenes/revision) each
pick their own provider + model, so you can mix (e.g. Claude planning + local
drafting) or run fully local.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


# Length presets steer the outline (scene count) and per-scene token budget.
# words is the rough target the prompts aim for.
LENGTH_PRESETS: dict[str, dict[str, int]] = {
    "flash": {"scenes": 1, "words": 1000, "max_tokens": 4000},
    "short": {"scenes": 4, "words": 3500, "max_tokens": 6000},
    "novelette": {"scenes": 8, "words": 9000, "max_tokens": 8000},
    "novella": {"scenes": 14, "words": 20000, "max_tokens": 10000},
}
DEFAULT_LENGTH = "short"

ANTHROPIC = "anthropic"
LOCAL = "local"


def _provider(env_key: str) -> str:
    val = os.environ.get(env_key, ANTHROPIC).strip().lower()
    return val if val in (ANTHROPIC, LOCAL) else ANTHROPIC


@dataclass(frozen=True)
class Settings:
    planning_model: str = os.environ.get("MUSEAI_PLANNING_MODEL", "claude-opus-4-8")
    drafting_model: str = os.environ.get("MUSEAI_DRAFTING_MODEL", "claude-opus-4-8")
    planning_provider: str = field(default_factory=lambda: _provider("MUSEAI_PLANNING_PROVIDER"))
    drafting_provider: str = field(default_factory=lambda: _provider("MUSEAI_DRAFTING_PROVIDER"))
    # Local Ollama server. This is the exact URL posted to (its native generate
    # endpoint) — nothing is appended to the path.
    local_base_url: str = os.environ.get("MUSEAI_LOCAL_BASE_URL", "http://localhost:11434/api/generate")
    local_api_key: str = os.environ.get("MUSEAI_LOCAL_API_KEY", "ollama")
    db_path: str = os.environ.get("MUSEAI_DB_PATH", "museai.db")
    host: str = os.environ.get("MUSEAI_HOST", "localhost")
    port: int = int(os.environ.get("MUSEAI_PORT", "8000"))
    log_level: str = os.environ.get("MUSEAI_LOG_LEVEL", "INFO").upper()
    # Planning stages think hard; keep their output cap modest.
    planning_max_tokens: int = 16000
    # Upload guardrails.
    max_upload_bytes: int = 20 * 1024 * 1024
    allowed_upload_exts: tuple[str, ...] = field(
        default=(".txt", ".md", ".markdown", ".docx", ".pdf")
    )

    @property
    def uses_anthropic(self) -> bool:
        return ANTHROPIC in (self.planning_provider, self.drafting_provider)

    def config_ready(self) -> tuple[bool, str]:
        """Whether the configured providers can run. The only blocking case is an
        Anthropic stage with no API key — local stages need no key here."""
        if self.uses_anthropic and not os.environ.get("ANTHROPIC_API_KEY"):
            which = []
            if self.planning_provider == ANTHROPIC:
                which.append("planning")
            if self.drafting_provider == ANTHROPIC:
                which.append("drafting")
            stages = " and ".join(which)
            return False, (
                f"ANTHROPIC_API_KEY is not set, but the {stages} "
                f"stage{'s' if len(which) > 1 else ''} use{'s' if len(which) == 1 else ''} "
                "Claude. Set the key in .env, or switch those stages to a local provider."
            )
        return True, "Ready."

    def active_providers(self) -> str:
        def label(provider: str, model: str) -> str:
            name = "Claude" if provider == ANTHROPIC else "local"
            return f"{name} ({model})"

        return (
            f"Planning: {label(self.planning_provider, self.planning_model)} · "
            f"Drafting: {label(self.drafting_provider, self.drafting_model)}"
        )


def length_preset(name: str | None) -> dict[str, int]:
    return LENGTH_PRESETS.get((name or DEFAULT_LENGTH).lower(), LENGTH_PRESETS[DEFAULT_LENGTH])


_logging_configured = False


def configure_logging() -> None:
    """Set up MuseAI logging once. Safe to call repeatedly."""
    global _logging_configured
    if _logging_configured:
        return
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # The HTTP/SDK layers are chatty at DEBUG; keep them at WARNING.
    for noisy in ("httpx", "httpcore", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _logging_configured = True


settings = Settings()
