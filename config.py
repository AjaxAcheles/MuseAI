"""Central configuration for MuseAI.

All knobs live here so the pipeline, app, and storage layers share one source of
truth. Values come from the environment (loaded from a local .env if present).
"""
from __future__ import annotations

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


@dataclass(frozen=True)
class Settings:
    planning_model: str = os.environ.get("MUSEAI_PLANNING_MODEL", "claude-opus-4-8")
    drafting_model: str = os.environ.get("MUSEAI_DRAFTING_MODEL", "claude-opus-4-8")
    db_path: str = os.environ.get("MUSEAI_DB_PATH", "museai.db")
    host: str = os.environ.get("MUSEAI_HOST", "localhost")
    port: int = int(os.environ.get("MUSEAI_PORT", "8000"))
    # Planning stages think hard; keep their output cap modest.
    planning_max_tokens: int = 16000
    # Upload guardrails.
    max_upload_bytes: int = 20 * 1024 * 1024
    allowed_upload_exts: tuple[str, ...] = field(
        default=(".txt", ".md", ".markdown", ".docx", ".pdf")
    )

    @property
    def has_api_key(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))


def length_preset(name: str | None) -> dict[str, int]:
    return LENGTH_PRESETS.get((name or DEFAULT_LENGTH).lower(), LENGTH_PRESETS[DEFAULT_LENGTH])


settings = Settings()
