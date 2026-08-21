"""Shared test fixtures."""

from __future__ import annotations

import logging
import tempfile
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Redirect the rotating log files into a temp directory before anything else
# imports a museai module. `get_logger()` runs at module import time (e.g.
# `runtime.py` builds one at top level) and auto-configures the handlers against
# `LOG_DIR`, so a fixture would be far too late: the test suite would already
# have opened — and would go on appending to — the real `logs/fsm.log` and
# `logs/llm_io.log` that the running app writes to.
from museai.core import logging_setup as _logging_setup  # isort: skip

_logging_setup.LOG_DIR = Path(tempfile.mkdtemp(prefix="museai-test-logs-"))

# Belt and braces: if some earlier import already attached handlers to the real
# files, drop them and let the next `configure_logging` rebuild against the temp
# directory.
for _name in ("museai", "museai.llm_io"):
    _logger = logging.getLogger(_name)
    for _handler in [h for h in _logger.handlers if isinstance(h, RotatingFileHandler)]:
        _logger.removeHandler(_handler)
        _handler.close()
_logging_setup._configured = False

import pytest  # noqa: E402

from museai.core.config import AppConfig, EndpointConfig, GenerationConfig  # noqa: E402
from museai.memory.db import (  # noqa: E402
    connect_db,
    upsert_arc,
    upsert_beat,
    upsert_chapter,
    upsert_project,
)
from museai.web.app import create_app  # noqa: E402


_GENERATION_DEFAULTS = dict(
    word_count_target=1500,
    narrative_person=None,
    revision_retry_cap=3,
    max_agent_iterations=6,
    recent_prose_beats=4,
    context_token_budget=8000,
    passive_voice_threshold=0.25,
    critic_parse_retries=2,
    critic_degrade_threshold=3,
    planner_parse_retries=2,
    served_window_mismatch_fraction=0.9,
    repetition_threshold=0.9,
    repetition_min_run=3,
    repetition_min_phrase_words=6,
    repetition_allowlist=[],
    # Matches config.yaml. A lower test-only floor would green-light density
    # gates that the shipped config skips on the same prose.
    passive_min_sentences=6,
    audit_offender_list_chars=1200,
    critic_verdict_max_chars=700,
    critic_max_findings_per_code=2,
    emotion_word_threshold=0.3,
    tic_phrase_threshold=0.15,
    planner_intensity_retries=1,
    intensity_hot_threshold=0.6,
    intensity_flat_fraction=0.8,
    research_mode=False,
    tool_call_cap=3,
    tool_timeout=0.2,
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
            # Isolation is structural, not a rule each test must remember: before
            # `output_dir`/`draft_dir` existed as config keys, `export_manuscript`
            # and the manager's draft-salvage path wrote to the hardcoded, real
            # `data/output/` and `data/drafts/` — a pytest run once overwrote a
            # live 5,657-word manuscript with 25 words of fixture output because
            # of it (2026-07-25 postmortem, B5).
            output_dir=str(tmp_path / "output"),
            draft_dir=str(tmp_path / "drafts"),
            allow_reset=True,
        )
        base.update(overrides)
        return AppConfig(**base)

    return _make


class MockManager:
    """Stands in for GenerationManager so web tests never touch an LLM or the graph."""

    def __init__(self, status: str = "idle") -> None:
        self.status = status
        self.state = None
        self.started: list[str] = []
        self.reviews: list[tuple[str, str | None]] = []

    async def start(self, project_id: str) -> None:
        self.started.append(project_id)
        self.status = "running"

    def pause(self) -> None:
        self.status = "paused"

    async def resume(self) -> None:
        self.status = "running"

    def stop(self) -> None:
        self.status = "stopped"

    async def resolve_review(self, decision: str, edited_text: str | None = None) -> None:
        self.reviews.append((decision, edited_text))
        self.status = "running"


@pytest.fixture
async def web_app(tmp_path):
    """Yield a factory returning a started Quart app; the serving context is torn down after."""
    started: list = []

    async def _make(config: AppConfig):
        app = create_app(config_path=tmp_path / "config.yaml", test_config=config)
        test_app = app.test_app()
        await test_app.__aenter__()
        started.append(test_app)
        return app

    yield _make

    for test_app in started:
        await test_app.__aexit__(None, None, None)


def seed_project(config: AppConfig, *, word_count_target: int = 3000) -> None:
    """Write the minimum a seeded project needs: one project row and one active arc."""
    conn = connect_db(config.db_path)
    try:
        with conn:
            upsert_project(
                conn,
                id=config.project_id,
                genre="mystery",
                premise="A precise premise.",
                word_count_target=word_count_target,
            )
            upsert_arc(
                conn,
                id=f"{config.project_id}-arc-1",
                project_id=config.project_id,
                ordering=0,
                description="A first arc.",
                status="active",
            )
    finally:
        conn.close()


def add_beat(config: AppConfig, *, beat_id: str, ordering: int, status: str, prose: str | None, word_count: int = 0) -> None:
    """Attach a beat (and its chapter, once) to the seeded project's first arc."""
    conn = connect_db(config.db_path)
    try:
        with conn:
            upsert_chapter(
                conn,
                id=f"{config.project_id}-ch-1",
                arc_id=f"{config.project_id}-arc-1",
                ordering=0,
                description="Chapter one.",
                status="active",
            )
            upsert_beat(
                conn,
                id=beat_id,
                chapter_id=f"{config.project_id}-ch-1",
                ordering=ordering,
                prose=prose,
                word_count=word_count,
                status=status,
            )
    finally:
        conn.close()


def patch_planner_llm(monkeypatch, *, chapter=None, beat=None) -> None:
    """Install fake planner responses at the seam both planners now share.

    `plan_chapter` and `plan_beat` no longer call `call_llm` directly — they go
    through `museai.llm.planning.call_llm_for_json_array`, which owns the
    re-prompt-and-repair ladder. Patching that away would skip the ladder, so
    tests patch the `call_llm` *inside* it and dispatch on the `agent` kwarg.
    Since the planners gained tools, that inner call runs through the agent
    loop, so the loop's own `call_llm` is patched with the same dispatch.
    """
    from museai.fsm.tools import loop as loop_module
    from museai.llm import planning as planning_module

    async def dispatch(endpoint, messages, **kwargs):
        agent = kwargs.get("agent")
        fake = {"chapter_planner": chapter, "beat_planner": beat}.get(agent)
        if fake is None:
            raise AssertionError(f"no planner fake installed for agent={agent!r}")
        return await fake(endpoint, messages, **kwargs)

    monkeypatch.setattr(planning_module, "call_llm", dispatch)
    monkeypatch.setattr(loop_module, "call_llm", dispatch)
