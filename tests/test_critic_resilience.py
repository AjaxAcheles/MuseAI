"""The critic must survive a model that cannot follow its schema.

On 2026-07-10 a live run died because `openbmb/minicpm5` answered the continuity
critic with the *schema template itself*: placeholder values, `offarming_text`
for `offending_text`, and no `critic_source`. `FailureObject` forbids extra keys,
so parsing raised and the exception killed a run holding two planned chapters and
a finished 336-word draft.

`PRODUCTION_FAILURE` below is that exact reply, byte for byte. Everything here
exists so it can never take a run down again.
"""

from __future__ import annotations

import logging

import pytest

from museai.core.stream_bus import bus
from museai.fsm.nodes import critics as critics_module
from museai.fsm.nodes.critics import adversarial_critics
from museai.fsm.nodes.deps import set_node_config
from museai.fsm.state import FSM_Pointer, make_initial_state
from museai.llm.structured import StructuredOutputError, parse_failure_objects

# The reply that killed the run, recovered verbatim from logs/llm_io.log.
PRODUCTION_FAILURE = (
    '```json\n'
    '[\n'
    '  {\n'
    '    "error_code": "CONTRADICTS_PRIOR_PROSE",\n'
    '    "offarming_text": "The exact sentence or phrase from the draft.",\n'
    '    "suggested_fix": "A concrete rewrite or correction. The text about Chloe '
    'feeling caught between loyalty and understanding is consistent with her character '
    'description and does not contradict established facts or context items."\n'
    '  }\n'
    ']\n'
    '```'
)

CORRECTED = """```json
[
  {
    "error_code": "CONTRADICTS_CHARACTER",
    "offending_text": "Mara lied about the letter.",
    "suggested_fix": "Mara never lies; have her say nothing."
  }
]
```"""

CLEAN = "[]"

DRAFT = "The sun stood at noon. Mara lied about the letter."

PACKAGE = {
    "beat": {"id": "arc-1-c01-b01", "ordering": 1},
    "pad_constraint": "Energy with nowhere to go.",
    "chapter": {"id": "arc-1-c01", "description": "Mara catalogs the letters.", "obligations": []},
    "threads": [],
    "characters": [
        {"id": "char-mara", "name": "Mara", "description": "She never lies.",
         "pad": {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0}}
    ],
    "recent_prose": [],
    "budget": {"budget": 8000, "tokens": 400, "tokens_before": 400,
               "dropped_prose_passages": 0, "dropped_threads": 0, "over_budget": False},
}


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tool_calls: list = []


def state_with(**overrides):
    return make_initial_state(
        "test-project",
        FSM_Pointer(arc_id="arc-1", chapter_id="arc-1-c01", beat_index=0),
        active_context_package=PACKAGE,
        current_draft_text=DRAFT,
        **overrides,
    )


@pytest.fixture
def configure(config_factory):
    """Install a node config with explicit retry/degrade budgets."""

    def _configure(**overrides):
        set_node_config(config_factory(**overrides))

    return _configure


@pytest.fixture
def scripted_loop(monkeypatch):
    """Replace `run_agent_loop` with a scripted sequence of critic replies.

    The last reply repeats once the script runs out, so a test can say "it never
    gets it right" without counting attempts.
    """
    calls: list[list[dict]] = []

    def _install(*replies: str):
        async def fake_loop(endpoint, messages, tools, tool_impls, max_iterations, on_event=None, **kwargs):
            calls.append(list(messages))
            index = min(len(calls) - 1, len(replies) - 1)
            return _Response(replies[index])

        monkeypatch.setattr(critics_module, "run_agent_loop", fake_loop)
        return calls

    return _install


@pytest.fixture
def health_events(monkeypatch):
    captured: list[dict] = []
    original = bus.publish

    async def spy(event_type, data):
        if event_type == "critic_health":
            captured.append(data)
        await original(event_type, data)

    monkeypatch.setattr(bus, "publish", spy)
    yield captured
    bus.last_snapshot.clear()


# ------------------------------------------------------- the production payload


def test_the_production_payload_still_fails_strict_parsing():
    """The regression's raw material. If this ever parses, the fixture is stale."""
    with pytest.raises(StructuredOutputError) as excinfo:
        parse_failure_objects(PRODUCTION_FAILURE)
    message = str(excinfo.value)
    assert "offending_text" in message  # the error names the field, for the re-prompt
    assert "offarming_text" in message


def test_lenient_mode_drops_the_production_payload_rather_than_faking_it():
    """`offarming_text` is ignored, which leaves no `offending_text` — so drop it."""
    assert parse_failure_objects(PRODUCTION_FAILURE, lenient=True) == []


async def test_the_production_payload_no_longer_kills_the_run(configure, scripted_loop, health_events):
    configure(critic_parse_retries=2, critic_degrade_threshold=3)
    scripted_loop(PRODUCTION_FAILURE)

    delta = await adversarial_critics(state_with())  # must not raise

    assert delta["critic_parse_failure_streak"] == 1
    assert "critic_failures" not in delta
    assert health_events[-1]["degraded"] is False


# ----------------------------------------------------------------- re-prompting


async def test_a_corrected_retry_succeeds_and_resets_the_streak(configure, scripted_loop):
    configure(critic_parse_retries=2, critic_degrade_threshold=3)
    calls = scripted_loop(PRODUCTION_FAILURE, CORRECTED)

    delta = await adversarial_critics(state_with(critic_parse_failure_streak=2))

    assert len(calls) == 2
    assert len(delta["critic_failures"]) == 1
    assert delta["critic_failures"][0].offending_text == "Mara lied about the letter."
    assert delta["critic_parse_failure_streak"] == 0


async def test_the_retry_shows_the_model_its_reply_and_the_validation_error(configure, scripted_loop):
    configure(critic_parse_retries=1, critic_degrade_threshold=3)
    calls = scripted_loop(PRODUCTION_FAILURE, CLEAN)

    await adversarial_critics(state_with())

    correction = calls[1]
    assert correction[-2]["role"] == "assistant"
    assert correction[-2]["content"] == PRODUCTION_FAILURE
    assert correction[-1]["role"] == "user"
    assert "offending_text" in correction[-1]["content"]  # the actual pydantic error
    assert "Do not return the example values" in correction[-1]["content"]


async def test_retries_are_bounded_by_the_config_key(configure, scripted_loop):
    configure(critic_parse_retries=2, critic_degrade_threshold=99)
    calls = scripted_loop(PRODUCTION_FAILURE)

    await adversarial_critics(state_with())

    assert len(calls) == 3, "one initial attempt plus two retries"


async def test_zero_retries_is_legal(configure, scripted_loop):
    configure(critic_parse_retries=0, critic_degrade_threshold=99)
    calls = scripted_loop(PRODUCTION_FAILURE)

    delta = await adversarial_critics(state_with())

    assert len(calls) == 1
    assert delta["critic_parse_failure_streak"] == 1


async def test_a_clean_critic_never_retries(configure, scripted_loop):
    configure(critic_parse_retries=2, critic_degrade_threshold=3)
    calls = scripted_loop(CLEAN)

    delta = await adversarial_critics(state_with())

    assert len(calls) == 1
    assert delta["critic_parse_failure_streak"] == 0


# ---------------------------------------------------------------- the streak


async def test_the_streak_accumulates_across_beats(configure, scripted_loop, health_events):
    """The streak spans beats: one bad beat is noise, three is a broken critic."""
    configure(critic_parse_retries=0, critic_degrade_threshold=3)
    scripted_loop(PRODUCTION_FAILURE)

    streak = 0
    for expected in (1, 2, 3):
        delta = await adversarial_critics(state_with(critic_parse_failure_streak=streak))
        streak = delta["critic_parse_failure_streak"]
        assert streak == expected

    assert [event["degraded"] for event in health_events] == [False, False, True]


async def test_one_readable_reply_clears_the_streak(configure, scripted_loop, health_events):
    configure(critic_parse_retries=0, critic_degrade_threshold=3)
    scripted_loop(CLEAN)

    delta = await adversarial_critics(state_with(critic_parse_failure_streak=5))

    assert delta["critic_parse_failure_streak"] == 0
    assert health_events[-1]["degraded"] is False


async def test_a_degraded_run_still_tries_strict_first(configure, scripted_loop, health_events):
    """Recovery must be possible: a well-formed reply clears a degraded run."""
    configure(critic_parse_retries=0, critic_degrade_threshold=2)
    scripted_loop(CORRECTED)

    delta = await adversarial_critics(state_with(critic_parse_failure_streak=7))

    assert delta["critic_parse_failure_streak"] == 0
    assert health_events[-1]["degraded"] is False
    assert len(delta["critic_failures"]) == 1


# ------------------------------------------------------------------- degrading


async def test_crossing_the_threshold_degrades_and_reports(configure, scripted_loop, health_events):
    configure(critic_parse_retries=0, critic_degrade_threshold=2)
    scripted_loop(PRODUCTION_FAILURE)

    delta = await adversarial_critics(state_with(critic_parse_failure_streak=1))

    assert delta["critic_parse_failure_streak"] == 2
    event = health_events[-1]
    assert event["degraded"] is True
    assert event["streak"] == 2
    assert event["threshold"] == 2
    # The production payload salvages to nothing, so lenient parsing found no findings.
    assert event["lenient_used"] is False
    assert "critic_failures" not in delta


async def test_degraded_mode_salvages_what_it_can(configure, scripted_loop, health_events):
    """An unknown key is ignored; the readable element survives."""
    salvageable = """```json
    [
      {
        "error_code": "CONTRADICTS_CHARACTER",
        "offending_text": "Mara lied about the letter.",
        "suggested_fix": "Mara never lies.",
        "severity": "high"
      }
    ]
    ```"""
    configure(critic_parse_retries=0, critic_degrade_threshold=1)
    scripted_loop(salvageable)

    delta = await adversarial_critics(state_with())

    assert health_events[-1]["degraded"] is True
    assert health_events[-1]["lenient_used"] is True
    assert len(delta["critic_failures"]) == 1
    assert delta["critic_failures"][0].critic_source == "continuity_critic"


async def test_degraded_mode_never_invents_an_empty_offending_text(configure, scripted_loop):
    """An empty needle makes `revise.locate` return None and rewrite the whole draft."""
    configure(critic_parse_retries=0, critic_degrade_threshold=1)
    scripted_loop(PRODUCTION_FAILURE)

    delta = await adversarial_critics(state_with())

    assert "critic_failures" not in delta


async def test_unparseable_prose_is_not_rescued_by_degrading(configure, scripted_loop, health_events):
    configure(critic_parse_retries=0, critic_degrade_threshold=1)
    scripted_loop("Honestly, the draft reads fine to me.")

    delta = await adversarial_critics(state_with())

    assert health_events[-1]["degraded"] is True
    assert health_events[-1]["lenient_used"] is False
    assert "critic_failures" not in delta


# ------------------------------------------------------------ audit interaction


async def test_programmatic_audit_failures_survive_a_broken_critic(configure, scripted_loop):
    """A dead critic must not launder away what `audit` already found."""
    from museai.fsm.state import FailureObject

    audit_failure = FailureObject(
        error_code="PASSIVE_VOICE_DENSITY",
        offending_text="The door was opened by Mara.",
        suggested_fix="Rewrite the passive clauses.",
        critic_source="programmatic_audit",
    )
    configure(critic_parse_retries=0, critic_degrade_threshold=3)
    scripted_loop(PRODUCTION_FAILURE)

    delta = await adversarial_critics(state_with(critic_failures=[audit_failure]))

    # The node omits `critic_failures` when it found none, so the reducer keeps audit's.
    assert "critic_failures" not in delta
    assert delta["best_seen_failure_count"] == 1


# ------------------------------------------------------ crash salvage (db side)


def test_reset_active_beats_only_touches_prose_less_active_beats(config_factory):
    """A beat with prose is committed or awaiting review. Neither is ours to undo."""
    from museai.memory.db import connect_db, init_db, reset_active_beats

    from conftest import add_beat, seed_project

    config = config_factory()
    init_db(config.db_path)
    seed_project(config)
    add_beat(config, beat_id="b-active", ordering=1, status="active", prose=None)
    add_beat(config, beat_id="b-done", ordering=2, status="completed", prose="Committed.", word_count=1)
    add_beat(config, beat_id="b-review", ordering=3, status="active", prose="Awaiting review.", word_count=2)
    add_beat(config, beat_id="b-planned", ordering=4, status="planned", prose=None)

    conn = connect_db(config.db_path)
    try:
        with conn:
            reset = reset_active_beats(conn, config.project_id)
        rows = {r["id"]: r["status"] for r in conn.execute("SELECT id, status FROM Beats")}
    finally:
        conn.close()

    assert reset == 1
    assert rows["b-active"] == "planned"
    assert rows["b-done"] == "completed"
    assert rows["b-review"] == "active"
    assert rows["b-planned"] == "planned"


def test_reset_active_beats_is_idempotent(config_factory):
    from museai.memory.db import connect_db, init_db, reset_active_beats

    from conftest import add_beat, seed_project

    config = config_factory()
    init_db(config.db_path)
    seed_project(config)
    add_beat(config, beat_id="b-active", ordering=1, status="active", prose=None)

    conn = connect_db(config.db_path)
    try:
        with conn:
            assert reset_active_beats(conn, config.project_id) == 1
        with conn:
            assert reset_active_beats(conn, config.project_id) == 0
    finally:
        conn.close()


# --------------------------------------------------- crash salvage (manager side)


def test_a_failed_run_writes_its_draft_and_frees_the_beat(config_factory, tmp_path, monkeypatch):
    from museai.fsm.manager import GenerationManager
    from museai.memory.db import connect_db, init_db

    from conftest import add_beat, seed_project

    monkeypatch.chdir(tmp_path)  # drafts are written to ./data/drafts
    config = config_factory()
    init_db(config.db_path)
    seed_project(config)
    add_beat(config, beat_id="arc-1-c01-b01", ordering=1, status="active", prose=None)

    manager = GenerationManager(config)
    manager.state = make_initial_state(
        config.project_id,
        FSM_Pointer(arc_id="arc-1", chapter_id="arc-1-c01", beat_index=0),
        active_context_package=PACKAGE,
        current_draft_text="The lamp held against the fog.",
    )

    draft_path = manager._salvage_failed_run()

    assert draft_path is not None
    written = (tmp_path / draft_path).read_text(encoding="utf-8")
    assert written == "The lamp held against the fog."

    conn = connect_db(config.db_path)
    try:
        status = conn.execute("SELECT status FROM Beats WHERE id='arc-1-c01-b01'").fetchone()[0]
    finally:
        conn.close()
    assert status == "planned"


def test_salvage_prefers_the_best_seen_draft(config_factory, tmp_path, monkeypatch):
    from museai.fsm.manager import GenerationManager
    from museai.memory.db import init_db

    from conftest import seed_project

    monkeypatch.chdir(tmp_path)
    config = config_factory()
    init_db(config.db_path)
    seed_project(config)

    manager = GenerationManager(config)
    manager.state = make_initial_state(
        config.project_id,
        FSM_Pointer(arc_id="arc-1", chapter_id="arc-1-c01", beat_index=0),
        active_context_package=PACKAGE,
        current_draft_text="A worse later draft.",
        best_seen_draft="The best draft seen.",
    )

    draft_path = manager._salvage_failed_run()
    assert (tmp_path / draft_path).read_text(encoding="utf-8") == "The best draft seen."


def test_salvage_writes_nothing_when_there_is_no_draft(config_factory, tmp_path, monkeypatch):
    from museai.fsm.manager import GenerationManager
    from museai.memory.db import init_db

    from conftest import seed_project

    monkeypatch.chdir(tmp_path)
    config = config_factory()
    init_db(config.db_path)
    seed_project(config)

    manager = GenerationManager(config)
    manager.state = make_initial_state(
        config.project_id,
        FSM_Pointer(arc_id="arc-1", chapter_id="arc-1-c01", beat_index=0),
        active_context_package=PACKAGE,
    )

    assert manager._salvage_failed_run() is None
    assert not (tmp_path / "data" / "drafts").exists()


def test_salvage_never_masks_the_original_failure(config_factory, tmp_path, monkeypatch):
    """A fault inside the handler must not replace the real exception's diagnosis."""
    from museai.fsm import manager as manager_module
    from museai.fsm.manager import GenerationManager
    from museai.memory.db import init_db

    from conftest import seed_project

    monkeypatch.chdir(tmp_path)
    config = config_factory()
    init_db(config.db_path)
    seed_project(config)

    def boom(*args, **kwargs):
        raise OSError("disk is on fire")

    monkeypatch.setattr(manager_module, "reset_active_beats", boom)

    manager = GenerationManager(config)
    manager.state = make_initial_state(
        config.project_id,
        FSM_Pointer(arc_id="arc-1", chapter_id="arc-1-c01", beat_index=0),
        active_context_package=PACKAGE,
        current_draft_text="Salvage me.",
    )

    draft_path = manager._salvage_failed_run()  # must not raise
    assert (tmp_path / draft_path).read_text(encoding="utf-8") == "Salvage me."


async def test_a_healthy_critic_does_not_warn(configure, scripted_loop, caplog):
    """Health fires once per beat. Warning on every clean beat trains you to skim."""
    configure(critic_parse_retries=0, critic_degrade_threshold=2)
    scripted_loop("[]")

    with caplog.at_level(logging.DEBUG, logger="museai"):
        await adversarial_critics(state_with(critic_parse_failure_streak=0))

    health = [r for r in caplog.records if "event=health" in r.getMessage()]
    assert len(health) == 1
    assert health[0].levelno == logging.INFO


async def test_a_degraded_critic_warns(configure, scripted_loop, caplog):
    configure(critic_parse_retries=0, critic_degrade_threshold=2)
    scripted_loop(PRODUCTION_FAILURE)

    with caplog.at_level(logging.DEBUG, logger="museai"):
        await adversarial_critics(state_with(critic_parse_failure_streak=1))

    health = [r for r in caplog.records if "event=health" in r.getMessage()]
    assert len(health) == 1
    assert health[0].levelno == logging.WARNING

    failed = [r for r in caplog.records if "event=parse_failed" in r.getMessage()]
    assert failed and all(r.levelno == logging.WARNING for r in failed)
