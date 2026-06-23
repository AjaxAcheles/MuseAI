"""Module: M02 (Persistent Memory Stores & Interfaces)
Synthetic tests for the append-only event log (`memory/event_log.py`, exercising
04.04–04.05): `.jsonl` round-trip, in-order iteration, chronological tailing, and
the design rule that PAD state travels nested inside a `beat_commit` payload (never
as a standalone event).

All fixtures are file-local under ``tmp_path`` — these tests never write to the
real ``data/`` directory. Crash replay, branch restore, `_apply_event`, snapshot
logic, and SQLite relational writes are out of scope here.
"""

import json

import memory.event_log as event_log
from memory.event_log import iter_events, tail_events, write_event

# Three ordered synthetic events; distinct `seq` values make write order checkable.
EVENTS = [
    {"event": "scene_open", "seq": 1, "scene_id": "s1"},
    {"event": "beat_draft", "seq": 2, "beat_id": "b1"},
    {"event": "beat_commit", "seq": 3, "beat_id": "b1"},
]

# A beat_commit carrying the per-character PAD snapshot nested inside the payload,
# matching the design's "PAD travels inside the beat commit" atomicity rule.
BEAT_COMMIT = {
    "event": "beat_commit",
    "beat_id": "b1",
    "fsm_pointer": {"arc_id": "a1", "chapter_id": "c1", "scene_id": "s1", "beat_index": 0},
    "prose_delta": "The door clicked shut behind her.",
    "thread_updates": [{"id": "t1", "status": "progressing"}],
    "pad_states": {
        "char1": {"pleasure": 0.4, "arousal": -0.2, "dominance": 0.7},
        "char2": {"pleasure": -0.8, "arousal": 0.9, "dominance": -0.6},
    },
}


def _log_path(tmp_path):
    """A nested temp log path — proves write_event creates parents, stays in tmp_path."""
    return tmp_path / "data" / "events.jsonl"


def test_write_event_appends_exactly_three_json_lines(tmp_path):
    log = _log_path(tmp_path)
    for event in EVENTS:
        write_event(log, event)

    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    assert all(isinstance(obj, dict) for obj in parsed)
    assert parsed == EVENTS


def test_iter_events_returns_all_events_in_write_order(tmp_path):
    log = _log_path(tmp_path)
    for event in EVENTS:
        write_event(log, event)

    assert list(iter_events(log)) == EVENTS
    assert [e["seq"] for e in iter_events(log)] == [1, 2, 3]


def test_tail_events_returns_last_two_in_chronological_order(tmp_path):
    log = _log_path(tmp_path)
    for event in EVENTS:
        write_event(log, event)

    tail = tail_events(log, 2)
    assert tail == EVENTS[-2:]
    assert [e["seq"] for e in tail] == [2, 3]  # original order, not reversed


def test_tail_events_non_positive_limit_returns_empty(tmp_path):
    log = _log_path(tmp_path)
    for event in EVENTS:
        write_event(log, event)

    assert tail_events(log, 0) == []
    assert tail_events(log, -1) == []


def test_missing_log_iterates_and_tails_empty(tmp_path):
    missing = _log_path(tmp_path)  # never written
    assert list(iter_events(missing)) == []
    assert tail_events(missing, 5) == []


def test_beat_commit_with_nested_pad_round_trips_as_one_event(tmp_path):
    log = _log_path(tmp_path)
    write_event(log, BEAT_COMMIT)

    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1  # the whole commit is a single ledger record

    events = list(iter_events(log))
    assert events == [BEAT_COMMIT]
    # PAD state is carried inside the beat_commit payload, not as its own record.
    assert events[0]["pad_states"]["char1"] == {
        "pleasure": 0.4,
        "arousal": -0.2,
        "dominance": 0.7,
    }


def test_no_public_helper_writes_standalone_pad_events():
    public_names = [name for name in dir(event_log) if not name.startswith("_")]
    # No public symbol exists for writing PAD as its own event — PAD only ever
    # travels nested inside a beat_commit payload via write_event.
    assert [name for name in public_names if "pad" in name.lower()] == []
    assert "write_event" in public_names
