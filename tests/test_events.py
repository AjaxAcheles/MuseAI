"""Tests for the append-only event log."""

from __future__ import annotations

from museai.core.events import append_event, replay_events


def test_append_then_replay_preserves_order(tmp_path):
    path = tmp_path / "events.jsonl"
    events = [
        {"type": "beat_commit", "beat_id": "b0", "word_count": 1},
        {"type": "beat_commit", "beat_id": "b1", "word_count": 2},
        {"type": "note", "text": "unicode ✦ ok"},
    ]
    for ev in events:
        append_event(path, ev)

    replayed = list(replay_events(path))
    assert replayed == events


def test_replay_missing_log_is_empty(tmp_path):
    assert list(replay_events(tmp_path / "nope.jsonl")) == []


def test_append_creates_parent_dir(tmp_path):
    path = tmp_path / "nested" / "dir" / "events.jsonl"
    append_event(path, {"type": "x"})
    assert path.exists()
    assert list(replay_events(path)) == [{"type": "x"}]


def test_replay_skips_torn_final_line(tmp_path):
    """A crash mid-append leaves a torn line; recovery must survive it."""
    path = tmp_path / "events.jsonl"
    append_event(path, {"type": "beat_commit", "beat_id": "b0"})
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"type": "beat_commit", "beat_id": "b1", "word_co')

    assert list(replay_events(path)) == [{"type": "beat_commit", "beat_id": "b0"}]


def test_replay_skips_non_object_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('"just a string"\n')
        fh.write('{"type": "ok"}\n')

    assert list(replay_events(path)) == [{"type": "ok"}]
