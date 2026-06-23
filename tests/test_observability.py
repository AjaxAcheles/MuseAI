"""Module: M14 (Configuration, Startup & Observability)
Synthetic tests for durable logging and resource lifecycle behavior.
"""

import json
import logging
from pathlib import Path

import pytest

import core.llm_io_logger as llm_io_logger
import core.logger as node_logger
import core.runtime as runtime


ENDPOINT_ENV_NAMES = (
    "PLANNER_API_KEY",
    "DRAFTER_API_KEY",
    "CRITIC_API_KEY",
    "PAD_TRANSLATOR_API_KEY",
    "CRAFT_CONSULTANT_API_KEY",
)


@pytest.fixture
def endpoint_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_name in ENDPOINT_ENV_NAMES:
        monkeypatch.setenv(env_name, f"{env_name.lower()}-secret")


def clear_logger(name: str) -> None:
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def test_node_logger_emits_structured_json_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endpoint_secrets: None
) -> None:
    del endpoint_secrets
    monkeypatch.setattr(node_logger, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(node_logger, "LOG_FILE", tmp_path / "logs" / "fsm.log")
    logger_name = "node_observability_test"
    clear_logger(logger_name)

    logger = node_logger.get_logger(logger_name)
    events = [
        ({"arc": "a1", "chapter": "c1", "scene": "s1", "beat": 1}, 12.5, "success", None),
        ({"arc": "a1", "chapter": "c1", "scene": "s1", "beat": 2}, 8.25, "escalated", None),
        ({"arc": "a1", "chapter": "c1", "scene": "s1", "beat": 3}, 3.0, "failure", "synthetic failure"),
    ]

    for pointer, duration_ms, outcome, error in events:
        node_logger.log_node_event(logger, pointer, duration_ms, outcome, error)
    for handler in logger.handlers:
        handler.flush()

    lines = node_logger.LOG_FILE.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(events)
    for line, (pointer, duration_ms, outcome, error) in zip(lines, events, strict=True):
        record = json.loads(line)
        assert set(record) == {
            "node_name",
            "fsm_pointer",
            "duration_ms",
            "outcome",
            "error",
        }
        assert record["node_name"] == logger_name
        assert record["fsm_pointer"] == pointer
        assert record["duration_ms"] == duration_ms
        assert record["outcome"] == outcome
        assert record["error"] == error


def test_inference_io_logger_emits_one_final_response_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm_io_logger, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(llm_io_logger, "LOG_FILE", tmp_path / "logs" / "llm_io.log")
    clear_logger(llm_io_logger.LOGGER_NAME)

    logger = llm_io_logger.get_llm_io_logger()
    request_payload = {
        "messages": [{"role": "user", "content": "synthetic prompt"}],
        "model": "synthetic-model",
        "temperature": 0.4,
        "max_tokens": 128,
        "grammar_constraint": {"type": "json_schema", "name": "Synthetic"},
    }
    llm_io_logger.log_llm_call(logger, request_payload, "final assembled response", 21.75)
    for handler in logger.handlers:
        handler.flush()

    lines = llm_io_logger.LOG_FILE.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert set(record) == {"request", "response", "duration_ms"}
    assert record["request"] == request_payload
    assert record["response"] == "final assembled response"
    assert record["duration_ms"] == 21.75
    assert "chunks" not in record
    assert "stream_chunks" not in record
    assert "token_chunks" not in record
    assert not (tmp_path / "logs" / "fsm.log").exists()


def test_lifecycle_init_reset_and_crash_sentinel_are_temp_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(runtime, "DATA_DIR", data_dir)
    monkeypatch.setattr(runtime, "LOG_DIR", log_dir)
    monkeypatch.setattr(runtime, "SQLITE_DB_PATH", data_dir / "fictionwriter.db")
    monkeypatch.setattr(runtime, "GRAPHITI_DB_PATH", data_dir / "graphiti.db")
    monkeypatch.setattr(runtime, "CHROMA_STORE_DIR", data_dir / "chroma")
    monkeypatch.setattr(runtime, "STYLE_STORE_DIR", data_dir / "styles")
    monkeypatch.setattr(runtime, "SNAPSHOT_DIR", data_dir / "snapshots")
    monkeypatch.setattr(runtime, "EVENT_LOG_PATH", data_dir / "events.jsonl")
    config = object()

    runtime.init_resources(config)
    assert data_dir.is_dir()
    assert log_dir.is_dir()
    assert runtime.SQLITE_DB_PATH.is_file()
    assert runtime.GRAPHITI_DB_PATH.is_dir()
    assert runtime.CHROMA_STORE_DIR.is_dir()
    assert runtime.STYLE_STORE_DIR.is_dir()
    assert runtime.SNAPSHOT_DIR.is_dir()
    assert runtime.EVENT_LOG_PATH.is_file()

    runtime.SQLITE_DB_PATH.write_text("database payload", encoding="utf-8")
    (runtime.GRAPHITI_DB_PATH / "graph-artifact").write_text("graph", encoding="utf-8")
    (runtime.CHROMA_STORE_DIR / "index-artifact").write_text("vector", encoding="utf-8")
    style_artifact = runtime.STYLE_STORE_DIR / "style_author.json"
    style_artifact.write_text("{}", encoding="utf-8")
    snapshot_artifact = runtime.SNAPSHOT_DIR / "chapter-001.zip"
    snapshot_artifact.write_text("snapshot", encoding="utf-8")
    runtime.EVENT_LOG_PATH.write_text("event", encoding="utf-8")

    before_scan = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    runtime.scan_startup_crash_sentinel(config)
    after_scan = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert after_scan == before_scan

    runtime.reset_resources(config)
    assert runtime.SQLITE_DB_PATH.is_file()
    # init_db now builds a real relational schema (not the empty placeholder of the
    # stub era), so the reset-then-reinit artifact is a freshly initialized SQLite
    # database — proving the stale "database payload" text written above was wiped.
    assert runtime.SQLITE_DB_PATH.read_bytes().startswith(b"SQLite format 3\x00")
    assert runtime.GRAPHITI_DB_PATH.is_dir()
    assert not (runtime.GRAPHITI_DB_PATH / "graph-artifact").exists()
    assert runtime.CHROMA_STORE_DIR.is_dir()
    assert not (runtime.CHROMA_STORE_DIR / "index-artifact").exists()
    assert runtime.STYLE_STORE_DIR.is_dir()
    assert not style_artifact.exists()
    assert runtime.SNAPSHOT_DIR.is_dir()
    assert not snapshot_artifact.exists()
    assert runtime.EVENT_LOG_PATH.is_file()
    assert runtime.EVENT_LOG_PATH.read_text(encoding="utf-8") == ""

    for _ in range(2):
        runtime.init_resources(config)
        runtime.reset_resources(config)

    assert runtime.SQLITE_DB_PATH.is_file()
    assert runtime.GRAPHITI_DB_PATH.is_dir()
    assert runtime.CHROMA_STORE_DIR.is_dir()
    assert runtime.STYLE_STORE_DIR.is_dir()
    assert runtime.SNAPSHOT_DIR.is_dir()
    assert runtime.EVENT_LOG_PATH.is_file()
