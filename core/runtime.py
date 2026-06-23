"""Module: M14 (Configuration, Startup & Observability)
Manage application resource lifecycle and development reset behavior.

Store initializers are labelled no-op placeholders until their owning modules
implement real schemas, clients, indexes, and replay behavior.
"""

from pathlib import Path
import shutil

from memory.branch_manager import init_snapshot_store
from memory.chroma_client import init_chroma_store
from memory.event_log import init_event_log
from memory.graphiti_client import init_graphiti_store
from memory.raptor import init_raptor_store
from memory.sqlite_db import init_db
from memory.style_store import init_style_stores

DATA_DIR = Path("data")
LOG_DIR = Path("logs")
SQLITE_DB_PATH = DATA_DIR / "fictionwriter.db"
GRAPHITI_DB_PATH = DATA_DIR / "graphiti.db"
CHROMA_STORE_DIR = DATA_DIR / "chroma"
STYLE_STORE_DIR = DATA_DIR / "styles"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
EVENT_LOG_PATH = DATA_DIR / "events.jsonl"


def init_resources(config) -> None:
    """Initialize runtime directories and placeholder store artifacts."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    init_db(SQLITE_DB_PATH)
    init_graphiti_store(config, GRAPHITI_DB_PATH)
    init_chroma_store(config, CHROMA_STORE_DIR)
    init_style_stores(config, STYLE_STORE_DIR)
    init_snapshot_store(config, SNAPSHOT_DIR)
    init_event_log(config, EVENT_LOG_PATH)
    init_raptor_store(config)
    scan_startup_crash_sentinel(config)


def reset_resources(config) -> None:
    """Delete file-based store artifacts and re-run resource initialization."""
    _remove_artifact(SQLITE_DB_PATH)
    _remove_artifact(GRAPHITI_DB_PATH)
    _remove_artifact(CHROMA_STORE_DIR)
    _remove_artifact(EVENT_LOG_PATH)
    _remove_style_artifacts()
    _remove_snapshot_archives()
    init_resources(config)


def scan_startup_crash_sentinel(config) -> None:
    """No-op hook until CommitIntent crash-sentinel scanning exists.

    The real scan depends on the relational store and intent records built in a
    later commit/crash increment. For now, startup remains non-blocking.
    """
    del config


def _remove_style_artifacts() -> None:
    if not STYLE_STORE_DIR.exists():
        return
    for path in STYLE_STORE_DIR.glob("style_*.json"):
        _remove_artifact(path)


def _remove_snapshot_archives() -> None:
    if not SNAPSHOT_DIR.exists():
        return
    for path in SNAPSHOT_DIR.glob("*.zip"):
        _remove_artifact(path)


def _remove_artifact(path: Path) -> None:
    _assert_safe_artifact_path(path)
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _assert_safe_artifact_path(path: Path) -> None:
    resolved = path.resolve(strict=False)
    allowed_roots = (
        DATA_DIR.resolve(strict=False),
        LOG_DIR.resolve(strict=False),
    )
    if not any(_is_relative_to(resolved, root) for root in allowed_roots):
        raise ValueError(f"Refusing to delete path outside runtime artifacts: {path}")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
