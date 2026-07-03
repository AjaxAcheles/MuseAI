"""Module: M14 (Configuration, Startup & Observability)
Manage application resource lifecycle, the shared runtime-resource container,
and development reset behavior.

Two entry points share the ``init_resources`` name for compatibility:

* ``init_resources(config)`` — the original M14 artifact-lifecycle path
  (02.05/INT-A·B contract): create runtime directories and store artifacts at
  the module-constant paths. Returns ``None``. Existing callers/tests keep
  working unchanged.
* ``init_resources()`` / ``init_resources(project_root=...)`` — the vertical
  slice's process-wide :class:`RuntimeResources` container: loads the strict
  typed config (applying ``.env`` endpoint secrets), initializes the real
  stores under ``<project_root>/data``, and returns the idempotent singleton
  that routes access through :func:`get_resources`.

The relational hub, event log, and provisional store initialize for real; the
remaining stores (Graphiti, RAPTOR, Chroma, style, snapshot archive) stay
honest no-op placeholders until their owning modules land, and the container
labels them ``stub`` so no surface can present them as working.

Nothing here runs at import time: no resource creation, no asyncio tasks.
"""

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import shutil
from typing import Any

from core.app_log import get_app_logger, log_app_event
from core.stream_bus import StreamBus
from memory.branch_manager import init_snapshot_store
from memory.chroma_client import init_chroma_store
from memory.event_log import init_event_log
from memory.graphiti_client import init_graphiti_store
from memory.provisional_store import init_provisional_store
from memory.raptor import init_raptor_store
from memory.sqlite_db import init_db
from memory.style_store import init_style_stores

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
LOG_DIR = Path("logs")
SQLITE_DB_PATH = DATA_DIR / "fictionwriter.db"
GRAPHITI_DB_PATH = DATA_DIR / "graphiti.db"
CHROMA_STORE_DIR = DATA_DIR / "chroma"
STYLE_STORE_DIR = DATA_DIR / "styles"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
EVENT_LOG_PATH = DATA_DIR / "events.jsonl"

# Sentinel distinguishing the legacy positional-config call from the new
# no-argument container call (both are named init_resources by contract).
_UNSET = object()

# The process-wide resource container. Never populated at import time.
_ACTIVE_RESOURCES: "RuntimeResources | None" = None


@dataclass
class StoreHandle:
    """One named store's honest status inside the resource container."""

    name: str
    kind: str  # "real" | "stub"
    path: Path | None = None
    note: str = ""


@dataclass
class RuntimeResources:
    """Explicit container for everything the running app shares.

    ``generation_manager`` is attached by the app factory (it lives in
    ``core/generation_manager.py`` and takes this container as input, so the
    dependency points one way and no import cycle forms).
    """

    config: Any
    project_root: Path
    data_dir: Path
    event_bus: StreamBus
    generation_manager: Any | None = None
    stores: dict[str, StoreHandle] = field(default_factory=dict)
    prompt_loader: Any | None = None
    logs_dir: Path | None = None
    app_logger: logging.Logger | None = None

    def log(self, event: str, **fields: Any) -> None:
        """Write one structured line to the app JSONL log (no-op if unwired)."""
        if self.app_logger is not None:
            log_app_event(self.app_logger, event, **fields)


def _load_env_file(path: Path) -> None:
    """Apply ``KEY=VALUE`` lines from a ``.env`` file; the shell env always wins."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


def _build_resources(
    project_root: Path,
    data_dir: Path,
    logs_dir: Path,
    config: Any = None,
) -> RuntimeResources:
    """Create one RuntimeResources container with real + labelled-stub stores."""
    repo_root = Path(__file__).resolve().parents[1]
    _load_env_file(project_root / ".env")
    if project_root != repo_root:
        _load_env_file(repo_root / ".env")
    if config is None:
        from core.config_loader import load_config

        config = load_config(repo_root / "config.yaml")

    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    sqlite_path = data_dir / "fictionwriter.db"
    event_log_path = data_dir / "events.jsonl"
    provisional_path = data_dir / "provisional_claims.db"
    init_db(sqlite_path)
    init_event_log(config, event_log_path)
    init_provisional_store(provisional_path)

    graphiti_path = data_dir / "graphiti.db"
    chroma_dir = data_dir / "chroma"
    style_dir = data_dir / "styles"
    snapshot_dir = data_dir / "snapshots"
    init_graphiti_store(config, graphiti_path)
    init_chroma_store(config, chroma_dir)
    init_style_stores(config, style_dir)
    init_snapshot_store(config, snapshot_dir)
    init_raptor_store(config)
    scan_startup_crash_sentinel(config)

    stores = {
        "sqlite": StoreHandle("sqlite", "real", sqlite_path, "relational hub (M02)"),
        "event_log": StoreHandle(
            "event_log", "real", event_log_path, "append-only .jsonl trace (M02)"
        ),
        "provisional": StoreHandle(
            "provisional", "real", provisional_path, "provisional-claim store (M02)"
        ),
        "graphiti": StoreHandle(
            "graphiti", "stub", graphiti_path, "no-op until the temporal graph lands"
        ),
        "raptor": StoreHandle(
            "raptor", "stub", None, "no-op until RAPTOR clustering lands"
        ),
        "chroma": StoreHandle(
            "chroma", "stub", chroma_dir, "no-op until the vector index lands"
        ),
        "style": StoreHandle(
            "style", "stub", style_dir, "no-op until stylometric baselines land"
        ),
        "snapshots": StoreHandle(
            "snapshots", "stub", snapshot_dir, "no-op until commit/crash snapshots land"
        ),
    }
    for handle in stores.values():
        if handle.kind == "stub":
            logger.info("runtime store %r initialized as a STUB (%s)", handle.name, handle.note)

    from prompts.prompt_loader import PromptLoader  # local import: avoid fsm/prompt cycles

    event_bus = StreamBus()
    event_bus.seed_snapshot(
        planning_execution_mode=config.planning.execution_mode,
        approval_mode=config.planning.approval_mode,
    )
    app_logger = get_app_logger(logs_dir / "app.jsonl")
    resources = RuntimeResources(
        config=config,
        project_root=project_root,
        data_dir=data_dir,
        event_bus=event_bus,
        generation_manager=None,
        stores=stores,
        prompt_loader=PromptLoader(),
        logs_dir=logs_dir,
        app_logger=app_logger,
    )
    resources.log(
        "runtime_init",
        data_dir=str(data_dir),
        stores={name: handle.kind for name, handle in stores.items()},
        execution_mode=config.planning.execution_mode,
        approval_mode=config.planning.approval_mode,
    )
    return resources


def init_resources(
    config: Any = _UNSET, *, project_root: Path | None = None
) -> "RuntimeResources | None":
    """Initialize runtime resources.

    Called with a positional ``config`` (the legacy M14 lifecycle contract), this
    creates runtime directories and store artifacts at the module-constant paths
    and returns ``None`` — exactly the pre-container behavior.

    Called with no positional argument, this builds (idempotently) and returns
    the process-wide :class:`RuntimeResources` container rooted at
    ``project_root`` (default: the repository root — never a hardcoded absolute
    path). Repeated calls return the same container.
    """
    if config is not _UNSET:
        _init_store_artifacts(config)
        return None

    global _ACTIVE_RESOURCES
    if _ACTIVE_RESOURCES is not None:
        return _ACTIVE_RESOURCES
    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    _ACTIVE_RESOURCES = _build_resources(
        project_root=root, data_dir=root / "data", logs_dir=root / "logs"
    )
    return _ACTIVE_RESOURCES


def get_resources() -> RuntimeResources:
    """Return the active RuntimeResources container (routes' access path)."""
    if _ACTIVE_RESOURCES is None:
        raise RuntimeError(
            "Runtime resources are not initialized; call init_resources() first."
        )
    return _ACTIVE_RESOURCES


def reset_resources_for_tests(
    tmp_path: Path, config: Any = None
) -> RuntimeResources:
    """Replace the active container with one rooted in an isolated temp tree.

    ``config`` may inject a synthetic typed config; when omitted the real
    ``config.yaml`` is loaded (endpoint secrets must be present in the env).
    """
    global _ACTIVE_RESOURCES
    root = Path(tmp_path).resolve()
    _ACTIVE_RESOURCES = _build_resources(
        project_root=root,
        data_dir=root / "data",
        logs_dir=root / "logs",
        config=config,
    )
    return _ACTIVE_RESOURCES


def _init_store_artifacts(config) -> None:
    """Initialize runtime directories and placeholder store artifacts (legacy path)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Relational hub: real M02 schema initialization (Arcs/Chapters/Scenes/Beats/
    # Threads/Characters/CharacterEmotions/CommitIntent/RaptorNodes) at the
    # documented data/fictionwriter.db path. The store initializers below remain
    # honest no-op stubs until their owning modules are built.
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
    _init_store_artifacts(config)


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
