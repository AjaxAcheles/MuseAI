"""Module: M03 (Context Assembly & Budgeting)
Assemble layered, unpruned context packages from the available memory stores.
"""

from collections.abc import Callable
from copy import deepcopy
from typing import Any, TypeVar

import core.runtime as runtime
from memory import provisional_store, sqlite_db
from memory.chroma_client import ChromaClient
from memory.graphiti_client import GraphitiClient
from typing_extensions import TypedDict


LayerResult = TypeVar("LayerResult")

# First entry is lowest priority for the later budgeting increment; relational
# truth remains last because canonical SQLite facts are always retained.
DROP_ORDER: list[str] = [
    "flavour",
    "summaries",
    "coreference_candidates",
    "temporal",
    "macro_constraints",
    "relational",
]


class LayerMeta(TypedDict):
    """Serializable read status for one context layer."""

    available: bool
    reason: str


class ContextPackage(TypedDict):
    """Serializable context payload consumed by later drafting nodes."""

    relational: dict[str, Any]
    summaries: dict[str, Any]
    temporal: dict[str, Any]
    flavour: list[dict[str, Any]]
    coreference_candidates: list[dict[str, Any]]
    macro_constraints: dict[str, Any]
    meta: dict[str, Any]


def build_context_package(state: dict[str, Any]) -> ContextPackage:
    """Build the layered context package for the state's active pointer."""

    pointer = state["fsm_pointer"]
    db_path = state.get("sqlite_db_path", runtime.SQLITE_DB_PATH)
    provisional_path = state.get(
        "provisional_store_path", provisional_store.DEFAULT_PROVISIONAL_PATH
    )
    meta: dict[str, Any] = {"drop_order": list(DROP_ORDER)}

    relational, meta["relational"] = _safe_store_read(
        "relational",
        lambda: _read_relational_layer(db_path, pointer),
        empty_factory=dict,
    )
    summaries, meta["summaries"] = _safe_store_read(
        "summaries",
        lambda: _read_summary_layer(db_path),
        empty_factory=dict,
    )
    flavour, meta["flavour"] = _safe_store_read(
        "flavour",
        lambda: _read_flavour_layer(pointer),
        empty_factory=list,
    )
    temporal, meta["temporal"] = _safe_store_read(
        "temporal",
        lambda: _read_temporal_layer(pointer),
        empty_factory=_empty_temporal_layer,
    )
    provisional_candidates, meta["coreference_candidates"] = _safe_store_read(
        "coreference_candidates",
        lambda: provisional_store.list_pending_claims(provisional_path),
        empty_factory=list,
    )
    coreference_candidates = [
        *provisional_candidates,
        *temporal.get("coreference_candidates", []),
    ]

    macro_constraints: dict[str, Any] = {}
    meta["macro_constraints"] = {
        "available": False,
        "reason": "planning subsystem not yet implemented",
    }

    return {
        "relational": relational,
        "summaries": summaries,
        "temporal": temporal,
        "flavour": flavour,
        "coreference_candidates": coreference_candidates,
        "macro_constraints": macro_constraints,
        "meta": meta,
    }


async def node_assemble_context(state: dict[str, Any]) -> dict[str, Any]:
    """Overwrite active context on the state and return the state."""

    package = build_context_package(state)
    state["active_context_package"] = package
    return state


def _safe_store_read(
    layer_name: str,
    read_callable: Callable[[], LayerResult],
    *,
    empty_factory: Callable[[], LayerResult],
) -> tuple[LayerResult, LayerMeta]:
    """Read one layer, degrading only for documented not-yet-built stores."""

    try:
        result = read_callable()
    except NotImplementedError:
        return empty_factory(), {
            "available": False,
            "reason": f"{layer_name} store not yet implemented",
        }
    return result, {"available": True, "reason": "read succeeded"}


def _read_relational_layer(db_path: Any, pointer: Any) -> dict[str, Any]:
    """Read canonical SQLite facts around the active pointer."""

    arc_id = _pointer_value(pointer, "arc_id")
    chapter_id = _pointer_value(pointer, "chapter_id")
    scene_id = _pointer_value(pointer, "scene_id")
    beat_index = _pointer_value(pointer, "beat_index")

    chapters = sqlite_db.get_chapters_for_arc(db_path, arc_id)
    scenes = sqlite_db.get_scenes_for_chapter_ordered(db_path, chapter_id)
    beats = sqlite_db.get_beats_for_scene_ordered(db_path, scene_id)

    return {
        "pointer": _pointer_to_dict(pointer),
        "arc": sqlite_db.get_arc(db_path, arc_id),
        "chapters_for_arc": chapters,
        "current_chapter": _first_matching(chapters, "id", chapter_id),
        "scenes_for_chapter": scenes,
        "current_scene": _first_matching(scenes, "id", scene_id),
        "beats_for_scene": beats,
        "current_beat": _first_matching(beats, "beat_index", beat_index),
        "committed_beats": [
            beat
            for beat in beats
            if beat.get("status") == "completed" or beat.get("prose") is not None
        ],
        "remaining_beats": sqlite_db.get_remaining_beats_for_scene(db_path, scene_id),
        "remaining_scenes": sqlite_db.get_remaining_scenes_for_chapter(
            db_path, chapter_id
        ),
        "remaining_chapters": sqlite_db.get_remaining_chapters_for_arc(db_path, arc_id),
        "open_threads": sqlite_db.get_open_threads(db_path),
        "latest_pad_by_character": sqlite_db.get_latest_pad_for_scene(
            db_path, scene_id
        ),
        "pending_commit_intents": sqlite_db.get_pending_commit_intents(db_path),
    }


def _read_summary_layer(db_path: Any) -> dict[str, Any]:
    """Read persisted RAPTOR summary-tree nodes without clustering."""

    return {
        "roots": sqlite_db.get_raptor_nodes_by_parent(db_path, None),
        "by_level": {
            level: sqlite_db.get_raptor_nodes_by_level(db_path, level)
            for level in ("global", "arc", "chapter", "scene", "beat")
        },
    }


def _read_flavour_layer(pointer: Any) -> list[dict[str, Any]]:
    """Read associative flavour context once the vector store is implemented."""

    result = ChromaClient().query(pointer=pointer)
    return result if result is not None else []


def _read_temporal_layer(pointer: Any) -> dict[str, Any]:
    """Read temporal graph context once the graph store is implemented."""

    result = GraphitiClient().query(pointer=pointer)
    if result is None:
        return _empty_temporal_layer()
    if isinstance(result, dict):
        return {
            "records": result.get("records", result),
            "coreference_candidates": _extract_temporal_coreference_candidates(result),
        }
    return {"records": result, "coreference_candidates": []}


def _empty_temporal_layer() -> dict[str, Any]:
    """Return the serializable empty shape for deferred temporal reads."""

    return {"records": [], "coreference_candidates": []}


def _extract_temporal_coreference_candidates(
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collect raw temporal coreference links without tiering confidence."""

    candidates = result.get("coreference_candidates", [])
    if not candidates:
        candidates = result.get("coreference_links", [])
    if isinstance(candidates, dict):
        return [candidates]
    return list(candidates)


def _pointer_value(pointer: Any, field_name: str) -> Any:
    """Return a pointer field from either a Pydantic object or plain dict."""

    if isinstance(pointer, dict):
        return pointer[field_name]
    return getattr(pointer, field_name)


def _pointer_to_dict(pointer: Any) -> dict[str, Any]:
    """Return a serializable pointer snapshot."""

    if hasattr(pointer, "model_dump"):
        return pointer.model_dump()
    if isinstance(pointer, dict):
        return deepcopy(pointer)
    return {
        "arc_id": _pointer_value(pointer, "arc_id"),
        "chapter_id": _pointer_value(pointer, "chapter_id"),
        "scene_id": _pointer_value(pointer, "scene_id"),
        "beat_index": _pointer_value(pointer, "beat_index"),
    }


def _first_matching(
    rows: list[dict[str, Any]], key: str, value: Any
) -> dict[str, Any] | None:
    """Return the first row matching ``key == value`` from an ordered result."""

    for row in rows:
        if row.get(key) == value:
            return row
    return None
