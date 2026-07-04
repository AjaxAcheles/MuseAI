"""Module: M07 (Quality Gauntlet)

Fast-path conditional edge run immediately after ``node_programmatic_audit``. For a
clearly-clean draft it bypasses ``node_adversarial_critics`` entirely — saving
approximately three LLM critic calls per beat — routing straight to
``node_commit_transaction``; every other state takes the standard critics path.

Pure read-only routing: no state mutation, no store access, no LLM/network contact.
Stays cycle-free (no node/graph module imports) per the router/node import-cycle split
in ``Architecture_Map.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

# Fixed structural safety margin applied below the routing threshold
# (Configuration_Reference.md §1/§4) — not a config key. It keeps the fast path
# reachable only on clearly clean drafts, not merely borderline-acceptable ones.
_FAST_PATH_MARGIN = 0.7

_COMMIT_ROUTE = "node_commit_transaction"
_CRITICS_ROUTE = "node_adversarial_critics"


def _resolve_config(state: dict[str, Any]) -> Any:
    """Return the active typed config (mirrors the built nodes' resolution)."""
    config = state.get("app_config") or state.get("config")
    if config is not None:
        return config
    from core.config_loader import load_config

    return load_config(state.get("config_path", CONFIG_PATH))


def route_programmatic(state: dict[str, Any]) -> str:
    """Select the next node after the Stage-1 programmatic audit.

    First-match, in order: fast-path bypass to commit when ``retry_count == 0`` AND
    ``critic_failures`` is empty AND ``stylometric_distance`` is below the base
    threshold times the fixed ``_FAST_PATH_MARGIN``; otherwise the standard critics
    path. ``base_threshold`` is ``transient_dc_override`` when set, else
    ``config.thresholds.stylometric_drift_threshold``.
    """
    override = state.get("transient_dc_override")
    if override is not None:
        base_threshold = override
    else:
        config = _resolve_config(state)
        base_threshold = config.thresholds.stylometric_drift_threshold

    retry_count = state.get("retry_count", 0)
    critic_failures = state.get("critic_failures") or []
    distance = state.get("stylometric_distance", 0.0)

    is_fast_path = (
        retry_count == 0
        and len(critic_failures) == 0
        and distance < base_threshold * _FAST_PATH_MARGIN
    )
    return _COMMIT_ROUTE if is_fast_path else _CRITICS_ROUTE
