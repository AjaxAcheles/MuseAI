"""Shared dependencies for FSM node functions.

A LangGraph node's signature is fixed — ``async def node(state) -> dict`` — so
configuration cannot arrive as an argument, and ``OrchestratorState`` carries
only what the graph needs to serialize (a project id and a pointer), not an
``AppConfig``. This module is the seam: bring-up calls :func:`set_node_config`
once, and every node reads the same instance through :func:`get_node_config`.

Left unset, :func:`get_node_config` loads ``config.yaml`` on first use, so a node
invoked outside a full bring-up still gets the same strict, validated config.
"""

from __future__ import annotations

from museai.core.config import AppConfig, load_config

_config: AppConfig | None = None


class PlanningError(RuntimeError):
    """A planning node could not proceed: missing outline rows or a bad plan.

    Distinct from ``StructuredOutputError`` (the model's JSON was unreadable) and
    ``LLMCallError`` (the endpoint failed).
    """


class DraftingError(RuntimeError):
    """A drafting node could not proceed: no assembled context, or empty prose."""


def set_node_config(config: AppConfig) -> None:
    """Install the ``AppConfig`` every node reads. Called once at bring-up."""
    global _config
    _config = config


def get_node_config() -> AppConfig:
    """Return the installed config, loading ``config.yaml`` if none was set."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
