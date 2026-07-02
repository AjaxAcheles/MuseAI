"""Module: M06 (Drafting & Generation)

Stream LLM prose for the current beat via the drafter endpoint.

Minimal-but-real implementation: it resolves the current beat's plan (objective +
PAD/behavioural constraints), renders the ``node_draft_prose`` template with the
assembled context, and streams a completion from the role-resolved ``drafter``
endpoint. Every token is forwarded to an injected ``on_token`` seam (the web
driver wires this to the SSE stream bus) and accumulated into the returned draft.

The inference timeout is read from ``config.runtime.inference_timeout_seconds`` and
passed to ``call_llm`` so a hung local model can never stall a run. The node writes
no persistence — committing the beat is ``node_commit_transaction``'s job. It
returns a state *delta* (the driver merges it), so it is agnostic to whether the
orchestrator is the linear driver or a future LangGraph graph.

Quality passes (adversarial critics, programmatic audit, revision) are separate
nodes and are not invoked here; this node only produces the first draft.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable

import core.runtime as runtime
from llm.call_llm import call_llm
from prompts.prompt_loader import PromptLoader

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
_NODE_NAME = "node_draft_prose"

# Fallback per-beat length target when the beat plan carries none. A short beat
# keeps the local round-trip bounded; real sizing comes from the beat plan.
_DEFAULT_BEAT_WORD_TARGET = 250

TokenSink = Callable[[str], Any | Awaitable[Any]]


def _resolve_config(state: dict[str, Any]) -> Any:
    """Return the active typed config (mirrors the planner nodes' resolution)."""
    config = state.get("app_config") or state.get("config")
    if config is not None:
        return config
    from core.config_loader import load_config

    return load_config(state.get("config_path", CONFIG_PATH))


def _relational_facts(context_package: dict[str, Any]) -> list[str]:
    """Best-effort flatten of the assembled relational layer into fact strings."""
    relational = context_package.get("relational") if context_package else None
    if not isinstance(relational, dict):
        return []
    facts: list[str] = []
    for value in relational.values():
        if isinstance(value, list):
            for item in value:
                facts.append(item if isinstance(item, str) else str(item))
        elif value:
            facts.append(str(value))
    return facts


async def node_draft_prose(
    state: dict[str, Any],
    *,
    on_token: TokenSink | None = None,
    loader: PromptLoader | None = None,
    call: Any = None,
) -> dict[str, Any]:
    """Draft prose for the beat at ``fsm_pointer.beat_id`` and return a state delta.

    ``on_token`` receives each streamed token (the web driver forwards it to the
    SSE bus). ``loader``/``call`` are injectable seams for testing; by default the
    real :class:`PromptLoader` and :func:`call_llm` are used.
    """
    config = _resolve_config(state)
    pointer = state["fsm_pointer"]
    beat_id = getattr(pointer, "beat_id", "") or ""
    beat_plan = (state.get("beat_plan_by_id") or {}).get(beat_id, {})
    project_metadata = state.get("project_metadata") or {}

    target_words = beat_plan.get("target_words") or _DEFAULT_BEAT_WORD_TARGET
    context = {
        "genre": project_metadata.get("genre", ""),
        "premise_seed": project_metadata.get("premise_seed", ""),
        "scene_description": beat_plan.get("scene_description", ""),
        "relational_facts": _relational_facts(state.get("active_context_package") or {}),
        "recent_prose": state.get("last_committed_prose", ""),
        "beat_objective": beat_plan.get("objective", "Advance the scene."),
        "pad_constraint": beat_plan.get("pad_constraint", ""),
        "physical_constraints": beat_plan.get("physical_constraints", ""),
        "target_words": target_words,
    }

    loader = loader or PromptLoader()
    rendered = loader.render(_NODE_NAME, context)
    messages = [{"role": "user", "content": rendered}]
    endpoint = config.endpoints.drafter
    timeout_seconds = float(getattr(config.runtime, "inference_timeout_seconds", 120))

    chunks: list[str] = []

    async def _sink(token: str) -> None:
        chunks.append(token)
        if on_token is not None:
            result = on_token(token)
            if hasattr(result, "__await__"):
                await result

    call_fn = call or call_llm
    response = await call_fn(
        messages,
        endpoint,
        stream=True,
        on_token=_sink,
        timeout_seconds=timeout_seconds,
    )

    draft_text = getattr(response, "text", None)
    if not draft_text:
        draft_text = "".join(chunks)

    return {"current_draft_text": draft_text, "streaming_buffer": ""}
