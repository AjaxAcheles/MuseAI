"""The shared tool registry: every tool's spec and impl, and each agent's roster.

One module per tool under ``museai/fsm/tools/``, each exporting an OpenAI
function spec and an implementation; this module registers them and declares
each agent's roster. Nodes ask for their agent's tools by name —
:func:`tool_specs_for` / :func:`tool_impls_for` — so what an agent may call is
decided here, in one place, and an agent is never handed an impl its spec list
does not offer.

The default rosters are **story-canon tools**: the seed contract, the pointer,
the outline, threads, character state, and the committed manuscript. The one
tool that leaves the project — ``web_search`` — is not a default; it is offered
only when ``generation.research_mode`` is on.

Every DB-backed tool is read-only (``PRAGMA query_only``, see ``project_db``)
and scoped to the active project.
"""

from __future__ import annotations

from typing import Any, Callable

from museai.fsm.nodes.deps import get_node_config
from museai.fsm.tools.check_draft import CHECK_DRAFT_TOOL_SPEC, check_draft
from museai.fsm.tools.check_plan_node import (
    CHECK_PLAN_NODE_TOOL_SPEC,
    check_plan_node,
)
from museai.fsm.tools.find_repetition import (
    FIND_REPETITION_TOOL_SPEC,
    find_repetition,
)
from museai.fsm.tools.get_canonical_state import (
    GET_CANONICAL_STATE_TOOL_SPEC,
    get_canonical_state,
)
from museai.fsm.tools.get_chapter_context import (
    GET_CHAPTER_CONTEXT_TOOL_SPEC,
    get_chapter_context,
)
from museai.fsm.tools.get_character_emotion_history import (
    GET_CHARACTER_EMOTION_HISTORY_TOOL_SPEC,
    get_character_emotion_history,
)
from museai.fsm.tools.get_character_sheet import (
    GET_CHARACTER_SHEET_TOOL_SPEC,
    get_character_sheet,
)
from museai.fsm.tools.get_current_pointer_context import (
    GET_CURRENT_POINTER_CONTEXT_TOOL_SPEC,
    get_current_pointer_context,
)
from museai.fsm.tools.get_full_outline import (
    GET_FULL_OUTLINE_TOOL_SPEC,
    get_full_outline,
)
from museai.fsm.tools.get_recent_commits import (
    GET_RECENT_COMMITS_TOOL_SPEC,
    get_recent_commits,
)
from museai.fsm.tools.get_seed_contract import (
    GET_SEED_CONTRACT_TOOL_SPEC,
    get_seed_contract,
)
from museai.fsm.tools.get_thread_history import (
    GET_THREAD_HISTORY_TOOL_SPEC,
    get_thread_history,
)
from museai.fsm.tools.get_thread_status import (
    GET_THREAD_STATUS_TOOL_SPEC,
    get_thread_status,
)
from museai.fsm.tools.search_manuscript import (
    SEARCH_MANUSCRIPT_TOOL_SPEC,
    search_manuscript,
)
from museai.fsm.tools.verify_replacement import (
    VERIFY_REPLACEMENT_TOOL_SPEC,
    verify_replacement,
)
from museai.fsm.tools.web_search import WEB_SEARCH_TOOL_SPEC, web_search

_REGISTERED: list[tuple[dict[str, Any], Callable[..., Any]]] = [
    (WEB_SEARCH_TOOL_SPEC, web_search),
    (SEARCH_MANUSCRIPT_TOOL_SPEC, search_manuscript),
    (GET_SEED_CONTRACT_TOOL_SPEC, get_seed_contract),
    (GET_CURRENT_POINTER_CONTEXT_TOOL_SPEC, get_current_pointer_context),
    (GET_FULL_OUTLINE_TOOL_SPEC, get_full_outline),
    (GET_CHAPTER_CONTEXT_TOOL_SPEC, get_chapter_context),
    (GET_CANONICAL_STATE_TOOL_SPEC, get_canonical_state),
    (GET_THREAD_HISTORY_TOOL_SPEC, get_thread_history),
    (GET_THREAD_STATUS_TOOL_SPEC, get_thread_status),
    (GET_CHARACTER_EMOTION_HISTORY_TOOL_SPEC, get_character_emotion_history),
    (GET_CHARACTER_SHEET_TOOL_SPEC, get_character_sheet),
    (GET_RECENT_COMMITS_TOOL_SPEC, get_recent_commits),
    (CHECK_PLAN_NODE_TOOL_SPEC, check_plan_node),
    (CHECK_DRAFT_TOOL_SPEC, check_draft),
    (VERIFY_REPLACEMENT_TOOL_SPEC, verify_replacement),
    (FIND_REPETITION_TOOL_SPEC, find_repetition),
]

TOOL_SPECS: dict[str, dict[str, Any]] = {
    spec["function"]["name"]: spec for spec, _ in _REGISTERED
}
TOOL_IMPLS: dict[str, Callable[..., Any]] = {
    spec["function"]["name"]: impl for spec, impl in _REGISTERED
}

# The canon rosters. Planners read seed/outline/thread/canonical state and can
# validate their own plan elements; the drafter reads the committed manuscript
# and the cast's voices; the reviser checks its drafts and splices; the critic
# gets the broadest read-only continuity surface.
AGENT_TOOLS: dict[str, tuple[str, ...]] = {
    # The seed, open threads, and cast are already in the planner's <context>
    # block, so get_seed_contract / get_thread_status are omitted here: a weak
    # model that "gathers context" by calling them just burns its bounded
    # iterations re-fetching what it was handed. What remains surfaces detail the
    # context block does not: cross-arc outline, per-thread history, canonical
    # records by id, and plan-node validation.
    "chapter_planner": (
        "get_full_outline",
        "get_thread_history",
        "get_canonical_state",
        "check_plan_node",
    ),
    "beat_planner": (
        "get_current_pointer_context",
        "get_chapter_context",
        "get_character_emotion_history",
        "get_canonical_state",
        "check_plan_node",
    ),
    "drafter": (
        "get_current_pointer_context",
        "get_recent_commits",
        "search_manuscript",
        "get_character_sheet",
        "find_repetition",
    ),
    "reviser": (
        "check_draft",
        "verify_replacement",
        "search_manuscript",
        "find_repetition",
    ),
    # find_repetition is deliberately absent: the critic's own error-code list
    # (CRITIC_ERROR_CODES) has no repetition category, and `audit`'s
    # paragraph_overlaps guard already runs this exact check against the whole
    # committed manuscript before the critic sees the draft — its verdict
    # reaches the prompt as `repetition_overlap_count` instead. See
    # `museai/fsm/nodes/critics.py:critic_messages`.
    "critic": (
        "search_manuscript",
        "get_full_outline",
        "get_thread_status",
        "get_thread_history",
        "get_canonical_state",
        "get_current_pointer_context",
        "get_recent_commits",
    ),
}


def _roster(agent: str) -> list[str]:
    try:
        names = list(AGENT_TOOLS[agent])
    except KeyError:
        known = ", ".join(sorted(AGENT_TOOLS))
        raise ValueError(f"no tool roster for agent {agent!r}; known: {known}") from None
    # The web is opt-in, never a default reflex: agents ground themselves in
    # the story's own canon unless research mode is explicitly on.
    if get_node_config().generation.research_mode:
        names.append("web_search")
    return names


def tool_specs_for(agent: str) -> list[dict[str, Any]]:
    """The tool schemas offered to ``agent``, in roster order."""
    return [TOOL_SPECS[name] for name in _roster(agent)]


def tool_impls_for(agent: str) -> dict[str, Callable[..., Any]]:
    """Name -> callable for exactly the tools ``agent`` is offered."""
    return {name: TOOL_IMPLS[name] for name in _roster(agent)}
