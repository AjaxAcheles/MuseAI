"""The draft → audit → revise loop's exit router.

A LangGraph conditional edge: pure, synchronous, and side-effect free. It reads
state and names the next node. First match wins.

1. An unreadable critic below its degradation threshold → ``retry_critic``.
2. An unreadable critic at the threshold with no salvageable findings → ``review``.
3. No outstanding failures → ``commit``.
4. Failures, and the revision budget is not spent → ``revise``.
5. Failures, and the budget *is* spent → ``review``.

``review`` is a safe boundary, not a verdict. The router does not discard the
draft, does not accept it, and does not restore ``best_seen_draft`` — it only
routes. Whether the beat is accepted as-is, rolled back to the best draft seen,
or rewritten by hand is a human's decision, made through the web layer. Nothing
here blocks a thread waiting for it.

There is no drift gate. Routing turns on failure count, retry count, and the
single critic parser's bounded health streak.
"""

from __future__ import annotations

from museai.core.logging_setup import log_node_event
from museai.fsm.nodes.deps import get_node_config
from museai.fsm.state import OrchestratorState

COMMIT = "commit"
REVISE = "revise"
REVIEW = "review"
RETRY_CRITIC = "retry_critic"


def mode_selector(state: OrchestratorState) -> str:
    """Route the beat to ``commit``, ``revise``, or ``review``."""
    config = get_node_config()
    failures = len(state["critic_failures"])
    retry_count = state["retry_count"]
    cap = config.generation.revision_retry_cap
    parse_streak = state["critic_parse_failure_streak"]
    degrade_threshold = config.generation.critic_degrade_threshold

    if 0 < parse_streak < degrade_threshold:
        # A parser failure is not a clean verdict. Retry the critic itself at a
        # bounded graph edge rather than committing or rewriting sound prose.
        destination = RETRY_CRITIC
    elif parse_streak >= degrade_threshold and failures == 0:
        # Degraded parsing recovered nothing. The uncertainty belongs at the
        # explicit human-review boundary, never in a silent commit.
        destination = REVIEW
    elif failures == 0:
        destination = COMMIT
    elif retry_count < cap:
        destination = REVISE
    else:
        destination = REVIEW

    log_node_event(
        "mode_selector",
        event="route",
        destination=destination,
        failures=failures,
        retry_count=retry_count,
        revision_retry_cap=cap,
        critic_parse_failure_streak=parse_streak,
        critic_degrade_threshold=degrade_threshold,
    )
    return destination
