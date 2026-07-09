"""The draft → audit → revise loop's exit router.

A LangGraph conditional edge: pure, synchronous, and side-effect free. It reads
state and names the next node. First match wins.

1. No outstanding failures → ``commit``.
2. Failures, and the revision budget is not spent → ``revise``.
3. Failures, and the budget *is* spent → ``review``.

``review`` is a safe boundary, not a verdict. The router does not discard the
draft, does not accept it, and does not restore ``best_seen_draft`` — it only
routes. Whether the beat is accepted as-is, rolled back to the best draft seen,
or rewritten by hand is a human's decision, made through the web layer. Nothing
here blocks a thread waiting for it.

There is no escalation ladder and no drift gate. Routing turns on the failure
count and the retry count, and on nothing else.
"""

from __future__ import annotations

from museai.core.logging_setup import log_node_event
from museai.fsm.nodes.deps import get_node_config
from museai.fsm.state import OrchestratorState

COMMIT = "commit"
REVISE = "revise"
REVIEW = "review"


def mode_selector(state: OrchestratorState) -> str:
    """Route the beat to ``commit``, ``revise``, or ``review``."""
    config = get_node_config()
    failures = len(state["critic_failures"])
    retry_count = state["retry_count"]
    cap = config.generation.revision_retry_cap

    if failures == 0:
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
    )
    return destination
