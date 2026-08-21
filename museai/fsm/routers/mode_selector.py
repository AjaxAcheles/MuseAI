"""The draft → audit → revise loop's exit router.

A LangGraph conditional edge: pure, synchronous, and side-effect free. It reads
state and names the next node. First match wins.

1. An unreadable critic below its degradation threshold → ``retry_critic``.
2. An unreadable critic at the threshold with no salvageable findings → ``review``.
3. No outstanding failures → ``commit``.
4. A revise already ran, and the findings it was told to fix are still present,
   with at least one outstanding finding critic-sourced → ``review``.
   Audit-only findings instead spend the configured revision budget.
5. Failures, and the revision budget is not spent → ``revise``.
6. Failures, and the budget *is* spent → ``review``.

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
from museai.fsm.nodes.audit import CRITIC_SOURCE as AUDIT_CRITIC_SOURCE
from museai.fsm.nodes.deps import get_node_config
from museai.fsm.state import OrchestratorState

COMMIT = "commit"
REVISE = "revise"
REVIEW = "review"
RETRY_CRITIC = "retry_critic"


def mode_selector(state: OrchestratorState) -> str:
    """Route the beat to ``commit``, ``revise``, or ``review``."""
    config = get_node_config()
    outstanding_failures = state["critic_failures"]
    failures = len(outstanding_failures)
    retry_count = state["retry_count"]
    cap = config.generation.revision_retry_cap
    parse_streak = state["critic_parse_failure_streak"]
    degrade_threshold = config.generation.critic_degrade_threshold
    audit_only = bool(outstanding_failures) and all(
        failure.critic_source == AUDIT_CRITIC_SOURCE
        for failure in outstanding_failures
    )
    # A revise already ran and the findings it was told to fix are still there:
    # another revise would spend a full critic pass re-running the same
    # non-improving cycle (e.g. a beat stuck at an unchanged passive_density
    # across four retries). `last_cycle_improved` is critics.py's verdict
    # against the count and signatures that revise recorded, so a critic re-score
    # of unchanged prose is not a failed cycle and a new finding after a real fix
    # is ordinary iteration. The `retry_count > 0` guard is redundant with that
    # baseline but says the precondition out loud. Seven observed parks all
    # occurred at retry_count=1 while revision_retry_cap=4 was never approached,
    # so deterministic, audit-only measurements are allowed to spend that
    # existing budget.
    no_progress = (
        retry_count > 0
        and not state["last_cycle_improved"]
        and not audit_only
    )

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
    elif no_progress:
        destination = REVIEW
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
        last_cycle_improved=state["last_cycle_improved"],
        audit_only=audit_only,
        no_progress=no_progress,
    )
    return destination
