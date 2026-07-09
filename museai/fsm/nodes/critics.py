"""The continuity critic.

v1 has exactly one critic and it runs alone. There is no dialogue critic, no
pacing critic, no craft consultant, and no panel to run them in parallel — one
critic, called serially, with one tool.

The critic is agentic: it drives :func:`run_agent_loop`, reaching for
``web_search`` when the draft asserts something checkable about the real world.
The loop is bounded by ``generation.max_agent_iterations`` and always terminates
with prose, so this node's final text is always a candidate JSON array.

**A response this node cannot parse is a hard error.** A critic whose findings
are unreadable has not found nothing; it has failed. Returning ``[]`` there
would launder a broken critic into a clean pass and commit the draft.
"""

from __future__ import annotations

from museai.core.logging_setup import log_node_event
from museai.core.stream_bus import bus
from museai.fsm.nodes.deps import get_node_config
from museai.fsm.state import OrchestratorState
from museai.fsm.tools.loop import run_agent_loop
from museai.fsm.tools.web_search import TOOL_IMPLS, WEB_SEARCH_TOOL_SPEC
from museai.llm.prompts import render_messages
from museai.llm.structured import parse_failure_objects

PHASE = "Auditing"
CRITIC_NAME = "continuity_critic"


def critic_messages(draft_text: str, package: dict) -> list[dict]:
    """Render the continuity-critic prompt for a draft and its context."""
    return render_messages(
        CRITIC_NAME,
        draft_text=draft_text,
        chapter=package["chapter"],
        threads=package["threads"],
        characters=package["characters"],
        recent_prose=package["recent_prose"],
    )


async def adversarial_critics(state: OrchestratorState) -> dict:
    """Run the continuity critic over ``current_draft_text``.

    Returns the state delta. ``critic_failures`` is **omitted entirely** when the
    critic is clean: its reducer reads an explicit ``[]`` as a reset, and
    resetting here would erase the programmatic failures ``audit`` just found.
    Only ``revise`` is entitled to clear that list.
    """
    config = get_node_config()
    package = state["active_context_package"]
    draft = state["current_draft_text"]
    beat_id = package["beat"]["id"]

    log_node_event("critics", event="start", critic=CRITIC_NAME, beat_id=beat_id)
    log_node_event("critics", event="phase_change", phase=PHASE, beat_id=beat_id)
    await bus.publish(
        "phase_change", {"phase": PHASE, "node": "critics", "beat_id": beat_id}
    )

    async def on_tool_call(event: dict) -> None:
        log_node_event(
            "critics",
            event="tool_call",
            beat_id=beat_id,
            tool=event["tool"],
            args=event["arguments"],
        )
        await bus.publish("critic_tool", {"beat_id": beat_id, **event})

    response = await run_agent_loop(
        config.endpoint,
        critic_messages(draft, package),
        [WEB_SEARCH_TOOL_SPEC],
        TOOL_IMPLS,
        config.generation.max_agent_iterations,
        on_event=on_tool_call,
    )

    await bus.publish(
        "critic_reasoning",
        {"beat_id": beat_id, "critic": CRITIC_NAME, "text": response.text},
    )

    # A StructuredOutputError propagates: an unreadable critic is a failure of
    # the critic, not a clean draft.
    failures = parse_failure_objects(
        response.text, retry_cap=config.generation.revision_retry_cap
    )

    # `audit`'s programmatic failures are already in state; the draft's true
    # score is both sets together.
    total = len(state["critic_failures"]) + len(failures)
    best_count = state["best_seen_failure_count"]
    improved = best_count is None or total < best_count

    delta: dict = {
        "best_seen_draft": draft if improved else state["best_seen_draft"],
        "best_seen_failure_count": total if improved else best_count,
    }
    if failures:
        delta["critic_failures"] = failures

    summary = f"{total} issues found" if total else "clean"
    log_node_event(
        "critics",
        event="critiqued",
        beat_id=beat_id,
        critic_failures=len(failures),
        total_failures=total,
        best_seen_failure_count=delta["best_seen_failure_count"],
        improved=improved,
        summary=summary,
    )
    await bus.publish(
        "critic_summary",
        {
            "beat_id": beat_id,
            "critic": CRITIC_NAME,
            "summary": summary,
            "total_failures": total,
            "critic_failures": [f.model_dump() for f in failures],
        },
    )

    return delta
