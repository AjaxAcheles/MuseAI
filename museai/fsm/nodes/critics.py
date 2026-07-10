"""The continuity critic.

v1 has exactly one critic and it runs alone. There is no dialogue critic, no
pacing critic, no craft consultant, and no panel to run them in parallel — one
critic, called serially, with one tool.

The critic is agentic: it drives :func:`run_agent_loop`, reaching for
``web_search`` when the draft asserts something checkable about the real world.
The loop is bounded by ``generation.max_agent_iterations`` and always terminates
with prose, so this node's final text is always a candidate JSON array.

**An unreadable response is re-prompted, then survived.** A weak model will
answer with the schema template itself — placeholder values, a misspelled key,
a missing field. Feeding the validation error back fixes that most of the time,
so the critic gets ``generation.critic_parse_retries`` chances to say it again
correctly.

When those are spent the run *continues* rather than dying: a broken critic must
not destroy a multi-hour generation and the draft in flight. But it is not
laundered into a clean pass either. Each beat whose critic stayed unreadable
increments ``critic_parse_failure_streak``, and once the streak reaches
``generation.critic_degrade_threshold`` the run is **degraded**: element
validation loosens, and every ``critic_health`` event says so, loudly, in the UI.
Degraded beats commit with only the programmatic ``audit`` behind them.
"""

from __future__ import annotations

import logging

from museai.core.logging_setup import log_node_event
from museai.core.stream_bus import bus
from museai.fsm.nodes.deps import get_node_config
from museai.fsm.state import OrchestratorState
from museai.fsm.tools.loop import run_agent_loop
from museai.fsm.tools.web_search import TOOL_IMPLS, WEB_SEARCH_TOOL_SPEC
from museai.llm.prompts import render_messages
from museai.llm.structured import StructuredOutputError, parse_failure_objects

PHASE = "Auditing"
CRITIC_NAME = "continuity_critic"

# What the model is told when its last reply would not parse. It carries the
# validation error verbatim: a model that misspelled `offending_text` needs to
# see which key it got wrong, not a generic scolding.
_CORRECTION_TEMPLATE = (
    "Your previous reply could not be parsed: {error}\n\n"
    "Return only the fenced JSON array described in the output format. Every "
    "element needs error_code, offending_text, and suggested_fix. Use the exact "
    "key names. Do not return the example values from the schema — quote the "
    "real offending text from the draft. If the draft is clean, return []."
)


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


async def _publish_health(
    *,
    beat_id: str,
    streak: int,
    threshold: int,
    degraded: bool,
    lenient_used: bool,
    error: str,
) -> None:
    """Report the critic's schema health after every beat.

    Published unconditionally, including when healthy, so the browser's banner is
    a pure function of the latest event and clears itself on recovery. It lands
    in ``bus.last_snapshot``, so a reloading page hydrates the current state.
    """
    # Health fires once per beat. WARNING only once the continuity gate has
    # actually stopped working — a warning on every healthy beat is a warning
    # nobody reads.
    log_node_event(
        "critics",
        level=logging.WARNING if degraded else logging.INFO,
        event="health",
        beat_id=beat_id,
        streak=streak,
        threshold=threshold,
        degraded=degraded,
        lenient_used=lenient_used,
    )
    await bus.publish(
        "critic_health",
        {
            "beat_id": beat_id,
            "critic": CRITIC_NAME,
            "streak": streak,
            "threshold": threshold,
            "degraded": degraded,
            "lenient_used": lenient_used,
            "error": error[:300] if error else "",
        },
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

    messages = critic_messages(draft, package)
    generation = config.generation

    # Every attempt validates strictly, even in a degraded run: a model that has
    # started answering correctly again must be able to clear the streak. Lenient
    # parsing is the last resort below, never the loop's contract.
    failures: list | None = None
    last_error = ""
    last_text = ""

    for attempt in range(generation.critic_parse_retries + 1):
        response = await run_agent_loop(
            config.endpoint,
            messages,
            [WEB_SEARCH_TOOL_SPEC],
            TOOL_IMPLS,
            generation.max_agent_iterations,
            on_event=on_tool_call,
            agent="critic",
        )
        last_text = response.text
        await bus.publish(
            "critic_reasoning",
            {"beat_id": beat_id, "critic": CRITIC_NAME, "text": response.text},
        )

        try:
            failures = parse_failure_objects(response.text)
            break
        except StructuredOutputError as exc:
            last_error = str(exc)
            log_node_event(
                "critics",
                level=logging.WARNING,
                event="parse_failed",
                beat_id=beat_id,
                attempt=attempt + 1,
                retries=generation.critic_parse_retries,
                error=last_error[:200],
            )
            if attempt == generation.critic_parse_retries:
                break
            # Show the model its own reply and exactly what was wrong with it.
            messages = [
                *messages,
                {"role": "assistant", "content": response.text},
                {"role": "user", "content": _CORRECTION_TEMPLATE.format(error=last_error)},
            ]

    lenient_used = False
    if failures is not None:
        streak = 0
    else:
        # Every retry spent and still unreadable. The run continues — see the
        # module docstring — but the streak climbs and the UI is told.
        streak = state["critic_parse_failure_streak"] + 1
        failures = []
        if streak >= generation.critic_degrade_threshold:
            try:
                failures = parse_failure_objects(last_text, lenient=True)
            except StructuredOutputError:
                failures = []
            # "Salvaged" only if something was actually recovered. Relaxed parsing
            # that yielded nothing salvaged nothing, and the UI must not claim it did.
            lenient_used = bool(failures)

    degraded = streak >= generation.critic_degrade_threshold
    await _publish_health(
        beat_id=beat_id,
        streak=streak,
        threshold=generation.critic_degrade_threshold,
        degraded=degraded,
        lenient_used=lenient_used,
        error=last_error,
    )

    # `audit`'s programmatic failures are already in state; the draft's true
    # score is both sets together.
    total = len(state["critic_failures"]) + len(failures)
    best_count = state["best_seen_failure_count"]
    improved = best_count is None or total < best_count

    delta: dict = {
        "best_seen_draft": draft if improved else state["best_seen_draft"],
        "best_seen_failure_count": total if improved else best_count,
        "critic_parse_failure_streak": streak,
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
