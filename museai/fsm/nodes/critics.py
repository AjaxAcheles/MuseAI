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
An unreadable verdict is never treated as clean: the graph retries this critic
until the degradation threshold, then routes to the explicit review boundary.
"""

from __future__ import annotations

import hashlib
import json
import logging

from museai.core.logging_setup import log_node_event
from museai.core.stream_bus import bus
from museai.fsm.nodes.context_budget import window_budget
from museai.fsm.nodes.deps import get_node_config
from museai.fsm.nodes.audit import longest_shared_verbatim_word_run
from museai.fsm.nodes.revise import locate
from museai.fsm.state import FailureObject, OrchestratorState
from museai.fsm.tools.loop import AgentLoopError, run_agent_loop
from museai.fsm.tools.registry import tool_impls_for, tool_specs_for
from museai.llm.client import LLMCallError
from museai.llm.prompts import render_messages
from museai.llm.tokenizer import count_message_tokens
from museai.llm.structured import (
    StructuredOutputError,
    UnlocatableFindingError,
    parse_failure_objects,
    response_truncation_remedy,
)

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

# What the model is told after a truncation (finish_reason == "length" with no
# usable text). Unlike a schema failure, the model did nothing here it can read
# back and fix — it spent its whole reply budget on reasoning and never reached
# an answer. Showing it an empty assistant turn plus operator-facing advice
# about context windows or endpoint configuration (see `response_truncation_
# remedy`) only grows the prompt for a model that has nothing to correct. This
# retry asks for less, not more: no operator diagnosis, no echoed empty reply.
_TRUNCATION_RETRY_MESSAGE = (
    "Your previous reply was cut off before it produced any JSON — the whole "
    "reply went to reasoning, with nothing left for the answer. Skip the "
    "reasoning this time: respond with only the fenced JSON array described in "
    "the output format, or [] if the draft is clean."
)

# The second truncation in a row. The message above is rebuilt from the same
# two constants every time, so re-sending it would put a byte-identical request
# on the wire — measured six times in a row on one live beat, ~62 s each, every
# one of them returning zero characters. The retry is still worth taking (a
# fresh sample answered roughly a third of the time), so this escalates the ask
# instead of repeating it: same demand, stated more narrowly, and explicitly
# permitting a partial answer so the model has somewhere to stop short of a
# complete audit.
#
# It does not offer "[]" as an escape from deliberating. An unreadable verdict
# is never laundered into a clean pass — that is the streak's job, and inviting
# a shortcut to "[]" here would fake exactly the signal the streak exists to
# report.
_TRUNCATION_FINAL_MESSAGE = (
    "Your reply was cut off before producing any JSON again. Do not deliberate "
    "at all this time. Write the fenced JSON array immediately, as your very "
    "first output. If you have only found one problem so far, report just that "
    "one — a partial array is far better than another cut-off reply."
)


def _fingerprint(value: object) -> str:
    """A stable identity for a prompt, used to spot a retry that repeats one."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _tool_turns(conversation: list[dict]) -> int:
    """How many tool results a conversation carries — its evidence, in short."""
    return sum(1 for message in conversation if message.get("role") == "tool")


def _fits_window(conversation: list[dict], endpoint) -> bool:
    """Whether a retry prompt still leaves the endpoint's output reservation free.

    ``assemble_context`` trimmed the *original* prompt to the window once. Every
    parse retry then piles on: the loop's tool-call turns, its tool results, the
    unreadable reply, and the correction. Nothing re-measures that, so on a
    16k window a couple of carried search results can eat the room the verdict
    needs and produce the truncated reply the retry exists to fix.
    """
    budget = window_budget(endpoint)
    if budget is None:
        return True
    tokens = count_message_tokens(
        conversation, endpoint.tokenizer_family, endpoint.model_name
    )
    return tokens <= budget


def _locatable_findings(
    draft: str, findings: list[FailureObject], beat_id: str
) -> list[FailureObject]:
    """Keep critic findings whose quoted text can be located in the draft."""
    kept: list[FailureObject] = []
    for finding in findings:
        if locate(draft, finding.offending_text) is not None:
            kept.append(finding)
            continue
        log_node_event(
            "critics",
            level=logging.WARNING,
            event="finding_discarded",
            beat_id=beat_id,
            error_code=finding.error_code,
            offending_text=finding.offending_text[:120],
        )
    return kept


_SANITIZED_FIX = (
    "Rewrite the offending text in fresh prose that carries this beat forward. "
    "Do not reuse wording from prose already committed earlier in the manuscript."
)


def _sanitize_borrowed_fixes(
    findings: list[FailureObject],
    committed_prose: list[str],
    beat_id: str,
    max_borrowed_words: int,
) -> list[FailureObject]:
    """Replace remedies that paste the critic's recent-prose context verbatim."""
    sanitized: list[FailureObject] = []
    for finding in findings:
        borrowed_words = longest_shared_verbatim_word_run(
            finding.suggested_fix, committed_prose
        )
        if borrowed_words < max_borrowed_words:
            sanitized.append(finding)
            continue
        log_node_event(
            "critics",
            level=logging.WARNING,
            event="fix_sanitized",
            beat_id=beat_id,
            error_code=finding.error_code,
            borrowed_words=borrowed_words,
            suggested_fix=finding.suggested_fix[:120],
        )
        sanitized.append(
            finding.model_copy(update={"suggested_fix": _SANITIZED_FIX})
        )
    return sanitized


def critic_messages(
    draft_text: str, package: dict, repetition_overlap_count: int
) -> list[dict]:
    """Render the continuity-critic prompt for a draft and its context.

    ``repetition_overlap_count`` is ``audit``'s own paragraph-repetition
    verdict for this exact draft (see ``OrchestratorState.repetition_overlap_
    count``), carried into the prompt so the critic is told that question is
    already answered rather than offered a tool for it.
    """
    return render_messages(
        CRITIC_NAME,
        draft_text=draft_text,
        beat=package["beat"],
        project=package.get("project") or {},
        chapter=package["chapter"],
        threads=package["threads"],
        characters=package["characters"],
        recent_prose=package["recent_prose"],
        research_mode=get_node_config().generation.research_mode,
        repetition_overlap_count=repetition_overlap_count,
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

    messages = critic_messages(draft, package, state["repetition_overlap_count"])
    generation = config.generation
    endpoint = config.endpoint_for("critic")

    # Every attempt validates strictly, even in a degraded run: a model that has
    # started answering correctly again must be able to clear the streak. Lenient
    # parsing is the last resort below, never the loop's contract.
    failures: list[FailureObject] = []
    unreadable = True
    last_error = ""
    last_text = ""
    last_truncated = False
    unlocatable_findings = False

    # Carries forward across parse retries so a bad reply doesn't discard the
    # tool results the previous attempt already paid for; run_agent_loop fills
    # conversation_out with its working messages right up to (not including)
    # the final reply, and each retry builds on top of that instead of the
    # original 2-message prompt.
    conversation: list[dict] = list(messages)

    # ...and the same courtesy across *node* invocations. `retry_critic` routes
    # back here with the draft untouched, so whatever the last pass looked up is
    # still true. The fingerprint is taken against the draft, so a revise makes
    # the carry stale and this falls back to the bare prompt; `_fits_window`
    # refuses a carry that would no longer leave room to answer.
    draft_fingerprint = _fingerprint(draft)
    carried = state["critic_evidence"] or {}
    reused_evidence = False
    if (
        carried.get("draft_fingerprint") == draft_fingerprint
        and carried.get("conversation")
        and _fits_window(carried["conversation"], endpoint)
    ):
        conversation = [dict(message) for message in carried["conversation"]]
        reused_evidence = True
        log_node_event(
            "critics",
            event="evidence_reused",
            beat_id=beat_id,
            tool_results=_tool_turns(conversation),
        )

    # The richest tool-bearing conversation this pass produced, saved for the
    # next one if the verdict turns out unreadable.
    evidence: list[dict] = conversation if reused_evidence else []

    # Every prompt the endpoint has already *answered* this pass. A retry that
    # would reproduce one exactly is a call whose outcome is already known.
    #
    # Answered, not merely sent: a call that died in transport produced no reply
    # to learn anything from, and re-sending the same prompt after one is the
    # entire point of the transient-retry path below.
    answered: set[str] = set()

    # How many truncations have already been answered with a re-ask, so the
    # second one escalates the wording instead of repeating it verbatim.
    truncation_asks = 0

    # Set once a truncation forces the *next* attempt to be a short, tool-free
    # ask instead of the normal agent loop: withholding the tool schemas means
    # the retry cannot burn its budget on another tool round-trip on top of
    # reasoning, and starting the model has nothing left to investigate anyway
    # since `conversation` was just rebuilt from the original prompt.
    retry_bare = False

    for attempt in range(generation.critic_parse_retries + 1):
        prompt_fingerprint = _fingerprint(conversation)
        if prompt_fingerprint in answered:
            # Every way of re-asking has been tried and the wording has nowhere
            # left to go. Sending this again would be a paid call with a known
            # answer; stop and let the streak report the pass as unreadable,
            # exactly as an exhausted retry budget does.
            log_node_event(
                "critics",
                level=logging.WARNING,
                event="retry_skipped_identical",
                beat_id=beat_id,
                attempt=attempt + 1,
                retries=generation.critic_parse_retries,
            )
            break

        loop_conversation: list[dict] = []
        try:
            response = await run_agent_loop(
                endpoint,
                conversation,
                [] if retry_bare else tool_specs_for("critic"),
                {} if retry_bare else tool_impls_for("critic"),
                generation.max_agent_iterations,
                on_event=on_tool_call,
                agent="critic",
                tool_call_cap=generation.tool_call_cap,
                tool_timeout=generation.tool_timeout,
                conversation_out=loop_conversation,
            )
        except (LLMCallError, AgentLoopError) as exc:
            # A broken critic must not destroy a multi-hour generation: treat
            # a failed call exactly like an unreadable verdict rather than
            # letting the exception propagate out of this node. `AgentLoopError`
            # is here for the stuck-model case specifically — an endpoint that
            # keeps emitting tool calls after the schemas were withheld, which
            # the strike-out now reaches sooner than the iteration cap did.
            last_error = f"critic call failed: {exc}"
            log_node_event(
                "critics",
                level=logging.WARNING,
                event="call_failed",
                beat_id=beat_id,
                attempt=attempt + 1,
                retries=generation.critic_parse_retries,
                error=last_error[:200],
            )
            if attempt == generation.critic_parse_retries:
                break
            continue

        answered.add(prompt_fingerprint)
        conversation = loop_conversation or conversation
        # Keep whichever pass looked up the most, not merely the most recent —
        # the truncation branch below rebuilds from the bare prompt, so the last
        # conversation is routinely the emptiest one.
        if _tool_turns(conversation) > _tool_turns(evidence):
            evidence = [dict(message) for message in conversation]
        last_text = response.text
        last_truncated = getattr(response, "finish_reason", None) == "length"
        await bus.publish(
            "critic_reasoning",
            {"beat_id": beat_id, "critic": CRITIC_NAME, "text": response.text},
        )

        # A reasoning model can spend its entire output grant on a <think>
        # block and hand back empty text with finish_reason == "length" — the
        # exact failure mode B1 in the 2026-07-25 postmortem quantified at 42%
        # of post-fix critic wall clock, and which used to be indistinguishable
        # from a context-window overrun (see `response_truncation_remedy`).
        # Logged as its own event, greppable as `event=empty_reply`, so the
        # rate is visible directly rather than re-derived from token arithmetic
        # the way it had to be for that report.
        if last_truncated and not last_text.strip():
            log_node_event(
                "critics",
                level=logging.WARNING,
                event="empty_reply",
                beat_id=beat_id,
                attempt=attempt + 1,
                thinking_chars=len(getattr(response, "thinking", "") or ""),
                served_completion_tokens=getattr(
                    response, "served_completion_tokens", None
                ),
            )

        try:
            if last_truncated:
                raise StructuredOutputError(
                    "critic reply was cut off (finish_reason='length'): "
                    + response_truncation_remedy(
                        response, config.endpoint_for("critic")
                    )
                )
            parsed_failures = parse_failure_objects(response.text)
            failures = _locatable_findings(draft, parsed_failures, beat_id)
            failures = _sanitize_borrowed_fixes(
                failures,
                package["recent_prose"],
                beat_id,
                generation.critic_fix_max_borrowed_words,
            )
            if parsed_failures and not failures:
                raise UnlocatableFindingError(
                    "critic quoted text that is not in the draft"
                )
            unreadable = False
            break
        except StructuredOutputError as exc:
            unlocatable_findings = isinstance(exc, UnlocatableFindingError)
            last_error = str(exc)
            log_node_event(
                "critics",
                level=logging.WARNING,
                event="parse_failed",
                beat_id=beat_id,
                attempt=attempt + 1,
                retries=generation.critic_parse_retries,
                error=last_error[:200],
                truncated=last_truncated,
            )
            if attempt == generation.critic_parse_retries:
                break
            if last_truncated:
                # The model did nothing here it can read back and fix. Rebuilt
                # from the original prompt rather than appended to the growing
                # one, so this attempt is shorter than the last, not longer —
                # and free of the operator-facing diagnosis from
                # `response_truncation_remedy`, which names config knobs, not
                # anything a continuity critic should be reasoning about.
                #
                # The wording escalates with each truncation. Rebuilding from
                # two fixed constants meant the second re-ask reproduced the
                # first one byte for byte, so the endpoint was asked a question
                # it had already answered — at full price, and always with the
                # same non-answer.
                ask = (
                    _TRUNCATION_RETRY_MESSAGE
                    if truncation_asks == 0
                    else _TRUNCATION_FINAL_MESSAGE
                )
                truncation_asks += 1
                conversation = [*messages, {"role": "user", "content": ask}]
                retry_bare = True
            else:
                retry_bare = False
                # Show the model its own reply and exactly what was wrong with it.
                correction = [
                    {"role": "assistant", "content": response.text},
                    {"role": "user", "content": _CORRECTION_TEMPLATE.format(error=last_error)},
                ]
                conversation = [*conversation, *correction]
                if not _fits_window(conversation, endpoint):
                    # All-or-nothing: dropping individual tool turns would orphan
                    # an assistant message's tool_calls from its results, which
                    # some endpoints reject outright. Losing this attempt's tool
                    # reuse costs a re-search; overflowing the window costs the
                    # verdict.
                    conversation = [*messages, *correction]
                    log_node_event(
                        "critics",
                        level=logging.WARNING,
                        event="retry_context_dropped",
                        beat_id=beat_id,
                        attempt=attempt + 1,
                        budget=window_budget(endpoint),
                    )

    lenient_used = False
    if not unreadable:
        streak = 0
    else:
        # Every retry spent and still unreadable. The run continues — see the
        # module docstring — but the streak climbs and the UI is told.
        streak = state["critic_parse_failure_streak"] + 1
        failures = []
        if (
            streak >= generation.critic_degrade_threshold
            and not last_truncated
            and not unlocatable_findings
        ):
            try:
                failures = parse_failure_objects(last_text, lenient=True)
            except StructuredOutputError:
                failures = []
            else:
                failures = _locatable_findings(draft, failures, beat_id)
                failures = _sanitize_borrowed_fixes(
                    failures,
                    package["recent_prose"],
                    beat_id,
                    generation.critic_fix_max_borrowed_words,
                )
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

    # "Is this the best draft so far" and "did the last revise help" are
    # different questions and need different baselines. `best_seen_failure_count`
    # is a running minimum that this pass updates, so scoring the *same* draft
    # twice — which the retry_critic edge does, routing back here with no revise
    # in between — would read the second reading as a regression and send a beat
    # to review with its revision budget untouched. `pre_revise_failure_count`
    # only moves when revise runs, so a re-score compares against the same
    # baseline the first reading did.
    pre_revise = state["pre_revise_failure_count"]
    progressed = pre_revise is None or total < pre_revise

    delta: dict = {
        "best_seen_draft": draft if improved else state["best_seen_draft"],
        "best_seen_failure_count": total if improved else best_count,
        "last_cycle_improved": progressed,
        "critic_parse_failure_streak": streak,
        # Only an unreadable pass has a successor to hand this to: `retry_critic`
        # is the one edge that comes back to this node with the same draft. A
        # readable verdict routes on to commit, revise, or review, so holding the
        # evidence would just carry a stale transcript through the rest of the run.
        "critic_evidence": (
            {"draft_fingerprint": draft_fingerprint, "conversation": evidence}
            if unreadable and evidence
            else None
        ),
    }
    if failures:
        delta["critic_failures"] = failures

    if unreadable and not failures:
        summary = "critic output unreadable"
    else:
        summary = f"{total} issues found" if total else "clean"
    log_node_event(
        "critics",
        event="critiqued",
        beat_id=beat_id,
        critic_failures=len(failures),
        error_codes=[f.error_code for f in failures],
        total_failures=total,
        best_seen_failure_count=delta["best_seen_failure_count"],
        improved=improved,
        pre_revise_failure_count=pre_revise,
        progressed=progressed,
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
