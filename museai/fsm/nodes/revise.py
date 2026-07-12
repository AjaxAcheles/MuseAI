"""Revision node.

Takes the failures the audit and the critic found and rewrites the draft to
answer them. It works at the smallest scope that can fix the problem:

* **Span mode.** Every failure's ``offending_text`` was located in the draft, so
  each span is rewritten on its own and spliced back in. Prose the critic did
  not fault is never sent to the model to be regenerated, and so cannot drift.
* **Full mode.** At least one failure could not be located — the critic
  paraphrased, or faulted the beat as a whole — so the beat is rewritten in one
  pass against the complete failure list.

Locating is three-tier: an exact ``str.find``, then a fuzzy scan over sliding
windows (difflib ratio ≥ ``FUZZY_THRESHOLD``) to survive a critic that
normalised a quotation mark, then failure. Overlapping spans also fall to full
mode: splicing two rewrites of the same sentence would produce neither.

The reviser prompt is budgeted in two tiers. Full context first; if the rendered
messages exceed ``generation.context_token_budget``, the context collapses to
the draft, the failures, and the hard constraints — the beat spec, the
``pad_constraint``, and the chapter's obligations. Those three are never
dropped: a revision written without them fixes one problem and creates another.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from museai.core.logging_setup import get_fsm_logger, log_node_event
from museai.core.stream_bus import bus
from museai.fsm.nodes.deps import DraftingError, get_node_config
from museai.fsm.state import FailureObject, OrchestratorState
from museai.fsm.tools.loop import run_agent_loop
from museai.fsm.tools.registry import tool_impls_for, tool_specs_for
from museai.llm.prompts import render_messages
from museai.llm.tokenizer import count_message_tokens

PHASE = "Drafting"

# A critic quoting the draft rarely mangles it beyond this. Below the threshold
# the "span" is likelier a paraphrase, and rewriting the wrong sentence is worse
# than rewriting the beat.
FUZZY_THRESHOLD = 0.8

# Fuzzy windows start on word boundaries: an offending span begins at a word.
_MIN_SPAN_CHARS = 4


def _word_starts(text: str) -> list[int]:
    """Offsets of every word start in ``text`` — the candidate span origins."""
    starts = [0] if text[:1].strip() else []
    starts.extend(i + 1 for i, char in enumerate(text[:-1]) if char.isspace() and not text[i + 1].isspace())
    return starts


def fuzzy_find(draft: str, needle: str, threshold: float = FUZZY_THRESHOLD) -> tuple[int, int] | None:
    """Best near-match for ``needle`` in ``draft``, or ``None``.

    Slides a window of the needle's length across the draft's word boundaries and
    keeps the highest-scoring window at or above ``threshold``.
    """
    if len(needle) < _MIN_SPAN_CHARS or not draft:
        return None

    matcher = SequenceMatcher(autojunk=False)
    matcher.set_seq2(needle)

    best_ratio = threshold
    best: tuple[int, int] | None = None

    for start in _word_starts(draft):
        window = draft[start : start + len(needle)]
        if len(window) < _MIN_SPAN_CHARS:
            break
        matcher.set_seq1(window)
        # quick_ratio is a cheap upper bound; skip windows that cannot win.
        if matcher.quick_ratio() < best_ratio:
            continue
        ratio = matcher.ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best = (start, start + len(window))

    return best


def locate(draft: str, offending_text: str) -> tuple[int, int] | None:
    """Find ``offending_text`` in ``draft``: exact first, then fuzzy."""
    needle = offending_text.strip()
    if not needle:
        return None

    index = draft.find(needle)
    if index != -1:
        return index, index + len(needle)

    return fuzzy_find(draft, needle)


def _overlapping(spans: list[tuple[int, int]]) -> bool:
    ordered = sorted(spans)
    return any(a[1] > b[0] for a, b in zip(ordered, ordered[1:]))


# A span rewrite the model padded with copies of the surrounding prose splices
# in as duplicated paragraphs — artifacts that have reached exported
# manuscripts. Implausible rewrites are rejected and the beat falls back to a
# full rewrite instead.
_SPAN_GROWTH_LIMIT = 3
_SPAN_GROWTH_SLACK = 400  # chars — a very short span may legitimately grow more
_ECHO_MIN_WORDS = 8


def _verbatim_echo(replacement: str, surrounding: str) -> str | None:
    """A run of ``_ECHO_MIN_WORDS`` words from ``replacement`` found verbatim
    in the prose around the span, or ``None``. Whitespace-normalised on both
    sides so a reflowed line still matches."""
    words = replacement.split()
    if len(words) < _ECHO_MIN_WORDS:
        return None
    surrounding_norm = " ".join(surrounding.split())
    for start in range(len(words) - _ECHO_MIN_WORDS + 1):
        window = " ".join(words[start : start + _ECHO_MIN_WORDS])
        if window in surrounding_norm:
            return window
    return None


def replacement_rejection(draft: str, span: tuple[int, int], replacement: str) -> str | None:
    """Why this span rewrite must not be spliced, or ``None`` when it is safe."""
    start, end = span
    span_len = end - start
    limit = max(_SPAN_GROWTH_LIMIT * span_len, span_len + _SPAN_GROWTH_SLACK)
    if len(replacement) > limit:
        return f"replacement grew a {span_len}-char span to {len(replacement)} chars"
    echo = _verbatim_echo(replacement, draft[:start] + draft[end:])
    if echo is not None:
        return f"replacement repeats surrounding prose: {echo[:80]!r}"
    return None


def _reviser_messages(
    *,
    mode: str,
    draft_text: str,
    failures: list[FailureObject],
    span_text: str,
    package: dict,
    collapsed: bool,
) -> list[dict]:
    """Render the reviser prompt at one of the two budget tiers."""
    return render_messages(
        "reviser",
        mode=mode,
        draft_text=draft_text,
        span_text=span_text,
        failures=[f.model_dump() for f in failures],
        beat=package["beat"],
        pad_constraint=package["pad_constraint"],
        chapter=package["chapter"],
        # The collapsed tier keeps the beat spec, the pad constraint, and the
        # chapter obligations; everything else is context the revision can lose.
        threads=[] if collapsed else package["threads"],
        characters=[] if collapsed else package["characters"],
        recent_prose=[] if collapsed else package["recent_prose"],
        research_mode=get_node_config().generation.research_mode,
    )


def _budgeted_messages(config, **kwargs) -> tuple[list[dict], bool]:
    """Render at full context, collapsing to the hard constraints if over budget."""
    endpoint = config.endpoint_for("reviser")
    budget = config.generation.context_token_budget

    messages = _reviser_messages(collapsed=False, **kwargs)
    tokens = count_message_tokens(messages, endpoint.tokenizer_family, endpoint.model_name)
    if tokens <= budget:
        return messages, False

    collapsed = _reviser_messages(collapsed=True, **kwargs)
    get_fsm_logger().info(
        "node=revise context collapsed to hard constraints: tokens=%d budget=%d",
        tokens,
        budget,
    )
    return collapsed, True


async def _rewrite(config, messages: list[dict], what: str, beat_id: str) -> str:
    """One reviser call. Empty prose is a hard failure, never a silent no-op."""

    async def on_tool_call(event: dict) -> None:
        log_node_event(
            "revise",
            event="tool_call",
            beat_id=beat_id,
            tool=event["tool"],
            args=event["arguments"],
        )
        await bus.publish("reviser_tool", {"beat_id": beat_id, **event})

    response = await run_agent_loop(
        config.endpoint_for("reviser"),
        messages,
        tool_specs_for("reviser"),
        tool_impls_for("reviser"),
        config.generation.max_agent_iterations,
        on_event=on_tool_call,
        agent="reviser",
        tool_call_cap=config.generation.tool_call_cap,
    )
    revised = response.text.strip()
    # Truncation first: it explains an empty reply as readily as a half-written
    # one, and unlike "no prose" it names the knob. Splicing a truncated rewrite
    # into the draft is worse than failing.
    if response.finish_reason == "length":
        raise DraftingError(
            f"the revision of {what} was truncated at the endpoint's output "
            f"token limit (finish_reason='length'); raise "
            f"endpoint.max_output_tokens or leave it unset to omit the cap"
        )
    if not revised:
        raise DraftingError(
            f"the endpoint returned no prose revising {what} "
            f"(finish_reason={response.finish_reason!r})"
        )
    return revised


async def revise_prose(state: OrchestratorState) -> dict:
    """Rewrite the draft to answer every outstanding failure.

    Returns the state delta. ``critic_failures`` is an explicit ``[]``, which
    resets the reducer: the findings described prose that no longer exists.
    """
    config = get_node_config()
    package = state["active_context_package"]
    draft = state["current_draft_text"]
    failures = list(state["critic_failures"])
    beat_id = package["beat"]["id"]

    if not failures:
        raise DraftingError(
            f"revise_prose ran for beat {beat_id!r} with no failures to fix"
        )

    log_node_event(
        "revise",
        event="start",
        beat_id=beat_id,
        failures=len(failures),
        retry_count=state["retry_count"],
    )

    located = [(failure, locate(draft, failure.offending_text)) for failure in failures]
    spans = [span for _, span in located if span is not None]
    span_mode = len(spans) == len(failures) and not _overlapping(spans)
    collapsed = False

    if span_mode:
        revised = draft
        # Splice from the end so an earlier rewrite cannot shift a later offset.
        for failure, span in sorted(located, key=lambda pair: pair[1], reverse=True):
            start, end = span
            messages, collapsed = _budgeted_messages(
                config,
                mode="span",
                draft_text=draft,
                failures=[failure],
                span_text=draft[start:end],
                package=package,
            )
            replacement = await _rewrite(config, messages, f"span {start}:{end}", beat_id)
            rejection = replacement_rejection(draft, (start, end), replacement)
            if rejection is not None:
                # Splicing this would duplicate prose. The full rewrite below
                # regenerates the beat against every failure instead.
                log_node_event(
                    "revise",
                    event="span_rejected",
                    beat_id=beat_id,
                    span=f"{start}:{end}",
                    reason=rejection,
                )
                span_mode = False
                break
            revised = revised[:start] + replacement + revised[end:]

    if not span_mode:
        messages, collapsed = _budgeted_messages(
            config,
            mode="full",
            draft_text=draft,
            failures=failures,
            span_text="",
            package=package,
        )
        revised = await _rewrite(config, messages, f"beat {beat_id!r}", beat_id)

    retry_count = state["retry_count"] + 1
    mode = "span" if span_mode else "full"

    log_node_event(
        "revise",
        event="revised",
        beat_id=beat_id,
        mode=mode,
        failures_fixed=len(failures),
        spans_located=len(spans),
        collapsed_context=collapsed,
        retry_count=retry_count,
        words=len(revised.split()),
    )
    await bus.publish(
        "revision",
        {
            "beat_id": beat_id,
            "mode": mode,
            "failures_fixed": len(failures),
            "retry_count": retry_count,
            "text": revised,
        },
    )

    return {
        "current_draft_text": revised,
        "streaming_buffer": revised,
        "retry_count": retry_count,
        "critic_failures": [],
    }
