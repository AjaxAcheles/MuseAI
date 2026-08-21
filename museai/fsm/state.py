"""LangGraph state surface for MuseAI v1.

Carries only the fields v1 actually uses. There is no scene pointer, no
escalation/replan/drift fields — those features are not in v1.
"""

from __future__ import annotations

from typing import Annotated, List

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict


class FSM_Pointer(BaseModel):
    """Where in the outline the drafter currently is.

    Chapter + Beat granularity only — v1 has no scene-level planning.
    """

    model_config = ConfigDict(extra="forbid")

    arc_id: str
    chapter_id: str
    beat_index: int


# The critic's error code for a draft that failed its own mandate: exit state
# never reached, required change not produced, a listed obligation or thread
# movement not delivered. The commit node reads it to withhold story-state
# advancement the prose did not earn.
UNFULFILLED_OBLIGATION = "UNFULFILLED_OBLIGATION"


class FailureObject(BaseModel):
    """A single continuity-critic finding against the current draft."""

    model_config = ConfigDict(extra="forbid")

    error_code: str
    offending_text: str
    suggested_fix: str
    # v1 has exactly one critic. Requiring the model to echo a constant back
    # bought nothing and cost a whole re-prompt whenever it forgot.
    critic_source: str = "continuity_critic"
    # True when the finding is a measurement over the entire draft (a density
    # breach) rather than a fault in one locatable span. A whole-draft failure
    # cannot be fixed by rewriting the single sentence quoted in
    # ``offending_text``, so the reviser must regenerate the beat against the
    # full offender list.
    whole_draft: bool = False


def accumulate_or_reset(
    current: List[FailureObject], incoming: List[FailureObject]
) -> List[FailureObject]:
    """LangGraph reducer for ``critic_failures``.

    Append when ``incoming`` is non-empty; return a fresh empty list when
    ``incoming`` is an explicit empty list (reset semantics).
    """
    if not incoming:
        return []
    return list(current) + list(incoming)


class OrchestratorState(TypedDict):
    """The full v1 orchestrator state passed between graph nodes."""

    project_id: str
    fsm_pointer: FSM_Pointer
    active_context_package: dict
    current_draft_text: str
    streaming_buffer: str
    critic_failures: Annotated[List[FailureObject], accumulate_or_reset]
    retry_count: int
    best_seen_draft: str | None
    best_seen_failure_count: int | None
    # The total failure count the last `revise` was handed, for this beat. Only
    # `revise` writes it, which is the point: the `retry_critic` edge routes
    # back into `critics` with no revise in between, and a baseline that moved
    # on every critic pass would read that re-score of unchanged prose as a
    # regression. None until a revise has run.
    pre_revise_failure_count: int | None
    # Whether the most recent critics cycle beat `pre_revise_failure_count` —
    # i.e. whether the last revise actually helped. `best_seen_failure_count`
    # cannot answer this: it is a running minimum, already updated to include
    # this cycle by the time mode_selector runs. This field is what
    # mode_selector's no-progress rule reads. True until a revise has run.
    last_cycle_improved: bool
    # Consecutive beats whose critic output stayed unparseable after every retry.
    # Spans beats within a run; a single readable critic resets it to 0.
    critic_parse_failure_streak: int
    # What the critic's last unreadable pass already looked up, so the pass the
    # `retry_critic` edge is about to start does not pay for it twice.
    #
    # `{"draft_fingerprint": str, "conversation": [...]}` — the loop's working
    # messages (assistant tool-call turns and their tool results, never a final
    # reply). `retry_critic` re-enters `critics` with the draft *unchanged*, and
    # the node used to rebuild from the bare two-message prompt, so the model
    # re-derived the same continuity facts from scratch every time; one live beat
    # searched the same kettle/door prose across four separate passes.
    #
    # The fingerprint is the safety catch: it is taken against the draft the
    # evidence was gathered from, so any revise makes it stale automatically and
    # the next pass starts clean. Cleared whenever the verdict was readable —
    # nothing is coming back to reuse it.
    critic_evidence: dict | None
    # How many paragraphs `audit`'s own repetition guard (`paragraph_overlaps`)
    # flagged against the current draft. Set fresh by every `audit` call, never
    # by `critics`; `retry_critic` re-enters `critics` directly (see
    # `graph.py`), so the count from the last real audit stays correct for a
    # draft that hasn't changed. Carried into the critic's prompt so it can be
    # told the manuscript-repetition question is already answered, instead of
    # offering `find_repetition` as a tool for a claim its own error-code list
    # (`CRITIC_ERROR_CODES`) has no category for.
    repetition_overlap_count: int
    # set when the revision cap is exhausted; parks at the interactive review state
    review_requested: bool
    pause_requested: bool
    hard_stop_asserted: bool


def make_initial_state(
    project_id: str, fsm_pointer: FSM_Pointer, **overrides: object
) -> OrchestratorState:
    """Build a fully-initialized ``OrchestratorState``.

    Every field is populated with a fresh default (empty dict/str/list, counters
    at 0, nullable fields at ``None``, bools ``False``). Unknown override keys
    raise ``KeyError`` clearly. Mutable defaults are created fresh per call.
    """
    state: OrchestratorState = {
        "project_id": project_id,
        "fsm_pointer": fsm_pointer,
        "active_context_package": {},
        "current_draft_text": "",
        "streaming_buffer": "",
        "critic_failures": [],
        "retry_count": 0,
        "best_seen_draft": None,
        "best_seen_failure_count": None,
        "pre_revise_failure_count": None,
        "last_cycle_improved": True,
        "critic_parse_failure_streak": 0,
        "critic_evidence": None,
        "repetition_overlap_count": 0,
        "review_requested": False,
        "pause_requested": False,
        "hard_stop_asserted": False,
    }

    unknown = set(overrides) - set(state)
    if unknown:
        raise KeyError(
            f"unknown override key(s) for OrchestratorState: {sorted(unknown)}"
        )
    state.update(overrides)  # type: ignore[typeddict-item]
    return state
