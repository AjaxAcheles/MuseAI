"""Module: M05 (Hierarchical Planning Cascade)
Deterministic, programmatic validators over a proposed plan dict, plus the
ValidationResult type the deliberation loop and planner nodes read.

These are the harness-authoritative checks: a planner LLM *proposes* a plan and may
self-report a `self_check`, but acceptance of a plan (especially a `finalize_plan`) is
decided here by structural inspection, never by the model's self-grading
(Context_Injection_Algorithms.md Phase 1 preamble; Module_Ability_Specification §5 —
required validators cannot be skipped). Every function in this module is pure and
side-effect free: it inspects its inputs and returns a structured pass/fail with a
reason. No LLM calls, no store reads, no network, and a validator never mutates the plan.

This module defines both the individual validators and `run_validators`, the
per-level required-check runner: it reads `config.planning.planner_required_checks[level]`
(`Configuration_Reference.md` §3a; `LangGraph_Nodes.md` Phase B preamble) and runs
exactly those checks via `REQUIRED_CHECK_REGISTRY`. Required checks cannot be skipped — a
configured check-name with no registered implementation is a hard error (fail closed),
never a silent pass. Wiring `run_validators` into the bounded deliberation loop is a later
M05 increment. Tunable expectations (required fields, required threads, expected depth
markers, hard annotations, continuity facts) are passed in by the caller (from the
compiled constraint package / continuity context) rather than hardcoded, so this module
guesses no numeric thresholds.
"""

from typing import Any, Callable, Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field

# The five planner levels, coarsest → finest (Context_Injection_Algorithms.md Phase 1).
PLANNING_LEVELS: tuple[str, ...] = ("global", "arc", "chapter", "scene", "beat")

# Field names that carry committed/draft *prose* rather than structural planning data.
# The planning proposal surface must never carry these — that boundary is what
# validate_no_drafting protects (proposal surface vs committed prose, Data_Structures §2.7).
_PROSE_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "prose",
        "draft",
        "draft_text",
        "prose_text",
        "narrative_text",
        "body_text",
        "committed_prose",
        "current_draft_text",
        "streaming_buffer",
    }
)

# Structural granularity contract per level (presence-only, no numeric thresholds). The
# design states two explicitly: a chapter plan carries obligations and *not* final scene
# rows; a scene plan carries entry/exit state. Other levels keep a minimal "non-empty
# dict" structural contract here and are refined by config / Phase B in a later increment.
_DEPTH_CONTRACT: dict[str, dict[str, tuple[str, ...]]] = {
    "global": {"required": (), "forbidden": ()},
    "arc": {"required": (), "forbidden": ()},
    "chapter": {"required": ("obligations",), "forbidden": ("scenes",)},
    "scene": {"required": ("entry_state", "exit_state"), "forbidden": ()},
    "beat": {"required": (), "forbidden": ()},
}


class ValidationResult(BaseModel):
    """Outcome of one (or several) deterministic plan validators.

    `passes` is the gate the loop reads; `failed_checks` names each validator that
    failed (so an aggregator can report all failures); `details` carries per-check
    diagnostics (missing fields, contradictions, the offending values) for the planner's
    next turn and for observability.
    """

    model_config = ConfigDict(extra="forbid")

    passes: bool
    failed_checks: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


def _pass(check: str, **details: Any) -> ValidationResult:
    """Build a passing result for `check`."""

    return ValidationResult(passes=True, failed_checks=[], details={check: details})


def _fail(check: str, **details: Any) -> ValidationResult:
    """Build a failing result naming `check`."""

    return ValidationResult(passes=False, failed_checks=[check], details={check: details})


def _is_empty(value: Any) -> bool:
    """Treat None and empty string/collection as "not provided"."""

    if value is None:
        return True
    if isinstance(value, (str, bytes, list, tuple, set, dict)) and len(value) == 0:
        return True
    return False


def validate_schema(
    plan: Any, level: str, *, required_keys: Iterable[str] | None = None
) -> ValidationResult:
    """The plan has the structural shape required for its level.

    Structural only: the plan must be a (non-empty) dict for a known level, and — when
    the caller supplies the level's `required_keys` — every one of those keys must be
    present. Key *values* are checked by validate_required_fields, not here.
    """

    if level not in PLANNING_LEVELS:
        return _fail("validate_schema", reason="unknown_level", level=level)
    if not isinstance(plan, dict):
        return _fail("validate_schema", reason="plan_not_dict", plan_type=type(plan).__name__)
    if len(plan) == 0:
        return _fail("validate_schema", reason="empty_plan")
    if required_keys:
        missing = [k for k in required_keys if k not in plan]
        if missing:
            return _fail("validate_schema", reason="missing_keys", missing_keys=missing)
    return _pass("validate_schema", level=level)


def validate_required_fields(
    plan: Any, level: str, *, required_fields: Iterable[str] | None = None
) -> ValidationResult:
    """Every required field for the level is present and non-empty.

    `required_fields` is supplied by the caller (compiled constraints / config); with
    none supplied there is nothing to require and the check passes.
    """

    if not isinstance(plan, dict):
        return _fail(
            "validate_required_fields", reason="plan_not_dict", plan_type=type(plan).__name__
        )
    if not required_fields:
        return _pass("validate_required_fields", level=level, required_fields=[])
    missing = [f for f in required_fields if f not in plan or _is_empty(plan.get(f))]
    if missing:
        return _fail("validate_required_fields", level=level, missing_or_empty=missing)
    return _pass("validate_required_fields", level=level, checked=list(required_fields))


def _collect_addressed_thread_ids(plan: dict) -> set[str]:
    """Collect thread ids the plan addresses, from documented structural keys.

    Recognises `thread_updates` / `threads` / `addressed_threads` / `thread_coverage`,
    each either a list of ids or a list of dicts carrying `thread_id`/`id`.
    """

    addressed: set[str] = set()
    for key in ("thread_updates", "threads", "addressed_threads", "thread_coverage"):
        entries = plan.get(key)
        if not isinstance(entries, (list, tuple, set)):
            continue
        for entry in entries:
            if isinstance(entry, str):
                addressed.add(entry)
            elif isinstance(entry, dict):
                tid = entry.get("thread_id", entry.get("id"))
                if isinstance(tid, str):
                    addressed.add(tid)
    return addressed


def validate_thread_coverage(
    plan: Any, *, required_threads: Iterable[str] | None = None
) -> ValidationResult:
    """Required/open threads for the scope are addressed by the plan.

    `required_threads` is the set of thread ids the caller (open-thread query / compiled
    constraints) says this scope must address. Coverage is structural: each required id
    must appear among the plan's addressed thread references.
    """

    if not isinstance(plan, dict):
        return _fail(
            "validate_thread_coverage", reason="plan_not_dict", plan_type=type(plan).__name__
        )
    required = {t for t in (required_threads or []) if isinstance(t, str)}
    if not required:
        return _pass("validate_thread_coverage", required_threads=[])
    addressed = _collect_addressed_thread_ids(plan)
    uncovered = sorted(required - addressed)
    if uncovered:
        return _fail(
            "validate_thread_coverage",
            uncovered_threads=uncovered,
            addressed_threads=sorted(addressed),
        )
    return _pass("validate_thread_coverage", required_threads=sorted(required))


def _collect_annotation_outcomes(plan: dict) -> dict[str, str]:
    """Map annotation id → outcome the plan records (applied / partially_applied / rejected).

    Reads `annotation_outcomes` (a {id: outcome} map, or a list of
    {annotation_id|id, outcome} dicts) — the structure the revision diff records (A14).
    """

    outcomes: dict[str, str] = {}
    raw = plan.get("annotation_outcomes")
    if isinstance(raw, dict):
        for ann_id, outcome in raw.items():
            if isinstance(ann_id, str) and isinstance(outcome, str):
                outcomes[ann_id] = outcome
    elif isinstance(raw, (list, tuple)):
        for entry in raw:
            if isinstance(entry, dict):
                ann_id = entry.get("annotation_id", entry.get("id"))
                outcome = entry.get("outcome")
                if isinstance(ann_id, str) and isinstance(outcome, str):
                    outcomes[ann_id] = outcome
    return outcomes


def validate_user_annotations(
    plan: Any, *, hard_annotations: Iterable[dict] | None = None
) -> ValidationResult:
    """Every applicable hard user annotation is satisfied by the plan.

    `hard_annotations` are the already-compiled hard constraints for this node (the
    annotation compiler resolves precedence and contradictions upstream — A13). A hard
    annotation is satisfied only when the plan records its id with an `applied` outcome;
    a missing, rejected, or merely partially_applied hard annotation fails.
    """

    if not isinstance(plan, dict):
        return _fail(
            "validate_user_annotations", reason="plan_not_dict", plan_type=type(plan).__name__
        )
    annotations = list(hard_annotations or [])
    if not annotations:
        return _pass("validate_user_annotations", hard_annotations=0)
    outcomes = _collect_annotation_outcomes(plan)
    unsatisfied: list[dict[str, Any]] = []
    for ann in annotations:
        ann_id = ann.get("annotation_id", ann.get("id")) if isinstance(ann, dict) else None
        outcome = outcomes.get(ann_id) if isinstance(ann_id, str) else None
        if outcome != "applied":
            unsatisfied.append({"annotation_id": ann_id, "outcome": outcome})
    if unsatisfied:
        return _fail("validate_user_annotations", unsatisfied=unsatisfied)
    return _pass("validate_user_annotations", satisfied=len(annotations))


def _normalize_facts(facts: Iterable[Any]) -> dict[tuple[str, str], set[str]]:
    """Index facts by (subject, attribute) → set of asserted string values.

    Each fact is a dict with `subject` and `attribute` (or `predicate`) and `value`.
    Facts missing any part are ignored (nothing to contradict).
    """

    indexed: dict[tuple[str, str], set[str]] = {}
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        subject = fact.get("subject")
        attribute = fact.get("attribute", fact.get("predicate"))
        value = fact.get("value")
        if not (isinstance(subject, str) and isinstance(attribute, str)):
            continue
        indexed.setdefault((subject, attribute), set()).add(str(value))
    return indexed


def validate_continuity_warnings(
    plan: Any, *, continuity_facts: Iterable[dict] | None = None
) -> ValidationResult:
    """The plan does not contradict the supplied continuity facts.

    Operates only on the continuity context handed in (no store reads here). The plan's
    own asserted facts (`asserted_facts`, same {subject, attribute, value} shape) are
    compared against `continuity_facts`; a contradiction is the same (subject, attribute)
    asserted with a different value.
    """

    if not isinstance(plan, dict):
        return _fail(
            "validate_continuity_warnings", reason="plan_not_dict", plan_type=type(plan).__name__
        )
    facts = _normalize_facts(continuity_facts or [])
    plan_claims_raw = plan.get("asserted_facts")
    plan_claims = _normalize_facts(plan_claims_raw if isinstance(plan_claims_raw, list) else [])
    if not facts or not plan_claims:
        return _pass("validate_continuity_warnings", facts=len(facts), plan_claims=len(plan_claims))
    contradictions: list[dict[str, Any]] = []
    for key, plan_values in plan_claims.items():
        known = facts.get(key)
        if known and not (plan_values & known):
            contradictions.append(
                {
                    "subject": key[0],
                    "attribute": key[1],
                    "plan_values": sorted(plan_values),
                    "known_values": sorted(known),
                }
            )
    if contradictions:
        return _fail("validate_continuity_warnings", contradictions=contradictions)
    return _pass("validate_continuity_warnings", checked_claims=len(plan_claims))


def validate_depth(
    plan: Any,
    level: str,
    *,
    required_markers: Iterable[str] | None = None,
    forbidden_markers: Iterable[str] | None = None,
) -> ValidationResult:
    """The plan is decomposed to the level's expected granularity.

    Presence-only structural check against the per-level granularity contract (e.g. a
    chapter plan carries `obligations` and not final `scenes`; a scene plan carries
    `entry_state`/`exit_state`). Callers may override the contract via
    `required_markers` / `forbidden_markers`; otherwise the documented default applies.
    """

    if level not in PLANNING_LEVELS:
        return _fail("validate_depth", reason="unknown_level", level=level)
    if not isinstance(plan, dict):
        return _fail("validate_depth", reason="plan_not_dict", plan_type=type(plan).__name__)
    contract = _DEPTH_CONTRACT.get(level, {"required": (), "forbidden": ()})
    required = tuple(required_markers) if required_markers is not None else contract["required"]
    forbidden = tuple(forbidden_markers) if forbidden_markers is not None else contract["forbidden"]
    if not required and not forbidden and len(plan) == 0:
        return _fail("validate_depth", reason="empty_plan", level=level)
    missing = [m for m in required if m not in plan or _is_empty(plan.get(m))]
    present_forbidden = [m for m in forbidden if not _is_empty(plan.get(m))]
    if missing or present_forbidden:
        return _fail(
            "validate_depth",
            level=level,
            missing_granularity=missing,
            forbidden_granularity=present_forbidden,
        )
    return _pass("validate_depth", level=level)


def _ordering_values(collection: Any) -> list[int] | None:
    """Extract integer order values from a list of items, or None if not orderable.

    Prefers an `ordering` field, falling back to `beat_index` (the two ordered columns
    in the schema). Returns None when the collection isn't a list of dicts carrying an
    integer order key.
    """

    if not isinstance(collection, (list, tuple)):
        return None
    values: list[int] = []
    for item in collection:
        if not isinstance(item, dict):
            return None
        order = item.get("ordering", item.get("beat_index"))
        if not isinstance(order, int) or isinstance(order, bool):
            return None
        values.append(order)
    return values


def validate_ordering(
    plan: Any, *, collection_keys: Iterable[str] | None = None
) -> ValidationResult:
    """Any ordered collection has a strictly monotonic, gapless ordering.

    Checks each ordered collection in the plan (default keys: `scenes`, `beats`): its
    order values must be distinct and form consecutive integers (no duplicates, no gaps)
    regardless of whether numbering starts at 0 or 1. Empty/absent collections pass.
    """

    if not isinstance(plan, dict):
        return _fail("validate_ordering", reason="plan_not_dict", plan_type=type(plan).__name__)
    keys = list(collection_keys) if collection_keys is not None else ["scenes", "beats"]
    bad: list[dict[str, Any]] = []
    checked: list[str] = []
    for key in keys:
        if key not in plan:
            continue
        values = _ordering_values(plan.get(key))
        if values is None:
            bad.append({"collection": key, "reason": "unorderable_items"})
            continue
        if not values:
            continue
        checked.append(key)
        ordered = sorted(values)
        expected = list(range(ordered[0], ordered[0] + len(ordered)))
        if ordered != expected:
            bad.append({"collection": key, "reason": "not_monotonic_gapless", "ordering": values})
    if bad:
        return _fail("validate_ordering", offending=bad)
    return _pass("validate_ordering", checked=checked)


def _find_prose_fields(value: Any, path: str = "") -> list[str]:
    """Recursively collect dotted paths of any prose/draft field names present."""

    found: list[str] = []
    if isinstance(value, dict):
        for key, sub in value.items():
            here = f"{path}.{key}" if path else str(key)
            if key in _PROSE_FIELD_NAMES:
                found.append(here)
            found.extend(_find_prose_fields(sub, here))
    elif isinstance(value, (list, tuple)):
        for idx, sub in enumerate(value):
            found.extend(_find_prose_fields(sub, f"{path}[{idx}]"))
    return found


def validate_no_drafting(plan: Any) -> ValidationResult:
    """The plan contains structural planning fields only — no prose/draft narrative text.

    Guards the proposal-surface-vs-committed-prose boundary: a planning dict must never
    carry a prose/draft field (`prose`, `draft_text`, `current_draft_text`, …) anywhere,
    including nested structures. Presence of any such field fails the check.
    """

    if not isinstance(plan, dict):
        return _fail("validate_no_drafting", reason="plan_not_dict", plan_type=type(plan).__name__)
    prose_fields = _find_prose_fields(plan)
    if prose_fields:
        return _fail("validate_no_drafting", prose_fields=prose_fields)
    return _pass("validate_no_drafting")


# ---------------------------------------------------------------------------
# Per-level required-check runner (Configuration_Reference.md §3a;
# LangGraph_Nodes.md Phase B preamble).
#
# Each registered check has the uniform signature (plan, level, constraints, continuity)
# -> ValidationResult, so run_validators can drive them generically. Generic-validator
# check-names route to the eight validators above; level-specific semantic check-names
# (arc_coverage, major_promise_payoff, escalation, thread_distribution, chapter_function,
# pacing, scene_function, entry_exit_state, draftability, pad_grounding) are implemented
# as deterministic *structural* checks — presence / coverage / ordering / declared-
# promise-has-payoff / beat-has-PAD-target+behavioural-string — never LLM calls.
# ---------------------------------------------------------------------------

CheckFn = Callable[[Any, str, dict, dict], ValidationResult]


def _not_dict(plan: Any, check: str) -> ValidationResult | None:
    """Return a failing result if `plan` is not a dict, else None."""

    if not isinstance(plan, dict):
        return _fail(check, reason="plan_not_dict", plan_type=type(plan).__name__)
    return None


def _first_present(plan: dict, names: Iterable[str]) -> tuple[str | None, Any]:
    """Return the first (name, value) among `names` present and non-empty in plan."""

    for name in names:
        if name in plan and not _is_empty(plan.get(name)):
            return name, plan.get(name)
    return None, None


def _collect_ids(entries: Any, *, id_keys: tuple[str, ...]) -> set[str]:
    """Collect string ids from a list of ids or dicts carrying one of `id_keys`."""

    found: set[str] = set()
    if not isinstance(entries, (list, tuple, set)):
        return found
    for entry in entries:
        if isinstance(entry, str):
            found.add(entry)
        elif isinstance(entry, dict):
            for key in id_keys:
                value = entry.get(key)
                if isinstance(value, str):
                    found.add(value)
                    break
    return found


def _required_threads(constraints: dict) -> list[str] | None:
    value = constraints.get("required_threads") if isinstance(constraints, dict) else None
    return list(value) if isinstance(value, (list, tuple, set)) else None


def _hard_annotations(constraints: dict) -> list[dict] | None:
    value = constraints.get("hard_annotations") if isinstance(constraints, dict) else None
    return list(value) if isinstance(value, (list, tuple)) else None


def _continuity_facts(continuity: Any) -> list:
    if isinstance(continuity, dict):
        facts = continuity.get("facts", continuity.get("continuity_facts"))
        return list(facts) if isinstance(facts, (list, tuple)) else []
    if isinstance(continuity, (list, tuple)):
        return list(continuity)
    return []


# --- generic-validator adapters (uniform signature) ---


def _run_schema(plan: Any, level: str, constraints: dict, continuity: dict) -> ValidationResult:
    return validate_schema(plan, level, required_keys=constraints.get("required_keys"))


def _run_required_fields(plan: Any, level: str, constraints: dict, continuity: dict) -> ValidationResult:
    return validate_required_fields(plan, level, required_fields=constraints.get("required_fields"))


def _run_thread_coverage(plan: Any, level: str, constraints: dict, continuity: dict) -> ValidationResult:
    return validate_thread_coverage(plan, required_threads=_required_threads(constraints))


def _run_user_annotations(plan: Any, level: str, constraints: dict, continuity: dict) -> ValidationResult:
    return validate_user_annotations(plan, hard_annotations=_hard_annotations(constraints))


def _run_continuity(plan: Any, level: str, constraints: dict, continuity: dict) -> ValidationResult:
    return validate_continuity_warnings(plan, continuity_facts=_continuity_facts(continuity))


def _run_depth(plan: Any, level: str, constraints: dict, continuity: dict) -> ValidationResult:
    return validate_depth(plan, level)


def _run_ordering(plan: Any, level: str, constraints: dict, continuity: dict) -> ValidationResult:
    return validate_ordering(plan)


def _run_no_drafting(plan: Any, level: str, constraints: dict, continuity: dict) -> ValidationResult:
    return validate_no_drafting(plan)


# --- level-specific structural semantic checks ---


def _check_arc_coverage(plan: Any, level: str, constraints: dict, continuity: dict) -> ValidationResult:
    """Global: the plan declares a non-empty arc set, each arc carrying an id."""

    guard = _not_dict(plan, "arc_coverage")
    if guard:
        return guard
    arcs = plan.get("arcs")
    if not isinstance(arcs, list) or len(arcs) == 0:
        return _fail("arc_coverage", reason="no_arcs_declared")
    uncovered = [
        i for i, a in enumerate(arcs)
        if not isinstance(a, dict) or _is_empty(a.get("arc_id", a.get("id")))
    ]
    if uncovered:
        return _fail("arc_coverage", reason="arcs_missing_id", indices=uncovered)
    return _pass("arc_coverage", arc_count=len(arcs))


def _check_major_promise_payoff(plan: Any, level: str, constraints: dict, continuity: dict) -> ValidationResult:
    """Global: every declared major promise has a payoff (inline or in `payoffs`)."""

    guard = _not_dict(plan, "major_promise_payoff")
    if guard:
        return guard
    promises = plan.get("promises")
    if not isinstance(promises, list) or len(promises) == 0:
        return _fail("major_promise_payoff", reason="no_promises_declared")
    payoff_ids = _collect_ids(plan.get("payoffs"), id_keys=("promise_id", "id"))
    unpaid: list[str | None] = []
    for promise in promises:
        if not isinstance(promise, dict):
            unpaid.append(None)
            continue
        pid = promise.get("id", promise.get("promise_id"))
        has_payoff = not _is_empty(promise.get("payoff")) or (isinstance(pid, str) and pid in payoff_ids)
        if not has_payoff:
            unpaid.append(pid)
    if unpaid:
        return _fail("major_promise_payoff", unpaid_promises=unpaid)
    return _pass("major_promise_payoff", promises=len(promises))


def _check_escalation(plan: Any, level: str, constraints: dict, continuity: dict) -> ValidationResult:
    """Arc: declares ordered milestones; any integer tension values do not de-escalate."""

    guard = _not_dict(plan, "escalation")
    if guard:
        return guard
    milestones = plan.get("milestones")
    if not isinstance(milestones, list) or len(milestones) == 0:
        return _fail("escalation", reason="no_milestones")
    tensions = [
        m.get("tension", m.get("intensity"))
        for m in milestones
        if isinstance(m, dict)
    ]
    ints = [t for t in tensions if isinstance(t, int) and not isinstance(t, bool)]
    if ints and any(ints[i] > ints[i + 1] for i in range(len(ints) - 1)):
        return _fail("escalation", reason="de_escalation", tensions=ints)
    return _pass("escalation", milestones=len(milestones))


def _check_thread_distribution(plan: Any, level: str, constraints: dict, continuity: dict) -> ValidationResult:
    """Arc: required threads (if supplied) are covered; else a thread structure is declared."""

    guard = _not_dict(plan, "thread_distribution")
    if guard:
        return guard
    required = _required_threads(constraints)
    if required:
        coverage = validate_thread_coverage(plan, required_threads=required)
        if not coverage.passes:
            return _fail("thread_distribution", **coverage.details.get("validate_thread_coverage", {}))
        return _pass("thread_distribution", required_threads=sorted(required))
    addressed = _collect_addressed_thread_ids(plan)
    if not addressed and _is_empty(plan.get("thread_distribution")):
        return _fail("thread_distribution", reason="no_thread_distribution_declared")
    return _pass("thread_distribution", addressed_threads=sorted(addressed))


def _check_chapter_function(plan: Any, level: str, constraints: dict, continuity: dict) -> ValidationResult:
    """Chapter: declares a non-empty dramatic/chapter function."""

    guard = _not_dict(plan, "chapter_function")
    if guard:
        return guard
    name, _ = _first_present(plan, ("dramatic_function", "function", "chapter_function"))
    if name is None:
        return _fail("chapter_function", reason="no_dramatic_function")
    return _pass("chapter_function", field=name)


def _check_pacing(plan: Any, level: str, constraints: dict, continuity: dict) -> ValidationResult:
    """Chapter: declares pacing / expected emotional-shift information."""

    guard = _not_dict(plan, "pacing")
    if guard:
        return guard
    name, _ = _first_present(plan, ("pacing", "expected_emotional_shift", "emotional_shift"))
    if name is None:
        return _fail("pacing", reason="no_pacing_declared")
    return _pass("pacing", field=name)


def _check_scene_function(plan: Any, level: str, constraints: dict, continuity: dict) -> ValidationResult:
    """Scene: declares a non-empty scene function."""

    guard = _not_dict(plan, "scene_function")
    if guard:
        return guard
    name, _ = _first_present(plan, ("scene_function", "function"))
    if name is None:
        return _fail("scene_function", reason="no_scene_function")
    return _pass("scene_function", field=name)


def _check_entry_exit_state(plan: Any, level: str, constraints: dict, continuity: dict) -> ValidationResult:
    """Scene: carries non-empty entry_state and exit_state."""

    guard = _not_dict(plan, "entry_exit_state")
    if guard:
        return guard
    missing = [k for k in ("entry_state", "exit_state") if _is_empty(plan.get(k))]
    if missing:
        return _fail("entry_exit_state", missing=missing)
    return _pass("entry_exit_state")


def _check_draftability(plan: Any, level: str, constraints: dict, continuity: dict) -> ValidationResult:
    """Beat: carries an immediate objective and physical constraints so it is draftable."""

    guard = _not_dict(plan, "draftability")
    if guard:
        return guard
    obj, _ = _first_present(plan, ("immediate_objective", "objective"))
    phys, _ = _first_present(plan, ("physical_constraints", "physical_blocking"))
    missing = []
    if obj is None:
        missing.append("immediate_objective")
    if phys is None:
        missing.append("physical_constraints")
    if missing:
        return _fail("draftability", missing=missing)
    return _pass("draftability")


def _check_pad_grounding(plan: Any, level: str, constraints: dict, continuity: dict) -> ValidationResult:
    """Beat: carries a PAD target and a grounded behavioural-constraint string."""

    guard = _not_dict(plan, "pad_grounding")
    if guard:
        return guard
    pad, _ = _first_present(plan, ("pad_target", "pad", "pad_state"))
    behaviour, _ = _first_present(
        plan,
        ("behavioral_constraint", "behavioural_constraint", "behavioral_constraint_string", "behavioural_constraint_string"),
    )
    missing = []
    if pad is None:
        missing.append("pad_target")
    if behaviour is None:
        missing.append("behavioral_constraint")
    if missing:
        return _fail("pad_grounding", missing=missing)
    return _pass("pad_grounding")


# Maps the config check-name strings (Configuration_Reference.md §3a) to deterministic
# check callables. Generic-validator aliases are included so a config edit can reference
# any of the eight validators by name as well.
REQUIRED_CHECK_REGISTRY: dict[str, CheckFn] = {
    # generic validators (and aliases)
    "schema": _run_schema,
    "required_fields": _run_required_fields,
    "thread_coverage": _run_thread_coverage,
    "user_annotations": _run_user_annotations,
    "annotation_satisfaction": _run_user_annotations,
    "continuity": _run_continuity,
    "continuity_warnings": _run_continuity,
    "depth": _run_depth,
    "ordering": _run_ordering,
    "no_drafting": _run_no_drafting,
    # level-specific structural checks
    "arc_coverage": _check_arc_coverage,
    "major_promise_payoff": _check_major_promise_payoff,
    "escalation": _check_escalation,
    "thread_distribution": _check_thread_distribution,
    "chapter_function": _check_chapter_function,
    "pacing": _check_pacing,
    "scene_function": _check_scene_function,
    "entry_exit_state": _check_entry_exit_state,
    "draftability": _check_draftability,
    "pad_grounding": _check_pad_grounding,
}


def _required_check_names(config: Any, level: str) -> list[str]:
    """Read the configured required check-names for `level` (fail closed if absent)."""

    if level not in PLANNING_LEVELS:
        raise ValueError(f"unknown planning level: {level!r}")
    planning = getattr(config, "planning", None)
    if planning is None:
        raise ValueError("config has no `planning` section")
    mapping = getattr(planning, "planner_required_checks", None)
    if not isinstance(mapping, Mapping):
        raise ValueError("config.planning.planner_required_checks is missing or not a mapping")
    if level not in mapping:
        raise ValueError(f"no planner_required_checks configured for level {level!r}")
    return list(mapping[level])


def run_validators(
    level: str,
    plan: Any,
    constraints: Mapping | None,
    continuity: Mapping | None,
    config: Any,
) -> ValidationResult:
    """Run exactly the configured required checks for `level` and combine the results.

    Reads `config.planning.planner_required_checks[level]` and runs every named check via
    REQUIRED_CHECK_REGISTRY. This is harness-authoritative — it is the single source of
    truth for whether a finalize is allowed, and it ignores any planner-reported
    `self_check`. A configured check-name with no registered implementation raises a hard
    error (fail closed) before any check runs, so a required check can never be silently
    skipped. `passes` is true only if every required check passes; `failed_checks` names
    every failing config check; `details` carries each check's diagnostics. Pure and
    side-effect free.
    """

    constraints_d = dict(constraints) if isinstance(constraints, Mapping) else {}
    continuity_d = dict(continuity) if isinstance(continuity, Mapping) else {}

    names = _required_check_names(config, level)

    # Fail closed: resolve every name up front; an unknown check-name is a hard error,
    # raised even if other checks would have failed first.
    resolved: list[tuple[str, CheckFn]] = []
    unknown: list[str] = []
    for name in names:
        fn = REQUIRED_CHECK_REGISTRY.get(name)
        if fn is None:
            unknown.append(name)
        else:
            resolved.append((name, fn))
    if unknown:
        raise ValueError(
            f"unknown required check name(s) for level {level!r}: {unknown}; "
            "a configured required check must have a registered implementation (fail closed)"
        )

    failed: list[str] = []
    details: dict[str, Any] = {}
    for name, fn in resolved:
        result = fn(plan, level, constraints_d, continuity_d)
        details[name] = result.details
        if not result.passes:
            failed.append(name)
    return ValidationResult(passes=(len(failed) == 0), failed_checks=failed, details=details)
