# D1–D6 Rebalance — Completion Report

Date: 2026-07-12. Scope: the plot-over-emotion correction plan from the
prompt & context architecture audit. All six defect groups are now addressed
in code. Full test suite: **583 passed** (`python -m pytest tests/ -q`).

## Baseline (already shipped before this pass)

- **D1** — the continuity critic receives `beat` (`critics.py`) and a
  `<beat_goal>` block, making `UNFULFILLED_OBLIGATION` operable.
- **D2** — the drafter prompt leads with `<this_beat_must_deliver>`, PAD is
  demoted to `<manner>`, all 27 `pad_baselines.json` strings rewritten shorter
  and subordinate, and the audit's emotion remediation no longer asks for more
  embodied emotional prose.

## Phase 3 — concrete beat obligations

Every beat spec can now carry, alongside `intent`/`entry_state`/`exit_state`/
`target_pad`/`thread_updates`:

- `required_change` — the state transition the beat must accomplish. Always
  stored non-empty: a plan that omits it falls back to the beat's exit state
  (logged at WARNING), so no beat declares an empty change.
- `observable_event` — the on-page anchor of the change; interior beats stay
  legitimate but need something the reader can see.
- `beat_function` — why the beat exists structurally (reveal, reversal,
  confrontation, decision, discovery, consequence, reaction, setup, payoff…).
- `discharges` — which chapter obligations the beat delivers. Entries are
  matched deterministically (case-insensitive, whitespace-normalized) against
  the chapter's obligations; unmatched entries are dropped. After planning, an
  obligation assigned to no beat raises a WARNING log and a
  `planner_obligation_gap` bus event.

Plumbing: `plan_beat.py` parses/validates and stores the fields;
`check_plan_node` (the planner's self-validation tool) requires non-empty
`required_change` and `observable_event` and well-formed `discharges`;
`assemble_context.py` surfaces all of them (plus `thread_updates`) into the
context package; the **drafter**, **reviser**, and **critic** templates render
them inside `<this_beat_must_deliver>` / `<beat_goal>`, always above
`<manner>`. The reviser's `<focal_character_constraint>` element is gone —
renamed/demoted to `<manner>` with an explicit rule that fixes must preserve
the required change and never add emotional description.

Backward compatible: pre-existing specs read back as empty fields and the
templates render nothing for them.

Note: the word "escalation" is deliberately absent from every template —
`test_prompts.py` forbids it (it guards against promising the removed
escalation-ladder feature). The planner vocabulary uses "raising the stakes".

## Phase 4 — commit gating

`commit_transaction` now withholds story-state advancement the prose did not
earn. The gate trips when:

- the state carries an `UNFULFILLED_OBLIGATION` critic finding against the
  draft being committed, or
- the critic was unreliable for this beat (`critic_parse_failure_streak > 0`
  — its output stayed unreadable after every retry).

When tripped: the prose still commits (a multi-hour run must not die at the
boundary, and mode_selector already prevents the automatic path from reaching
commit with outstanding failures), but planned `thread_updates` are **not**
applied. The durable `beat_commit` event records
`thread_updates_withheld: {reason, requested, unfulfilled_findings}`, a
WARNING is logged, and a `commit_gate` bus event is published. Reconcile
replays stay honest because withheld updates never enter `thread_updates`.

The human review "accept" path (`manager.resolve_review`) now preserves
`UNFULFILLED_OBLIGATION` findings into the commit — accepting the writing does
not launder unearned story state — while clearing every other finding as
before. Committed prose remains only `current_draft_text`; critic reasoning
and discarded drafts never enter the manuscript (unchanged, verified by
existing tests).

How the blocking works end-to-end: in the automatic path an
`UNFULFILLED_OBLIGATION` finding routes to revise (and to review when the
retry budget is spent) via the existing `mode_selector`, so a draft that
failed its mandate cannot reach commit at all; the new gate covers the two
paths that bypass that router — a degraded critic and a human accept.

## Phase 5 — per-agent inference configuration

`config.py` gains `AGENT_ROLES = (chapter_planner, beat_planner, drafter,
reviser, critic)`, an `AgentEndpointOverride` model (every endpoint field
optional, `${VAR}` api_key resolution included), an `agents:` map on
`AppConfig` validated against the known roles, and
`AppConfig.endpoint_for(role)` which merges sparse overrides onto the shared
endpoint. No provider, model, or port names appear anywhere in logic.

All call sites route through it: `plan_chapter` → `chapter_planner`,
`plan_beat` → `beat_planner`, `draft_prose` → `drafter`, `critics` → `critic`,
`revise` → `reviser` (both the token-budget measurement and the call), and
`assemble_context`'s budget counter measures against the drafter's endpoint.
A config with no `agents:` section behaves byte-for-byte as before
(`endpoint_for` returns the shared `endpoint` object itself); the live
`config.yaml` gains a commented example block only.

## Phase 6 — test coverage

New/extended tests (all passing):

- `test_planners.py` — new fields stored in `beat_spec`; unmatched
  `discharges` dropped; `required_change` fallback to exit state;
  `planner_obligation_gap` published for unassigned obligations.
- `test_context.py` — the full mandate travels into the context package.
- `test_prompts.py` — drafter and reviser read `required_change` before
  `<manner>`; reviser system prompt preserves the required change and the old
  `<focal_character_constraint>` element is gone; critic `<beat_goal>` carries
  change/event/discharges and the system prompt names `UNFULFILLED_OBLIGATION`.
- `test_critics.py` — an `UNFULFILLED_OBLIGATION` response parses and lands in
  state.
- `test_commit.py` — an unfulfilled obligation withholds the thread advance
  (prose commits, thread stays, event records why); an unreliable critic
  withholds; unrelated failures do not gate.
- `test_review.py` — end-to-end: accepting a beat the critic flagged as
  unfulfilled commits the prose but leaves the planned thread closure
  unapplied.
- `test_agent_endpoints.py` (new) — fallback identity, sparse merge, unknown
  role fatal at boot, unknown override field fatal, and node-level routing for
  drafter and critic.
- `test_agent_tools.py` — `check_plan_node` requires the new fields; a
  changeless beat is named.
- Recalibrations: two `test_context.py` token budgets (the drafter's protected
  core grew with the mandate block) and the `test_critic_resilience.py` beat
  fixture (now shaped like a real package).

Fixture note: "the-last-postcard" is the live project in `data/museai.db` —
runtime data, not a test fixture — so it was not modified; its existing beat
specs read back unchanged through the backward-compatible field defaults. The
synthetic test fixtures were updated instead to exercise concrete plot
movement (discoveries, discharged obligations, thread closures).

## Remaining risks / follow-ups

- The obligation-coverage check warns and publishes rather than re-prompting
  the planner; if gaps prove common in live runs, a bounded corrective
  re-prompt (mirroring the intensity retry) is the next cheap step.
- `beat_function` is free-form by design (weak models garble enums); it is
  rendered for the drafter but nothing validates the vocabulary.
- Per-agent temperatures/models are plumbed but deliberately unset in
  `config.yaml` — values are to be chosen by testing, per the audit's
  validation strategy. The 0.88 shared temperature on JSON emitters remains a
  suspect for parse failures until tuned.
- End-to-end quality measurement (baseline vs. rebalanced on the live
  project, both model tiers) still requires the live endpoint and human
  rating; nothing in this pass claims prose improvement — it claims the
  system now plans, demands, checks, and gates on plot movement.
