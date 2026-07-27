# PROGRESS_LEDGER.md — MuseAI v1 build ledger

One prompt is built per session. After a prompt passes its done-check, flip its
row to `done` and move the "Next up" line forward. See `AGENTS.md` for the
session ritual.

| Build ID | Scope | Status |
| -------- | ----- | ------ |
| v1.01 | Skeleton, rules, config, logging, stream bus, FSM state, persistence, events, reconcile, runtime, seed loader | done |
| v1.02 | LLM boundary: tokenizer, async adapter client, streaming, retry, tool-calling, prompt rendering, structured output | done |
| v1.03 | FSM nodes: chapter planner, beat planner, deterministic PAD table, context assembly, prose drafter | done |
| v1.04 | The quality loop: web_search, bounded agent loop, audit, continuity critic, revise, mode_selector | done |
| v1.05 | Commit/router, compiled graph, review boundary, background manager, manuscript export, headless engine checkpoint | done |
| v1.06 | Async Quart web server, SSE dashboard stream, controls, settings, seed routes, guarded reset | done |
| v1.07 | UI/UX overhaul, full-codebase audit, and stabilization fixes | done |
| v1.08 | Repository housekeeping: remove tracked empty placeholder files | done |
| v1.09 | Test-run UI polish and short default seed | done |
| v1.10 | Frontend rebuild: tabbed local-only UI, committed/live separation, seed gating, story rail, Database/Logs/Exports tabs | done |
| v1.11 | View Chat tab: every agent's LLM traffic (prompts, thinking, streamed responses) live + replayed; all-agent streaming; committed-story chapter label fix | done |
| v1.12 | Critic resilience: re-prompt + degrade instead of killing the run; PAD focal-character resolution; crash salvage; test log isolation | done |
| v1.13 | Planner resilience: re-prompt + JSON quote repair; idempotent planners (resume no longer destroys committed prose); ERROR/WARNING log levels | done |
| v1.14 | Web UI polish: compact story rail + instant node tooltips; View Chat keeps in-flight thinking across reloads and auto-scrolls it; richer seed timeline; Database tab remembers its record type | done |
| v1.15 | Manuscript-quality fixes: repetition guard + emotion-tell audit; beat-planner positional context + thread advancement (dead path reconnected); intensity-arc re-prompt; physical-continuity critic line | done |
| v1.16 | Export/project sync (seed rewrites config.yaml; export keyed to the run); span-splice guard; structured tool calls in View Chat; per-beat word target removed; optional global target (0 = unlimited); web_search for every agent; arc/epigraph export format | done |
| v1.17 | Story-canon agent tools: shared registry with per-agent rosters; 15 read-only project-scoped tools; web_search gated behind research_mode; per-tool call caps; structured tool-error objects; tool-call activity events for every agent | done |

Next up: v1 complete

## v1.01 — done

Foundation: repo skeleton, standing rules, strict config, dual logging, SSE
stream bus, the FSM state surface, the full persistence layer, append-only
event log, crash reconciler, runtime bring-up, and the JSON/CLI seed loader.

**Files produced / implemented**

- Project: `pyproject.toml`, `.gitignore`, `config.yaml`, `config.example.yaml`,
  `README.md`, `AGENTS.md`, `PROGRESS_LEDGER.md`
- Core: `museai/core/config.py`, `museai/core/logging_setup.py`,
  `museai/core/stream_bus.py`, `museai/core/events.py`, `museai/core/runtime.py`
- FSM: `museai/fsm/state.py`
- Memory: `museai/memory/db.py`, `museai/memory/reconcile.py`
- Seed: `museai/seed/loader.py`, `seeds/example.json`
- Tests: `tests/conftest.py`, `tests/test_db.py`, `tests/test_events.py`,
  `tests/test_reconcile.py`, `tests/test_seed.py`
- Scaffold placeholders (docstring only, reserved for later build IDs):
  remaining `museai/llm/*`, `museai/fsm/{graph,manager}.py`,
  `museai/fsm/tools/*`, `museai/fsm/nodes/*`, `museai/fsm/routers/*`,
  `museai/web/**`, `run.py`

**Done-check** — `uv run pytest -q` → `14 passed`.
`uv run python -m museai.seed.loader seeds/example.json` → row counts printed
for every table (projects 1, arcs 2, threads 3, characters 2,
character_emotions 2).

## v1.02 — done

The LLM boundary: one async, provider-agnostic inference entry point, the
tokenizer behind it, the four XML prompt templates, and the critic's
structured-output parser. Nothing is wired into FSM nodes yet — that is v1.03.

**Files produced / implemented**

- LLM: `museai/llm/tokenizer.py`, `museai/llm/client.py`,
  `museai/llm/prompts.py`, `museai/llm/structured.py`
- Prompts: `museai/prompts/{chapter_planner,beat_planner,drafter,continuity_critic}.xml.j2`
- Tests: `tests/test_tokenizer.py`, `tests/test_client.py`, `tests/test_prompts.py`

**Design notes**

- `call_llm` takes `transport` and `retry_backoff` keyword arguments purely so
  tests can inject an `httpx.MockTransport` and collapse the 5s/15s sleeps. No
  test opens a socket.
- Retry is transient-only: connect/read timeouts, dropped connections, and 5xx.
  A 4xx raises immediately. A stream that faults *after* emitting a token does
  not retry — replaying it would deliver those tokens to the caller twice.
- `FailureObject` is reused from `museai/fsm/state.py` rather than redefined.
  Because it sets `extra='forbid'`, a hallucinated field is a hard parse failure.
- `parse_failure_objects` performs exactly two bounded passes (strict, then
  fence-stripping + first-balanced-span extraction) and never loops. The
  `retry_cap` is the *caller's* re-prompt budget; it is validated and echoed in
  the raised error.
- Templates render under `StrictUndefined`, so a mistyped context key is fatal,
  matching the config layer's `extra='forbid'` posture.

**Done-check** — `uv run pytest -q` → `115 passed`.

The micro-check prints `3`, and importing `museai.llm.client` / `.tokenizer`
opens no socket (verified with `socket.connect`/`getaddrinfo` patched to raise).

**Live endpoint check: PASSED.** Run against a local OpenAI-compatible server
(`config.yaml` carries the endpoint; `MUSEAI_API_KEY` lives in the gitignored
`.env`). The one-liner printed `ready`. Beyond it, the full boundary was
exercised live:

- Streaming assembled `['one',' two',' three',' four',' five']` into
  `"one two three four five"`, equal to `"".join(chunks)`, `finish_reason=stop`.
- Tool-calling returned `finish_reason=tool_calls` with one parsed call:
  `web_search({"query": "when was the Eiffel Tower completed"})`.
- `continuity_critic.xml.j2` → `call_llm` → `parse_failure_objects` caught both
  planted contradictions (`CONTRADICTS_PRIOR_PROSE` on a noon-sun/midnight
  clash, `CONTRADICTS_CHARACTER` on a lie by a character who "never lies") and
  returned `[]` on a clean draft.
- `chapter_planner.xml.j2` returned a fenced JSON array of 4 chapters with
  contiguous `ordering` starting at 1.

**Wire-format finding (relevant to planner/drafter callers).** Reasoning-style models
stream a non-standard `reasoning` field in the delta alongside `content: ""`,
and those reasoning tokens are billed against `max_tokens`. A too-small
`max_tokens` is therefore consumed before any prose is emitted, yielding empty
text with `finish_reason="length"`. The client deliberately ignores `reasoning`
— surfacing it would be a vendor conditional at the boundary — but keeps every
raw chunk in `LLMResponse.raw` for callers that want it. **Nodes must give
`max_tokens` real headroom over the prose target, or budget nothing at all.**
Regression-tested in `tests/test_client.py::test_nonstandard_delta_fields_are_ignored`.

## v1.03 — done

Four FSM nodes carrying a beat from an arc to finished prose, the deterministic
PAD translation table, and a logging pass over the whole codebase. `plan_chapter`
breaks the pointer's arc into ordered chapters; `plan_beat` breaks the active
chapter — and only that chapter — into ordered beats, each carrying a grounded
PAD behavioural constraint; `assemble_context` gathers and prunes the drafting
context; `draft_prose` streams the beat.

**Files produced / implemented**

- FSM nodes: `museai/fsm/nodes/plan_chapter.py`, `museai/fsm/nodes/plan_beat.py`,
  `museai/fsm/nodes/assemble_context.py`, `museai/fsm/nodes/draft_prose.py`,
  `museai/fsm/nodes/deps.py`
- PAD: `museai/fsm/pad.py`, `museai/fsm/pad_baselines.json` (27 regions)
- Config: **new `generation.context_token_budget` key** in `museai/core/config.py`,
  `config.yaml`, and `config.example.yaml` — the soft token ceiling
  `assemble_context` prunes against. Omitting it is fatal at boot.
- LLM: `museai/llm/structured.py` (added `parse_json_array`),
  `museai/llm/client.py` (request/response/error log records)
- Logging: `museai/core/logging_setup.py`, and log lines added to
  `museai/core/events.py`, `museai/core/runtime.py`,
  `museai/memory/reconcile.py`, `museai/seed/loader.py`
- Tests: `tests/test_planners.py`, `tests/test_context.py`, `tests/test_draft.py`,
  `tests/test_slice_plan_to_draft.py`, `tests/test_client.py` (`TestLogging`),
  `tests/conftest.py` (generation-key overrides)

**Design notes**

- **PAD translation is a pure lookup — no model, no fallback.** Raw coordinates
  never reach an LLM: asked to interpret `pleasure=-0.6, arousal=0.7,
  dominance=-0.5` cold, a model answers differently every time. Each axis
  quantizes into `neg`/`neu`/`pos` at `PAD_BAND_THRESHOLD = 0.33`, forming a
  `"P_A_D"` key into the full 3×3×3 grid of 27 authored behavioural constraints
  in `pad_baselines.json`. The string is injected verbatim. Same coordinate,
  same constraint, always. `plan_beat` therefore makes exactly one LLM call: the
  beat plan itself.
- The table must be complete. `load_pad_baselines` validates all 27 keys at
  first use and `resolve_pad_constraint` raises a `KeyError` naming the missing
  region — a beat with no constraint is not a state the drafter can be in.
- Chapter and beat ids are derived from their parent id and their 1-based
  position (`{arc_id}-c01`, `{chapter_id}-b01`), so re-planning overwrites the
  same rows. The upserts stay idempotent under an event-log replay.
- A bad plan is never a silent empty result. `parse_json_array` raises
  `StructuredOutputError` on unparseable text *and on an empty array* — for a
  planner, `[]` is a failed plan, not an empty one. Both nodes parse before they
  write, so a failed plan leaves no rows behind.
- Neither node sends `max_tokens`, per the v1.02 wire-format finding: a
  reasoning-style endpoint bills hidden reasoning against that budget, and a plan
  truncated mid-array parses to nothing.
- `deps.py` exists because a LangGraph node's signature is fixed at
  `(state) -> dict` and `OrchestratorState` carries no `AppConfig`. Bring-up
  calls `set_node_config` once; unset, it falls back to `load_config()`.
- **`assemble_context` is model-free and reads SQLite only.** Over
  `context_token_budget` it drops context cheapest-first: recent prose oldest
  passage first (the newest is what the beat continues from, so it goes last),
  then open threads by ascending `priority_score`. The beat spec, the
  `pad_constraint`, and the chapter's obligations are never dropped — a draft
  written without them is not shorter, it is wrong. If the protected core alone
  exceeds the budget the node proceeds and logs a WARNING; mutilating the
  instructions to hit a number would be the worse failure.
- `drafter_messages` lives in `assemble_context`, not `draft_prose`, so the
  budget is counted against the exact messages the drafter later sends. One
  renderer means the counted prompt and the transmitted prompt cannot diverge.
- `draft_prose` publishes one `beat_start` before the first token, then a `token`
  event per token. The *log* does not follow suit — `llm_io.log` records the call
  once, on the assembled text. Tokens are for the browser, not the disk.

**Logging**

- `llm_io.log` now carries **one `request` record when a call goes out and one
  `response` record when it returns**, plus an `error` record per failed attempt
  (with `retrying`). A retry logs its own request, so an in-flight or hung call
  is visible. A streamed call still logs exactly one response, on the assembled
  text — never per token.
- `fsm.log` carries a record at every notable event: node start, context
  assembled, chapters/beats planned, phase change, PAD update, event-log append
  (after the fsync, so a line means the event is durable), recovery scan and each
  recovered/cleared intent, seed load, and runtime bring-up.
- **Fixed: `get_logger()` wrote to a void.** The `fsm.log` handler was attached
  to `museai.fsm`, which does not propagate, and no handler existed on the
  `museai` root — so `museai.core.runtime`'s log lines went nowhere. The handler
  now lives on the `museai` root (a single owner per file; two rotating handlers
  on one path race on rollover) and `museai.fsm` reaches it by propagation.
  `museai.llm_io` keeps its own non-propagating handler, so the two logs stay
  separate.

**Done-check** — `uv run pytest -q` → `154 passed`.
The micro-check prints `27 PAD regions`. No test opens a socket: `call_llm` is
replaced in each node's namespace, and the PAD lookup needs no mocking at all.

`tests/test_slice_plan_to_draft.py` is the isolation check that the four nodes
compose. It seeds `seeds/example.json` into a temp DB, runs `plan_chapter` →
`plan_beat` → `assemble_context` → `draft_prose` with a faked endpoint, and
asserts the pointer walked to beat 0 of chapter 1, that `Chapters` and `Beats`
hold contiguous ordered rows with the first of each `active`, that both beats'
`pad_constraint` came from the static table, that `current_draft_text` is
non-empty and equals the streamed tokens joined, and that the phase changes fire
in order: `Planning` → `Drafting` → `Auditing`.

## v1.04 — done

The whole quality loop, built in two passes: first the critic's one tool, the
reusable agent loop, and the deterministic audit gate; then the single agentic
continuity critic, the revision node, and the router that closes the loop —
including the interactive-review branch.

`draft → audit → critics → mode_selector → {commit | revise | review}` now
composes end to end.

**Files produced / implemented**

- Tools: `museai/fsm/tools/web_search.py`, `museai/fsm/tools/loop.py`
- FSM nodes: `museai/fsm/nodes/audit.py`, `museai/fsm/nodes/critics.py`,
  `museai/fsm/nodes/revise.py`
- Router: `museai/fsm/routers/mode_selector.py`
- Prompts: `museai/prompts/reviser.xml.j2` (a fifth template — see below)
- Config: **new `generation.passive_voice_threshold`** (the audit gate) and
  **new top-level `web_search_timeout`** (seconds the search tool waits) in
  `museai/core/config.py`, `config.yaml`, `config.example.yaml`. Omitting
  `passive_voice_threshold` is fatal at boot; a value outside `0..1` is too.
- Tests: `tests/test_web_search.py`, `tests/test_agent_loop.py`,
  `tests/test_audit.py`, `tests/test_critics.py`, `tests/test_revise.py`,
  `tests/test_mode_selector.py`, `tests/test_slice_quality.py`,
  `tests/test_prompts.py` (reviser coverage), `tests/conftest.py` (new key)

**Design notes**

- **`web_search` never raises.** It runs inside a loop a model drives, and a
  tool that throws would abort a draft over a rate limit or a transient DNS
  failure. A blank query, a timeout, a broken upstream scraper, an unreadable
  `config.yaml`, an empty result set — all collapse to `[]` plus one INFO line
  in `fsm.log`. The model reads "no results" and writes around it, which is what
  a critic should do with a fact it cannot confirm. The config read is inside
  the `try` for exactly this reason.
- `ddgs` needs no credential, which is why it is the whole tool surface. There
  is no headless browser, no page fetcher, no API key, and no second tool.
- Rows are normalized to `{"title", "url", "snippet"}`. `ddgs` spells them
  `href`/`body` but some of its engines emit `url`/`snippet`, so both are
  accepted rather than pinning the tool to one scraper's vocabulary. A row with
  no URL is dropped: a citation the critic cannot follow is not evidence.
- **`run_agent_loop` is generic** — endpoint, messages, tool schemas, a
  name→callable registry. It carries no continuity or critic specifics. This is
  the seam future tools plug into.
- The loop is bounded and *always* terminates with a plain answer. After
  `max_iterations` tool-enabled turns that each came back asking for another
  tool, it makes one final `call_llm` with `tools` omitted entirely, leaving the
  endpoint no way to reply but prose. The returned `LLMResponse` therefore never
  carries pending tool calls.
- **A tool fault is data, not a crash.** An unknown tool name, an unparseable
  arguments blob, a bad keyword, or an exception inside a tool all become an
  error *string* handed back as that call's result. Models recover from being
  told a tool failed; they cannot recover from a traceback. Only `call_llm` may
  raise. The caller's `messages` list is never mutated.
- **The audit node computes no drift or stylometric metric** — those are not in
  v1, and a number nobody computes is worse than none. It checks one thing:
  passive-voice density, a proportion of sentences and never a raw count. One
  passive sentence in a long beat is prose; half the beat is a draft that reads
  limp. A breach raises exactly one `PASSIVE_VOICE_DENSITY` `FailureObject` for
  the beat, not one per sentence.
- The passive detector is a regex, not a parser, and **every ambiguity resolves
  toward not flagging.** A be/get form plus a past participle (regular `-ed` by
  shape, irregulars enumerated) with an optional adverb or negation between.
  Common predicate adjectives (`was tired`, `was worried`) are excused by name —
  no regex distinguishes them from `was seized` structurally. A missed passive
  costs one limp sentence; a false one sends the reviser to rewrite prose that
  was already fine. Sentence over-splitting on `Dr.` can only grow the
  denominator, so it too errs toward forgiveness.
- `audit` returns `{'critic_failures': [...]}`. An empty list resets the
  `accumulate_or_reset` reducer, which is correct for the first check against a
  fresh draft: the previous cycle's findings describe prose that no longer
  exists.
- **One critic, run serially.** No dialogue critic, no pacing critic, no craft
  consultant, no panel and no concurrency. `adversarial_critics` is one agentic
  call driving `run_agent_loop` with the one tool.
- **A critic response that will not parse is a hard error.** `StructuredOutputError`
  propagates out of `adversarial_critics`. A critic whose findings are unreadable
  has not found nothing — it has failed — and returning `[]` there would launder
  a broken critic into a clean pass and commit the draft.
- **`adversarial_critics` omits `critic_failures` from its delta when clean**
  rather than returning `[]`. The reducer reads an explicit `[]` as a *reset*, so
  returning it after a clean continuity pass would silently erase the
  passive-voice failure `audit` had just found, and the router would commit a
  draft that failed the audit. Only `revise` is entitled to clear the list.
  Regression-tested in
  `test_critics.py::test_a_clean_critic_does_not_erase_programmatic_failures`.
- `best_seen_draft` scores the draft on **programmatic + critic failures
  together**, and only a strictly lower count replaces it. The first draft always
  wins the slot, since `best_seen_failure_count` starts `None`.
- **`revise_prose` works at the smallest scope that can fix the problem.** Every
  `offending_text` located (exact `str.find`, then a fuzzy sliding-window scan at
  difflib ratio ≥ 0.8) → *span mode*: each span is rewritten alone and spliced
  back, so prose the critic did not fault is never regenerated and cannot drift.
  Any span unlocated, or two spans overlapping → *full mode*, one rewrite of the
  beat against the whole failure list. Splices are applied back-to-front so an
  earlier rewrite cannot shift a later offset.
- The reviser prompt is budgeted in two tiers: full context, then — if the
  rendered messages exceed `context_token_budget` — a collapse to the draft, the
  failures, and the hard constraints (beat spec, `pad_constraint`, chapter
  obligations). Those three are never dropped; a revision written without them
  fixes one problem and creates another.
- **A fifth prompt template, `reviser.xml.j2`.** The build note said to reuse the
  drafter template in a "revision" framing, but `drafter.xml.j2` has no slot for
  a draft, a failure list, or a target span, and adding optional variables would
  have meant guarding them with `is defined` — quietly undoing the
  `StrictUndefined` posture that makes a mistyped context key fatal. A dedicated
  template carries the same context and both revision modes. `tests/test_prompts.py`
  now asserts **five** templates by name, so the no-absent-feature guard still
  bites.
- **`mode_selector` is pure and decides nothing.** First match wins: no failures
  → `commit`; failures under `revision_retry_cap` → `revise`; otherwise
  `review`. It does not discard the draft, accept it, restore `best_seen_draft`,
  or set `review_requested` — the review branch is a safe boundary, and the
  verdict belongs to the human via the web layer. There is no escalation ladder
  and no drift gate: routing turns on the failure count and the retry count and
  nothing else.

**Done-check** — `uv run pytest -q` → `264 passed`.

No test opens a socket: `DDGS` is replaced in the `web_search` namespace,
`call_llm` in the drafter's/reviser's, and `run_agent_loop` in the critic's —
verified by re-running the new files with `socket.connect`/`create_connection`/
`getaddrinfo` patched to raise.

`tests/test_slice_quality.py` is the isolation check that the quality loop
closes. It seeds `seeds/example.json` into a temp DB and walks
seed → plan → draft → audit → critics → revise → `mode_selector` with real nodes
and a faked endpoint, applying each delta through the real reducer. Two runs:
the critic finds one problem, `revise` splices in the repair, the critic returns
clean, `best_seen_failure_count` falls to 0, and the router reaches `commit`;
and the critic keeps finding the same problem until `revision_retry_cap` is
spent, whereupon the router reaches `review` with the draft and
`best_seen_draft` both intact and `review_requested` still `False`.

**Live check: PASSED.**
`uv run python -c "from museai.fsm.tools.web_search import web_search; print(len(web_search('Perseid meteor shower peak', 3)))"`
printed `3` against the real search backend.

## v1.05 — done

The v1 engine now runs headlessly end to end: it plans chapters and beats,
drafts, audits, runs the single continuity critic, revises or parks for review,
commits through SQLite plus the append-only event log, routes from SQLite ground
truth, and exports a single Markdown manuscript from committed prose only.

**Files produced / implemented**

- FSM node: `museai/fsm/nodes/commit.py`
- FSM router: `museai/fsm/routers/commit_router.py`
- FSM graph: `museai/fsm/graph.py`
- FSM manager: `museai/fsm/manager.py`
- Manuscript export: `museai/fsm/export.py`
- Headless CLI: `run.py`
- Recovery hardening: `museai/memory/reconcile.py`
- Tests: `tests/test_commit.py`, `tests/test_commit_router.py`,
  `tests/test_graph.py`, `tests/test_review.py`, `tests/test_slice_headless.py`

**Design notes**

- `commit_transaction` writes a pending `CommitIntent` before commit writes,
  updates the active `Beats` row with committed prose/word count/status, upserts
  the focal character's current PAD from the beat's target PAD, applies only
  explicit forward thread-status updates, marks chapter/arc activity or
  completion, appends a durable `beat_commit` event, and only then flips the
  intent to `committed`.
- The transient graph state reset is explicit: retry count, critic failures,
  best-seen draft/failure count, draft text, streaming buffer, and review request
  are cleared. The project word-count bus event is computed from completed beats
  in SQLite, never from the live token stream.
- `commit_router` is synchronous and queries SQLite for every branch. First match
  wins: next planned beat in the active chapter → `assemble`; next planned
  chapter with no beats in the active arc → `plan_beat`; next planned arc →
  `plan_chapter`; otherwise target/all-complete → `export`. When it advances, it
  writes the new active row and mutates `state['fsm_pointer']` explicitly.
- Recovery now preserves existing beat metadata (`beat_spec`, `pad_constraint`,
  `word_target`, ordering, and chapter) when replaying an appended `beat_commit`
  event, so a crash after the event append cannot erase planning columns.
- `build_graph(config)` compiles a `StateGraph` over `OrchestratorState` with
  the required static edges: `plan_chapter → plan_beat → assemble → draft →
  audit → critics`, plus `revise → audit`.
- Conditional routing after `critics` uses `mode_selector`: clean drafts route to
  `commit`, failures under the cap to `revise`, and exhausted failures to
  `review`.
- Conditional routing after `commit` uses `commit_router`: the next planned beat
  loops to `assemble`, a planned chapter with no beats routes to `plan_beat`, a
  planned arc routes to `plan_chapter`, and export ends the graph.
- The `review` node is a safe boundary. It sets `review_requested=True`,
  publishes `review_needed` with the pointer and `best_seen_draft`, then ends the
  graph run. There is no hidden wait and no busy-loop.
- `GenerationManager` owns lifecycle state (`idle|running|paused|review|stopped|done`),
  starts runs in a background asyncio task, tracks the current
  `OrchestratorState`, publishes run status to the stream bus, and honors pause
  and stop requests at safe boundaries between nodes.
- Review resumption is explicit: `resolve_review("accept")` commits the edited
  text or `best_seen_draft`; `resolve_review("regenerate")` resets retry/failure
  review state and resumes at `assemble` to re-draft the same beat.
- `export_manuscript(config)` writes exactly one Markdown file at
  `data/output/<project_id>.md`, with light chapter headers and completed beat
  prose in narrative order. It never exports live stream text, uncommitted drafts,
  or best-seen review candidates.
- `run.py --headless --seed <path>` loads config, aligns `config.project_id` with
  the seed project id, initializes resources, loads the seed, runs the
  `GenerationManager`, prints the manuscript path on completion, and parks with
  a clear operator message if human review is required. Headless mode does not
  auto-accept review drafts.
- The implementation remains v1-only: no non-v1 stores, snapshots, global
  replanning, timeline/bible/heatmap artifacts, or implied features were added.

**Done-check** — `uv run python -c "from museai.core.config import load_config; from museai.fsm.graph import build_graph; g=build_graph(load_config()); print('graph compiled')"` → `graph compiled`.

Focused tests: `uv run pytest -q tests/test_slice_headless.py tests/test_graph.py tests/test_review.py` → `6 passed`.

Full suite: `uv run pytest -q` → `277 passed`.

Headless live run note: `tests/test_slice_headless.py` proves the complete
headless path with mocked endpoint calls. A real-endpoint
`uv run python run.py --headless --seed seeds/example.json` run is deferred until
the configured endpoint is available and approved for a live generation run.

## v1.06 — done

The live web UI is implemented as a narrow async Quart app plus browser UI:
dashboard, SSE hydration/streaming, generation start/status, manager controls,
review resolution, guarded development reset, settings view/save/test, seed
intake, and vanilla-JS telemetry panes. No routes, pages, panels, or controls for
absent v1 features were added.

**Files produced / implemented**

- Web app factory: `museai/web/app.py`
- Web routes: `museai/web/routes/{dashboard,control,settings,seed}.py`
- Templates: `museai/web/templates/{base,dashboard,settings,seed,error}.html`
- Static UI: `museai/web/static/css/theme.css`, `museai/web/static/js/main.js`
- Web entry point: `run.py` now serves the Quart app by default and keeps
  `--headless --seed` for non-browser runs
- Config: `allow_reset` in `museai/core/config.py`, `config.yaml`, and
  `config.example.yaml`
- Tests: `tests/test_web.py`, plus `tests/conftest.py` fixture coverage for the
  new config key

**Design notes**

- `create_app()` registers only the v1 route surface and configures a clean
  top-level exception handler so requests return JSON or a plain HTML error page,
  never a stack-trace page.
- Startup loads strict config, initializes resources (including recovery), and
  installs a single module-level `GenerationManager` for the process.
- `/stream` is a Quart `Response` over an async generator. It subscribes to the
  core bus, emits a hydration event from `bus.last_snapshot`, streams subsequent
  bus events, and unsubscribes in `finally` when the client disconnects.
- `/generate` refuses to start when no project has been seeded and otherwise
  starts the manager only from idle/done/stopped states.
- `/status` reports manager status, pointer, and committed project word total
  from SQLite via `committed_word_count`; it does not derive totals from live
  token events.
- `/control/reset` is guarded by `allow_reset`; when false it returns 403 with a
  clear JSON message.
- Settings save validates the submitted document through the same Pydantic v2
  config models with `extra='forbid'`, writes `config.yaml`, reloads it, and
  rebuilds runtime handles. Endpoint testing uses the single `call_llm` boundary.
- Seed submission accepts either raw JSON or the form textarea, validates the
  v1 seed shape, then calls `load_seed`.
- `base.html` loads Bootstrap 5.3, `marked.js`, and Chart.js from CDNs and keeps
  project styling in `theme.css` with the required design-token custom
  properties.
- `dashboard.html` uses a two-column layout: sticky status ribbon, manuscript
  stream, real Chart.js PAD radar, critic reasoning/tool panels, controls, and a
  hidden-until-needed review panel.
- `main.js` opens one `EventSource('/stream')`, hydrates from the first snapshot,
  and has handlers for the real bus events only: `run_status`, `phase_change`,
  `chapters_planned`, `beats_planned`, `pad_update`, `beat_start`, `token`,
  `audit`, `critic_tool`, `critic_reasoning`, `critic_summary`, `revision`,
  `word_count`, `pointer_update`, `review_needed`, and `manuscript_ready`. Empty
  panels stay honest until real events arrive.
- Settings and review/control actions use vanilla `fetch` with `AbortController`;
  the UI uses no `localStorage`.

**Done-check** — route micro-check:
`uv run python -c "import asyncio; from museai.web.app import create_app; app=create_app(); print([r.rule for r in app.url_map.iter_rules()])"`
→ `['/static/<path:filename>', '/dashboard', '/', '/stream', '/generate',
'/status', '/control/pause', '/control/resume', '/control/stop',
'/control/review', '/control/reset', '/settings', '/settings/save',
'/settings/test_endpoint', '/seed', '/seed/submit']`.

Focused tests: `uv run pytest -q tests/test_web.py` → `6 passed`.

Full suite: `uv run pytest -q` → `283 passed`.

`uv run python run.py` now serves the browser UI at the configured host/port.

**v1 complete.**

## v1.07 — done

UI/UX overhaul + full-codebase audit and stabilization.

**Front end (rebuilt).** Single "Studio" workspace: one app bar owns navigation
(Studio / Seed / Settings) with the run-state chip; a single command bar holds
Generate / Pause–Resume / Stop (state-aware — only valid actions are shown),
the phase tracker, the beat pointer, and a word-count progress bar. The
manuscript stream is center stage in serif type; the review banner and the
manuscript-done card render as workspace states instead of hidden sidebar
panels. The PAD radar was removed from the UI (engine PAD untouched). Bootstrap
and `marked` are vendored under `static/vendor/` (plus DOMPurify), so the UI
works offline; Chart.js was dropped. The seed page validates JSON live
(mirroring the server rules) with Format / Reset-to-example helpers.

**Bugs fixed in the audit.**
- XSS: LLM/critic output was injected via `innerHTML` unsanitized; markdown now
  renders through `marked` + DOMPurify, labels through `textContent`.
- `settings/save` wrote the *resolved* API key into `config.yaml` in plaintext,
  destroying the `${MUSEAI_API_KEY}` env reference; the raw on-disk reference
  is now preserved when the key field is left blank.
- A rejected seed submit re-rendered with `example_seed=""`, erasing the user's
  edits; the submitted text now survives the error round-trip.
- A crash mid-append left a torn last line in `events.jsonl` and
  `replay_events` raised `JSONDecodeError` during the recovery it exists for;
  malformed lines are now skipped with a warning.
- A node exception (e.g. endpoint down) killed the manager task silently and
  the UI showed "Running" forever; the manager now parks in a new `error`
  status, publishes the failure, and `/generate` allows a retry from it.
- `revision_retry_cap: 0` (legal config, "straight to review") crashed every
  critic pass on `parse_failure_objects`'s `>=1` guard; the call site floors it.
- `plan_beat` silently attributed PAD to the alphabetically-first character
  when the model returned an unknown `focal_character_id`; it now resolves
  names/ids case-insensitively and otherwise attributes to nobody, logged.
- `web_search` (blocking `ddgs` HTTP, up to 10 s) ran on the event loop and
  froze SSE streaming during critic searches; sync tools now run via
  `asyncio.to_thread`.
- Malformed character `pad` values in a seed surfaced as `AttributeError` /
  `IntegrityError` (HTTP 500); the loader validates shape and −1..1 range and
  the web validator checks thread/character required fields (HTTP 400).
- Stream-bus subscriber queues were unbounded; they now cap at 1024 events,
  dropping oldest (reconnecting clients re-hydrate from the snapshot).
- `/status` now includes `word_target`; the SSE response carries
  `Cache-Control: no-cache`; every fetch in `main.js` handles network and
  non-JSON failures instead of throwing unhandled rejections.

**Done-check.** `uv run pytest -q` → `293 passed` (283 prior + 10 new
regression tests covering each fix above).

## v1.08 — done

Repository housekeeping: removed tracked empty placeholder files from the source
tree. The deleted files were package-marker-only `__init__.py` files under
`museai/`; Python 3 namespace packages keep imports working without them.

**Files removed**

- `museai/__init__.py`
- `museai/core/__init__.py`
- `museai/fsm/__init__.py`
- `museai/fsm/nodes/__init__.py`
- `museai/fsm/routers/__init__.py`
- `museai/fsm/tools/__init__.py`
- `museai/llm/__init__.py`
- `museai/memory/__init__.py`
- `museai/prompts/__init__.py`
- `museai/seed/__init__.py`
- `museai/web/__init__.py`
- `museai/web/routes/__init__.py`

**Done-check.** `uv run pytest -q` → `293 passed`.

## v1.09 — done

Test-run UI polish and short default seed.

**Changes made**

- Reworked `seeds/example.json` into a compact one-arc literary mystery seed
  targeting a complete ~1–2k word local test story (`word_count_target: 1500`).
- Aligned `config.yaml`, `config.example.yaml`, and test fixtures around the
  short-run defaults (`word_count_target: 1500`, `beat_word_target: 400`) and
  corrected the example config comment to describe a manuscript target.
- Improved first-run UI copy: the dashboard empty state now points users to the
  short test seed, and the seed page labels the reset action as "Reset to test
  seed" with explicit 1,500-word test-run guidance.
- Improved dashboard status behaviour: `/status` reports no word target before a
  project is loaded, then reports the loaded project's target rather than a
  misleading fallback. The word progress control now has progressbar semantics
  and renders "No project loaded" before seed load.
- Reduced awkward wrapping on narrow screens for the app bar, command bar, run
  metrics, seed toolbar, and review actions; the continuity critic column is
  narrower on desktop and naturally stacks on smaller viewports.
- Added/updated regression tests for the short default seed, seed-page copy,
  dashboard progress semantics, and the pre-seed status response.

**Done-check.** Focused web/seed/slice check:
`uv run pytest -q tests/test_seed.py tests/test_web.py tests/test_slice_headless.py tests/test_slice_plan_to_draft.py tests/test_slice_quality.py`
→ `19 passed`.

Full suite: `uv run pytest -q` → `295 passed`.

## v1.10 — done

Frontend rebuild. The UI became a six-tab local-only interface that separates
committed manuscript from live engine activity, gates generation on a real seed,
and surfaces the story's shape as a graphic rail. The generation engine was not
rewritten; every backend change below is additive.

**The four problems this fixed**

1. The Generate button was always enabled. `/generate` rejected an unseeded
   project server-side, but `/status` carried no seed flag, so the browser could
   not know until the click failed.
2. Committed story and raw model output shared one panel: `main.js` appended
   `token` SSE events straight into the manuscript, so discarded drafts and
   revisions were displayed as though they had committed.
3. Arc/chapter/beat structure existed in SQLite but was never shown.
4. The database, the rotating logs, and the working `export_manuscript()`
   function were unreachable from the browser.

**Backend changes (additive; no existing route, key, or manager behaviour changed)**

- `museai/memory/db.py`: added `get_committed_beats(conn, project_id)` —
  `status='completed' AND prose IS NOT NULL`, in arc → chapter → beat order.
- `museai/web/routes/dashboard.py`: `/status` gained `seed_loaded`,
  `can_generate`, `project`, `counts`, `last_commit`, and `endpoint`. The five
  pre-existing keys are unchanged. Added `GET /committed` (committed prose only)
  and `GET /outline` (rail structure; carries no prose).
- `museai/web/routes/seed.py`: `GET /setup` alias for the Seed & Plan page;
  `_example_seed` became the shared `example_seed_text()`.
- `museai/web/routes/settings.py`: `_settings_view` now also reports the real
  `request_timeout`, `temperature`, `log_level`, `web_search_timeout`, and
  `allow_reset` fields. The API key is still reported as present, never returned.
- New read-only blueprints: `database.py` (`/database`, `/database/records`,
  `/database/event-log`), `logs.py` (`/logs`, `/logs/tail` with credential
  redaction), `exports.py` (`/exports`, `POST /exports/manuscript`,
  `/exports/download`).
- `museai/web/app.py`: registered the three blueprints and routed their JSON
  sub-paths through the JSON error handler.

**Frontend**

- Tabs: Dashboard · Seed & Plan · Database · Settings · Logs · Exports.
- `static/js/ui.js` (new): fetch helpers, Markdown sanitizing, and vanilla
  controllers for tabs, the offcanvas drawer, and toasts. Bootstrap's CSS is
  vendored but its JS bundle is not, so these toggle Bootstrap's own class names.
- `static/js/seed.js` (new): `parseSeedJson`, `formatSeedJson`,
  `validateSeedClientSide`, `summarizeSeed`, `renderSeedTimeline`,
  `submitSeedToBackend`, `applyPremise`. Shared by the Seed & Plan page and the
  Dashboard's Load Seed drawer via the `_seed_workspace.html` macro, so the two
  surfaces cannot drift.
- `static/js/main.js` (rewritten): per-page controllers. Committed Story is
  written *only* from `/committed`. Tokens, revisions, audits, critic messages,
  and the best-seen review draft are confined to Live Activity. Word count comes
  from the backend, never from client-side counting.
- Added a client handler for `pad_update`, which the server has always published
  from `plan_beat.py` and the old client silently dropped. It renders as a text
  row in Live Activity — not a chart.
- Settings has four sections: Endpoint, Generation, Quality, Runtime. No
  Planning section and no critics/antislop/drift toggles were built, because
  those config keys do not exist and `AGENTS.md` forbids implying them.

**Local-only asset contract**

- Added `museai/web/static/vendor/README.md` with version, license, and SHA-256
  for each of the three vendored bundles.
- Added `tests/test_frontend_assets.py`, which fails the build on any `http://`,
  `https://`, protocol-relative URL, `cdnjs`, `jsdelivr`, `unpkg`, or Google
  Fonts reference in a template, stylesheet, or first-party script, and on any
  `@import`/`@font-face`. It passed against the pre-rebuild tree before any other
  file was touched, so it locks in an invariant rather than papering over a fix.
- No new vendored asset was added.

**Files produced**

- New: `museai/web/routes/{database,logs,exports}.py`,
  `museai/web/templates/{_seed_workspace,database,logs,exports}.html`,
  `museai/web/static/js/{ui,seed}.js`,
  `museai/web/static/vendor/README.md`,
  `tests/{test_frontend_assets,test_dashboard_ui,test_setup_seed_ui,test_database_routes,test_logs_exports}.py`
- Modified: `museai/memory/db.py`, `museai/web/app.py`,
  `museai/web/routes/{dashboard,seed,settings}.py`,
  `museai/web/templates/{base,dashboard,seed,settings}.html`,
  `museai/web/static/css/theme.css`, `museai/web/static/js/main.js`,
  `tests/conftest.py`, `tests/test_web.py`, `README.md`

**Test-coupling repaired.** `tests/test_web.py::test_dashboard_renders` asserted
on the string `"short test seed"`, which the rebuilt dashboard no longer prints.
It now asserts the unseeded dashboard shows "No seed loaded" and offers "Load
Seed" — the behaviour the test was actually there to protect.

**Done-check.** Targeted, then full:

```
uv run pytest tests/test_frontend_assets.py -q   → 27 passed
uv run pytest tests/test_dashboard_ui.py -q      → 12 passed
uv run pytest tests/test_setup_seed_ui.py -q     → 11 passed
uv run pytest tests/test_database_routes.py -q   →  9 passed
uv run pytest tests/test_logs_exports.py -q      → 12 passed
uv run pytest tests/test_web.py -q               → 12 passed
uv run pytest -q                                 → 366 passed
```

Baseline before this build was `295 passed`. Zero failures, zero skips.

**Manual smoke test.** Ran `uv run python run.py` against an isolated config and
database. Verified: all eight routes return 200; `/status` reports
`seed_loaded=false, can_generate=false` and `POST /generate` returns 400 before a
seed; loading `seeds/example.json` flips both flags and reports 1 arc / 2 threads
/ 2 characters; `/outline` returns arcs with zero chapters before planning and
never carries prose; inserting one `completed` beat and one `planned` beat with
prose showed the committed beat in `/committed` and `/exports` while the planned
draft appeared in neither; `/database/records?type=Bogus` returns 400 JSON;
`/logs/tail?source=../../etc/passwd` returns 400; a planted `sk-…` token and the
configured API key were both replaced with `[REDACTED]` in `/logs/tail`; a full
settings save round-tripped and left `${MUSEAI_API_KEY}` on disk as a reference
rather than a resolved secret; `/stream` emitted its `hydration` frame; and every
`<script>`/`<link>` on all six pages resolved under `/static/` and returned 200.

**Known limitations**

- The seed schema has no chapters or beats, so the seed **Preview Timeline**
  draws a project node, one node per arc, and thread/character count badges.
  Chapter and beat nodes appear on the Dashboard rail only after the planners
  have run and written them to SQLite.
- The endpoint health dot reads "Configured" from `config.yaml`; it only becomes
  ok/failed after **Test connection** is pressed. There is no background probe.
- The Database tab is strictly read-only. No edits, no deletes.
- Retry/revision count is available only from the `revision` SSE event; it is not
  exposed on `/status`, so a page reload loses it until the next revision.
- `export_manuscript` and `committed_word_count` key off `config.project_id`. A
  seed whose `project.id` differs from the configured `project_id` will load and
  display, but exports will not find it. This predates the rebuild and was left
  as-is rather than change engine semantics.
- Browser console output was not verified programmatically: no headless browser
  is available in this environment. What was verified instead is that all three
  scripts parse under `node --check`, that all 84 `byId(...)` lookups in
  `main.js` resolve against the rendered HTML of the page that owns them, and
  that every referenced asset returns 200 from the local server.

## v1.11 — done

The View Chat tab plus a committed-story rendering fix. Every prompt the engine
sends to the model — planner, drafter, critic, reviser, endpoint test — now
appears in a full-screen chat interface that streams thinking and response
tokens live and replays history from an on-disk transcript.

**The bug fixed first**

The Committed Story panel appeared to be missing a chapter: planners write
chapter `ordering` 1-based (`plan_chapter.py` enumerates from 1), but `main.js`
rendered `Chapter ${ordering + 1}`, labelling seven chapters "Chapter 2" through
"Chapter 8" with no "Chapter 1". Verified against the live database
(`love-thy-doppelganger`: 23 committed beats across 7 chapters, all returned by
`/committed`). The label is now `ordering` as-is. `pointer.beat_index + 1` was
audited at the same time and is correct — `beat_index` really is 0-based
(`commit.py` maps it to a 1-based `ordering`).

**Thinking capture (llm/client.py)**

The client previously discarded reasoning. It now captures both formats an
OpenAI-compatible endpoint may emit, per-call, without faking anything:

- a `reasoning_content` / `reasoning` delta or message field (DeepSeek, vLLM,
  OpenRouter style);
- a literal `<think>…</think>` block opening the content (Ollama serving qwen3
  / r1-distill class models) — handled by a stateful splitter that survives
  tags split across SSE chunk boundaries. Only a block that *opens* the reply
  counts; a `<think>` mid-prose stays prose.

Thinking lands in the new `LLMResponse.thinking` and never reaches `on_token`,
`text`, or token counts, so prose consumers (drafter word counts, JSON parsers)
see cleaner text than before — think blocks no longer leak into drafts.

**Chat instrumentation (llm/client.py → stream bus + transcript)**

`call_llm` gained an `agent` label and publishes per call: `chat_start` (full
untruncated prompt messages), `chat_token` (`kind: thinking|response` deltas
while streaming), and `chat_end` (full thinking + text, token counts, finish
reason, or the error). Start/end records are also appended to
`data/chat.jsonl` (`core/chat_log.py`, configured in `init_resources`,
torn-line-tolerant replay, deleted by reset). Keys travel in HTTP headers, never
in message bodies, so the transcript carries no credentials — verified by test
and by smoke.

**All agents stream now**

`plan_chapter`, `plan_beat`, `revise`, and both calls inside the critic's tool
loop flipped to `stream=True` with agent labels (`chapter_planner`,
`beat_planner`, `reviser`, `critic`); the drafter and the settings
`endpoint_test` call were labelled too. `run_agent_loop` passes its `agent`
through, so each critic turn in the tool loop is its own chat bubble.

**The View Chat page**

`/chat` (nav tab "View Chat") + `/chat/history?limit=` replaying the
transcript. `templates/chat.html`, `static/js/chat.js`, chat styles in
`theme.css`. One bubble per call: agent chip, model, time, live state pill;
prompt collapsed to "Prompt · N messages · ~X tokens" and expandable to the
full messages; a Thinking section that streams italic and auto-collapses when
the answer lands; the response streamed as tokens that each fade in (220ms) and
re-rendered as sanitized Markdown on completion. Filter bar by agent. The SSE
connection opens before history is fetched and buffers events until history
renders, so nothing is lost or duplicated across that window; joining mid-call
shows an honest "Joined mid-call" note instead of a fabricated prompt.

**Files changed**

- Modified: `museai/llm/client.py`, `museai/core/runtime.py`,
  `museai/fsm/nodes/{plan_chapter,plan_beat,draft_prose,revise,critics}.py`,
  `museai/fsm/tools/loop.py`, `museai/web/app.py`,
  `museai/web/routes/settings.py`, `museai/web/templates/base.html`,
  `museai/web/static/js/main.js`, `museai/web/static/css/theme.css`,
  `tests/{test_critics,test_review,test_graph,test_slice_quality,test_slice_headless}.py`
  (fakes updated to accept the new `agent` kwarg).
- Created: `museai/core/chat_log.py`, `museai/web/routes/chat.py`,
  `museai/web/templates/chat.html`, `museai/web/static/js/chat.js`,
  `tests/test_chat.py`.

**Done-check (all actually run)**

- `uv run pytest tests/test_chat.py -q` → `22 passed`.
- `uv run pytest tests/test_client.py -q` → `41 passed` (unchanged behaviour
  for existing callers).
- Full suite: `uv run pytest -q` → `391 passed` (v1.10 baseline was 366).
- Live smoke against a fake OpenAI-compatible endpoint returning a `<think>`
  block: `/chat` renders with filters; `/chat/history` empty → test-endpoint
  call → history holds `chat_start`/`chat_end` with
  `thinking: "pondering the reply carefully"` split from `text: "ok"`;
  `chat_start`/`chat_end` observed on `/stream`; `data/chat.jsonl` created; the
  configured API key appears nowhere in the page, stream, history, or
  transcript.

**Known limitations**

- `data/chat.jsonl` grows without bound on long runs (full prompts are several
  KB per call); reset clears it, nothing rotates it.
- History replay caps at the last 1000 records server-side (200 by default).
- Once any token (thinking included) has streamed, a mid-stream transient
  fault is not retried — previously non-streaming planner/critic/reviser calls
  had all three retry attempts available. The trade is deliberate: replaying a
  partially-streamed call would duplicate tokens in the chat.
- Streaming with tools (the critic loop) requires an endpoint that supports
  `stream: true` alongside `tools`; current Ollama does, very old builds may not.
- Browser console output still cannot be verified headlessly in this
  environment; `node --check` passes on all scripts and every `byId` target in
  `chat.js` resolves against the rendered page.

## v1.12 — done

A live run died and sat dead for 41 minutes. This build makes that failure mode
survivable, and fixes three latent bugs found while diagnosing it.

**The run-killer**

At 2026-07-10 00:39:10 the continuity critic (`openbmb/minicpm5`) answered with
the *schema template itself*: the placeholder value `"The exact sentence or
phrase from the draft."`, `offarming_text` in place of `offending_text`, and no
`critic_source`. `FailureObject` sets `extra="forbid"`, so `parse_failure_objects`
raised `StructuredOutputError`, which propagated through `critics.py` into
`manager.py`'s catch-all and set `status="error"`. Two planned chapters, two
planned beats, and a finished 336-word draft were discarded; beat `b01` was left
stuck at `status='active'`.

`parse_failure_objects(raw, retry_cap=3)` advertised `retry_cap` as "the caller's
re-prompt budget", but nothing ever re-prompted — not the parser, not the node.
The parameter was dead, and its docstring's claim that "re-prompting won't fix
the schema" was exactly wrong: a typo'd key and a missing field are what a
re-prompt fixes.

**What replaces it**

- `parse_failure_objects(raw, *, lenient=False)`. The dead `retry_cap` is gone;
  the caller owns the budget. The raised error now carries the pydantic
  validation detail, because that text is what gets fed back to the model.
- `FailureObject.critic_source` defaults to `"continuity_critic"`. v1 has one
  critic; making the model echo a constant cost a round-trip and bought nothing.
- `critics.py` re-prompts up to `generation.critic_parse_retries`, showing the
  model its own reply and the validation error.
- Retries exhausted → **the run continues**. `critic_parse_failure_streak` (new,
  in `OrchestratorState`) increments and spans beats. At
  `generation.critic_degrade_threshold` the run degrades: element validation
  loosens (`lenient=True`, unknown keys ignored) and a `critic_health` event
  fires. One readable reply resets the streak to 0 and clears the degrade.
- Lenient mode **skips** an element it cannot read rather than defaulting
  `offending_text` to `""` — an empty needle makes `revise.locate()` return
  `None`, silently escalating the beat to a whole-draft rewrite off a finding
  nobody wrote.
- `lenient_used` is only true when findings were actually recovered. Relaxed
  parsing that salvaged nothing salvaged nothing, and the banner does not claim
  otherwise. (A test caught this; the first implementation reported `True`.)
- Web UI: an orange `#critic-warning` banner (new `--warn` token, distinct from
  the amber `--run-pause` — a paused run is chosen, a degraded critic is not),
  plus a one-shot toast on the transition into degraded and a `warnings`-category
  Live Activity row. The banner states plainly that beats are committing with
  only the programmatic audit behind them.

**PAD attribution had never worked**

`beat_planner.xml.j2`'s output template showed the literal placeholder
`"focal_character_id": "character-id"`, so the model answered
`character-id=love-thy-doppelganger-char-1`, `ch-1`, `cl-2`. None resolved, every
beat's PAD was dropped, and `CharacterEmotions` still held its seeded values. The
template now shows a real id from the context, and `resolve_focal_character()`
normalises a `key=` prefix and quotes before matching exact id → casefolded id →
casefolded name → unique embedded id. It returns `""` when two ids could match:
attributing a beat's PAD to the wrong character corrupts that character for the
rest of the book, which is worse than attributing it to nobody.

**Crash salvage**

`db.reset_active_beats()` returns prose-less `active` beats to `planned`.
`manager._salvage_failed_run()` writes the best-seen draft to
`data/drafts/<beat_id>-<utc>.md` and frees the beat, then reports `draft_path` on
the error `run_status`. The draft goes to a *file*, not `Beats.prose`: `/committed`,
`get_committed_beats`, and `export_manuscript` all select on `status='completed'
AND prose IS NOT NULL`, so an unreviewed draft in that column is one status flip
from being shipped as manuscript. Both salvage steps are individually guarded —
a fault inside the error handler must never replace the real exception's diagnosis.

**pytest was writing to the live log files**

`LOG_DIR = Path("logs")` is hardcoded, and `get_logger()` auto-configures at
module import (`runtime.py` builds one at top level), so by the time any fixture
ran the suite had already opened the app's real `logs/fsm.log` and
`logs/llm_io.log`. Test traffic (`example.invalid`, `test-model`, fake reconcile
warnings) was interleaved with the running app's output and shown in the Logs tab.
`tests/conftest.py` now redirects `LOG_DIR` to a temp directory *before* the first
museai import, and drops any handler that beat it there.

**Files changed**

- Modified: `museai/core/config.py`, `config.yaml`, `config.example.yaml`,
  `museai/llm/structured.py`, `museai/fsm/state.py`, `museai/fsm/nodes/critics.py`,
  `museai/fsm/nodes/plan_beat.py`, `museai/fsm/manager.py`, `museai/memory/db.py`,
  `museai/prompts/beat_planner.xml.j2`, `museai/web/templates/dashboard.html`,
  `museai/web/static/js/main.js`, `museai/web/static/css/theme.css`,
  `tests/conftest.py`, `tests/test_critics.py`, `tests/test_prompts.py`,
  `tests/test_planners.py`, `tests/test_frontend_assets.py`, `README.md`
- Created: `tests/test_critic_resilience.py`

**Done-check (all actually run)**

- `uv run pytest tests/test_critic_resilience.py -q` → `22 passed`. The exact
  production reply is fixtured verbatim in `PRODUCTION_FAILURE`.
- `uv run pytest tests/test_prompts.py tests/test_critics.py tests/test_planners.py -q`
  → all pass.
- Full suite: `uv run pytest -q` → **`426 passed`** (v1.11 baseline was 391).
- Log isolation verified by byte-diffing `logs/fsm.log` and `logs/llm_io.log`
  across a full suite run: **0 bytes** added to each.
- Live smoke against a fake OpenAI-compatible endpoint hard-coded to return the
  production payload for every critic call, and the malformed
  `character-id=<id>` for every beat plan:
  - with `critic_degrade_threshold: 3` the run reached `status: done` instead of
    `status: error`, committed 1 beat / 180 words, and published
    `critic_health {streak: 1, degraded: false}`;
  - `CharacterEmotions` moved from the seeded `(0.4, 0.6, -0.2)` to the beat's
    target `(0.1, 0.5, -0.2)` — the malformed focal id resolved;
  - with `critic_degrade_threshold: 1` the same run published
    `critic_health {streak: 1, degraded: true, lenient_used: false}` and still
    reached `done`;
  - zero `unknown focal character` warnings in the smoke window.

**Known limitations**

- Degraded mode commits beats with only the programmatic passive-voice audit
  behind them. The banner says exactly that; the manuscript is genuinely less
  checked.
- `critic_parse_failure_streak` lives in `OrchestratorState`, so it spans beats
  within a run but resets when the process restarts.
- `data/drafts/` is never pruned; nothing rotates it.
- The continuity critic has made **zero tool calls with any real model, ever**
  (129 gemma + 8 minicpm responses across both log generations). This predates
  the v1.11 streaming change and is untouched here — `web_search` is wired,
  tested, and offered to the model, which simply never asks for it. Unexplained.
- The smoke server exercises the real app, so it writes to the real `logs/`.
  Only the test suite is isolated.

---

## v1.13 — done

**What broke.** A generation of `lantern-keeper` died at `2026-07-10 10:43:49`
after 17 minutes, 7 committed beats and 2745 words. The beat planner wrote
dialogue inside a JSON string value without escaping the quotes:

```
"exit_state": "Mara expresses vague regret ("It was meant to be stronger") while ...",
```

`json.loads` stopped at the inner quote (`Expecting ',' delimiter: line 19
column 50`), both of `parse_json_array`'s extraction passes failed,
`StructuredOutputError` reached `manager`'s catch-all, and the run went to
`status="error"`. `finish_reason` was `stop` — nothing was truncated. The full
reply is only recoverable from `data/chat.jsonl`; `llm_io.log` elides it at
`_LOG_CONTENT_PREVIEW_CHARS`.

This is the failure class v1.12 fixed for the critic. The planners never got it:
`plan_chapter.py` and `plan_beat.py` each called `parse_json_array` once, bare.

**The bug underneath.** `manager.start()` always re-enters the graph at
`plan_chapter`, which re-called the model and upserted every chapter with
`status="planned"`; `upsert_chapter` did `status = excluded.status`. The graph
then ran `plan_beat`, which upserted beats without passing `prose`, and
`upsert_beat` did `prose = excluded.prose`. Beat ids are deterministic, so the
rows collided exactly. **Pressing Generate after the crash would have destroyed
the manuscript.** Replaying the real upserts against a copy of `data/museai.db`
erased 979 of 2745 words on chapter 1 alone, and the graph would have walked on
into chapter 2.

**Why it went unnoticed, twice.** `log_node_event` hardcoded `logger.info`, so
`run_failed` — the only line that says a run is over — sat at INFO among
thousands of routine node lines. `fsm.log` contained no ERROR or WARNING line at
any severity, ever.

**What replaced it**

- `repair_json_text` (`llm/structured.py`) escapes unescaped double quotes inside
  single-line JSON string values. Line-oriented, because JSON forbids a raw
  newline in a string: the closing quote is the last quote on the line. A
  character scanner using `,`/`}` lookahead would mangle
  `"he said "yes", then left"`; this does not. Valid JSON is returned
  byte-identical. Reached via `parse_json_array(..., repair=True)`, a third pass
  after the two existing ones.
- `llm/planning.py::call_llm_for_json_array` — the shared planner ladder. Ask,
  parse strictly; on failure re-prompt with the model's own reply and the
  verbatim error, up to `generation.planner_parse_retries` (new required config
  key, default 2); when those are spent, repair. **Repair is last, never first**
  — a re-prompt returns the model's words, the repair returns our reconstruction
  of them, and the difference matters when the text becomes a beat's `intent`.
  Every repair logs `event=json_repaired` at WARNING and publishes
  `planner_repaired` to the stream, which the dashboard renders under
  **Warnings** with an orange toast.
- A planner **cannot degrade** the way the critic can: `parse_json_array` treats
  an empty array as a hard failure, and there is no honest empty plan. Exhausting
  every rung still raises.
- **Idempotent planners.** `plan_chapter` reuses an arc's existing chapters and
  `plan_beat` reuses a chapter's existing beats, skipping the model entirely.
  `beat_spec` already stores the planned dict as JSON, so the reuse path
  reconstitutes it exactly. The active chapter/beat becomes the first one whose
  status is not `completed`, so `plan_beat` returns a pointer at the first
  *unwritten* beat rather than always beat 0. The PAD target is **not**
  republished on the reuse path: it was applied when the beat was first planned,
  and re-applying it would drag the character's emotional state backwards.
- **Defensive floor** in `memory/db.py`: `upsert_beat` now does
  `prose = COALESCE(excluded.prose, Beats.prose)`, keeps `word_count` with the
  text it counts, and never downgrades a `completed` beat; `upsert_chapter` never
  downgrades a `completed` chapter. `commit` still overwrites prose, because it
  passes real prose — so review → Regenerate → recommit still works.
- **Log levels.** `log_node_event(node, *, level=logging.INFO, **fields)`.
  ERROR: `manager: run_failed`, `draft_salvage_failed`, `beat_reset_failed`.
  WARNING: `critics: parse_failed`, `critics: health` *only when degraded*,
  `plan_*: parse_failed` and `json_repaired`, `review: review_needed`.
  Everything else stays INFO — a warning on every healthy beat is a warning
  nobody reads. `run_failed` also gained `after_node`, since `entry_point` says
  where the run began, not where it broke.

**Files**

- Created: `museai/llm/planning.py`, `tests/test_planner_resilience.py`.
- Modified: `museai/llm/structured.py`, `museai/core/config.py`,
  `museai/core/logging_setup.py`, `museai/memory/db.py`, `museai/fsm/manager.py`,
  `museai/fsm/graph.py`, `museai/fsm/nodes/plan_chapter.py`,
  `museai/fsm/nodes/plan_beat.py`, `museai/fsm/nodes/critics.py`,
  `museai/prompts/beat_planner.xml.j2`, `museai/prompts/chapter_planner.xml.j2`,
  `museai/web/static/js/main.js`, `config.yaml`, `config.example.yaml`,
  `tests/conftest.py` (+ `patch_planner_llm`), `tests/test_planners.py`,
  `tests/test_graph.py`, `tests/test_review.py`, `tests/test_slice_quality.py`,
  `tests/test_slice_plan_to_draft.py`, `tests/test_slice_headless.py`,
  `tests/test_manager_error.py`, `tests/test_logs_exports.py`,
  `tests/test_critic_resilience.py`, `README.md`.

**Done-check**

- `uv run pytest tests/test_planner_resilience.py -q` → `20 passed`. The
  production reply is fixtured verbatim in `PRODUCTION_BEAT_PLAN`.
- Full suite: `uv run pytest -q` → **`453 passed`** (v1.12 baseline was 426).
- Log isolation still holds: **0 bytes** added to `logs/fsm.log` and
  `logs/llm_io.log` across a full suite run.
- **Data-loss regression, run against a copy of the real `data/museai.db`:**
  `plan_chapter` then `plan_beat` on an already-drafted chapter, with `call_llm`
  replaced by a function that raises if called. Result: 7 beats / 2745 words
  unchanged, chapter statuses unchanged, no model call, and the pointer resumed
  at `lantern-keeper-arc-1-c03`. Before this change the same replay erased 979
  words.
- **Live smoke** against a stub OpenAI-compatible endpoint whose beat planner
  *always* returns the malformed payload: the run reached `status: done` (it
  previously died), `fsm.log` shows three `parse_failed` WARNINGs (one attempt
  plus two retries) followed by `json_repaired`, one `planner_repaired` event was
  published, and the committed beat's `exit_state` reads
  `Mara expresses vague regret ("It was meant to be stronger") while she works.`
  — the dialogue quotes preserved as content.
- `config.yaml` / `config.example.yaml` generation keys at exact parity;
  `planner_parse_retries: -1` is rejected at boot.
- `node --check museai/web/static/js/main.js` parses.

**Known limitations**

- `repair_json_text` is a heuristic. It fires only after re-prompts are spent and
  always logs a WARNING, but a repaired plan is not literally what the model
  wrote. It handles one field per line; a string value containing a newline is
  not JSON and is not repaired.
- Idempotent planners mean a story can only be re-planned by **Reset**. That is
  the intended trade: the alternative destroyed the manuscript.
- `_first_unfinished` returns the last element when everything is complete. No
  caller reaches that branch (a completed chapter is never selected as active),
  but the fallback is a floor, not a guarantee.
- `critic_parse_failure_streak` still lives in `OrchestratorState` and resets
  when the process restarts.
- `data/drafts/` is never pruned; nothing rotates it.
- The continuity critic has still made **zero tool calls with any real model,
  ever** — `tool_calls=0` on all 40 responses of the `lantern-keeper` run.
  `web_search` is wired, tested, and offered; the model never asks. Untouched
  again, and still unexplained.
- The smoke harness exercises the real app, so it writes to its own `logs/`
  directory under the scratchpad. Only the test suite is isolated from `logs/`.

## v1.14 — done

Four user-reported web UI fixes. No engine changes.

**Dashboard: story progress rail.** The rail was a headline-sized panel whose
information density did not justify it. It is now a slim strip: SVG height
100 → 58, node radii 15/9 → 10/6, gap 110 → 72, and the panel header sits on
one compact line. Hover behaviour is real now: the sluggish native SVG
`<title>` tooltips are replaced by an instant custom tooltip
(`MuseAI.attachNodeTooltips` in `ui.js`, one shared element per document)
showing description, status, committed words, and beats-committed per node,
with a visible hover highlight on the node itself.

**View Chat: thinking tokens survived only in the moment.** Leaving the tab
mid-call and returning rebuilt the page from `data/chat.jsonl`, which only
knows a call once it *ends* — everything already streamed was gone until
`chat_end` arrived. Fix: `client.py` keeps an in-memory registry of in-flight
calls (`_LIVE_CALLS`) accumulating thinking/response text; `GET /chat/history`
returns it as `partials`; `chat.js` renders partials after history and flips
them back from "interrupted" to "streaming". Each `chat_token` now carries a
per-call monotonic `seq`, and the partial records the last seq it includes, so
tokens buffered while history was loading are not applied twice. `chat_end`
still rewrites the full text, so any residual gap self-heals. The thinking box
also auto-scrolls now: pinned to the newest tokens unless the reader scrolled
up (same near-bottom rule as the transcript itself).

**Seed & Plan: preview timeline.** Arc nodes carry the same instant tooltips
(full description, position N of M; the Start node shows genre, word target,
and premise), each arc is captioned with the start of its description, and
below the badges an itemised preview lists every arc, thread (status,
priority), and character (with PAD) the seed declares — strictly seed
contents, no invented chapters or beats, so it is richer than the dashboard
rail by exactly the data it legitimately has.

**Database: record type resets on every visit.** The selected record type now
persists in `localStorage` (guarded — privacy modes without storage just start
fresh) and is restored on load only if it still matches an existing option.

`MuseAI.escapeHtml` now also escapes double quotes: it was already being
interpolated into attribute values (`aria-label`, now `data-tip`), where a
bare `"` in story text — dialogue — would have ended the attribute.

**Done-check**

- `uv run pytest -q` → **457 passed** (baseline 453), zero failures. New tests:
  chat_token seq monotonicity, mid-stream `live_chat_calls` visibility +
  emptiness after end, failed-call registry cleanup, `/chat/history` partials
  (and the exact-shape empty-history test updated for the new key).
- `node --check` parses all four modified JS files.
- Live boot: `/chat/history` serves `partials`; `/`, `/chat`, `/setup`,
  `/database` and all changed static assets serve 200.

**Known limitations**

- No headless browser is available in this environment, so hover tooltips,
  thinking auto-scroll, and the localStorage restore were verified by code
  path and served assets, not by driving a real browser.
- `_LIVE_CALLS` is in-process state: a server restart mid-call loses the
  partial (the transcript never had it), and the call shows as interrupted —
  which is then true.
- The seq guard dedupes only against the history partial; it does not attempt
  general SSE replay protection (EventSource reconnects already re-subscribe
  cleanly through `bus.last_snapshot`).

## v1.15 — done

Five reader-reported manuscript faults from the `lantern-keeper` run, traced to
source and fixed in code. Two were genuine engine bugs; three are model-quality
issues given the strongest deterministic mitigation available. All fixes are
model-agnostic and additive — no graph, routing, or commit changes.

**#1 Verbatim copy-paste (code).** Confirmed in the DB, not the exporter: beat
c03-b04 reproduced 5 whole paragraphs from c03-b03. The drafter was handed prior
committed prose under "continue seamlessly" and a weak model reproduced it; the
audit only checked passive voice, and the critic's only codes were
contradictions (a copy contradicts nothing). Fix: a size-gated paragraph-overlap
check in `audit` (`paragraph_overlaps`, `difflib.SequenceMatcher`) that faults a
drafted paragraph duplicating committed prose (or an earlier draft paragraph) and
feeds it to the existing revise loop as `PARAGRAPH_OVERLAP`. A short line stays
under the size gate (`repetition_min_run`), so a deliberate refrain passes. The
beat planner may declare an `intended_refrain`, stored in `beat_spec`, which the
audit exempts — and every declaration is logged + published (`planner_refrain`)
so a human can veto it. Only the planner can write the allowlist, never the
drafter, so a copy-happy model cannot exempt its own paste. The drafter and
beat-planner prompts also now wrap recent prose with an explicit "continue from —
do not restate" note.

**#2 Pacing loop / no forward motion (code, two parts).** (A) The beat planner
saw only its own one-line chapter — never the arc, its position, sibling
chapters, or what earlier chapters dramatized — so ch4 re-planned the confession
ch3 had committed. It now receives `story_position`, `sibling_chapters`, and
`already_dramatized` (prior beat intents), read straight from the DB. (B) Threads
never advanced: `commit._apply_thread_updates` was fully built but the beat
schema never emitted `thread_updates` and `plan_beat` stripped the field. Both
reconnected — the beat schema asks for `thread_updates`, `plan_beat` whitelists
it into `beat_spec`, and the planner is shown all threads (open · progressing ·
closed, via new `get_threads_for_project`) with closed ones marked resolved.

**#3 Erratic physical movement (model).** No cheap deterministic spatial check
exists. Added an `INCOHERENT_BLOCKING` line to the continuity critic's checklist;
model-dependent, and relieved indirectly by #2 not re-staging the same room.

**#4 Emotional register pinned at max (split).** The planner chose high-arousal
PAD for ~all 16 beats. Added an arc rule to both planner prompts, plus a
post-plan check in `plan_beat`: if more than `intensity_flat_fraction` of beats
exceed `intensity_hot_threshold` arousal, re-prompt once
(`planner_intensity_retries`) for a varied arc, then accept. Emits
`planner_intensity`.

**#5 Tell-after-show (model, partial).** The drafter prompt already forbade it
and the model did it anyway (capability limit). Added an emotion-word-density
check to `audit` (`EMOTION_TELL`, mirrors passive-voice) over a config vocabulary
(`emotion_words`), faulting a beat that names emotions in more than
`emotion_word_threshold` of its sentences.

**Config (new `GenerationConfig` keys, in both config files + conftest):**
`repetition_threshold`, `repetition_min_run`, `repetition_allowlist` (default
`[]`), `emotion_word_threshold`, `emotion_words` (defaulted vocabulary),
`planner_intensity_retries`, `intensity_hot_threshold`, `intensity_flat_fraction`.

**Files:** `museai/fsm/nodes/audit.py`, `museai/fsm/nodes/plan_beat.py`,
`museai/fsm/nodes/assemble_context.py`, `museai/memory/db.py`,
`museai/core/config.py`, `config.yaml`, `config.example.yaml`,
`museai/prompts/{beat_planner,chapter_planner,drafter,continuity_critic}.xml.j2`,
`museai/web/static/js/main.js` (new event handlers), `tests/conftest.py`,
`tests/{test_audit,test_planners,test_db,test_prompts}.py`, `README.md`.

**Done-check**

- `uv run pytest -q` → **479 passed** (baseline 457), zero failures. New tests:
  paragraph-overlap catches the real c03-b03/b04 payload and respects the size
  gate + allowlist; intra-draft repetition; emotion density + whole-word matching;
  `thread_updates` round-trips planner→`beat_spec`→commit→`Threads` (thread-1
  closed); `intended_refrain` stored + announced; intensity re-prompt fires once
  then accepts, and a varied plan is accepted on the first call; planner receives
  positional/sibling/thread context; `get_threads_for_project` ordering.
- Live gemma4 run of `seeds/example.json` against a scratch DB (the live
  `data/museai.db` manuscript untouched): see metrics recorded on completion.

**Known limitations**

- The repetition allowlist is agent-writable by the *planner* (per the user's
  decision). Mitigations: the drafter — which produces the copies — cannot write
  it; the planner declares refrains prospectively; every addition is logged and
  published for human veto. Not a cryptographic guarantee.
- The overlap check is paragraph-level: a 3-sentence copy spliced into the middle
  of an otherwise-new paragraph will not trip it. The observed bug was
  whole-paragraph/multi-paragraph copies, which it catches.
- `PARAGRAPH_OVERLAP` and `EMOTION_TELL` are heuristics; `INCOHERENT_BLOCKING` is
  fully model-dependent. #3, #4-diction, and #5-subtle cases improve materially
  only with a stronger model.
- The intensity re-prompt is bounded to one round, then accepts the plan.
- Threads advance only if the planner emits updates; a planner that never does
  leaves them `open` — no worse than before, and now visible in its context.
- The stronger-model comparison is left for the user to run: point the endpoint
  at a stronger model and rerun the same seed; the fixes need no code change.

## v1.16 — done

Debugging pass driven by the `love-thy-doppelganger` run analysis: the empty
`lantern-keeper.md` export, the span-splice duplication artifacts in the
manuscript, invisible tool calls in View Chat, and a pacing rework.

**Fixes / changes**

- **Export/project sync.** Loading a seed now rewrites `project_id` in
  `config.yaml` (raw-YAML edit, so the `${MUSEAI_API_KEY}` reference survives)
  and reloads the runtime (`museai/web/routes/seed.py`,
  `museai/core/config.py:persist_project_id`). Defense in depth:
  `export_manuscript`/`committed_word_count` take an explicit `project_id`; the
  manager exports the run's own `state["project_id"]`, the exports routes
  resolve the running/seeded project, and the dashboard's silent
  seeded-project fallback now logs a WARNING. Root cause of the 0-word
  `lantern-keeper.md`: export was keyed to a stale `config.project_id`.
- **Span-splice guard.** Span-mode revision rejects a replacement that grew
  implausibly or repeats ≥8 consecutive words of the prose around the span,
  and falls back to a full-beat rewrite (`revise.py:replacement_rejection`,
  `span_rejected` log event). The reviser's span prompt now forbids repeating
  surrounding sentences. This is the bug that left duplicated paragraphs in
  the exported manuscript.
- **Structured tool calls.** `chat_end` records carry `tool_calls` as
  `[{name, arguments}]` plus `tool_call_count` (`llm/client.py`); View Chat
  renders collapsible per-call cards and named, collapsed tool results
  (`chat.js`, `.chat-tool-call` in `theme.css`). Legacy integer records still
  render.
- **Per-beat word target removed.** `beat_word_target` config, the
  `Beats.word_target` column, planner/drafter/reviser prompt references, and
  all plumbing are gone; the drafter chooses length from the story's rhythm
  (recent committed prose stays injected). Old DBs need a reset.
- **Optional global target.** `generation.word_count_target` accepts 0/empty/
  None = no word limit; the commit router treats a falsy project target as
  "outline decides"; the dashboard reports `word_target: null` and the UI
  already shows "No target set."
- **web_search for every agent.** Drafter and reviser now run through
  `run_agent_loop` (with an `on_token` passthrough for the drafter's live
  stream); `call_llm_for_json_array` gained `tools=`/`tool_impls=` so both
  planners run the bounded loop inside the JSON parse ladder; all four prompt
  templates carry a "you have one tool" block. Design for the next tool wave:
  `docs/agent-tools.md`.
- **Export format.** Arc headings, continuous chapter numbering across arcs,
  and planner descriptions demoted to italic epigraphs under `## Chapter N`.

**Done-check**

- `uv run pytest -q` → **495 passed** (baseline 479), zero failures. New tests:
  seed submit rewrites config.yaml and preserves the api-key env ref; sync
  failure surfaces as 400; export follows the seeded project over a stale
  config; two-arc export formatting; splice-guard rejections and fallback;
  null/0 word target semantics; config target normalisation; structured
  chat_end tool calls; loop `on_token` passthrough.
- Live sandboxed headless run (llama3.2:3b via Ollama, scratch DB/logs — the
  real `data/museai.db` untouched): chapters and beats planned through the
  tool-enabled planner path, the drafter made real `web_search` calls
  (`node=draft_prose event=tool_call`), beats drafted with no word target,
  committed, and exported as `# Arc 1` / `## Chapter 1` / epigraph / prose
  under the correct project name with a real word count.

**Known limitations**

- llama3.2:3b's critic often emits tool-call JSON as *text*; the v1.12 parse
  ladder absorbs it (repair/degrade), but critic quality remains
  model-limited.
- The splice guard is deterministic and conservative: a rewrite that repeats
  old-and-new phrasing *within* the span (under 8 shared words with the
  surroundings) still splices; the prompt tightening is the mitigation there.
- The style-echo loop and event-less beat specs (root causes of the
  doppelganger manuscript's repetition/abstraction) are documented as
  follow-ups in `docs/agent-tools.md`, not fixed here.

## v1.17 — done

The agent-tool wave from `docs/agent-tools.md`, built to its implementation
notes: story-canon tools are the default grounding for every agent, and the
web is an explicit research mode.

**Fixes / changes**

- **Registry** (`museai/fsm/tools/registry.py`). One module per tool, each
  exporting an OpenAI spec + impl; `TOOL_SPECS`/`TOOL_IMPLS` plus per-agent
  rosters (`AGENT_TOOLS`, `tool_specs_for`/`tool_impls_for`). All five nodes
  fetch their roster from it; an agent is never handed an impl its spec list
  does not offer.
- **Fifteen read-only, project-scoped tools.** All DB access through
  `project_db.project_connection()` (`PRAGMA query_only`, scoped to
  `config.project_id`): `search_manuscript` (scored FTS, snippets only,
  `scope="committed_only"`), `get_seed_contract`,
  `get_current_pointer_context`, `get_full_outline`, `get_chapter_context`
  (structure, deliberately not prose — the full-text `get_chapter_prose` was
  rejected in design), `get_canonical_state` (beats as metadata, no prose),
  `get_thread_history`, `get_thread_status`,
  `get_character_emotion_history`, `get_character_sheet` (sampled committed
  dialogue), `get_recent_commits` (closing words, not dumps),
  `check_plan_node`, `check_draft` (the audit's exact heuristics as a tool),
  `verify_replacement` (the v1.16 splice guard as a tool), `find_repetition`
  (the "have I written this before?" check). Domain errors are structured
  payloads naming the valid ids/scopes.
- **web_search demoted to research mode.** `generation.research_mode`
  (default false) appends `web_search` to every roster; each prompt's
  web_search bullet renders only under `{% if research_mode %}`, so no prompt
  advertises a tool the roster does not carry.
- **Loop hardening** (`museai/fsm/tools/loop.py`). Tool faults are structured
  objects (`{"error": {"tool", "type", "message"}}` — `unknown_tool`,
  `bad_arguments`, `tool_failure`, `call_cap_exceeded`), and
  `generation.tool_call_cap` bounds how many times any one tool may run per
  loop.
- **Prompts.** All five templates carry a "read-only story tools" block
  matched to their roster; planners still end with "return the fenced JSON
  array and nothing else".
- **Frontend.** `planner_tool` and `reviser_tool` bus events join
  `drafter_tool`/`critic_tool`; the dashboard activity log renders any
  agent's tool call as `Tool call: name(args)` (`main.js:toolCallLabel` — the
  old handler assumed web_search's `query`). View Chat needed no change: the
  v1.16 collapsible cards render every new tool generically.
- **Planner parser: one level of array unwrapping.** The live rerun caught
  qwen2.5:3b returning a beat plan as `[[{...}]]`; `parse_json_array` now
  flattens exactly one unambiguous level (every element a list, everything
  inside an object) instead of failing the run. Deeper or mixed nesting is
  still a shape error.
- Config: `research_mode` + `tool_call_cap` in `config.yaml`,
  `config.example.yaml`, `GenerationConfig` (validated ≥ 1).

**Done-check**

- `uv run pytest -q` → **564 passed** (v1.16 baseline 495). New:
  `tests/test_agent_tools.py` (49 tests — registry/roster integrity, research
  -mode gating, read-only enforcement, cross-project leak checks, and
  behaviour of every tool including error payloads), loop call-cap and
  structured-error tests, per-node roster assertions (drafter, reviser, both
  planners, critic), prompt research-mode rendering tests, and
  nested-array unwrap tests for the planner parser.
- Live sandboxed headless run (qwen2.5:3b via Ollama, scratch config/DB/logs;
  the real `data/museai.db` untouched): the chapter planner made real
  `get_seed_contract`, `get_full_outline`, and `get_thread_history` calls
  (`node=plan_chapter event=tool_call`), the transcript carried structured
  `tool_calls: [{name, arguments}]` plus the named JSON tool results the View
  Chat cards render, two beats drafted (340 and 115 words) and committed, and
  the manual export produced 455 words in the arc/epigraph format under the
  correct project name.
**Known limitations**

- Small local models call the canon tools rarely and sometimes emit tool JSON
  as text; the parse ladders absorb it. The rosters are honest either way —
  the tools are there when the model reaches.
- `get_character_sheet` attributes dialogue at paragraph level (a mention
  heuristic), so its samples are voice reference, not a transcript.
- The style-echo loop and event-less beat specs remain the documented
  follow-ups in `docs/agent-tools.md`; `find_repetition` supplies the check,
  not the cure.

## Post-v1.17 maintenance — planner parser hardening

Recent logs showed repeated run-killing planner failures where a small local
model answered with tool-call-shaped JSON text instead of the required chapter or
beat plan array:

- `PlanningError: chapter 1 has no description` after a fenced
  `{"name": "get_full_outline", "parameters": ...}` reply.
- `PlanningError: beat 1 has no intent` after unwrapped tool-call objects such as
  `get_chapter_context` and `get_character_emotion_history`.

**Fixes / changes**

- `museai/llm/structured.py`: planner parsing now rejects objects that look like
  tool calls (`name` plus object-valued `parameters`) instead of wrapping them as
  one-element plans and letting planner nodes fail later with missing-field
  errors.
- `museai/llm/structured.py`: planner salvage extraction now prefers a balanced
  `[...]` span before falling back to the first balanced object span, so an
  explanatory object before a real array no longer masks the actual plan.
- `museai/llm/structured.py`: multi-object fake tool-call chains are now reported
  as `FakeToolCallTextError` before JSON repair, so the logs distinguish tool
  protocol confusion from ordinary malformed JSON.
- `museai/llm/planning.py`: fake tool-call text logs as
  `event=fake_tool_call_text` and gets a targeted correction that says no tool
  was executed, tells the model to use the real tool-call channel if needed, and
  omits the bad transcript from the retry conversation so the model is not shown
  a long fake-call example to imitate.
- `tests/test_planner_resilience.py`: added production-shaped regressions for a
  fenced chapter-planner tool call, multiple unwrapped beat-planner tool calls,
  the 14:28–14:29 multi-tool fake-call chain, the safer retry prompt, and
  array-preferred salvage.

**Done-check**

- `uv run pytest -q tests/test_planner_resilience.py` → `29 passed`.
- `uv run pytest -q tests/test_planner_resilience.py tests/test_planners.py tests/test_prompts.py tests/test_critic_resilience.py` → `163 passed`.

## Post-v1.17 maintenance — adversarial LLM boundary hardening

The structured-output, tool-call, streaming, and prose boundaries now reject
ambiguous or incomplete model replies instead of silently accepting a usable-
looking fragment.

**Fixes / changes**

- **Strict JSON extraction** (`museai/llm/structured.py`). Ordinary fences,
  surrounding prose, Unicode content, and escaped newlines remain supported.
  Duplicate keys, `NaN`/`Infinity`, bare objects where arrays are required,
  truncated outer arrays, and concatenated/repeated valid arrays fail loudly.
  Multiple valid candidates are an explicit ambiguity error rather than
  "first payload wins". Critic findings require the exact field set, source,
  and supported error-code enum.
- **Planner schemas** (`plan_chapter.py`, `plan_beat.py`, `llm/planning.py`).
  Schema validation runs inside the bounded correction path, before a response
  can be persisted. Chapter fields and ordering are exact; beat core fields,
  PAD axes/types/ranges, optional text/list fields, thread-update shape, and
  thread status enums are validated without string/boolean coercion. Invalid
  extras no longer disappear, and invalid thread updates no longer get dropped.
- **Prose-only boundary** (`draft_prose.py`, `revise.py`). Empty prose,
  markdown fences, leaked reasoning blocks, JSON/tool-call text, and common
  assistant preambles/sign-offs raise a drafting error. Literal prose
  newlines, smart quotes, and non-ASCII story text remain valid.
- **Critic uncertainty routing** (`critics.py`, `mode_selector.py`, `graph.py`).
  An unreadable critic response is not a clean verdict. The graph retries the
  single critic up to its configured degradation threshold, then parks at the
  explicit review boundary when nothing was salvageable. A balanced JSON
  response with `finish_reason="length"` is still treated as truncated.
- **Tool protocol and execution** (`fsm/tools/loop.py`). Malformed envelopes,
  malformed/duplicate-key/non-object arguments, unknown tools, tool faults,
  and per-tool call-cap failures become structured tool results. The new
  positive `generation.tool_timeout` setting bounds each execution and returns
  `tool_timeout` to the model. Tool calls returned from the forced tool-free
  turn are a loud bounded protocol error.
- **Transport and SSE validation** (`llm/client.py`). HTTP 200 bodies must have
  the expected object/choices/message/content/finish/tool-call shapes; server
  error objects and malformed tool calls cannot become empty successes.
  Stream chunks are validated, fragmented tool calls require coherent indices
  and IDs, malformed SSE and clean EOF without a terminal finish reason use
  bounded retries, and all terminal failures close live chat state. Stream
  retries expose `on_restart` so callback consumers can discard partial tokens.
- **Endpoint probe** (`web/routes/settings.py`). The probe opts into empty-
  response retries and succeeds only on the requested `ok` reply.

**Done-check**

- `.venv/win/Scripts/python.exe -m compileall -q museai` → success.
- `.venv/win/Scripts/python.exe -m pytest -q --basetemp .venv/pytest-final2 -o cache_dir=.venv/pytest-cache-final2` → **710 passed** in 33.49s.
- `git diff --check` → clean.

## Post-v1.17 maintenance — chapter obligation placeholder rejection

Chapter planning now rejects generic output-schema placeholders before they can
be persisted and handed to beat planning as story obligations.

**Fixes / changes**

- Added `museai/fsm/plan_validation.py`, a pure shared validator that normalizes
  whitespace and rejects only the observed generic output-format placeholders;
  it deliberately does not make subjective judgments about story-specific prose.
- `museai/fsm/nodes/plan_chapter.py` now requires each chapter to include a
  non-empty obligation array of concrete strings. Placeholder output enters the
  existing bounded planner correction path; exhausted retries write no chapters.
- `museai/fsm/tools/check_plan_node.py` uses the same validator, so the
  planner-visible self-check reports the exact issue before an answer is sent.
- Added regressions for normalization, placeholder rejection, correction and
  persistence of a repaired chapter plan, retry exhaustion without DB writes,
  and the self-check response.

**Done-check**

- `uv run pytest -q tests/test_plan_validation.py tests/test_planners.py tests/test_agent_tools.py` → **114 passed**.
- `uv run pytest -q` → **714 passed, 9 failed**. The failures are pre-existing
  bundled-seed/UI expectation mismatches (`lantern-keeper`/Mara/Tomas/1500
  expected while the current example seed supplies `last-train-signal`/Imani/
  Daniel/no target); none exercise the changed validation boundary.

## Post-v1.17 maintenance — critic latency: futile retries, discarded evidence

`reports/2026-07-25-critic-latency-investigation.md` measured the 2026-07-25 live
run: the critic was 86.9% of LLM wall clock (51.1 of 60.3 min) and only 13 of its
81 calls produced any answer text. This lands the four contained fixes from that
report. The wall-clock ceiling (F-B) was deliberately **not** taken — it adds
human-review parking, and the goal is an unattended run — and reasoning
suppression (F-C, a second `/api/chat` wire adapter) is still deferred.

**Fixes / changes**

- **No attempt repeats a question already answered** (`fsm/nodes/critics.py`).
  A truncation rebuilt `conversation` from two module constants, so the second
  re-ask reproduced the first byte for byte; one live prompt went out six times
  in a row, ~62 s each, returning zero characters every time. Each truncation now
  escalates the ask, and a fingerprint of every *answered* prompt stops a retry
  that would repeat one. Answered, not sent: a transport failure must still be
  retried on the same prompt, which the existing suite caught immediately.
- **Tool results survive a `retry_critic` hop** (`fsm/nodes/critics.py`,
  `fsm/state.py`). New `critic_evidence` state field carries the richest
  tool-bearing conversation to the next pass, keyed by a fingerprint of the draft
  it was gathered against, so a revise invalidates it and a readable verdict
  clears it. One live beat re-searched the same kettle/door prose across four
  passes because each restarted from the bare two-message prompt.
- **Invented tool names strike out** (`fsm/tools/loop.py`). Strikes for a
  nonexistent tool are pooled under one key instead of counted per name: the run
  fabricated four distinct names and never struck out once. The limit stays at 2
  and over-cap strikes stay per name, so a model that hallucinates once still gets
  its chance to answer with the schemas on the table.
- **The chat transcript is read from its tail** (`core/chat_log.py`).
  `replay` parsed the whole (3.32 MB, growing) file on the FSM's own event loop to
  return the last 200 records. Window sized from observed bytes-per-record, since
  a fixed block plus blind doubling was 2.4x *slower* than the full parse at the
  default limit. Byte-identical output verified against the real transcript.
- **The critic prompt's deliberation surface is now measured**
  (`tests/test_critic_prompt_deliberation.py`). The prompt trim itself was not
  taken; this file pins the prompt's shape and prices candidate cuts against the
  real template. Its finding: the suspect sections are only 181 and 326 tokens
  (7.3% / 13.2%), so a trim cannot pay off on prompt size — only on reasoning
  length, which needs a live run to measure.

**Done-check**

- `./.venv/bin/python -m pytest -q` -> **818 passed** in 12.12s (was 797).
- The three F-A regressions verified to fail against the pre-fix code, which
  reproduced the six-identical-attempt sequence from the live log.
- `git diff --check` -> clean.
