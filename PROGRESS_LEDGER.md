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
