# PROGRESS_LEDGER.md — MuseAI v1 build ledger

One prompt is built per session. After a prompt passes its done-check, flip its
row to `done` and move the "Next up" line forward. See `AGENTS.md` for the
session ritual.

| Build ID | Scope | Status |
| -------- | ----- | ------ |
| v1.01 | Skeleton, rules, config, logging, stream bus, FSM state, persistence, events, reconcile, runtime, seed loader | done |
| v1.02 | LLM boundary: tokenizer, async adapter client, streaming, retry, tool-calling, prompt rendering, structured output | done |
| v1.03 | FSM nodes: chapter planner, beat planner, deterministic PAD table, context assembly, prose drafter | done |
| v1.04 | (reserved) | pending |
| v1.05 | (reserved) | pending |
| v1.06 | (reserved) | pending |

Next up: v1.04

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
