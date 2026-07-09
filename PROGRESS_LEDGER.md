# PROGRESS_LEDGER.md — MuseAI v1 build ledger

One prompt is built per session. After a prompt passes its done-check, flip its
row to `done` and move the "Next up" line forward. See `AGENTS.md` for the
session ritual.

| Build ID | Scope | Status |
| -------- | ----- | ------ |
| v1.01 | Skeleton, rules, config, logging, stream bus, FSM state, persistence, events, reconcile, runtime, seed loader | done |
| v1.02 | LLM boundary: tokenizer, async adapter client, streaming, retry, tool-calling, prompt rendering, structured output | done |
| v1.03 | (reserved) | pending |
| v1.04 | (reserved) | pending |
| v1.05 | (reserved) | pending |
| v1.06 | (reserved) | pending |

Next up: v1.03

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

**Wire-format finding (relevant to v1.03 callers).** Reasoning-style models
stream a non-standard `reasoning` field in the delta alongside `content: ""`,
and those reasoning tokens are billed against `max_tokens`. A too-small
`max_tokens` is therefore consumed before any prose is emitted, yielding empty
text with `finish_reason="length"`. The client deliberately ignores `reasoning`
— surfacing it would be a vendor conditional at the boundary — but keeps every
raw chunk in `LLMResponse.raw` for callers that want it. **Nodes must give
`max_tokens` real headroom over the prose target, or budget nothing at all.**
Regression-tested in `tests/test_client.py::test_nonstandard_delta_fields_are_ignored`.
