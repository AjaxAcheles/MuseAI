# PROGRESS_LEDGER.md — MuseAI v1 build ledger

One prompt is built per session. After a prompt passes its done-check, flip its
row to `done` and move the "Next up" line forward. See `AGENTS.md` for the
session ritual.

| Build ID | Scope | Status |
| -------- | ----- | ------ |
| v1.01 | Skeleton, rules, config, logging, stream bus, FSM state, persistence, events, reconcile, runtime, seed loader | done |
| v1.02 | (reserved) | pending |
| v1.03 | (reserved) | pending |
| v1.04 | (reserved) | pending |
| v1.05 | (reserved) | pending |
| v1.06 | (reserved) | pending |

Next up: v1.02

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
