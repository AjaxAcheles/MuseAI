# Progress Ledger

The living record of what is real, what is a stub, and what a session left half-done. The agent updates this in-repo copy at the end of every session, and it is read at the start of every session.

## Current position
- **Last completed:** `02.04`
- **Next up:** `02.05`
- **Current session block:** S2

## Status key
`done` - built, done-check passes | `stub` - placeholder with docstring, no real logic | `half-done` - started, not passing (see notes)

## Ledger
| ID | Module | Produced | Status | Notes / open items | Done-check |
|----|--------|----------|--------|--------------------|------------|
| 01.01 | Scaffold | Full directory tree, `.gitignore`, `.gitkeep` placeholders | done | Existing repo metadata was preserved. | Every directory in the merged tree exists; Git is initialized; `.gitignore` lists `_design/`. |
| 01.02 | Scaffold | `pyproject.toml`, `.env.example`, `uv.lock`, package markers, importable source stubs, no-op `app.py` entry point | done | Native Windows cannot install `falkordblite`; WSL is the intended runtime. User confirmed `app.py` boots in WSL. | Stubs import without error; `uv run app.py` starts and stops cleanly in WSL. |
| 01.03 | Scaffold | Root `AGENTS.md`, `CLAUDE.md`, and `PROGRESS_LEDGER.md` | done | In-repo ledger starts future-session tracking. | `AGENTS.md`, `CLAUDE.md`, and `PROGRESS_LEDGER.md` exist at repo root; Next up reads `02.01`. |
| 02.01 | Config/Startup | Typed Pydantic v2 config models (`AppConfig`/endpoints/thresholds/runtime/logging) + `load_config()` with env secret overrides | done | Permissive parse; `extra='forbid'` fail-fast deferred to 02.02. `api_key` sourced from `{ENDPOINT}_API_KEY` env (defaults `""`). `base_url` `${...}` placeholders left literal (secrets-only override per scope). Design's `supports_inference_antislop` not in config.yaml, so not modeled. | `load_config('config.yaml')` returns `AppConfig`; nested endpoint/threshold/runtime fields are typed values; clean under `-W error::UserWarning`. |
| 02.02 | Config/Startup | `extra="forbid"` on all six config models — fail-fast on unknown/mistyped keys at parse time | done | Pydantic `ValidationError` reports the offending key's `loc` (e.g. `('bogus_top_level_key',)`, `('endpoints','planner','base_url')`). `RuntimeConfig` keeps `protected_namespaces=()` alongside forbid. | Temp config with a bogus top-level key and a nested `base_url`→`base_uri` typo each raise `ValidationError` locating the key; real `config.yaml` still loads. |
| 02.03 | Config/Startup | Shared node-event logger: `get_logger(node_name)` writes JSON lines to a rotating `logs/fsm.log`; `log_node_event()` emits node, pointer, duration, outcome, and optional error fields | done | Handler attaches once per logger name, `propagate=False`, and level resolves from strict `config.yaml` logging config. Rotation envelope remains the section 3.1 file-size housekeeping default. | Strict config load verified; three events append as three valid JSON lines; a tiny-cap done-check rotates to `fsm.log.1`/`fsm.log.2` while every retained line parses as one JSON object. |
| 02.04 | Config/Startup | Dedicated inference-boundary logger: `get_llm_io_logger()` writes JSON lines to rotating `logs/llm_io.log`; `log_llm_call()` records full request payload, final response text, and duration | done | Separate from the FSM node-event logger and not wired into `call_llm()` yet. Records only the final assembled response, not streaming chunks. | Simulated call wrote exactly one valid JSON line with request, response, and duration; temp check confirmed no `fsm.log` was created. |
| 02.05 | Config/Startup | _pending_ | - | - | - |
| 02.T1 | Config/Startup (test) | _pending_ | - | - | - |
| 02.T2 | Config/Startup (test) | _pending_ | - | - | - |

*(Append a row per increment. Keep "Current position" accurate - it is read first each session.)*

## Stub inventory (deliberate no-ops, wired later)
- `init_resources()` store inits - no-op until the relational store lands (04).
- Startup crash-sentinel scan - no-op until intent records exist (13).
- `core.antislop.detect_slop()` and `core.antislop.resolve_slop()` - no-op passthroughs until anti-slop implementation.
- `memory.graphiti_client._apply_event()` - intentional early no-op for graph writes during crash-recovery scaffolding.
