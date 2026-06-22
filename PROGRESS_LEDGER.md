# Progress Ledger

The living record of what is real, what is a stub, and what a session left half-done. The agent updates this in-repo copy at the end of every session, and it is read at the start of every session.

## Current position
- **Last completed:** `01.03`
- **Next up:** `02.01`
- **Current session block:** S2

## Status key
`done` - built, done-check passes | `stub` - placeholder with docstring, no real logic | `half-done` - started, not passing (see notes)

## Ledger
| ID | Module | Produced | Status | Notes / open items | Done-check |
|----|--------|----------|--------|--------------------|------------|
| 01.01 | Scaffold | Full directory tree, `.gitignore`, `.gitkeep` placeholders | done | Existing repo metadata was preserved. | Every directory in the merged tree exists; Git is initialized; `.gitignore` lists `_design/`. |
| 01.02 | Scaffold | `pyproject.toml`, `.env.example`, `uv.lock`, package markers, importable source stubs, no-op `app.py` entry point | done | Native Windows cannot install `falkordblite`; WSL is the intended runtime. User confirmed `app.py` boots in WSL. | Stubs import without error; `uv run app.py` starts and stops cleanly in WSL. |
| 01.03 | Scaffold | Root `AGENTS.md`, `CLAUDE.md`, and `PROGRESS_LEDGER.md` | done | In-repo ledger starts future-session tracking. | `AGENTS.md`, `CLAUDE.md`, and `PROGRESS_LEDGER.md` exist at repo root; Next up reads `02.01`. |
| 02.01 | Config/Startup | _pending_ | - | - | - |
| 02.02 | Config/Startup | _pending_ | - | - | - |
| 02.03 | Config/Startup | _pending_ | - | - | - |
| 02.04 | Config/Startup | _pending_ | - | - | - |
| 02.05 | Config/Startup | _pending_ | - | - | - |
| 02.T1 | Config/Startup (test) | _pending_ | - | - | - |
| 02.T2 | Config/Startup (test) | _pending_ | - | - | - |

*(Append a row per increment. Keep "Current position" accurate - it is read first each session.)*

## Stub inventory (deliberate no-ops, wired later)
- `init_resources()` store inits - no-op until the relational store lands (04).
- Startup crash-sentinel scan - no-op until intent records exist (13).
- `core.antislop.detect_slop()` and `core.antislop.resolve_slop()` - no-op passthroughs until anti-slop implementation.
- `memory.graphiti_client._apply_event()` - intentional early no-op for graph writes during crash-recovery scaffolding.
