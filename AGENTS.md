# Conventions (canonical source for AGENTS.md)

This is the canonical guardrails doc. Increment `01.03` writes its content into the repo as **`AGENTS.md`** (read natively by Cursor *and* by Claude Code via a `CLAUDE.md` pointer), so it auto-loads at the start of every session in either tool. You never paste it. Edit it here, drop it in `_design/`, and re-sync if it changes.

## Which agent am I in?
These prompts are written to run identically in **Claude Code** and **Cursor**. Tool-specific mechanics (restart, rewind) live in each prompt's *Operator notes*, never in the pasted prompt body. The pasted body is always tool-agnostic.

## Environment
- **Linux (Ubuntu primary, all distros) and Windows via WSL2 only.** macOS is not supported. No macOS paths or branches.
- **Python 3.12+.** One dependency manager: `uv`. No Docker, no external servers, no system-level prerequisites.
- `uv sync` installs everything; `uv run app.py` starts the app.

## Context economy (read this — it's why sessions stay cheap)
- Design docs live in the git-ignored `_design/` folder. **Read only the specific file and section a prompt names.** Do not load the whole design corpus — it wastes the context window and dilutes attention.
- Prefer reading a file by path over having it pasted. Reference exact paths and, where useful, exact sections.
- Do not restate large documents back to the user. Act on them.

## Session & restart policy
- **New increment block, new session.** Each prompt's Operator notes say whether to continue the current chat or start a fresh one. Honor it.
- **Hard cap: ~4–5 small increments per session, then restart** — sooner if the tool warns the context window is filling (Claude Code: glance at the % / run `/context`; Cursor: the long-conversation notice).
- **Two failures = stop.** If the agent fails the same step twice, do not stack "that didn't work, try X" corrections — the failed attempt poisons the context. Rewind/revert to before the attempt and re-prompt, or abandon the session and restart from the prompt file.
- A fresh session is cheap here: `AGENTS.md` auto-loads and every prompt re-reads `PROGRESS_LEDGER.md`, so the agent re-orients in two file reads.

## Confirm before editing
- Before changing any file, restate the increment's goal and verify its preconditions in 2–4 bullets. If a precondition is missing, **stop and report** — never guess or build ahead.

## Module identity & file placement
- The repo is organized by technical layer, not one-folder-per-module. The mapping between the 17 conceptual modules and the files lives in **`_design/Module_Map.md`** — it is authoritative. When unsure where something goes, or which module a file belongs to, read it.
- **Every source file's docstring opens with its module tag**, e.g. `Module: M14 (Configuration, Startup & Observability)`, taken from `Module_Map.md`. Cross-cutting files name their primary module and note the role.
- Shared folders carry many modules: `fsm/nodes/` (seven modules — all nodes must share one graph), `fsm/routers/`, `core/` (cross-cutting infra), `prompts/` (M04 loader + template content for several modules). Do not "tidy" a file into a module-named folder — placement is dictated by framework and import-cycle constraints.
- **Module IDs (`M01–M17`) are not build-sequence numbers (`01–19`).** The first is conceptual identity; the second is build order. `Module_Map.md` has the crosswalk.

## Stub-first discipline
- The full directory tree exists from increment one. **Never delete a stub and never break an import.** Unimplemented work stays as a stub with a docstring stating its responsibility (opening with its `Module:` tag), raising `NotImplementedError` or acting as the specified no-op — never silently absent.
- Build only the increment you were given. Do not implement adjacent increments "while you're in there."

## Scope hygiene
- Touch only the files named in the increment's deliverable. If a change seems to need editing something out of scope, **stop and report** rather than expanding silently. No incidental refactors.

## Honesty (hard rule)
- Report status truthfully. A stub is a stub. **Never present a stub or a planned design as built and tested.** Never invent test results, benchmarks, or an experimental history.
- Docstrings describe real current status, not aspiration.

## Docs & comments
- Forward-looking only: no changelog language, no "I added X", no time estimates, no retrospective sections.
- Conceptual descriptions stay at technique-and-tradeoff altitude. Code is concrete (real libraries, real paths).
- In design-level prose, refer to models as **small-/mid-/high-tier** — never model names, sizes, VRAM, or where inference runs. Real endpoint details live only in `config.yaml`.

## Configuration
- Every numeric threshold is a **proposed default** read from `config.yaml` — never hardcoded. If logic needs a constant, it reads the config key.
- Config is validated strictly at boot: unknown or mistyped keys are fatal.

## Testing
- `pytest`. Test files are named `test_*.py` (never `*_test.py`).
- A module is tested in isolation with synthetic input before it is integrated with another module.

## Global policies (apply when their modules arrive)
- Concurrent critic calls are forbidden unless the active endpoint's capability flag allows them; default is strict serial.
- Event-log writes are append-only; cross-store fact writes are idempotent (replay-safe).
- The system never blocks waiting on a human; human input is always optional.

## Session ritual
Read `AGENTS.md` → read `PROGRESS_LEDGER.md` → confirm preconditions → do the one increment → run its done-check → update the ledger.
