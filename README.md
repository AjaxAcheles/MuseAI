# MuseAI v1

An outline-driven autonomous fiction drafter.

MuseAI v1 takes an outline and drafts a book chapter by chapter. It plans at the
Chapter and Beat level, drafts prose, audits each beat with a single
web-searching continuity critic, and revises in a draft → audit → revise loop.
Everything is persisted to SQLite plus an append-only event log, and progress is
watchable live in a web UI.

This is a **complete but deliberately narrow** product. It is not the full
system — see `AGENTS.md` for exactly what is in scope and what is intentionally
absent.

## Requirements

- Linux / WSL2.
- Python 3.12+.
- [`uv`](https://docs.astral.sh/uv/).
- An OpenAI-compatible LLM endpoint (local or hosted).

No Docker and no external servers are required.

## Setup

```bash
uv sync
cp config.example.yaml config.yaml   # then edit config.yaml
```

Set the endpoint API key in the environment (a `.env` file is loaded
automatically). The default `config.yaml` reads it from `${MUSEAI_API_KEY}`:

```bash
echo 'MUSEAI_API_KEY=your-key-here' >> .env
```

## Configuration

All thresholds, caps, targets, and endpoint details live in `config.yaml`.
Configuration is validated strictly at boot: an unknown or mistyped key is a
fatal error. See `config.example.yaml` for every available key.

## How to run

Start the web UI at the configured `host`/`port`:

```bash
uv run python run.py
```

The interface has seven tabs: **Dashboard · View Chat · Seed & Plan ·
Database · Settings · Logs · Exports**. A typical session:

1. Open **Seed & Plan** (`/setup`). Review or edit the example seed JSON — it
   validates live as you type. Use **Plain Text** to write a premise, **JSON
   Editor** for the full seed, and **Preview Timeline** to see the story's arcs
   before committing to them.
2. Press **Load Seed**. Until this succeeds, **Generate** stays disabled: the
   engine has no project to work on and the button says so.
3. Open the **Dashboard**. The seed pill reads "Seed loaded", the story rail
   renders the arcs, and Generate is now enabled.
4. Press **Generate**.
5. Watch the two panels, which never mix:
   - **Committed Story** shows only prose that passed review and committed. It
     refreshes when the backend reports a commit.
   - **Live Activity** shows what the engine is doing right now — the raw draft
     stream, planner and critic messages, audits, revisions, warnings. Filter it
     by Planner / Drafter / Critics / Commit / Warnings.

   If a weak model keeps answering the continuity critic with unreadable JSON,
   the run does not die: the critic is re-prompted (`critic_parse_retries`), and
   after `critic_degrade_threshold` consecutive unreadable beats an orange banner
   warns that continuity is no longer being checked. Beats keep committing, but
   with only the automated passive-voice audit behind them. Fix the model in
   **Settings** — the banner clears as soon as one reply parses.
6. If generation parks for review, edit the best-seen draft and choose
   **Accept** or **Regenerate**.
7. Open **View Chat** to watch the models themselves: every prompt any agent
   sends — chapter planner, beat planner, drafter, critic, reviser — appears as
   a chat bubble with the outgoing prompt (collapsed, expandable), the model's
   thinking, and the response streaming in token by token. Filter by agent.
   History persists in `data/chat.jsonl` and replays on reload.
8. Use the other tabs as needed:
   - **Database** — read-only view of every SQLite table and the event log.
   - **Logs** — the server's `fsm.log` and `llm_io.log`, with credentials
     stripped, plus the stream events this browser has seen.
   - **Settings** — endpoint, generation, quality, and runtime configuration.
   - **Exports** — write the committed manuscript to
     `data/output/<project_id>.md` and download it.

The run also exports the manuscript automatically when it completes.

Everything the browser loads is served from this machine. There are no CDN
assets, remote fonts, or remote scripts, so the UI works with the Internet
disabled. See `museai/web/static/vendor/README.md`.

For a non-browser run with the same v1 engine:

```bash
uv run python run.py --headless --seed seeds/example.json
```

## Layout

```
museai/
  core/    config, logging, runtime, events, stream bus
  memory/  SQLite persistence and event-log reconciliation
  llm/     endpoint-agnostic adapter, tokenizer, prompt rendering
  fsm/     LangGraph state machine: nodes, routers, tools
  seed/    outline / seed loading
  web/     Quart app, routes, templates, static assets
```

## Development

```bash
uv run pytest
```

Build progress is tracked in `PROGRESS_LEDGER.md`.
