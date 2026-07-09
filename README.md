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
