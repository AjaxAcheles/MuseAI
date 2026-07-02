# Running the local end-to-end spike

This is the **minimal** premise → short-story path wired for local testing: it plans
(global → arc → deterministic structure) and then streams drafted prose beat by beat
to a browser. It exercises the real planner/context/draft/commit/persistence code but
**omits** critics, revision, memory backends (Chroma/Graphiti/RAPTOR), and the full
LangGraph conditional graph — those remain to be built.

> Scope note: `fsm/nodes/node_plan_structure.py` is a deterministic scaffold standing
> in for the not-yet-built `node_plan_chapter` / `node_plan_beat` cascade planners.
> `fsm/graph.py::compile_graph` (the full graph) is still a stub; the run is driven by
> the linear orchestrator in `core/generation_manager.py`.

## Prerequisites

1. **Linux or WSL2**, Python 3.12+, and [`uv`](https://astral.sh/uv). (macOS is
   unsupported per `AGENTS.md`.)
2. A running **OpenAI-compatible model endpoint**. `config.yaml` targets Ollama at
   `http://localhost:11434/v1`. Pull a model and point `config.yaml`'s
   `endpoints.*.model_name` at it (all five roles may share one local model).
3. **Per-endpoint API keys** — config load strictly requires a non-empty
   `{ROLE}_API_KEY` for every endpoint, and `.env` is **not** auto-loaded. Export
   them (any placeholder works for a keyless local Ollama):
   ```bash
   export PLANNER_API_KEY=x DRAFTER_API_KEY=x CRITIC_API_KEY=x \
          PAD_TRANSLATOR_API_KEY=x CRAFT_CONSULTANT_API_KEY=x
   ```

## Run

```bash
uv sync
# (export the *_API_KEY vars and start your model server first)
uv run app.py
```

Open <http://127.0.0.1:8888> (override with `MUSEAI_PORT=NNNN` / `MUSEAI_HOST=...`),
enter a premise, set a small target word count
(e.g. 750–2000 for a quick run), and click **Start generation**. Prose streams in
live; status shows beats planned/committed. A short target keeps the round-trip
bounded — see the earlier inference-load estimate for how call volume scales.

## What happens under the hood

`GenerationManager.start()` spawns a background task that runs:

```
plan_global → plan_arc → plan_structure
  → for each planned beat: assemble_context → draft_prose → commit
```

Tokens and lifecycle events are published to the `StreamBus` and pushed to the
browser over SSE (`GET /events/<run_id>`). Committed beats are persisted to
`data/fictionwriter.db` (Beats rows) and appended to `data/events.jsonl`.

- `POST /control/start` — begin a run, returns `{run_id}`
- `GET  /control/status/<run_id>` — poll run status
- `POST /control/reset` — wipe local stores (dev convenience)

## Tests

```bash
uv run pytest tests/test_stream_bus.py tests/test_node_draft_prose.py \
              tests/test_node_plan_structure.py tests/test_node_commit_transaction.py \
              tests/test_generation_manager.py
```

These cover the driver sequencing, the stream bus, the structure scaffold, the
commit/pointer-advance path, and the draft node — all without a model server (the
model boundary is faked via injectable seams).
