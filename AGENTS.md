# AGENTS.md — Standing rules for MuseAI v1

Every session that touches this repository must follow these rules. They are not
suggestions; they define what v1 *is* and how it must be built.

## What MuseAI v1 is

MuseAI v1 is a **complete but deliberately narrow** product: an outline-driven
autonomous fiction drafter. Its scope is exactly:

- Chapter + Beat planning.
- A single continuity critic.
- A draft → audit → revise loop.
- Every LLM agent runs the bounded tool loop with a read-only, story-canon
  tool roster; `web_search` is opt-in via `generation.research_mode`
  (see `docs/agent-tools.md` for the rosters and plumbing).
- SQLite + an append-only event log.
- A live web UI.

That is the whole product. It is **not** the full system.

## The no-stubs / no-implied-features rule

Features not in v1 are **absent, not stubbed**. The following are explicitly out
of v1 and must not appear anywhere — no file, table column, config key, dropdown
option, label, or tooltip:

- Temporal knowledge graph (Graphiti and similar).
- RAPTOR summaries.
- Vector stores (Chroma and similar).
- Stylometric drift gating / style stores.
- Multi-critic panels.
- Scene-level planning / a `Scenes` table.
- Ingestion pipelines.
- Escalation ladders / escalation or drift fields.

Concretely:

- Never write `NotImplementedError`, a no-op passthrough, an empty-return
  placeholder, or any user-facing "not implemented / coming soon / stubbed" text.
- If a feature isn't in v1, it simply doesn't appear: no dead parameter, no
  unused table column, no dropdown option that errors.
- Never *imply* a capability that isn't there. No label, tooltip, or status
  message may claim book-wide continuity, drift analysis, or anything else v1
  does not actually do.

(The one allowed exception is the module-level docstrings on files reserved for a
later build ID in the scaffold — those exist only so the import tree resolves and
carry no logic. Any file worked on in a given prompt must be fully implemented.)

## Config-driven, never hardcoded

- All thresholds, caps, targets, and endpoint details live in `config.yaml`.
- Config is loaded through Pydantic v2 models with `extra='forbid'`: an unknown
  or mistyped key is **fatal at boot**.
- Do not add speculative config keys "for later."

## LLM boundary

- The LLM is reached only through an endpoint/model-agnostic adapter seam.
- No provider, model, or port names appear in logic — those details live in
  `config.yaml` only.

## Persistence

- The event log is **append-only**.
- SQLite writes are **idempotent upserts keyed by id**, so replaying the event
  log is always safe.

## FSM and human review

- The FSM never blocks a thread waiting on a human.
- Interactive review is a **safe-boundary state with explicit resume**, not a
  hidden wait.

## Deliverables and tests

- Deliver **complete files, not diffs**.
- Tests use `pytest`; test files are named `test_*.py`.

## Session ritual

Every session, in order:

1. Read `AGENTS.md`.
2. Read `PROGRESS_LEDGER.md`.
3. Build the one prompt that is next up.
4. Run that prompt's done-check.
5. Update `PROGRESS_LEDGER.md`.


# Graphify
This project has a knowledge graph at graphify-out/ with god nodes,
community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when
  graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for
  relationships and `graphify explain "<concept>"` for focused concepts.
  These return a scoped subgraph, usually much smaller than
  GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation
  instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review
  or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph
  current (AST-only, no API cost).

# Working agreements
- Make the smallest change that satisfies the task; do not refactor
  unrelated code.
- Follow the existing code style in the files you touch.
- Never edit files under graphify-out/ by hand; they are generated.