# Agent Tools

Every LLM call in MuseAI gives the model at least one useful tool. The default
tools are **story-canon tools**: agents ground themselves in the loaded seed,
the current pointer, the committed manuscript, the outline, threads, character
state, and canonical database records. The one tool that leaves the project —
`web_search` — is kept but is *not* a default: it is offered only when
`generation.research_mode` is on.

## The registry

One module per tool under `museai/fsm/tools/`, each exporting an OpenAI
function spec (`<NAME>_TOOL_SPEC`) and an implementation.
`museai/fsm/tools/registry.py` registers them all (`TOOL_SPECS` /
`TOOL_IMPLS`) and declares each agent's roster (`AGENT_TOOLS`); nodes fetch
their tools with `tool_specs_for(agent)` / `tool_impls_for(agent)`, so an agent
is never handed an impl its spec list does not offer, and `web_search` is
appended to a roster only in research mode.

Every DB-backed tool goes through `project_db.project_connection()`: a
short-lived connection pinned **read-only** (`PRAGMA query_only`) and scoped to
the active `config.project_id`. Tools return snippets and metadata, never
whole-prose dumps — context flooding feeds the style-echo loop. A bad argument
(unknown thread, character, chapter, scope) comes back as a structured error
payload naming the real options, never an exception.

## Rosters

| Agent | Tools |
|---|---|
| chapter_planner | get_seed_contract, get_full_outline, get_thread_history, get_thread_status, get_canonical_state, check_plan_node |
| beat_planner | get_seed_contract, get_current_pointer_context, get_chapter_context, get_character_emotion_history, get_canonical_state, check_plan_node |
| drafter | get_current_pointer_context, get_recent_commits, search_manuscript, get_character_sheet, find_repetition |
| reviser | check_draft, verify_replacement, search_manuscript, find_repetition |
| critic | search_manuscript, get_full_outline, get_thread_status, get_thread_history, get_canonical_state, get_current_pointer_context, get_recent_commits, find_repetition |

Planners read seed/outline/thread/canonical state and validate their own plan
elements; the drafter reads the committed manuscript and the cast's voices;
the reviser checks its drafts and splices; the critic gets the broadest
read-only continuity surface. `+ web_search` on every roster when
`research_mode: true`.

## The tools

- **`search_manuscript(query, limit=5, scope="committed_only")`** — scored
  full-text search over committed `Beats.prose`, returning beat/chapter ids and
  a snippet per hit. The fix for the last-`recent_prose_beats`-beats blindness
  (the character-referenced-before-introduction class of error). A verbatim
  phrase hit outranks scattered term hits.
- **`get_seed_contract()`** — genre, premise, target, seeded cast, and threads:
  the commitments every plan must honour.
- **`get_current_pointer_context()`** — the active arc, chapter, and beat as
  recorded (`status='active'` rows), with the beat's spec and PAD constraint.
- **`get_full_outline()`** — every arc and chapter with status, obligations,
  and beat progress.
- **`get_chapter_context(chapter_id)`** — one chapter's obligations and its
  beats' intents/exit states/word counts. Deliberately structure, not prose
  (the replacement for a rejected full-text `get_chapter_prose`).
- **`get_canonical_state(scope, ids=None)`** — records for one scope (arcs,
  chapters, beats, threads, characters); beats come back as metadata without
  prose; characters carry their current PAD.
- **`get_thread_history(thread_id)`** / **`get_thread_status()`** — a thread's
  committed advances (read from stored `beat_spec.thread_updates`), and all
  threads with status/priority.
- **`get_character_emotion_history(character_id)`** — current PAD plus the
  `target_pad` of every committed beat focused on the character, in story
  order; plans varied intensity against real data instead of triggering
  `intensity_reprompt`.
- **`get_character_sheet(name)`** — description plus dialogue lines sampled
  from committed paragraphs that mention the character (paragraph-level
  attribution — voice reference, not a transcript).
- **`get_recent_commits(limit=5)`** — the last committed beats' intents, exit
  states, word counts, and closing words. Where the story left off, without a
  prose dump.
- **`check_plan_node(plan_node)`** — deterministic validation of one planned
  chapter or beat: required fields, `target_pad` ranges, focal character
  resolution (the planners' own resolver), thread-update ids and statuses.
- **`check_draft(text)`** — the audit node's exact deterministic checks
  (passive density, paragraph overlap against recent committed prose, emotion
  tells) with the offending sentences quoted.
- **`verify_replacement(draft, span, replacement)`** — the span splice guard
  (`revise.replacement_rejection`) as a tool: rejects a rewrite that ballooned
  or echoes surrounding prose, with the reason, before the node would.
- **`find_repetition(text_or_query, scope="project")`** — near-duplicate
  paragraphs by the audit's similarity machinery, plus verbatim hits for short
  phrases; the "have I written this before?" check for the style-echo problem.
- **`web_search(query, max_results=5)`** *(research mode only)* — bounded ddgs
  search for checkable real-world facts; never for invented in-world facts.

## Plumbing

- The loop (`museai/fsm/tools/loop.py:run_agent_loop`) is bounded by
  `generation.max_agent_iterations`; after that a tool-free call forces a plain
  answer.
- **Structured errors**: a tool fault is never a loose prose string. Unknown
  tool, bad arguments, a raising impl, or a spent call cap all come back as
  `{"error": {"tool", "type", "message"}}`; the DB tools' own domain errors
  (`{"error": ..., "known_threads": [...]}`) name the valid options.
- **Per-tool call caps**: `generation.tool_call_cap` bounds how many times any
  one tool may run within a single loop; past it the model gets a
  `call_cap_exceeded` error and is told to answer with what it has.
- Planners keep their JSON parse/repair ladder: the loop runs *inside* each
  attempt of `museai/llm/planning.py:call_llm_for_json_array` (which forwards
  `on_tool_event` and `tool_call_cap` to the loop).
- Each prompt template carries a "read-only story tools" block; the
  `web_search` bullet renders only when `research_mode` is on, so a prompt
  never advertises a tool the roster does not carry.
- Every tool call is surfaced live: `drafter_tool` / `reviser_tool` /
  `planner_tool` / `critic_tool` bus events land in the dashboard activity log
  (`main.js:toolCallLabel`), and `chat_end` transcript records carry structured
  `tool_calls` (`[{name, arguments}]`) that the View Chat tab renders as
  collapsible cards with the `role:"tool"` results beneath them
  (`museai/web/static/js/chat.js`).

## Known architectural follow-ups (out of scope here)

- **Style echo loop**: the recent-prose window makes the drafter imitate its own
  last beats; whole runs converge on one register. `find_repetition` gives
  agents the check; the remaining work is a cross-beat repetition audit and
  style-variation guidance keyed to beat intent.
- **Beat specs are emotional abstractions**: specs carry intent/PAD but rarely
  concrete events, which yields interiority instead of dramatized action. A
  planner prompt rework should require an observable event per beat.
