# Agent Tools — current state and the next wave

Every LLM call in MuseAI must give the model at least one useful tool: something
that helps it reach its goal with better quality, faster, or with more context.
This document records what is wired today and the designed-but-not-built next
wave.

## Current state

All five agents run through the bounded tool loop
(`museai/fsm/tools/loop.py:run_agent_loop`) with **`web_search`**
(`museai/fsm/tools/web_search.py`):

| Agent | Call path | Tools |
|---|---|---|
| chapter_planner | `call_llm_for_json_array(..., tools=...)` → `run_agent_loop` | web_search |
| beat_planner | same | web_search |
| drafter | `run_agent_loop` (streams tokens via `on_token`) | web_search |
| reviser | `run_agent_loop` | web_search |
| critic | `run_agent_loop` | web_search |

Notes on the plumbing:

- The loop is bounded by `generation.max_agent_iterations`; after that a
  tool-free call forces a plain answer. Tools never raise into the loop — a
  fault becomes an error string the model can read.
- Planners keep their JSON parse/repair ladder: the loop runs *inside* each
  attempt of `museai/llm/planning.py:call_llm_for_json_array`.
- Each prompt template carries a "You have one tool available" block saying
  when to reach for it. web_search is for checkable real-world facts only,
  never invented in-world facts.
- `chat_end` transcript records carry structured `tool_calls`
  (`[{name, arguments}]`) plus `tool_call_count`, and the View Chat tab renders
  them as collapsible cards (`museai/web/static/js/chat.js`).

## Next wave (designed, not built)

Registry layout: one module per tool under `museai/fsm/tools/`, exporting a
spec (OpenAI function schema) and an impl, registered in a shared
`TOOL_IMPLS`-style mapping. All DB tools are read-only and scoped to the active
project.

### Shared

- **`search_manuscript(query, limit=5)`** — full-text search over committed
  `Beats.prose`. The single highest-value tool: today every agent sees only the
  last `recent_prose_beats` beats, which is how a character got referenced
  before ever being introduced. Give it to the drafter, reviser, critic, and
  both planners.

### chapter_planner

- `get_full_outline()` — arcs and chapters with status, so obligations build on
  what is actually planned/dramatized.
- `get_thread_history(thread_id)` — the thread's status plus every committed
  beat that advanced it.

### beat_planner

- `get_character_emotion_history(character_id)` — the PAD trajectory across
  committed beats, so intensity curves are planned against real data instead of
  triggering `intensity_reprompt`.
- `get_chapter_prose(chapter_id)` — the full committed text of one earlier
  chapter, for callbacks and continuity beyond the recent window.

### drafter

- `get_character_sheet(name)` — description plus sampled committed dialogue
  lines, for voice consistency.
- `check_draft(text)` — runs the deterministic audit heuristics
  (`museai/fsm/nodes/audit.py`: passive density, repetition, emotion tells) so
  the drafter can self-correct before entering the critic loop.

### reviser

- `verify_replacement(draft, span, replacement)` — tool form of the splice
  guard in `museai/fsm/nodes/revise.py:replacement_rejection`, so the model can
  check its own span rewrite before answering.
- `check_draft(text)` — as above.

### critic

- `get_thread_status()` — every thread with status and priority, so a
  thread-conflict claim is grounded.
- `search_manuscript` — verify continuity claims against the whole book, not
  the recent window.

## Known architectural follow-ups (out of scope here)

- **Style echo loop**: the recent-prose window makes the drafter imitate its own
  last beats; whole runs converge on one register. Candidate mitigations: a
  cross-beat repetition audit, explicit style-variation guidance keyed to beat
  intent, and `search_manuscript`-powered "have I said this before?" checks.
- **Beat specs are emotional abstractions**: specs carry intent/PAD but rarely
  concrete events, which yields interiority instead of dramatized scenes. A
  planner prompt rework should require an observable event per beat.
