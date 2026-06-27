# MuseAI Streamlit Test Frontends — Operation Guide

## Table of Contents

1. [What is Streamlit? Why is it here?](#what-is-streamlit-why-is-it-here)
2. [Architecture](#architecture)
3. [Launching](#launching)
4. [Per-Module Operation Reference](#per-module-operation-reference)
   - [M02 — Persistent Memory](#m02--persistent-memory)
   - [M03 — Coordinator / State](#m03--coordinator--state)
   - [M04 — Inference Boundary](#m04--inference-boundary)
   - [M05 — Prompt Loader](#m05--prompt-loader)
   - [M06 — Context Assembly & Budgeting](#m06--context-assembly--budgeting)
5. [Debugging Techniques](#debugging-techniques)
6. [Extending a Frontend for a New Module](#extending-a-frontend-for-a-new-module)
7. [Streamlit Frontends vs. pytest — When to Use What](#streamlit-frontends-vs-pytest--when-to-use-what)

---

## What is Streamlit? Why is it here?

**Streamlit** is an open-source Python framework that turns ordinary scripts into interactive web UIs with zero HTML, CSS, or JavaScript. You write normal Python — widgets like `st.button()`, `st.selectbox()`, `st.slider()` render as live controls, and every interaction triggers a top-to-bottom script rerun.

These frontends (`tests/frontends/m02_memory.py`, `m03_state.py`, etc.) are **interactive visual debuggers**, not automated test suites. Each one imports and calls **real production module code** against ephemeral temp workspaces. Their purpose:

- **Exploratory testing** — Tweak parameters, click buttons, inspect real output in milliseconds. No test-writing overhead while you're trying to understand how a function behaves.
- **Visualising intermediate data structures** — Context assembly packages, token distributions, pruning decisions, provisional-claim filtering — all rendered as JSON, dataframes, and charts.
- **Isolation without mocks** — Temp directories (`tempfile.TemporaryDirectory`) scoped to the Streamlit session. Real SQLite, real JSONL, real provisional stores — but against throwaway files.
- **Catching regressions during development** — Manually exercise the exact code path you're changing before you push.
- **Demonstration** — `run_all.py` presents a sidebar with every module visualizer. Onboard a new team member in five minutes.

---

## Architecture

### Layer diagram

```
┌─────────────────────────────────────────────────────┐
│                  streamlit run                       │
│  (script → web UI, every interaction = rerun)       │
├─────────────────────────────────────────────────────┤
│  run_all.py  (sidebar page selector)                │
│  m02_memory.py  ──┐                                 │
│  m03_state.py     ├── import real module code       │
│  m04_inference.py │    no mocks, no stubs           │
│  m05_prompt_loader│                                 │
│  m06_context...   ┘                                 │
├─────────────────────────────────────────────────────┤
│  shared.py  (workspace(), seed_narrative_data(),    │
│              visual_config(), context_defaults())    │
├─────────────────────────────────────────────────────┤
│  Production modules:                                │
│  memory/sqlite_db.py   memory/event_log.py          │
│  memory/provisional_store.py                        │
│  fsm/state.py           fsm/nodes/...              │
│  llm/tokenizer.py       llm/gbnf_compiler.py       │
│  prompts/prompt_loader.py                           │
├─────────────────────────────────────────────────────┤
│  Temp workspace  (/tmp/museai_frontends_XXXXX/)     │
│  ├── fictionwriter.db      (SQLite)                 │
│  ├── events.jsonl          (event log)              │
│  ├── provisional_claims.db (provisional store)      │
│  └── prompts/              (template dir)           │
└─────────────────────────────────────────────────────┘
```

### Workspace isolation

Every frontend calls `shared.workspace()` early. It creates one `tempfile.TemporaryDirectory` per Streamlit session and returns a dict of paths:

```python
paths = workspace()
# → {
#     "root": Path(tmpdir),
#     "db": Path(tmpdir)/"fictionwriter.db",
#     "event_log": Path(tmpdir)/"events.jsonl",
#     "provisional": Path(tmpdir)/"provisional_claims.db",
#     "templates": Path(tmpdir)/"prompts",
# }
```

`shared.reset_workspace()` cleans up the temp directory and calls `st.rerun()`, giving a clean slate without restarting the process.

### Synthetic data seeding

`shared.seed_narrative_data(db_path, provisional_path)` writes a substantial slice-of-life romance through real store APIs — Elena Marchetti revives The Paper Petal bookshop in Willow Creek and falls slowly for the town's quiet carpenter, Marcus Hale:

| Store | Content |
|-------|---------|
| SQLite | 1 arc, 4 chapters, 10 scenes, 6 characters, 5 threads (spanning open/progressing/closed), 12 committed beats (with evolving PAD vectors and thread updates), 6 RaptorNodes (global → arc → chapter → scene) |
| Provisional | 8 claims spread across the confidence bands — high (0.95/0.92), upper-mid (0.78), mid (0.60/0.55/0.45), and low (0.20/0.18) |

Stable IDs (`arc-1` / `chapter-1` / `scene-1`, plus a `scene-1` beat at `beat_index` 2) mean the M02 and M06 forms resolve against the seeded data out of the box. The spread of confidences is designed so filtering by threshold produces predictable subsets — useful for verifying coreference tiering.

### Config plumbing

`shared.context_defaults()` reads `config.yaml` directly to populate slider ranges. `shared.visual_config()` builds a `SimpleNamespace` that mimics the real `app_config` dict shape, so context-assembly code runs without modification.

### Import pattern

Every frontend imports real module functions directly:

```python
# m02_memory.py
from memory import event_log, provisional_store, sqlite_db

# m04_inference.py
from llm.tokenizer import count_tokens, count_message_tokens
from llm.gbnf_compiler import json_schema_to_gbnf
from fsm.state import FailureObject

# m06_context_assembly.py
from fsm.nodes.node_assemble_context import build_context_package
from fsm.state import FSM_Pointer
```

No mocks, no stubs, no monkey-patching. The code running under the UI is the same code that runs in production.

---

## Launching

### Single-module launch

```bash
uv run streamlit run tests/frontends/m02_memory.py
```

Replace `m02_memory` with `m03_state`, `m04_inference`, `m05_prompt_loader`, or `m06_context_assembly`.

### Launcher with sidebar navigation

```bash
uv run streamlit run tests/frontends/run_all.py
```

Opens a browser with a sidebar radio group to switch between all five visualizers.

### First-run

1. The terminal prints a URL (usually `http://localhost:8501`).
2. Open it in a browser. If the browser doesn't open automatically, Ctrl+click the URL.
3. You'll see the UI. Click buttons, adjust sliders, inspect output.

### Hover-help convention

The frontends are laid out as guided runbooks for new engineers:

1. **Top action strip** - seed or reset the temp workspace and confirm which temp directory is active.
2. **Inputs / run controls** - edit parameters on the left or in the first workflow panel.
3. **Primary outputs** - inspect the result returned by the real module function.
4. **Side effects / raw views** - open expanders or tabs for raw tables, JSON layers, logs, or schemas.

Small circular `i` icons appear beside testing areas and output labels. Hover over them for two-section help: a plain-English overview first, then technical detail on the next line. The sections are unlabeled and stay under 500 total words. Native Streamlit input help appears as hover text beside widgets such as text inputs, sliders, selectors, buttons, and submit controls.

### Killing the server

Ctrl+C in the terminal where you ran `streamlit run`.

---

## Per-Module Operation Reference

### M02 — Persistent Memory

**File:** `m02_memory.py`
**Modules exercised:** `memory/sqlite_db.py`, `memory/event_log.py`, `memory/provisional_store.py`

#### Tab 1: SQLite relational hub

| Control | Type | What it does |
|---------|------|-------------|
| **Seed synthetic stores** | Button | Writes deterministic test data via real store APIs. Must be pressed first. |
| **Reset temp workspace** | Button | Tears down the temp directory; next interaction creates a fresh one. |
| Temp path display | Code | Shows the current workspace root path. |

**Commit beat form:**

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `arc_id` | Text input | `arc-1` | Must exist in Arcs table |
| `chapter_id` | Text input | `chapter-1` | Must exist in Chapters table |
| `scene_id` | Text input | `scene-1` | Must exist in Scenes table |
| `beat_id` | Text input | `beat-ui` | Must be unique per scene |
| `beat_index` | Number input | `2` | Ordering within the scene |
| `committed prose` | Text area | synthetic prose | The prose content |
| `pad_states JSON` | Text area | JSON with char-ada PAD | Must be valid JSON |
| `thread_updates JSON` | Text area | JSON with thread-open | Must be valid JSON |
| **Run upsert_beat_commit** | Submit button | — | Calls `sqlite_db.upsert_beat_commit()` |

**Read helpers** — Six JSON displays that call read functions on whatever IDs you type:

| Column | Function |
|--------|----------|
| Left | `get_arc()`, `get_chapters_for_arc()` |
| Centre | `get_scenes_for_chapter_ordered()`, `get_beats_for_scene_ordered()` |
| Right | `get_beat()`, `get_latest_pad_for_scene()` |

**Tables** — Multi-select dropdown. Select any combination of the 9 tables and see them rendered as dataframes. Useful for verifying side effects of write operations.

#### Tab 2: Event log

| Control | Type | What it does |
|---------|------|-------------|
| `event payload JSON` | Text area | Must be valid JSON |
| **Run write_event** | Button | Calls `event_log.write_event()` |
| `tail_events limit` | Slider | 0–20, controls how many recent events to show |
| Event list | JSON | Result of `event_log.tail_events()` |
| **iter_events full stream** | Expander | Calls `event_log.iter_events()` and renders the full list |

#### Tab 3: Provisional claims

| Control | Type | What it does |
|---------|------|-------------|
| **Upsert claim** form | Group | `claim_id`, `claim_text`, `confidence` (slider), `source_ref` |
| **Run upsert_claim** | Submit button | Calls `provisional_store.upsert_claim()` |
| `min_confidence` / `max_confidence` | Sliders | Filter range for `list_claims_by_confidence()` |
| `status filter` | Select | Empty (all), provisional, reviewed, confirmed, rejected |
| Claims dataframe | Widget | Results of the filtered query |
| `claim_id to review` / status / note | Group | Review workflow via `mark_claim_reviewed()` |

**Edge cases to exercise:**

- Upsert a claim with `claim_id=None` (auto-generate)
- Upsert the same `claim_id` twice with different confidence — verify upsert semantics
- Set `min_confidence > max_confidence` — observe the SQL query behaviour
- Review a non-existent `claim_id` — observe the error
- Append an event with malformed JSON — observe the error
- Tail events when the log is empty
- Write a beat commit with empty `pad_states` (`{}`) — verify it doesn't crash

---

### M03 — Coordinator / State

**File:** `m03_state.py`
**Modules exercised:** `fsm/state.py` (`FSM_Pointer`, `FailureObject`, `accumulate_or_reset`, `make_initial_state`, `failure_object_json_schema`)

#### Layout

Two workflow tabs: **Initial state** builds and displays `make_initial_state()`, while **Reducer and schema** demonstrates `accumulate_or_reset()` and exposes the `FailureObject` schema. Each tab uses an input column and an output column.

#### Pointer input column

| Control | Type | What it does |
|---------|------|-------------|
| `project_id` | Text | Passed to `make_initial_state()` |
| `arc_id` / `chapter_id` / `scene_id` | Text | Used to construct `FSM_Pointer` |
| `beat_index` | Number | Integer 0+ |
| `pause_requested override` | Checkbox | Adds to `overrides` dict |
| `has_paradox override` | Checkbox | Adds to `overrides` dict |

#### State output column

Renders the full `make_initial_state()` result as JSON. The `fsm_pointer` is serialised via `.model_dump()` for clean display.

#### Reducer demo

A session-state list of `FailureObject` instances.

| Control | Type | What it does |
|---------|------|-------------|
| `error_code` | Text | e.g. `PACING_ISSUE`, `CONTRADICTION` |
| `offending_text` | Text | The text that triggered the failure |
| `suggested_fix` | Text | The proposed fix |
| `critic_source` | Text | Which critic raised this |
| **Append FailureObject** | Submit | Calls `accumulate_or_reset(failures, [new_failure])` |
| **Pass [] to reducer** | Button | Calls `accumulate_or_reset(failures, [])` — should reset the list |
| Failure list | JSON | Current list of serialised `FailureObject` models |

**FailureObject JSON schema** — Expandable section that calls `failure_object_json_schema()`.

**Edge cases to exercise:**

- Toggle `pause_requested` on a project that isn't in a paused state — verify no crash
- Append multiple failures in sequence, then pass `[]` — verify the list resets
- Set `beat_index` to a very large number — verify no crash from `FSM_Pointer` validation
- Leave all fields at default, observe initial state structure

---

### M04 — Inference Boundary

**File:** `m04_inference.py`
**Modules exercised:** `llm/tokenizer.py`, `llm/gbnf_compiler.py`, `llm/call_llm.py` (synchronous parsing functions only — no live LLM calls)

#### Tab 1: Tokenizer

| Control | Type | What it does |
|---------|------|-------------|
| `Text to count` | Text area | Input text for counting |
| `model_name for exact tokenizers` | Text | Passed to tiktoken and HF tokenizers (default `gpt-4o`) |
| Token counts | Dataframe | Rows for `char_heuristic`, `tiktoken`, `hf_auto` with token count or error |
| `Messages JSON` | Text area | Array of message dicts |
| `message tokenizer family` | Select | One of the three families |
| **Run count_message_tokens** | Button | Calls `count_message_tokens()` and displays a metric |

**Edge cases to exercise:**

- Empty string — all tokenizers should return 0
- Very long text (paste 10K+ chars) — compare `char_heuristic` vs `tiktoken`
- Special characters, emoji, Unicode — observe byte-level differences between tokenizer families
- Unknown model name for `tiktoken` — observe fallback behavior through the configured base encoding
- Messages JSON with invalid structure — observe the error

#### Tab 2: GBNF compiler

| Control | Type | What it does |
|---------|------|-------------|
| Example schema | Select | Five presets: `FailureObject`, `Simple object`, `Array`, `Unsupported pattern` |
| `JSON Schema` | Text area | The schema to compile (editable) |
| **Compile schema** | Button | Calls `json_schema_to_gbnf()` |

**Edge cases to exercise:**

- Select `Unsupported pattern` (contains a `pattern` field with a regex) — observe the compiler's fail-closed error
- Manually edit the schema to remove `additionalProperties: False` — verify the grammar changes
- Paste an empty object `{}` — observe the compilation
- Paste invalid JSON — observe the error

#### Tab 3: Structured output

| Control | Type | What it does |
|---------|------|-------------|
| `Raw model text` | Text area | Input text with JSON, possibly wrapped in markdown fences |
| **Strict parse** | Column | Calls `validate_structured_text()` — will fail on markdown-wrapped input |
| **First JSON object** | Column | Calls `extract_first_json_object()` — unwraps from fences |
| **Salvage parse** | Column | Calls `validate_with_salvage()` — tries extraction then parse |

**Edge cases to exercise:**

- Paste valid JSON without markdown fences — strict parse should work
- Paste JSON inside ` ```json ... ``` ` — strict parse should fail, salvage should succeed
- Paste plain text with no JSON anywhere — all three should fail gracefully
- Paste JSON with an extra field not in the schema — strict and salvage validation both reject the extra key
- Paste unbalanced braces — observe `extract_first_json_object` return None

---

### M05 — Prompt Loader

**File:** `m05_prompt_loader.py`
**Modules exercised:** `prompts/prompt_loader.py` (`PromptLoader`, `InvalidNodeNameError`)

#### Tab 1: Create template

| Control | Type | What it does |
|---------|------|-------------|
| `node_name` | Text | e.g. `node_draft_prose` — validated by `template_name_for_node()` |
| `template body` | Text area | Jinja2 template content with `{{ variables }}` |
| **Validate name and write template** | Button | Writes the template as `{node_name}.xml.j2` to the workspace |
| Template directory | JSON | Lists all `.xml.j2` files in the workspace |

#### Tab 2: Render template

| Control | Type | What it does |
|---------|------|-------------|
| `node_name to render` | Text | The node name |
| `context JSON` | Text area | Template variables as JSON dict |
| **Run PromptLoader.render** | Button | Calls `loader.render()` — uses `StrictUndefined` |

**Edge cases to exercise:**

- Render with all required variables provided — should succeed
- Omit a required variable from context — `StrictUndefined` raises `UndefinedError`, which is shown as an error
- Pass a variable that isn't in the template — Jinja2 ignores it (no error)
- Write a template with invalid Jinja2 syntax — observe the compilation error
- Write the template, delete it from the workspace, then try to render — observe the file-not-found error

#### Tab 3: Name guard

| Control | Type | What it does |
|---------|------|-------------|
| `candidate names` | Text area | One name per line — includes valid names and attack vectors |
| Results | Dataframe | Each name, whether it was accepted, the resolved template name, and any error |

**Names tested by default:**
- `node_draft_prose` — should be accepted
- `../secrets` — path traversal, must be rejected
- `node_draft_prose.xml.j2` — already has extension, should be rejected (or handled)
- `Node_Draft` — invalid casing, must be rejected
- `node_x/../../outside` — path traversal, must be rejected

**Edge cases to exercise:**

- Add names with spaces, special characters, empty strings
- Add `node_` prefix-only names without a valid suffix
- Add extremely long names

---

### M06 — Context Assembly & Budgeting

**File:** `m06_context_assembly.py`
**Modules exercised:** `fsm/nodes/node_assemble_context.py` (`build_context_package`), `fsm/state.py` (`FSM_Pointer`)

#### Controls

| Control | Type | What it does |
|---------|------|-------------|
| **Seed synthetic memory stores** | Button | Writes deterministic data (same as M02 seed) |
| **Reset temp workspace** | Button | Fresh workspace |
| `context.token_budget` | Slider | 1 to the default from config.yaml |
| `context.coreference_high_confidence` | Slider | 0.0–1.0 |
| `context.coreference_mid_confidence` | Slider | 0.0 to the high confidence value (capped) |
| `drafter.tokenizer_family` | Select | `char_heuristic` or `tiktoken` |
| FSM pointer fields | 4 inputs | `arc_id`, `chapter_id`, `scene_id`, `beat_index` |
| **Run build_context_package** | Primary button | Assembles the context package and stores it in `st.session_state.m06_package` |

#### Output sections

M06 is organized as a left-to-right runbook:

1. Seed or reset the temp stores.
2. Choose config-shaped budget and confidence controls.
3. Set the FSM pointer.
4. Run `build_context_package()`.
5. Inspect summary metrics, intermediate steps, and raw package layers.

**Metrics row** — Four metric cards:
- Initial tokens
- Final tokens
- Budget
- Over budget (boolean as string)

**Intermediate steps:**

| Panel | Content |
|-------|---------|
| Layer token counts | Bar chart of tokens per layer (relational, summaries, temporal, flavour, coreference, macro) |
| Pruning decisions | JSON list of which layers were pruned |
| Layer availability | JSON dict showing which layers are present vs unavailable |
| Coreference tiering | JSON showing high/mid/low claim assignment |

**Package layers** — Seven tabs (Relational, Summaries, Temporal, Flavour, Coreference, Macro, Meta) each showing the raw JSON of that layer.

**Edge cases to exercise:**

- Set token budget to 1 — prunable layers are dropped, while relational truth remains and `over_budget` stays true if needed
- Set high confidence above mid confidence, then drag mid to exceed high — observe the slider cap
- Run without seeding stores first — observe the error from missing data
- Run after seeding, change a parameter, run again — session state preserves the package
- Set `beat_index` to a value that doesn't exist in the seeded data — observe the assembly behaviour
- Select `tiktoken` without internet access (for the tokenizer) — observe graceful fallback to `char_heuristic`

---

## Debugging Techniques

### Viewing errors

Every frontend wraps module calls in try/except and displays errors via `st.error()`. If something fails, the error message includes the exception type and message. This is the first place to look.

### Inspecting session state

Streamlit's session state is not directly exposed in the UI. To inspect it:

```python
# Temporarily add to the frontend:
st.write(st.session_state)
```

Or use the Streamlit Developer Tools: append `?debug=true` to the URL (`http://localhost:8501/?debug=true`) to see the debug toolbar.

### Re-running manually

Press **R** in the browser to trigger a full rerun. This reloads all modules (since they're imported at the top of each script).

### Console logs

All `print()` calls go to the terminal where `streamlit run` is running. You can also use `st.write()`, `st.json()`, or `st.code()` to dump variables to the UI temporarily.

### Simulating multiple sessions

Open the same URL in two browser tabs. Each tab gets its own session with independent `st.session_state`, so their temp workspaces are separate. This is useful for comparing two parameter sets side by side.

---

## Extending a Frontend for a New Module

### Adding a new visualiser

1. **Create `tests/frontends/mXX_mymodule.py`** following the pattern:
   - Top docstring with `Module: MXX (Name)`
   - `render()` function with all UI logic
   - `main()` that calls `st.set_page_config()` then `render()`
   - `if __name__ == "__main__": main()` block
   - Import real module functions at the top

2. **Add shared helpers** if needed to `shared.py`:
   - New synthetic seeding functions
   - New config-shaped objects
   - New display helpers

3. **Register in `run_all.py`**:
   ```python
   import mXX_mymodule
   PAGES = {
       "MXX My Module": mXX_mymodule.render,
       # ... existing entries
   }
   ```

### Pattern for a typical UI section

```python
st.subheader("Function name")
with st.form("my_form"):
    param_a = st.text_input("param_a", "default")
    param_b = st.slider("param_b", 0.0, 1.0, 0.5)
    submitted = st.form_submit_button("Run function")
if submitted:
    try:
        result = real_module.some_function(param_a, param_b)
        st.json(result)
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")
```

### Important rules

- **Always wrap calls in try/except** with `st.error()`. The UI should never crash on bad input.
- **Use `st.form`** for groups of controls that should submit together, to avoid excessive reruns.
- **Use `workspace()`** for all temp file paths — never use fixed paths.
- **Import real module code**, not test doubles. The whole point is exercising the production code.

---

## Streamlit Frontends vs. pytest — When to Use What

| Situation | Tool | Reason |
|-----------|------|--------|
| First exploration of a module | **Streamlit** | Interactive; adjust parameters and see results instantly |
| Debugging a specific function's behaviour | **Streamlit** | Visualise intermediate data structures, try edge cases rapidly |
| Verifying a bug fix visually | **Streamlit** | Confirm the fix with your own eyes before writing a regression test |
| Adding a new feature | **Streamlit** first, **pytest** after | Explore behaviour interactively, then codify the expected behaviour in tests |
| Regression suite for CI/CD | **pytest** | Automated, deterministic, runs in CI |
| Exhaustive error-handling tests | **pytest** | Streamlit expects *some* input interaction; pytest can cover all code paths |
| Testing with large/fuzzed data | **pytest** | Faster to run programmatically; Streamlit is for human inspection |
| Documentation / demo | **Streamlit** | Show the module working to another human |

**The iterative cycle:**
1. While coding a module, run the corresponding Streamlit frontend to test manually.
2. When you discover an edge case, note it.
3. After the module stabilises, write pytest tests for the specific edge cases you found.
4. Add the tests to your CI pipeline. The Streamlit frontends remain as interactive documentation and exploratory tools.

---

*Last updated: 2026-06-27*
