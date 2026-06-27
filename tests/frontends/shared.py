"""Module: M02-M06 (Visualization Test Frontends)
Shared helpers for Streamlit visualizers that exercise real module code.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from html import escape
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory import provisional_store, sqlite_db

UX_CSS = """
<style>
.ux-title-row {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    margin: 0.2rem 0 0.45rem 0;
}
.ux-title-row h2,
.ux-title-row h3,
.ux-title-row h4 {
    margin: 0;
}
.ux-info {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.05rem;
    height: 1.05rem;
    border: 1px solid rgba(49, 51, 63, 0.35);
    border-radius: 50%;
    color: rgba(49, 51, 63, 0.78);
    font-size: 0.72rem;
    font-weight: 700;
    cursor: help;
    vertical-align: middle;
}
.ux-output-label {
    margin: 0.35rem 0 0.15rem 0;
    font-size: 0.88rem;
    font-weight: 650;
    color: rgba(49, 51, 63, 0.86);
}
.ux-callout {
    padding: 0.55rem 0.7rem;
    border: 1px solid rgba(49, 51, 63, 0.15);
    border-radius: 6px;
    background: rgba(49, 51, 63, 0.035);
    font-size: 0.9rem;
}
</style>
"""

TABLES = (
    "Arcs",
    "Chapters",
    "Scenes",
    "Beats",
    "Threads",
    "Characters",
    "CharacterEmotions",
    "CommitIntent",
    "RaptorNodes",
)


def two_section_help(overview: str, technical: str) -> str:
    """Combine an overview and technical section into a single tooltip string.

    Overview is a plain-English 'for-dummies' explanation. Technical holds the
    implementation-level detail (typically the existing help text). The two
    sections are unlabeled, separated only by one newline, and should stay under
    500 words in total.
    """
    return f"{overview.strip()}\n{technical.strip()}"


def install_ux() -> None:
    """Install shared CSS for the Streamlit visualizer pages."""

    st.markdown(UX_CSS, unsafe_allow_html=True)


def info_icon(help_text: str) -> str:
    """Return a small title-tooltip info icon for labels and headings."""

    safe = escape(help_text, quote=True)
    return f'<span class="ux-info" title="{safe}" aria-label="{safe}">i</span>'


def section_header(title: str, help_text: str, *, level: int = 3) -> None:
    """Render a heading with a small hover-help icon."""

    safe_title = escape(title)
    tag = f"h{level}"
    st.markdown(
        f'<div class="ux-title-row"><{tag}>{safe_title}</{tag}>{info_icon(help_text)}</div>',
        unsafe_allow_html=True,
    )


def output_label(title: str, help_text: str) -> None:
    """Render a compact label for a result region."""

    safe_title = escape(title)
    st.markdown(
        f'<div class="ux-output-label">{safe_title} {info_icon(help_text)}</div>',
        unsafe_allow_html=True,
    )


def page_intro(title: str, help_text: str) -> None:
    """Render the page title and install the shared visual language."""

    install_ux()
    section_header(title, help_text, level=2)


def workspace_strip(paths: dict[str, Path], *, seed_label: str | None = None):
    """Render a consistent top action strip and return action columns."""

    cols = st.columns([1, 1, 2])
    if seed_label:
        cols[0].caption(seed_label)
    if cols[1].button(
        "Reset temp workspace",
        width='stretch',
        help=two_section_help(
            "Clears all temporary files for this session and starts fresh. Use this when you want a clean slate without restarting the whole app.",
            "Deletes this Streamlit session's temporary files and recreates a clean workspace.",
        ),
    ):
        reset_workspace()
    with cols[2]:
        output_label(
            "Temp workspace",
            two_section_help(
                "Every frontend writes test data to a temporary folder, not your real project files. This session-scoped directory keeps experiments isolated and throwaway.",
                "All UI writes land in this session-scoped directory, not in the repo's real data folder.",
            ),
        )
        st.code(str(paths["root"]))
    return cols


def workspace() -> dict[str, Path]:
    """Return per-session temp paths for DB/log/template experiments."""

    if "frontend_tmpdir" not in st.session_state:
        st.session_state.frontend_tmpdir = tempfile.TemporaryDirectory(
            prefix="museai_frontends_"
        )
    root = Path(st.session_state.frontend_tmpdir.name)
    return {
        "root": root,
        "db": root / "fictionwriter.db",
        "event_log": root / "events.jsonl",
        "provisional": root / "provisional_claims.db",
        "templates": root / "prompts",
    }


def reset_workspace() -> None:
    """Drop the current temp workspace and create a fresh one on next rerun."""

    tmp = st.session_state.pop("frontend_tmpdir", None)
    if tmp is not None:
        tmp.cleanup()
    st.rerun()


def visual_config(
    *,
    token_budget: int | None = None,
    high_confidence: float | None = None,
    mid_confidence: float | None = None,
    tokenizer_family: str = "char_heuristic",
    model_name: str = "synthetic-drafter",
) -> SimpleNamespace:
    """Small config-shaped object used by context assembly visualizers."""

    defaults = context_defaults()
    return SimpleNamespace(
        endpoints=SimpleNamespace(
            drafter=SimpleNamespace(
                tokenizer_family=tokenizer_family,
                model_name=model_name,
            )
        ),
        context=SimpleNamespace(
            token_budget=token_budget or defaults["token_budget"],
            coreference_high_confidence=(
                high_confidence
                if high_confidence is not None
                else defaults["coreference_high_confidence"]
            ),
            coreference_mid_confidence=(
                mid_confidence
                if mid_confidence is not None
                else defaults["coreference_mid_confidence"]
            ),
        ),
    )


def context_defaults() -> dict[str, Any]:
    """Read context defaults directly from config.yaml without requiring secrets."""

    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
    return dict(raw["context"])


def parse_json(text: str, fallback: Any) -> Any:
    """Parse user JSON text, returning ``fallback`` for blank input."""

    if not text.strip():
        return fallback
    return json.loads(text)


def read_table(db_path: str | Path, table: str) -> list[dict[str, Any]]:
    """Read one SQLite table into plain dict rows for display."""

    if table not in TABLES:
        raise ValueError(f"unknown display table: {table}")
    sqlite_db.init_db(db_path)
    conn = sqlite_db.connect_db(db_path)
    try:
        return [dict(row) for row in conn.execute(f"SELECT * FROM {table}")]
    finally:
        conn.close()


def table_view(db_path: str | Path, table: str, *, help_text: str | None = None) -> None:
    """Render one SQLite table."""

    rows = read_table(db_path, table)
    output_label(
        f"{table}: {len(rows)} row(s)",
        help_text
        or two_section_help(
            "Raw database rows pulled from the temporary SQLite file. This shows exactly what the module's read helpers return after your writes.",
            "Raw rows read from the temp SQLite table after the current operation.",
        ),
    )
    st.dataframe(rows, width='stretch', hide_index=True)


def ensure_relational_parents(
    db_path: str | Path,
    *,
    arc_id: str,
    chapter_id: str,
    scene_id: str,
    characters: list[str] | None = None,
    threads: list[str] | None = None,
) -> None:
    """Seed the planner-owned parent rows needed by write helpers."""

    characters = characters or []
    threads = threads or []
    sqlite_db.init_db(db_path)
    conn = sqlite_db.connect_db(db_path)
    try:
        _upsert_parent_rows(conn, arc_id=arc_id, chapter_id=chapter_id, scene_id=scene_id)
        for character_id in characters:
            conn.execute(
                "INSERT OR IGNORE INTO Characters (id, name) VALUES (?, ?)",
                # Strip a leading "char-" so "char-elena" titles to "Elena".
                (character_id, character_id.removeprefix("char-").replace("-", " ").title()),
            )
        for thread_id in threads:
            conn.execute(
                "INSERT OR IGNORE INTO Threads "
                "(id, description, status, priority_score) VALUES (?, ?, ?, ?)",
                (
                    thread_id,
                    "A quiet thread of feeling running under the days at "
                    "The Paper Petal.",
                    "open",
                    0.50,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def seed_narrative_data(db_path: str | Path, provisional_path: str | Path) -> None:
    """Seed a substantial slice-of-life romance through the real store APIs.

    Elena Marchetti trades a glass-tower marketing career for The Paper Petal, a
    failing bookshop in the small town of Willow Creek, and -- between water-stained
    ledgers and custom shelving -- falls slowly for Marcus Hale, the reticent local
    carpenter, while the town quietly conspires to keep them in the same room. The
    seed spans one arc, four chapters, ten scenes, six characters, five threads, six
    RAPTOR nodes, twelve committed beats with PAD trajectories, and eight provisional
    claims across the confidence bands.

    IDs are stable so the M02/M06 forms (which pre-fill ``arc-1`` / ``chapter-1`` /
    ``scene-1`` and a ``scene-1`` beat at ``beat_index`` 2) resolve against the data.
    """

    sqlite_db.init_db(db_path)
    conn = sqlite_db.connect_db(db_path)
    try:
        # --- Arc + the first chapter/scene (the IDs the forms pre-fill) ---
        _upsert_parent_rows(
            conn,
            arc_id="arc-1",
            chapter_id="chapter-1",
            scene_id="scene-1",
            arc_description=(
                "Elena Marchetti leaves a glass-tower marketing career to revive "
                "The Paper Petal, a failing bookshop in the small town of Willow "
                "Creek, and falls slowly for Marcus Hale, the town's quiet "
                "carpenter, as Willow Creek conspires to keep them close."
            ),
            chapter_description=(
                "Opening Day \u2014 Elena unlocks The Paper Petal, meets the town, "
                "and crosses paths with Marcus over a set of unfinished shelves."
            ),
            scene_description=(
                "Elena unlocks The Paper Petal for the first time, heart hammering "
                "with excitement and fear."
            ),
        )

        # --- Chapters 2-4 ---
        chapters = (
            (
                "chapter-2",
                "Growing Closer \u2014 coffee, a rainy afternoon, and a book club give "
                "Elena and Marcus excuses to linger while Clara and Mrs. Pendleton nudge.",
                "active",
            ),
            (
                "chapter-3",
                "The Misunderstanding \u2014 a visit from Elena's city friend Rosa and a "
                "half-heard remark make Marcus pull back as the shop's finances strain.",
                "active",
            ),
            (
                "chapter-4",
                "Staying \u2014 Elena chooses Willow Creek, Marcus chooses to be seen, "
                "and the town gets its quiet happy ending.",
                "planned",
            ),
        )
        for chapter_id, description, status in chapters:
            conn.execute(
                "INSERT OR IGNORE INTO Chapters "
                "(id, arc_id, description, status) VALUES (?, ?, ?, ?)",
                (chapter_id, "arc-1", description, status),
            )

        # --- Scenes 2-10 (scene-1 already written by _upsert_parent_rows) ---
        scenes = (
            ("scene-2", "chapter-1",
             "Marcus arrives to measure the shop for custom shelves; their first "
             "conversation is awkward and electric.", 500, 1, "completed"),
            ("scene-3", "chapter-1",
             "Tom's caf\u00e9 across the square; Elena learns the town's rhythms over "
             "the worst-best coffee she has ever had.", 450, 2, "completed"),
            ("scene-4", "chapter-2",
             "A rainy afternoon traps Elena and Marcus inside the shop; Clara "
             "conveniently vanishes.", 500, 0, "completed"),
            ("scene-5", "chapter-2",
             "Mrs. Pendleton's book club meets at The Paper Petal; Elena and Marcus "
             "share a quiet moment that feels like a beginning.", 500, 1, "active"),
            ("scene-6", "chapter-2",
             "Marcus brings coffee, exactly right, and Elena lets herself hope.",
             400, 2, "active"),
            ("scene-7", "chapter-3",
             "Rosa visits from the city and misreads Marcus; a half-heard remark "
             "sends him quiet.", 500, 0, "active"),
            ("scene-8", "chapter-3",
             "Elena finds the shop's ledger deep in the red and wonders whether she "
             "has made a terrible mistake.", 450, 1, "planned"),
            ("scene-9", "chapter-4",
             "Marcus mends more than shelves \u2014 an apology shaped like a bookcase.",
             500, 0, "planned"),
            ("scene-10", "chapter-4",
             "Under the new shelves, Elena and Marcus finally say the thing out loud.",
             550, 1, "planned"),
        )
        for scene_id, chapter_id, description, budget, ordering, status in scenes:
            conn.execute(
                "INSERT OR IGNORE INTO Scenes "
                "(id, chapter_id, description, word_budget, ordering, status) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (scene_id, chapter_id, description, budget, ordering, status),
            )

        # --- Characters (all inserted before any PAD reference) ---
        for character_id, name in (
            ("char-elena", "Elena Marchetti"),
            ("char-marcus", "Marcus Hale"),
            ("char-clara", "Clara Wright"),
            ("char-pendleton", "Margaret Pendleton"),
            ("char-tom", "Tom Avery"),
            ("char-rosa", "Rosa Delgado"),
        ):
            conn.execute(
                "INSERT OR IGNORE INTO Characters (id, name) VALUES (?, ?)",
                (character_id, name),
            )

        # --- Threads spanning open/progressing/closed ---
        for thread_id, description, status, priority in (
            ("thread-slow-burn",
             "Elena and Marcus navigate their growing feelings through coffee, "
             "bookshelves, and unspoken glances.", "open", 0.95),
            ("thread-belonging",
             "Elena finds her place in Willow Creek and learns that small-town life "
             "holds more than she expected.", "progressing", 0.70),
            ("thread-shop-finances",
             "The Paper Petal is quietly bleeding money; Elena's savings will not "
             "last another slow season.", "open", 0.65),
            ("thread-matchmaker",
             "Mrs. Pendleton's subtle schemes to push Elena and Marcus together, "
             "with Clara as an unwitting accomplice.", "open", 0.40),
            ("thread-marcus-past",
             "Why Marcus keeps the world at arm's length \u2014 a loss the town "
             "remembers but does not name.", "closed", 0.30),
        ):
            conn.execute(
                "INSERT OR IGNORE INTO Threads "
                "(id, description, status, priority_score) VALUES (?, ?, ?, ?)",
                (thread_id, description, status, priority),
            )

        # --- RAPTOR nodes across global/arc/chapter/scene levels ---
        raptor_nodes = (
            ("raptor-global", None, "global",
             "A slow-burn small-town romance: a city woman revives a dying bookshop "
             "and learns, alongside a guarded carpenter, that staying is its own kind "
             "of courage."),
            ("raptor-arc", "raptor-global", "arc",
             "Elena Marchetti opens The Paper Petal in Willow Creek and begins a "
             "tentative romance with carpenter Marcus Hale while the town gently "
             "orchestrates their proximity and the shop's finances strain."),
            ("raptor-ch1", "raptor-arc", "chapter",
             "Opening day jitters, the first meeting over unfinished shelves, and a "
             "first cup of Tom's terrible coffee establish Elena in the town."),
            ("raptor-ch2", "raptor-arc", "chapter",
             "Rain, a book club, and a perfectly-guessed coffee order pull Elena and "
             "Marcus closer as Clara and Mrs. Pendleton meddle."),
            ("raptor-sc1", "raptor-ch1", "scene",
             "Elena unlocks the shop, meets Marcus, and is welcomed by Clara with "
             "pastries 'for luck.'"),
            ("raptor-sc4", "raptor-ch2", "scene",
             "A storm strands Elena and Marcus alone in the shop; the silence between "
             "them turns comfortable."),
        )
        for node_id, parent_id, level, summary in raptor_nodes:
            conn.execute(
                "INSERT OR REPLACE INTO RaptorNodes "
                "(id, parent_id, level, summary, updated_at) VALUES (?, ?, ?, ?, ?)",
                (node_id, parent_id, level, summary, "2026-01-01T00:00:00Z"),
            )
        conn.commit()
    finally:
        conn.close()

    # --- Beats: written via the real transactional commit helper. Each carries
    #     prose, an evolving PAD snapshot per on-screen character, and thread moves.
    beats = (
        # scene-1: opening day (three beats so the default M06 pointer at index 2 resolves)
        ("beat-0", "scene-1", 0, "2026-01-01T08:00:00Z",
         "Elena turned the sign on the door from 'Closed' to 'Open' and stepped back, "
         "heart hammering. The smell of old paper and fresh paint clung to the air, a "
         "mingling of past and future. She ran her fingers along the spine of a display "
         "copy of Persuasion and smiled. This was hers. All hers.",
         {"char-elena": {"pleasure": 0.85, "arousal": 0.60, "dominance": 0.40}},
         {"thread-belonging": "progressing"}),
        ("beat-1", "scene-1", 1, "2026-01-01T08:15:00Z",
         "He was taller than she'd expected, with sawdust dusting the shoulders of his "
         "flannel shirt and hands that looked like they'd shaped a thousand things by "
         "touch. 'I'm Marcus,' he said, and his voice was quiet, like he wasn't used to "
         "filling a room. Elena tucked a strand of hair behind her ear and tried to "
         "remember how to speak.",
         {"char-elena": {"pleasure": 0.60, "arousal": 0.70, "dominance": 0.30},
          "char-marcus": {"pleasure": 0.30, "arousal": 0.20, "dominance": 0.10}},
         {"thread-slow-burn": "progressing"}),
        ("beat-2", "scene-1", 2, "2026-01-01T17:30:00Z",
         "By closing time Clara had appeared from the bakery two doors down, a box of "
         "lemon cookies balanced on one hip. 'For luck,' she announced, setting them on "
         "the counter as if she'd done it a hundred times. 'Every new shop in Willow "
         "Creek gets cookies. House rule.' Elena laughed, surprised to find her eyes "
         "stinging.",
         {"char-elena": {"pleasure": 0.80, "arousal": 0.45, "dominance": 0.50},
          "char-clara": {"pleasure": 0.75, "arousal": 0.40, "dominance": 0.55}},
         {"thread-belonging": "progressing"}),
        # scene-2: shelves and the coffee
        ("beat-3", "scene-2", 0, "2026-01-02T10:00:00Z",
         "Marcus ran a hand along the pine plank, checking for splinters. 'The light in "
         "here's good,' he said, not quite looking at her. 'Morning light, I mean. For "
         "reading.' Elena pressed her lips together to hide a smile. 'I know what you "
         "meant.'",
         {"char-elena": {"pleasure": 0.70, "arousal": 0.65, "dominance": 0.50},
          "char-marcus": {"pleasure": 0.45, "arousal": 0.35, "dominance": 0.35}},
         None),
        ("beat-4", "scene-2", 1, "2026-01-03T14:30:00Z",
         "He showed up the next afternoon with a cup of coffee from the caf\u00e9 down "
         "the street. 'Didn't know how you take it,' he said, handing it over. 'So I "
         "guessed. Extra cream.' Elena wrapped her fingers around the warm cup. He'd "
         "guessed exactly right. 'Thanks,' she managed. He nodded once, almost a bow, and "
         "turned back to his measuring tape.",
         {"char-elena": {"pleasure": 0.90, "arousal": 0.75, "dominance": 0.60},
          "char-marcus": {"pleasure": 0.60, "arousal": 0.50, "dominance": 0.45}},
         {"thread-matchmaker": "progressing"}),
        # scene-3: Tom's caf\u00e9
        ("beat-5", "scene-3", 0, "2026-01-05T09:00:00Z",
         "Tom's coffee was, objectively, terrible \u2014 scorched and somehow both weak "
         "and bitter \u2014 but he poured it with such ceremony that Elena couldn't bring "
         "herself to say so. 'Marcus tell you about the shelves yet?' he asked, far too "
         "casual, wiping a clean counter. 'He's been talking about that job all week.'",
         {"char-elena": {"pleasure": 0.65, "arousal": 0.40, "dominance": 0.45},
          "char-tom": {"pleasure": 0.70, "arousal": 0.35, "dominance": 0.60}},
         {"thread-belonging": "progressing"}),
        # scene-4: the rainy afternoon
        ("beat-6", "scene-4", 0, "2026-01-08T15:00:00Z",
         "The storm came on fast, rattling the shop windows, and somewhere between the "
         "first thunderclap and the third Clara had remembered an urgent errand and "
         "vanished. Marcus stood by the poetry shelf, rain blurring the glass behind him. "
         "'I could keep working,' he said. 'Or I could not.' Elena reached for two mugs. "
         "'Stay,' she said, and the word felt enormous.",
         {"char-elena": {"pleasure": 0.78, "arousal": 0.68, "dominance": 0.55},
          "char-marcus": {"pleasure": 0.62, "arousal": 0.55, "dominance": 0.40}},
         {"thread-slow-burn": "progressing"}),
        # scene-5: the book club
        ("beat-7", "scene-5", 0, "2026-01-12T19:00:00Z",
         "Mrs. Pendleton had arranged the book-club chairs so that the only open seat was "
         "beside Marcus, a fact she pretended not to notice from behind her reading "
         "glasses. 'Sit, dear, sit,' she said, patting nothing in particular. Across the "
         "circle, Clara hid a grin behind her copy of the month's novel.",
         {"char-elena": {"pleasure": 0.72, "arousal": 0.60, "dominance": 0.50},
          "char-marcus": {"pleasure": 0.55, "arousal": 0.48, "dominance": 0.42},
          "char-pendleton": {"pleasure": 0.80, "arousal": 0.30, "dominance": 0.75}},
         {"thread-matchmaker": "progressing"}),
        # scene-6: the hope beat
        ("beat-8", "scene-6", 0, "2026-01-15T08:30:00Z",
         "There was a second cup waiting on the counter when she arrived \u2014 extra "
         "cream, already going cold, which meant he'd come early and not stayed. Tucked "
         "under it was a torn strip of receipt paper: a single line of a poem she'd "
         "mentioned weeks ago, copied out in his careful block letters. Elena read it "
         "three times before she trusted herself to breathe.",
         {"char-elena": {"pleasure": 0.92, "arousal": 0.70, "dominance": 0.58}},
         {"thread-slow-burn": "progressing"}),
        # scene-7: the misunderstanding
        ("beat-9", "scene-7", 0, "2026-01-20T13:00:00Z",
         "Rosa breezed in from the city in heels wrong for the cobblestones and looped an "
         "arm through Elena's. 'So this is the famous bookshop,' she said, loud and "
         "delighted, just as Marcus came through the back with an armload of cedar. 'And "
         "is this the handyman?' Something in his face closed like a door. He set the wood "
         "down and said he'd come back another time.",
         {"char-elena": {"pleasure": 0.35, "arousal": 0.65, "dominance": 0.30},
          "char-marcus": {"pleasure": -0.20, "arousal": 0.45, "dominance": 0.25},
          "char-rosa": {"pleasure": 0.60, "arousal": 0.50, "dominance": 0.65}},
         {"thread-slow-burn": "open"}),
        ("beat-10", "scene-7", 1, "2026-01-20T18:00:00Z",
         "He didn't bring coffee the next morning, or the one after. The half-built "
         "shelves sat in the corner under a drop cloth like something abandoned. Elena "
         "told herself it was only a job running late. She was not, she insisted to the "
         "empty shop, the kind of woman who waited by a window.",
         {"char-elena": {"pleasure": -0.10, "arousal": 0.40, "dominance": 0.35},
          "char-marcus": {"pleasure": -0.35, "arousal": 0.30, "dominance": 0.20}},
         # thread-marcus-past stays 'closed' so the seed shows all three thread statuses
         # at rest (open / progressing / closed) for the M02 visualizer.
         None),
        # scene-8: the ledger
        ("beat-11", "scene-8", 0, "2026-01-24T22:00:00Z",
         "The numbers didn't improve no matter how long she stared at them. Rent, "
         "stock, the loan \u2014 the red ran down the page like a tide. Elena pressed the "
         "heels of her hands to her eyes and thought, for one cowardly second, about the "
         "corner office she'd left behind, where nothing ever ached this much because "
         "nothing ever mattered this much.",
         {"char-elena": {"pleasure": -0.45, "arousal": 0.55, "dominance": 0.25}},
         {"thread-shop-finances": "progressing"}),
    )
    for beat_id, scene_id, beat_index, committed_at, prose, pad_states, thread_updates in beats:
        sqlite_db.upsert_beat_commit(
            db_path,
            beat_id=beat_id,
            scene_id=scene_id,
            beat_index=beat_index,
            prose=prose,
            status="completed",
            committed_at=committed_at,
            pad_states=pad_states,
            thread_updates=thread_updates,
        )

    # --- Provisional claims across the confidence bands (high / mid / low) ---
    provisional_store.init_provisional_store(provisional_path)
    for claim_id, claim_text, confidence in (
        ("claim-high",
         "HIGH_MARKER: 'she' (scene-1, 'she ran her fingers') -> char-elena", 0.95),
        ("claim-high-marcus",
         "HIGH_MARKER: 'He' (scene-2, 'He showed up the next afternoon') -> char-marcus",
         0.92),
        ("claim-medium-high",
         "HIGH_MARKER: 'her friend' (scene-7, the visitor from the city) -> char-rosa",
         0.78),
        ("claim-mid",
         "MID_MARKER: 'his voice' (scene-1, 'his voice was quiet') -> char-marcus", 0.60),
        ("claim-mid-pendleton",
         "MID_MARKER: 'the older woman' (scene-5, behind reading glasses) -> char-pendleton",
         0.55),
        ("claim-mid-thread",
         "MID_MARKER: 'the book club' (scene-5) -> thread-matchmaker (Pendleton's scheme)",
         0.45),
        ("claim-low",
         "LOW_MARKER: 'the shop' (scene-8 references to 'the shop') -> scene-1 "
         "(the Paper Petal, carryover reference)", 0.20),
        ("claim-low-tom",
         "LOW_MARKER: 'that man' (scene-3, ambiguous) -> char-tom (could also be char-marcus)",
         0.18),
    ):
        provisional_store.upsert_claim(
            provisional_path,
            claim_id=claim_id,
            claim_text=claim_text,
            confidence=confidence,
            status="provisional",
        )


def display_package_layers(package: dict[str, Any]) -> None:
    """Render context package layers in tabs."""

    section_header(
        "Raw package layers",
        two_section_help(
            "These seven tabs show every section of the assembled context package — the full document the model will see. Inspect each layer to understand what information goes into prose generation.",
            "These tabs show the complete context-package layers returned by build_context_package().",
        ),
    )
    tabs = st.tabs(
        [
            "Relational",
            "Summaries",
            "Temporal",
            "Flavour",
            "Coreference",
            "Macro",
            "Meta",
        ]
    )
    keys = (
        "relational",
        "summaries",
        "temporal",
        "flavour",
        "coreference_candidates",
        "macro_constraints",
        "meta",
    )
    for tab, key in zip(tabs, keys, strict=True):
        with tab:
            st.json(package.get(key, {}))


def _upsert_parent_rows(
    conn: sqlite3.Connection,
    *,
    arc_id: str,
    chapter_id: str,
    scene_id: str,
    arc_description: str | None = None,
    chapter_description: str | None = None,
    scene_description: str | None = None,
) -> None:
    """Insert the planner-owned Arc/Chapter/Scene parents for a write.

    Descriptions default to slice-of-life romance placeholders (Willow Creek /
    The Paper Petal) so that even form-driven parents read like real story data
    rather than bare ``synthetic`` markers.
    """

    arc_description = arc_description or (
        "Elena leaves the city to run The Paper Petal bookshop in Willow Creek "
        "and falls slowly for the town's quiet carpenter."
    )
    chapter_description = chapter_description or (
        "Days at the bookshop where Elena and Marcus keep finding reasons "
        "to share the same small room."
    )
    scene_description = scene_description or (
        "A quiet afternoon at The Paper Petal, dust motes in the window light."
    )
    conn.execute(
        "INSERT OR IGNORE INTO Arcs (id, description, status) VALUES (?, ?, ?)",
        (arc_id, arc_description, "active"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO Chapters (id, arc_id, description, status) "
        "VALUES (?, ?, ?, ?)",
        (chapter_id, arc_id, chapter_description, "active"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO Scenes "
        "(id, chapter_id, description, word_budget, ordering, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (scene_id, chapter_id, scene_description, 500, 0, "active"),
    )
