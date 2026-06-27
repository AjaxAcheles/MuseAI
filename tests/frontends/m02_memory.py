"""Module: M02 (Persistent Memory Stores & Interfaces)
Streamlit frontend for exercising real memory-store APIs.
"""

from __future__ import annotations

import json

import streamlit as st

from shared import (
    TABLES,
    context_defaults,
    ensure_relational_parents,
    output_label,
    page_intro,
    parse_json,
    reset_workspace,
    section_header,
    seed_narrative_data,
    table_view,
    two_section_help,
    workspace,
)
from memory import event_log, provisional_store, sqlite_db


def render() -> None:
    paths = workspace()
    page_intro(
        "M02 Persistent Memory",
        two_section_help(
            "This page lets you interact with the three real memory stores MuseAI uses to save story data: a SQLite database for structured info (arcs, chapters, beats), a JSONL event log for recording what happened when, and a provisional claims store for keeping track of uncertain inferences the system makes about the story.",
            "Interactive debugger for the real SQLite, JSONL event-log, and provisional-claim store APIs.",
        ),
    )

    action_cols = st.columns([1, 1, 2])
    if action_cols[0].button(
        "Seed synthetic stores",
        width='stretch',
        help=two_section_help(
            "Writes a pre-made set of test data — arcs, chapters, scenes, beats, character emotions, and claims — into the temporary stores so you have something to inspect and experiment with.",
            "Writes deterministic sample arcs, scenes, beats, PAD rows, threads, summaries, and provisional claims to temp files.",
        ),
    ):
        seed_narrative_data(paths["db"], paths["provisional"])
        st.success("Seeded temp relational and provisional stores.")
    if action_cols[1].button(
        "Reset temp workspace",
        width='stretch',
        help=two_section_help(
            "Deletes all temporary data for this session and starts fresh. Handy when you've made a mess and want to begin again.",
            "Deletes this Streamlit session's temporary memory artifacts and starts fresh.",
        ),
    ):
        reset_workspace()
    with action_cols[2]:
        output_label(
            "Temp workspace",
            two_section_help(
                "The path below shows where this session's temporary files live. Everything you write here stays local to this session — your real project data is safe.",
                "All writes in this page go here, not to the repo's production data directory.",
            ),
        )
        st.code(str(paths["root"]))

    sqlite_tab, log_tab, provisional_tab = st.tabs(
        ["SQLite relational hub", "Event log", "Provisional claims"]
    )
    with sqlite_tab:
        _render_sqlite(paths["db"])
    with log_tab:
        _render_event_log(paths["event_log"])
    with provisional_tab:
        _render_provisional(paths["provisional"])


def _render_sqlite(db_path) -> None:
    sqlite_db.init_db(db_path)
    section_header(
        "Beat commit workflow",
        two_section_help(
            "This is where you write a 'beat' — a chunk of story prose — into the database. You specify which arc, chapter, and scene it belongs to, along with character emotions (PAD values) and thread status updates. It's the primary way the system saves progress during drafting.",
            "Creates required parent rows, then calls memory.sqlite_db.upsert_beat_commit() against the temp SQLite hub.",
        ),
    )
    input_col, output_col = st.columns([1, 1])

    with input_col:
        section_header(
            "Inputs",
            two_section_help(
                "Fill in these fields to identify exactly where in the story this beat goes. Arc, chapter, and scene IDs are like mailing addresses — they tell the database which shelf to put the beat on. The JSON fields bundle up emotion data and thread statuses that get written alongside the beat.",
                "IDs identify the synthetic story location; JSON fields become bundled PAD and thread-update payloads.",
            ),
            level=4,
        )
        with st.form("commit_beat"):
            arc_id = st.text_input(
                "arc_id",
                "arc-1",
                help=two_section_help(
                    "The story arc this beat belongs to. The UI will create this arc in the temp database if it doesn't exist yet.",
                    "Parent arc id. The UI creates it in the temp DB if missing.",
                ),
            )
            chapter_id = st.text_input(
                "chapter_id",
                "chapter-1",
                help=two_section_help(
                    "The chapter within the arc. If this chapter doesn't exist yet, the UI creates it for you.",
                    "Parent chapter id. The UI creates it under arc_id if missing.",
                ),
            )
            scene_id = st.text_input(
                "scene_id",
                "scene-1",
                help=two_section_help(
                    "The scene this beat lives in. The beat commit writes the beat as a child of this scene.",
                    "Parent scene id. upsert_beat_commit writes the beat under this scene.",
                ),
            )
            beat_id = st.text_input(
                "beat_id",
                "beat-ui",
                help=two_section_help(
                    "A unique name for this beat. If you run the same beat_id twice it updates the existing row rather than creating a duplicate.",
                    "Primary key for the committed beat; running twice updates the same row.",
                ),
            )
            beat_index = st.number_input(
                "beat_index",
                min_value=0,
                value=2,
                step=1,
                help=two_section_help(
                    "Where this beat falls in the scene's sequence. Beats are ordered by this number.",
                    "Ordering key inside the scene.",
                ),
            )
            prose = st.text_area(
                "committed prose",
                "Marcus set the last shelf bracket and stepped back. 'There,' he said. "
                "'Room for a few more stories.' Elena didn't tell him she'd meant to "
                "leave that wall bare.",
                help=two_section_help(
                    "The actual story text for this beat. This gets saved in the Beats table for later retrieval.",
                    "Committed prose saved on the Beats row.",
                ),
            )
            pad_text = st.text_area(
                "pad_states JSON",
                json.dumps(
                    {
                        "char-elena": {
                            "pleasure": 0.80,
                            "arousal": 0.55,
                            "dominance": 0.50,
                        },
                        "char-marcus": {
                            "pleasure": 0.55,
                            "arousal": 0.40,
                            "dominance": 0.45,
                        },
                    },
                    indent=2,
                ),
                help=two_section_help(
                    "Character emotional states using PAD (Pleasure-Arousal-Dominance) coordinates. Each character gets three decimal values that describe their emotional state during this beat. These get stored for tracking emotional arcs.",
                    "Maps character_id to PAD axes. These rows are written inside the same beat commit.",
                ),
            )
            thread_text = st.text_area(
                "thread_updates JSON",
                json.dumps({"thread-slow-burn": "progressing"}, indent=2),
                help=two_section_help(
                    "Updates to story threads — think of threads as plot lines or character arcs. Each entry maps a thread ID to its new status (e.g. progressing, resolved). Existing threads are updated without creating duplicates.",
                    "Maps thread_id to its new status. Existing temp thread rows are updated idempotently.",
                ),
            )
            submitted = st.form_submit_button(
                "Run upsert_beat_commit",
                help=two_section_help(
                    "Executes the real database function that writes all this data — beat, emotions, and thread updates — in one transactional call.",
                    "Executes the real memory.sqlite_db.upsert_beat_commit() helper.",
                ),
            )
        if submitted:
            try:
                pad_states = parse_json(pad_text, {})
                thread_updates = parse_json(thread_text, {})
                ensure_relational_parents(
                    db_path,
                    arc_id=arc_id,
                    chapter_id=chapter_id,
                    scene_id=scene_id,
                    characters=list(pad_states),
                    threads=list(thread_updates),
                )
                sqlite_db.upsert_beat_commit(
                    db_path,
                    beat_id=beat_id,
                    scene_id=scene_id,
                    beat_index=int(beat_index),
                    prose=prose,
                    pad_states=pad_states,
                    thread_updates=thread_updates,
                )
                st.success("Committed beat through memory.sqlite_db.upsert_beat_commit().")
                st.session_state.m02_last_beat_id = beat_id
                st.session_state.m02_last_scene_id = scene_id
            except Exception as exc:  # noqa: BLE001 - UI should surface module errors.
                st.error(f"{type(exc).__name__}: {exc}")

    with output_col:
        section_header(
            "Primary output",
            two_section_help(
                "After committing a beat, these three read helpers show you what changed: the beat row itself, the latest PAD snapshot per character in the scene, and all currently open threads.",
                "These reads prove what changed after the commit: beat row, PAD snapshots, and open-thread state.",
            ),
            level=4,
        )
        last_beat = st.session_state.get("m02_last_beat_id", "beat-0")
        last_scene = st.session_state.get("m02_last_scene_id", "scene-1")
        output_label(
            "get_beat()",
            two_section_help(
                "Reads one beat row by its ID from the temp database. Shows you everything saved for that beat including prose, status, timestamps, and associated emotion data.",
                "Reads the committed Beats row by id from the temp SQLite store.",
            ),
        )
        st.json(sqlite_db.get_beat(db_path, last_beat))
        output_label(
            "get_latest_pad_for_scene()",
            two_section_help(
                "Shows the most recent emotional state (PAD values) for each character in the selected scene. This is how the system tracks how characters are feeling as the story progresses.",
                "Shows the latest PAD snapshot per character in the selected scene.",
            ),
        )
        st.json(sqlite_db.get_latest_pad_for_scene(db_path, last_scene))
        output_label(
            "get_open_threads()",
            two_section_help(
                "Lists all threads that are still marked as 'open' — plot lines that haven't been resolved yet. After a beat commit with thread updates, this shows you what's still in progress.",
                "Open threads after any thread status updates from the beat commit.",
        ),
        )
        st.json(sqlite_db.get_open_threads(db_path))

    section_header(
        "Read helper explorer",
        two_section_help(
            "Change the IDs below and click through the tabs to inspect any arc, chapter, scene, or beat in the database. This is read-only — nothing gets written when you use these.",
            "Change IDs and inspect deterministic read helpers without writing anything.",
        ),
    )
    read_col, result_col = st.columns([1, 2])
    with read_col:
        read_arc = st.text_input(
            "Read arc",
            "arc-1",
            key="m02_read_arc",
            help=two_section_help(
                "Type an arc ID and the Arc tab will show its details plus all chapters inside it.",
                "Used by get_arc() and get_chapters_for_arc().",
            ),
        )
        read_chapter = st.text_input(
            "Read chapter",
            "chapter-1",
            key="m02_read_chapter",
            help=two_section_help(
                "Type a chapter ID and the Chapter tab will show its scenes in order.",
                "Used by get_scenes_for_chapter_ordered().",
            ),
        )
        read_scene = st.text_input(
            "Read scene",
            "scene-1",
            key="m02_read_scene",
            help=two_section_help(
                "Type a scene ID and the Scene tab will show its beats in order plus the latest PAD emotions.",
                "Used by beat and PAD scene reads.",
            ),
        )
        read_beat = st.text_input(
            "Read beat",
            "beat-0",
            key="m02_read_beat",
            help=two_section_help(
                "Type a beat ID and the Beat tab will show that beat's full row — or null if it doesn't exist.",
                "Used by get_beat().",
            ),
        )
    with result_col:
        tabs = st.tabs(["Arc", "Chapter", "Scene", "Beat"])
        with tabs[0]:
            output_label(
                "Arc reads",
                two_section_help(
                    "Shows the arc details plus every chapter that belongs to it.",
                    "get_arc() plus chapters under the selected arc.",
                ),
            )
            st.json(
                {
                    "get_arc": sqlite_db.get_arc(db_path, read_arc),
                    "get_chapters_for_arc": sqlite_db.get_chapters_for_arc(db_path, read_arc),
                }
            )
        with tabs[1]:
            output_label(
                "Chapter reads",
                two_section_help(
                    "Shows all scenes in the selected chapter in their correct order, plus which scenes are still remaining (not yet completed).",
                    "Ordered scenes and remaining scenes for the selected chapter.",
                ),
            )
            st.json(
                {
                    "get_scenes_for_chapter_ordered": sqlite_db.get_scenes_for_chapter_ordered(db_path, read_chapter),
                    "get_remaining_scenes_for_chapter": sqlite_db.get_remaining_scenes_for_chapter(db_path, read_chapter),
                }
            )
        with tabs[2]:
            output_label(
                "Scene reads",
                two_section_help(
                    "Shows all beats in the selected scene in their correct order, plus the latest PAD emotional snapshot for each character.",
                    "Ordered beats and latest PAD snapshots for the selected scene.",
                ),
            )
            st.json(
                {
                    "get_beats_for_scene_ordered": sqlite_db.get_beats_for_scene_ordered(db_path, read_scene),
                    "get_latest_pad_for_scene": sqlite_db.get_latest_pad_for_scene(db_path, read_scene),
                }
            )
        with tabs[3]:
            output_label(
                "Beat read",
                two_section_help(
                    "Looks up a single beat by ID. If the beat doesn't exist you'll see null — a good way to check whether a beat was actually committed.",
                    "Exact beat lookup; missing beats return null here.",
                ),
            )
            st.json(sqlite_db.get_beat(db_path, read_beat))

    with st.expander("Raw SQLite tables"):
        selected = st.multiselect(
            "Tables",
            TABLES,
            default=["Beats", "CharacterEmotions", "Threads"],
            help=two_section_help(
                "Select one or more database tables to see their raw contents as a dataframe. Useful for checking what rows were actually written after running operations.",
                "Select temp SQLite tables to inspect after running write helpers.",
            ),
        )
        for table in selected:
            table_view(db_path, table)


def _render_event_log(log_path) -> None:
    section_header(
        "Append-only event workflow",
        two_section_help(
            "The event log is like a diary — every write operation appends a timestamped event to a JSONL file. You can append new events here and then read them back. This is how the system records what happened and when, for debugging and audit trails.",
            "Calls memory.event_log.write_event() and then reads the same JSONL file with tail_events() and iter_events().",
        ),
    )
    input_col, output_col = st.columns([1, 1])
    default_payload = {
        "event_type": "beat_commit",
        "beat_id": "beat-ui",
        "written_at": "2026-01-01T00:00:00Z",
        "pad_states": {"char-ada": {"pleasure": 0.1, "arousal": 0.2, "dominance": 0.3}},
    }
    with input_col:
        payload_text = st.text_area(
            "event payload JSON",
            json.dumps(default_payload, indent=2),
            help=two_section_help(
                "A JSON object describing the event. This gets appended as one line to the JSONL file. Include whatever fields you want to record — event type, IDs, timestamps, and any relevant data.",
                "Must be one JSON object. The writer appends it as one JSONL line.",
            ),
        )
        if st.button(
            "Run write_event",
            help=two_section_help(
                "Appends one event to the temp JSONL log file by calling the real write_event() function.",
                "Executes memory.event_log.write_event() against the temp events.jsonl file.",
            ),
        ):
            try:
                event_log.write_event(log_path, parse_json(payload_text, {}))
                st.success("Appended one event through memory.event_log.write_event().")
            except Exception as exc:  # noqa: BLE001
                st.error(f"{type(exc).__name__}: {exc}")
    with output_col:
        limit = st.slider(
            "tail_events limit",
            min_value=0,
            max_value=20,
            value=5,
            help=two_section_help(
                "How many of the most recent events to show. The log is append-only, so events are ordered oldest to newest.",
                "How many recent JSONL records to read back in chronological order.",
            ),
        )
        output_label(
            "tail_events()",
            two_section_help(
                "Shows the N most recent events from the log. Events are returned in chronological order (oldest first).",
                "Recent events read from the append-only temp log.",
            ),
        )
        st.json(event_log.tail_events(log_path, limit))
        with st.expander("iter_events full stream"):
            output_label(
                "iter_events()",
                two_section_help(
                    "Reads every event from the log file from beginning to end. Use this when you want the complete history.",
                    "Full forward scan of the same log file.",
                ),
            )
            try:
                st.json(list(event_log.iter_events(log_path)))
            except Exception as exc:  # noqa: BLE001
                st.error(f"{type(exc).__name__}: {exc}")


def _render_provisional(path) -> None:
    provisional_store.init_provisional_store(path)
    defaults = context_defaults()
    section_header(
        "Provisional-claim workflow",
        two_section_help(
            "Provisional claims are the system's best guesses about the story — things like 'this pronoun probably refers to character Ada'. They're kept separate from the main database because they might be wrong. Here you can add claims, adjust their confidence levels, browse them with filters, and mark them as reviewed or confirmed.",
            "Calls the real provisional store APIs; claims stay separate from canonical SQLite truth.",
        ),
    )
    input_col, output_col = st.columns([1, 1])
    with input_col:
        section_header(
            "Create or update",
            two_section_help(
                "Write a new provisional claim or update an existing one. The claim_id is optional — leave it blank and the store will generate one from a hash of the claim text.",
                "upsert_claim() inserts or updates one provisional row.",
            ),
            level=4,
        )
        with st.form("upsert_claim"):
            claim_id = st.text_input(
                "claim_id",
                "claim-ui",
                help=two_section_help(
                    "A unique name for this claim. If you leave this blank the store will derive an ID automatically from the text content. If you reuse an existing ID it overwrites the old claim.",
                    "Stable row key. Blank lets the store derive a deterministic content hash.",
                ),
            )
            claim_text = st.text_area(
                "claim_text",
                "MID_MARKER: 'he' (scene-6, the second coffee cup) -> char-marcus",
                help=two_section_help(
                    "The actual claim text — what the system inferred. For example 'the pronoun \"she\" in scene-2 refers to Ada'. This is stored in the provisional store, not in the main database tables.",
                    "The unconfirmed claim text stored separately from canonical truth.",
                ),
            )
            confidence = st.slider(
                "confidence",
                min_value=0.0,
                max_value=1.0,
                value=float(defaults["coreference_mid_confidence"]),
                help=two_section_help(
                    "How confident the system is that this claim is correct, from 0.0 (not at all) to 1.0 (certain). Later, context assembly uses threshold bands to decide which claims to treat as facts, beliefs, or ignore.",
                    "Stored verbatim. Interpretation into high/mid/low bands happens in context assembly.",
                ),
            )
            source_ref = st.text_input(
                "source_ref",
                "ui-source",
                help=two_section_help(
                    "An optional note about where this claim came from — which module or process made the inference. Helps with debugging provenance.",
                    "Optional provenance marker for where the provisional claim came from.",
                ),
            )
            submitted = st.form_submit_button(
                "Run upsert_claim",
                help=two_section_help(
                    "Executes the real provisional_store.upsert_claim() function to write or update the claim.",
                    "Executes memory.provisional_store.upsert_claim().",
                ),
            )
        if submitted:
            try:
                resolved = provisional_store.upsert_claim(
                    path,
                    claim_id=claim_id or None,
                    claim_text=claim_text,
                    confidence=float(confidence),
                    source_ref=source_ref or None,
                )
                st.success(f"Upserted claim_id={resolved}.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"{type(exc).__name__}: {exc}")

        section_header(
            "Review",
            two_section_help(
                "When a claim has been checked — either by a human or by another module — you can mark it as reviewed, confirmed, or rejected. The row stays in the database (it's never deleted) but its status changes so downstream logic knows what to do with it.",
                "mark_claim_reviewed() records a review decision without deleting the row.",
            ),
            level=4,
        )
        claim_to_review = st.text_input(
            "claim_id to review",
            "claim-ui",
            help=two_section_help(
                "The ID of an existing provisional claim that you want to review.",
                "Existing provisional claim id to review.",
            ),
        )
        review_status = st.selectbox(
            "review status",
            ["reviewed", "confirmed", "rejected"],
            help=two_section_help(
                "What to mark the claim as: reviewed (checked but not necessarily confirmed), confirmed (accepted as true), or rejected (determined to be false).",
                "Allowed terminal review states enforced by the store.",
            ),
        )
        note = st.text_input(
            "reviewer_note",
            "UI review note",
            help=two_section_help(
                "An optional note explaining why the claim was marked that way. Useful for human review workflows.",
                "Optional note saved on the reviewed claim.",
        ),
        )
        if st.button(
            "Run mark_claim_reviewed",
            help=two_section_help(
                "Executes the real provisional_store.mark_claim_reviewed() function to update the claim's status.",
                "Executes memory.provisional_store.mark_claim_reviewed().",
            ),
        ):
            try:
                st.session_state.m02_reviewed_claim = provisional_store.mark_claim_reviewed(
                    path,
                    claim_to_review,
                    status=review_status,
                    reviewer_note=note or None,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"{type(exc).__name__}: {exc}")

    with output_col:
        section_header(
            "Browse claims",
            two_section_help(
                "Use the filters below to search through provisional claims by confidence range and status. The results show you which claims exist at various confidence levels — useful for verifying that the filtering logic works correctly.",
                "Filters use list_claims_by_confidence() against the temp provisional DB.",
            ),
            level=4,
        )
        min_conf = st.slider(
            "min_confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            help=two_section_help(
                "Only show claims with confidence at or above this value. Set to 0 to include everything.",
                "Inclusive lower confidence bound for the query.",
            ),
        )
        max_conf = st.slider(
            "max_confidence",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            help=two_section_help(
                "Only show claims with confidence at or below this value. Set to 1 to include everything.",
                "Inclusive upper confidence bound for the query.",
            ),
        )
        status = st.selectbox(
            "status filter",
            ["", "provisional", "reviewed", "confirmed", "rejected"],
            help=two_section_help(
                "Filter by claim status. Leave blank to show all statuses. Use this to see only claims that are still provisional, or only ones that have been confirmed.",
                "Optional status predicate; blank means all statuses.",
            ),
        )
        claims = provisional_store.list_claims_by_confidence(
            path,
            min_confidence=float(min_conf),
            max_confidence=float(max_conf),
            status=status or None,
        )
        output_label(
            "Filtered claims",
            two_section_help(
                "The rows returned by your filter query. Each row shows the claim ID, text, confidence, status, and source.",
                "Rows returned by list_claims_by_confidence().",
            ),
        )
        st.dataframe(claims, width='stretch', hide_index=True)
        if "m02_reviewed_claim" in st.session_state:
            output_label(
                "Last reviewed claim",
                two_section_help(
                    "The updated claim row after your review action, returned by mark_claim_reviewed().",
                    "Updated row returned by mark_claim_reviewed().",
                ),
            )
            st.json(st.session_state.m02_reviewed_claim)


def main() -> None:
    st.set_page_config(page_title="M02 Memory", layout="wide")
    render()


if __name__ == "__main__":
    main()