"""Module: M01 (Coordinator & State Machine)
Streamlit frontend for exercising state schemas and reducers.
"""

from __future__ import annotations

import streamlit as st

from shared import output_label, page_intro, reset_workspace, section_header, two_section_help, workspace
from fsm.state import (
    FSM_Pointer,
    FailureObject,
    accumulate_or_reset,
    failure_object_json_schema,
    make_initial_state,
)


def render() -> None:
    paths = workspace()
    page_intro(
        "M03 Coordinator / State Machine",
        two_section_help(
            "This page lets you explore the state machine that drives MuseAI's drafting process. You can build an initial state with a story pointer and optional overrides, then test the reducer that accumulates errors (failures) from critics. It's like a control panel for the brain of the system.",
            "Interactive view of the real fsm.state schemas and reducer behavior; the code module is tagged M01.",
        ),
    )
    cols = st.columns([1, 2])
    if cols[0].button(
        "Reset temp workspace",
        help=two_section_help(
            "Clears the current session state for this visualizer. Useful if you want to start over with a clean state.",
            "Clears Streamlit session state for this visualization.",
        ),
        width='stretch',
    ):
        reset_workspace()
    with cols[1]:
        output_label(
            "Temp workspace",
            two_section_help(
                "This page mostly works with in-memory data rather than files. The path is shown here for consistency with the other module visualizers.",
                "This page is mostly in-memory; the path is shown for consistency with other module visualizers.",
            ),
        )
        st.code(str(paths["root"]))

    initial_tab, reducer_tab = st.tabs(["Initial state", "Reducer and schema"])
    with initial_tab:
        _render_initial_state()
    with reducer_tab:
        _render_reducer()


def _render_initial_state() -> None:
    section_header(
        "Initial state workflow",
        two_section_help(
            "Builds a story pointer (FSM_Pointer) that says where we are in the story — which arc, chapter, scene, and beat — and then constructs a full initial state dictionary. The overrides let you toggle flags like pausing or marking a paradox before drafting begins.",
            "Builds an FSM_Pointer and passes it to make_initial_state() with optional overrides.",
        ),
    )
    input_col, output_col = st.columns([1, 2])
    with input_col:
        section_header(
            "Inputs",
            two_section_help(
                "These values are validated by Pydantic — the same validation that runs in production. If you enter something invalid, you'll see the error immediately.",
                "These values are validated by FSM_Pointer and make_initial_state().",
            ),
            level=4,
        )
        project_id = st.text_input(
            "project_id",
            "project-ui",
            help=two_section_help(
                "Identifies which story project this state belongs to. Passed directly to make_initial_state().",
                "Project identifier passed directly to make_initial_state().",
            ),
        )
        arc_id = st.text_input(
            "arc_id", "arc-1",
            help=two_section_help(
                "The story arc the pointer is positioned at. Used to construct the FSM_Pointer.",
                "FSM_Pointer arc_id.",
            ),
        )
        chapter_id = st.text_input(
            "chapter_id", "chapter-1",
            help=two_section_help(
                "The chapter within the arc. Used to construct the FSM_Pointer.",
                "FSM_Pointer chapter_id.",
            ),
        )
        scene_id = st.text_input(
            "scene_id", "scene-1",
            help=two_section_help(
                "The scene within the chapter. Used to construct the FSM_Pointer.",
                "FSM_Pointer scene_id.",
            ),
        )
        beat_index = st.number_input(
            "beat_index",
            min_value=0,
            value=0,
            step=1,
            help=two_section_help(
                "The index of the current beat within the scene. Pydantic validates this is always a non-negative integer.",
                "FSM_Pointer beat index. Strict Pydantic validation keeps it an integer.",
            ),
        )
        overrides = {}
        if st.checkbox(
            "pause_requested override",
            help=two_section_help(
                "Check this to set pause_requested=True in the initial state. This tells the system to pause drafting before it starts.",
                "Adds pause_requested=True to the explicit override dict.",
            ),
        ):
            overrides["pause_requested"] = True
        if st.checkbox(
            "has_paradox override",
            help=two_section_help(
                "Check this to set has_paradox=True in the initial state. This tells the system there's already a known contradiction at startup.",
                "Adds has_paradox=True to the explicit override dict.",
            ),
        ):
            overrides["has_paradox"] = True

    with output_col:
        section_header(
            "Output",
            two_section_help(
                "The full OrchestratorState dictionary produced by make_initial_state(). This is what the drafting process starts with — it contains the pointer, default values, and any overrides you selected.",
                "Full OrchestratorState dictionary returned by make_initial_state().",
            ),
            level=4,
        )
        try:
            pointer = FSM_Pointer(
                arc_id=arc_id,
                chapter_id=chapter_id,
                scene_id=scene_id,
                beat_index=int(beat_index),
            )
            state = make_initial_state(project_id, pointer, **overrides)
            state["fsm_pointer"] = state["fsm_pointer"].model_dump()
            output_label(
                "FSM_Pointer",
                two_section_help(
                    "The validated pointer object serialized for display. Shows where in the story this state is positioned.",
                    "Validated pointer object serialized for display.",
                ),
            )
            st.json(pointer.model_dump())
            output_label(
                "make_initial_state()",
                two_section_help(
                    "The fresh state dictionary with all default fields populated and any overrides you selected applied.",
                    "Fresh state with default fields and selected overrides applied.",
                ),
            )
            st.json(state)
        except Exception as exc:  # noqa: BLE001
            st.error(f"{type(exc).__name__}: {exc}")


def _render_reducer() -> None:
    section_header(
        "Reducer workflow",
        two_section_help(
            "This demonstrates accumulate_or_reset() — the function that manages the list of failure objects (errors reported by critics). If you append one or more failures they get added to the list. If you pass an empty list, the whole list resets. This is how the system decides whether to keep drafting or stop and fix problems.",
            "Demonstrates accumulate_or_reset(): non-empty incoming lists append, an explicit [] resets.",
        ),
    )
    if "m03_failures" not in st.session_state:
        st.session_state.m03_failures = []
    input_col, output_col = st.columns([1, 2])
    with input_col:
        section_header(
            "Append a FailureObject",
            two_section_help(
                "Fill in the details of a failure — what went wrong, which text caused it, what the fix might be, and which critic found it. Each failure is validated against the real Pydantic FailureObject schema, just like in production.",
                "The form validates the real Pydantic FailureObject schema.",
            ),
            level=4,
        )
        with st.form("failure_form"):
            error_code = st.text_input(
                "error_code",
                "PACING_ISSUE",
                help=two_section_help(
                    "A code that identifies what kind of failure this is, like PACING_ISSUE or CONTRADICTION. Downstream logic uses this code to decide how to recover.",
                    "Free-form code used by downstream recovery routing.",
                ),
            )
            offending_text = st.text_input(
                "offending_text",
                "synthetic offending text",
                help=two_section_help(
                    "The exact text that triggered the failure. This is stored as verbatim text, not as a character offset or line number.",
                    "Verbatim text target, not a character offset.",
                ),
            )
            suggested_fix = st.text_input(
                "suggested_fix",
                "synthetic fix",
                help=two_section_help(
                    "A proposed repair or corrected version of the text. Stored on the failure object for later use by recovery logic.",
                    "Proposed repair text stored on the failure object.",
                ),
            )
            critic_source = st.text_input(
                "critic_source",
                "synthetic_critic",
                help=two_section_help(
                    "Which critic or checker module produced this failure. Helps trace where problems originate.",
                    "Which critic or checker produced the failure.",
                ),
            )
            append = st.form_submit_button(
                "Append FailureObject",
                help=two_section_help(
                    "Calls accumulate_or_reset() with the current list and the new failure appended. If the list was previously reset, this starts a new accumulation.",
                    "Calls accumulate_or_reset(current, [new_failure]).",
                ),
            )
        if append:
            try:
                failure = FailureObject(
                    error_code=error_code,
                    offending_text=offending_text,
                    suggested_fix=suggested_fix,
                    critic_source=critic_source,
                )
                st.session_state.m03_failures = accumulate_or_reset(
                    st.session_state.m03_failures,
                    [failure],
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"{type(exc).__name__}: {exc}")
        if st.button(
            "Pass [] to reducer",
            help=two_section_help(
                "Calls accumulate_or_reset() with an empty list, which clears the accumulated failures. Use this to simulate the system deciding everything is fine.",
                "Calls accumulate_or_reset(current, []), which resets the accumulated list.",
            ),
        ):
            st.session_state.m03_failures = accumulate_or_reset(
                st.session_state.m03_failures,
                [],
            )

    with output_col:
        output_label(
            "Current reducer state",
            two_section_help(
                "The accumulated list of FailureObjects after the latest reducer operation. If the list is empty, there are no unaddressed failures.",
                "Session-state list after the latest reducer operation.",
            ),
        )
        st.json([failure.model_dump() for failure in st.session_state.m03_failures])
        with st.expander("FailureObject JSON schema"):
            output_label(
                "failure_object_json_schema()",
                two_section_help(
                    "The JSON Schema that FailureObject instances must conform to. This schema is used by the structured output pipeline to validate that critic responses parse correctly.",
                    "Schema used by structured critic output validation.",
                ),
            )
            st.json(failure_object_json_schema())


def main() -> None:
    st.set_page_config(page_title="M03 State", layout="wide")
    render()


if __name__ == "__main__":
    main()