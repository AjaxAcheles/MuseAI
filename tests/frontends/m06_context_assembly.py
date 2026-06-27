"""Module: M03 (Context Assembly & Budgeting)
Streamlit frontend for real context-package assembly over temp stores.
"""

from __future__ import annotations

import streamlit as st

from shared import (
    context_defaults,
    display_package_layers,
    output_label,
    page_intro,
    reset_workspace,
    section_header,
    seed_narrative_data,
    two_section_help,
    visual_config,
    workspace,
)
from fsm.nodes.node_assemble_context import build_context_package
from fsm.state import FSM_Pointer


def render() -> None:
    paths = workspace()
    page_intro(
        "M06 Context Assembly & Budgeting",
        two_section_help(
            "This page runs the real context assembly pipeline — the process that gathers all relevant story data (arcs, beats, character emotions, summaries, claims) and packages it into a context document for the language model. You control the token budget, confidence thresholds, and story position, then inspect every layer of the resulting package.",
            "Runs build_context_package() over temp SQLite and provisional stores; deferred Graphiti/Chroma stores surface as unavailable.",
        ),
    )

    action_cols = st.columns([1, 1, 2])
    if action_cols[0].button(
        "Seed synthetic memory stores",
        width='stretch',
        help=two_section_help(
            "Writes deterministic test data — arcs, chapters, scenes, beats with prose and PAD emotions, summaries, and provisional claims — into the temporary stores so context assembly has something to work with.",
            "Writes deterministic M02 data used by the context package: arc, chapter, scenes, beats, summaries, and claims.",
        ),
    ):
        seed_narrative_data(paths["db"], paths["provisional"])
        st.success("Seeded temp stores.")
    if action_cols[1].button(
        "Reset temp workspace",
        width='stretch',
        help=two_section_help(
            "Deletes the temporary memory stores and clears the last assembled context package. Use this to start over with a clean environment.",
            "Deletes temp memory stores and clears the last assembled package.",
        ),
    ):
        st.session_state.pop("m06_package", None)
        reset_workspace()
    with action_cols[2]:
        output_label(
            "Temp workspace",
            two_section_help(
                "Context assembly reads from these temporary store paths. No production data is touched.",
                "Context assembly reads from these temp store paths only.",
            ),
        )
        st.code(str(paths["root"]))

    section_header(
        "Configure and run",
        two_section_help(
            "Set the token budget (how many tokens the context package is allowed), confidence thresholds (which provisional claims count as facts vs beliefs), and the story position (FSM pointer). Then run the real build_context_package() to see how the system assembles all this into a context document.",
            "Choose config-shaped values and an FSM pointer, then run the real build_context_package() function.",
        ),
    )
    config_col, pointer_col, run_col = st.columns([1, 1, 1])
    with config_col:
        defaults = context_defaults()
        token_budget = st.slider(
            "context.token_budget",
            min_value=1,
            max_value=int(defaults["token_budget"]),
            value=int(defaults["token_budget"]),
            step=10,
            help=two_section_help(
                "The maximum number of tokens the context package can use. If the assembled data exceeds this, prunable layers will be dropped. Production defaults are loaded from config.yaml.",
                "Temporary budget passed through visual_config(); production defaults still live in config.yaml.",
            ),
        )
        high_conf = st.slider(
            "context.coreference_high_confidence",
            0.0,
            1.0,
            float(defaults["coreference_high_confidence"]),
            help=two_section_help(
                "Claims with confidence at or above this threshold are treated as confirmed facts. They get promoted from provisional guesses to canonical truth for the current drafting pass.",
                "High-band threshold used to promote provisional claims to confirmed facts.",
            ),
        )
        mid_conf = st.slider(
            "context.coreference_mid_confidence",
            0.0,
            high_conf,
            min(float(defaults["coreference_mid_confidence"]), float(high_conf)),
            help=two_section_help(
                "Claims with confidence between mid and high thresholds are treated as plausible beliefs — worth including but not fully trusted. Claims below mid are ignored. The slider is capped to stay below the high threshold.",
                "Mid-band threshold used to retain provisional claims as unconfirmed beliefs.",
            ),
        )
        tokenizer_family = st.selectbox(
            "drafter.tokenizer_family",
            ["char_heuristic", "tiktoken"],
            help=two_section_help(
                "Which tokenizer to use when measuring per-layer token counts for budgeting. 'char_heuristic' is fast and always available; 'tiktoken' is more accurate but requires the model data.",
                "Tokenizer family used for per-layer sizing in the assembled package.",
            ),
        )
    with pointer_col:
        arc_id = st.text_input(
            "arc_id", "arc-1",
            help=two_section_help(
                "The story arc the pointer is positioned at. The relational layer reads data filtered to this arc.",
                "Pointer arc_id read by the relational layer.",
            ),
        )
        chapter_id = st.text_input(
            "chapter_id", "chapter-1",
            help=two_section_help(
                "The chapter within the arc. The relational layer reads data filtered to this chapter.",
                "Pointer chapter_id read by the relational layer.",
            ),
        )
        scene_id = st.text_input(
            "scene_id", "scene-1",
            help=two_section_help(
                "The scene within the chapter. Beat data and PAD helpers read from this scene.",
                "Pointer scene_id read by beat and PAD helpers.",
            ),
        )
        beat_index = st.number_input(
            "beat_index",
            min_value=0,
            value=0,
            step=1,
            help=two_section_help(
                "The current beat index within the scene. Used to determine which beat is the 'current' one for the drafting pass.",
                "Pointer beat_index used to identify the current beat within the scene.",
            ),
        )
    with run_col:
        output_label(
            "Real function call",
            two_section_help(
                "This button constructs an FSM_Pointer and a config-shaped object, then calls the real build_context_package() function — the same one used in production.",
                "The button constructs FSM_Pointer and calls fsm.nodes.node_assemble_context.build_context_package().",
            ),
        )
        if st.button(
            "Run build_context_package",
            type="primary",
            width='stretch',
            help=two_section_help(
                "Executes the full context assembly pipeline with the current settings. The result is stored in the session and displayed below.",
                "Executes context assembly with the current temp store paths and config-shaped controls.",
            ),
        ):
            try:
                config = visual_config(
                    token_budget=int(token_budget),
                    high_confidence=float(high_conf),
                    mid_confidence=float(mid_conf),
                    tokenizer_family=tokenizer_family,
                )
                state = {
                    "fsm_pointer": FSM_Pointer(
                        arc_id=arc_id,
                        chapter_id=chapter_id,
                        scene_id=scene_id,
                        beat_index=int(beat_index),
                    ),
                    "app_config": config,
                    "sqlite_db_path": str(paths["db"]),
                    "provisional_store_path": str(paths["provisional"]),
                }
                st.session_state.m06_package = build_context_package(state)
            except Exception as exc:  # noqa: BLE001
                st.error(f"{type(exc).__name__}: {exc}")

    package = st.session_state.get("m06_package")
    if not package:
        st.info("Seed the stores, then run context assembly.")
        return

    _render_metrics(package)
    _render_intermediate_steps(package)
    display_package_layers(package)


def _render_metrics(package: dict) -> None:
    section_header(
        "Assembly summary",
        two_section_help(
            "Four key metrics that tell you whether the context package fits within budget. Initial tokens are the total before any pruning; final tokens are what's left after dropping layers; budget is the cap; and 'over budget' warns if even the bare minimum data exceeds the limit.",
            "High-level token budget result from the latest context package.",
        ),
    )
    meta = package["meta"]
    sizing = meta["token_sizing"]
    cols = st.columns(4)
    cols[0].metric(
        "Initial tokens",
        meta.get("initial_token_total"),
        help=two_section_help(
            "Total tokens before pruning and final coreference classification. This is the raw assembled size before the budget enforcement pass.",
            "Token total before pruning and final coreference classification.",
        ),
    )
    cols[1].metric(
        "Final tokens",
        meta.get("final_token_total"),
        help=two_section_help(
            "Total tokens after pruning less-important layers and applying deterministic coreference tiering. This is what the model would actually see.",
            "Token total after pruning and deterministic coreference tiering.",
        ),
    )
    cols[2].metric(
        "Budget",
        sizing.get("context_token_budget"),
        help=two_section_help(
            "The budget value that was passed into this run. This comes from the config-shaped object, which reads defaults from config.yaml.",
            "Budget value passed into this run's config-shaped object.",
        ),
    )
    cols[3].metric(
        "Over budget",
        str(meta.get("over_budget")),
        help=two_section_help(
            "True means even the essential relational data (which can't be pruned) still exceeds the token budget. This signals that the budget may be too tight for the current story size.",
            "True when retained relational truth still exceeds the configured budget.",
        ),
    )


def _render_intermediate_steps(package: dict) -> None:
    meta = package["meta"]
    section_header(
        "Intermediate steps",
        two_section_help(
            "Details recorded during context assembly: how many tokens each layer uses, which layers were pruned to fit the budget, which stores were available, and how provisional claims were tiered into high/mid/low confidence bands.",
            "Budgeting, layer availability, and coreference-tier details recorded in package meta.",
        ),
    )
    col_a, col_b = st.columns(2)
    with col_a:
        output_label(
            "Layer token counts",
            two_section_help(
                "A bar chart showing how many tokens each layer of the context package consumed. Layers with taller bars took up more context space.",
                "Per-layer count from llm.tokenizer.count_tokens().",
            ),
        )
        st.bar_chart(meta["token_sizing"]["layers"])
        output_label(
            "Pruning decisions",
            two_section_help(
                "Which layers were dropped (pruned) to fit within the token budget. Layers are dropped in order of priority — less critical ones first.",
                "Drop-order decisions applied to fit the token budget.",
            ),
        )
        st.json(meta.get("pruned_layers", []))
    with col_b:
        output_label(
            "Layer availability",
            two_section_help(
                "Shows which data stores were readable (available) and which ones degraded gracefully to 'unavailable'. Not all stores are implemented yet — deferred modules show as unavailable without crashing.",
                "Shows which stores were readable and which deferred stores degraded to unavailable.",
            ),
        )
        st.json(
            {
                key: meta[key]
                for key in (
                    "relational",
                    "summaries",
                    "temporal",
                    "flavour",
                    "coreference_candidates",
                    "macro_constraints",
                )
            }
        )
        output_label(
            "Coreference tiering",
            two_section_help(
                "Counts and thresholds from the high/mid/low provisional-claim classification. Shows how claims were split across confidence bands using the thresholds you configured.",
                "Counts and thresholds from high/mid/low provisional-claim classification.",
            ),
        )
        st.json(meta.get("coreference_resolution", {}))


def main() -> None:
    st.set_page_config(page_title="M06 Context Assembly", layout="wide")
    render()


if __name__ == "__main__":
    main()