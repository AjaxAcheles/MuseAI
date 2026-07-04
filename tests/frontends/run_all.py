"""Module: M02-M07 (Visualization Test Frontends)
Streamlit launcher for completed-module visualizers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure the project root is on sys.path so bare imports of sibling modules
# (shared, m02_memory, etc.) work regardless of working directory or Streamlit's
# own path setup.  This mirrors the same guard in shared.py (line 19–21).
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import two_section_help

import m02_memory
import m03_state
import m04_inference
import m05_prompt_loader
import m06_context_assembly
import m07_planning
import m08_drafting

PAGES = {
    "M02 Persistent Memory": m02_memory.render,
    "M03 Coordinator / State": m03_state.render,
    "M04 Inference Boundary": m04_inference.render,
    "M05 Prompt Loader": m05_prompt_loader.render,
    "M06 Context Assembly": m06_context_assembly.render,
    "M07 Planning Cascade": m07_planning.render,
    "M08 Drafting": m08_drafting.render,
}


def main() -> None:
    st.set_page_config(page_title="MuseAI Module Visualizers", layout="wide")
    st.sidebar.title("MuseAI visualizers")
    st.sidebar.caption("Choose a module and use the hover help on each page for context.")
    page = st.sidebar.radio(
        "Module",
        list(PAGES),
        help=two_section_help(
            "Select which module visualizer to explore. Each page imports and runs real production code against isolated temporary files — no project data is touched.",
            "Each page runs real module code against temp files, never production stores.",
        ),
    )
    st.sidebar.code("uv run streamlit run tests/frontends/run_all.py")
    PAGES[page]()


if __name__ == "__main__":
    main()
