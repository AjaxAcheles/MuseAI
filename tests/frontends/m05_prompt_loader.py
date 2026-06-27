"""Module: M04 (LLM Inference Boundary)
Streamlit frontend for the strict prompt-template loader.
"""

from __future__ import annotations

import json

import streamlit as st
from jinja2 import UndefinedError

from shared import output_label, page_intro, parse_json, reset_workspace, section_header, two_section_help, workspace
from prompts.prompt_loader import InvalidNodeNameError, PromptLoader


def render() -> None:
    paths = workspace()
    template_dir = paths["templates"]
    template_dir.mkdir(parents=True, exist_ok=True)

    page_intro(
        "M05 Prompt Loader",
        two_section_help(
            "This page lets you test PromptLoader — the module that manages Jinja2 templates for drafting prompts. You can create templates, render them with variables (and see what happens when you forget one), and test the safety validation that prevents path-traversal attacks on template names.",
            "Interactive debugger for PromptLoader; templates are written only to this session's temp prompts directory.",
        ),
    )
    cols = st.columns([1, 2])
    if cols[0].button(
        "Reset temp workspace",
        width='stretch',
        help=two_section_help(
            "Deletes all temporary templates and creates a fresh workspace. Use this when you want to clear out your test templates.",
            "Deletes temp templates and recreates a clean workspace.",
        ),
    ):
        reset_workspace()
    with cols[1]:
        output_label(
            "Template directory",
            two_section_help(
                "The temporary directory where PromptLoader looks for templates. All templates you create here live in this session-scoped folder — your real prompt templates are safe.",
                "Temporary prompt root used by PromptLoader(template_dir=...).",
            ),
        )
        st.code(str(template_dir))

    create_tab, render_tab, guard_tab = st.tabs(
        ["1. Create template", "2. Render template", "3. Validate names"]
    )
    with create_tab:
        _render_create(template_dir)
    with render_tab:
        _render_render(template_dir)
    with guard_tab:
        _render_guard(template_dir)


def _render_create(template_dir) -> None:
    section_header(
        "Create a temp .xml.j2 template",
        two_section_help(
            "Write a new Jinja2 template file into the temporary directory. The node name must follow strict rules: start with 'node_', use only lowercase ASCII letters and underscores, and not contain dots or slashes. If the name is valid, the template is saved as {node_name}.xml.j2.",
            "Validates the node name with template_name_for_node(), then writes the template file into the temp directory.",
        ),
    )
    input_col, output_col = st.columns([1, 1])
    with input_col:
        node_name = st.text_input(
            "node_name",
            "node_draft_prose",
            help=two_section_help(
                "The canonical name for this template node. Must start with 'node_' and contain only lowercase ASCII letters, digits, and underscores. No dots, slashes, or uppercase allowed.",
                "Canonical node id. Must start with node_ and contain only lowercase ASCII words.",
            ),
        )
        body = st.text_area(
            "template body",
            "<instructions><topic>{{ topic }}</topic><tone>{{ tone }}</tone></instructions>",
            height=220,
            help=two_section_help(
                "The Jinja2 template content. Use {{ variable }} syntax for placeholders that will be filled in when rendering. The template is saved exactly as written.",
                "Jinja2 XML template body written exactly to the temp file.",
            ),
        )
        if st.button(
            "Validate name and write template",
            help=two_section_help(
                "First validates the node name using PromptLoader.template_name_for_node(), then writes the template file as {node_name}.xml.j2 to the temp directory.",
                "Calls PromptLoader.template_name_for_node(), then writes {node_name}.xml.j2.",
            ),
        ):
            try:
                loader = PromptLoader(template_dir=template_dir)
                template_name = loader.template_name_for_node(node_name)
                (template_dir / template_name).write_text(body, encoding="utf-8")
                st.success(f"Wrote {template_name}.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"{type(exc).__name__}: {exc}")
    with output_col:
        output_label(
            "Current templates",
            two_section_help(
                "Lists all .xml.j2 files currently available in the temp directory. These are the templates PromptLoader can load and render.",
                "Files currently available to PromptLoader.load() in the temp directory.",
            ),
        )
        st.json(sorted(path.name for path in template_dir.glob("*.xml.j2")))


def _render_render(template_dir) -> None:
    section_header(
        "Render with StrictUndefined",
        two_section_help(
            "This tests PromptLoader.render() with StrictUndefined enabled — meaning if you forget to provide a variable that the template expects, you get an error instead of silent undefined text. This catches missing variables before they reach the model.",
            "Loads the selected temp template and renders it; missing variables fail loudly instead of becoming blanks.",
        ),
    )
    loader = PromptLoader(template_dir=template_dir)
    input_col, output_col = st.columns([1, 1])
    with input_col:
        node_name = st.text_input(
            "node_name to render",
            "node_draft_prose",
            key="render_node",
            help=two_section_help(
                "The canonical node name of the template to render. PromptLoader resolves this to the corresponding .xml.j2 file after safety validation.",
                "PromptLoader resolves this to {node_name}.xml.j2 after safety validation.",
            ),
        )
        context_text = st.text_area(
            "context JSON",
            json.dumps({"topic": "synthetic topic", "tone": "spare"}, indent=2),
            height=200,
            help=two_section_help(
                "A JSON object providing values for the template's {{ variables }}. Leave out a variable that the template expects to see StrictUndefined in action — it will raise an error instead of silently producing blank output.",
                "JSON object used as Jinja2 render context. Omit a required key to see StrictUndefined.",
            ),
        )
        run_render = st.button(
            "Run PromptLoader.render",
            help=two_section_help(
                "Calls PromptLoader.render() with the specified node name and context. If a required variable is missing, you'll see an UndefinedError.",
                "Calls PromptLoader.render(node_name, context).",
            ),
        )
    with output_col:
        output_label(
            "Rendered XML",
            two_section_help(
                "The rendered XML output from PromptLoader.render(). If rendering succeeded, this shows the completed template with all variables filled in. If a variable was missing, you'll see the error instead.",
                "Text returned by the real PromptLoader.render() call.",
            ),
        )
        if run_render:
            try:
                rendered = loader.render(node_name, parse_json(context_text, {}))
                st.code(rendered, language="xml")
            except UndefinedError as exc:
                st.error(f"UndefinedError: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"{type(exc).__name__}: {exc}")
        else:
            st.info("Create a template, then run render to see output here.")


def _render_guard(template_dir) -> None:
    section_header(
        "Name guard workflow",
        two_section_help(
            "Tests the safety validator that checks node names before loading templates. It rejects names with path traversal (dots, slashes), wrong casing, or missing the 'node_' prefix. This prevents attacks where someone tries to load a file outside the template directory.",
            "Runs template_name_for_node() over candidate names to show accepted canonical names and rejected traversal attempts.",
        ),
    )
    loader = PromptLoader(template_dir=template_dir)
    input_col, output_col = st.columns([1, 1])
    with input_col:
        names_text = st.text_area(
            "candidate names",
            "\n".join(
                [
                    "node_draft_prose",
                    "../secrets",
                    "node_draft_prose.xml.j2",
                    "Node_Draft",
                    "node_x/../../outside",
                ]
            ),
            height=220,
            help=two_section_help(
                "One candidate node name per line. The validator checks for dots, slashes, uppercase characters, and the 'node_' prefix. Lines with rejected names will show the error in the results table.",
                "One candidate node name per line. The loader rejects dots, slashes, uppercase, and missing node_ prefix.",
            ),
        )
    rows = []
    for name in [line.strip() for line in names_text.splitlines() if line.strip()]:
        try:
            rows.append(
                {
                    "node_name": name,
                    "accepted": True,
                    "template_name": loader.template_name_for_node(name),
                    "error": "",
                }
            )
        except InvalidNodeNameError as exc:
            rows.append(
                {
                    "node_name": name,
                    "accepted": False,
                    "template_name": "",
                    "error": str(exc),
                }
            )
    with output_col:
        output_label(
            "Validation results",
            two_section_help(
                "Each row shows whether the name was accepted, what template filename it resolves to, and any error message. Rejected names show why they failed so you can understand the validator's rules.",
                "Each row is the real template_name_for_node() result or InvalidNodeNameError message.",
            ),
        )
        st.dataframe(rows, width='stretch', hide_index=True)

def main() -> None:
    st.set_page_config(page_title="M05 Prompt Loader", layout="wide")
    render()


if __name__ == "__main__":
    main()