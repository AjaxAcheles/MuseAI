"""Module: M04 (LLM Inference Boundary)
Streamlit frontend for tokenizer, GBNF, and structured-output helpers.
"""

from __future__ import annotations

import json

import streamlit as st

from shared import output_label, page_intro, parse_json, section_header, two_section_help
from fsm.state import FailureObject
from llm.call_llm import (
    StructuredOutputError,
    extract_first_json_object,
    validate_structured_text,
    validate_with_salvage,
)
from llm.gbnf_compiler import json_schema_to_gbnf
from llm.tokenizer import count_message_tokens, count_tokens

TOKENIZER_FAMILIES = ("char_heuristic", "tiktoken", "hf_auto")


def render() -> None:
    page_intro(
        "M04 LLM Inference Boundary",
        two_section_help(
            "This page exercises the tools MuseAI uses to talk to language models: counting tokens to stay within budget, compiling JSON schemas into GBNF grammars for structured model output, and validating that model responses parse correctly. No live LLM calls are made — everything runs locally.",
            "Runs synchronous inference-boundary helpers locally; this page never calls a live model endpoint.",
        ),
    )
    tokenizer_tab, gbnf_tab, structured_tab = st.tabs(
        ["Tokenizer pipeline", "GBNF compiler", "Structured output pipeline"]
    )
    with tokenizer_tab:
        _render_tokenizer()
    with gbnf_tab:
        _render_gbnf()
    with structured_tab:
        _render_structured()


def _render_tokenizer() -> None:
    section_header(
        "Token counting workflow",
        two_section_help(
            "Counts how many tokens a piece of text uses across three different tokenizer methods. Token counts are how the system measures context size — crucial for staying within the model's budget. The message token counter also estimates how many tokens a chat message payload uses.",
            "Counts raw text through each tokenizer family and optionally counts chat-message payloads.",
        ),
    )
    input_col, output_col = st.columns([1, 1])
    with input_col:
        text = st.text_area(
            "Text to count",
            "synthetic text for token counting",
            help=two_section_help(
                "The raw text you want to count tokens for. Paste in a sentence, a paragraph, or a whole document — each tokenizer family will report how many tokens it finds.",
                "Raw string passed to count_tokens() for each tokenizer family.",
            ),
        )
        model_name = st.text_input(
            "model_name for exact tokenizers",
            "gpt-4o",
            help=two_section_help(
                "The model name to use for tiktoken and HuggingFace tokenizers. Different models may use different tokenizers; char_heuristic ignores this completely.",
                "Passed to tiktoken and hf_auto. char_heuristic ignores it.",
            ),
        )
        messages_text = st.text_area(
            "Messages JSON",
            json.dumps([{"role": "user", "content": "synthetic message"}], indent=2),
            help=two_section_help(
                "A JSON array of chat messages (each with role and content). This is used by count_message_tokens() to estimate how many tokens a full chat payload uses — including per-message formatting overhead.",
                "Array of role/content dictionaries passed to count_message_tokens().",
            ),
        )
        message_family = st.selectbox(
            "message tokenizer family",
            TOKENIZER_FAMILIES,
            help=two_section_help(
                "Which tokenizer family to use when counting message tokens. Each family produces slightly different counts.",
                "Tokenizer family used for the message-token metric.",
            ),
        )
        run_messages = st.button(
            "Run count_message_tokens",
            help=two_section_help(
                "Parses the Messages JSON and runs count_message_tokens() to produce a single token-count metric.",
                "Parses Messages JSON and calls llm.tokenizer.count_message_tokens().",
            ),
        )
    with output_col:
        rows = []
        for family in TOKENIZER_FAMILIES:
            try:
                rows.append(
                    {
                        "family": family,
                        "tokens": count_tokens(text, family, model_name),
                        "error": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append({"family": family, "tokens": None, "error": str(exc)})
        output_label(
            "count_tokens() by family",
            two_section_help(
                "Each row shows how many tokens the same text produces with a different counting method. Compare them to see how the heuristic compares to the exact tokenizers.",
                "Each row is a real count_tokens() call over the same text.",
            ),
        )
        st.dataframe(rows, width='stretch', hide_index=True)
        output_label(
            "Heuristic note",
            two_section_help(
                "char_heuristic uses a simple formula (text length / 4) to estimate tokens quickly without loading any model data. It's a fast approximation, not an exact count — use tiktoken or hf_auto when you need precision.",
                "char_heuristic is deterministic budgeting math, not backend-exact accounting.",
            ),
        )
        st.caption("char_heuristic uses ceil(len(text) / 4); message counting adds stable wrapper overhead per message.")
        if run_messages:
            try:
                output_label(
                    "count_message_tokens()",
                    two_section_help(
                        "The estimated token count for the chat message payload, including per-message formatting tokens.",
                        "Token count for the parsed message payload.",
                    ),
                )
                st.metric(
                    "message tokens",
                    count_message_tokens(parse_json(messages_text, []), message_family, model_name),
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"{type(exc).__name__}: {exc}")


def _render_gbnf() -> None:
    section_header(
        "Schema to grammar workflow",
        two_section_help(
            "Converts a JSON Schema into a GBNF grammar — a set of rules that constrains the language model to output only valid JSON matching your schema. This is how MuseAI forces models to return structured data instead of free-form text.",
            "Compiles a supported JSON Schema subset into deterministic GBNF with json_schema_to_gbnf().",
        ),
    )
    examples = {
        "FailureObject": FailureObject.model_json_schema(),
        "Simple object": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"name": {"type": "string"}, "ok": {"type": "boolean"}},
            "required": ["name", "ok"],
        },
        "Array": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"scores": {"type": "array", "items": {"type": "number"}}},
            "required": ["scores"],
        },
        "Unsupported pattern": {"type": "string", "pattern": "^x+$"},
    }
    input_col, output_col = st.columns([1, 1])
    with input_col:
        selected = st.selectbox(
            "Example schema",
            list(examples),
            help=two_section_help(
                "Pick a pre-loaded schema to see how the compiler handles it. The 'Unsupported pattern' example demonstrates what happens when you use an unsupported JSON Schema feature like regex patterns.",
                "Preloaded schemas that demonstrate supported and unsupported compiler paths.",
            ),
        )
        schema_text = st.text_area(
            "JSON Schema",
            json.dumps(examples[selected], indent=2),
            height=320,
            help=two_section_help(
                "The JSON Schema to compile. You can edit this freely — any changes will be compiled when you click the button. The schema must use the supported subset (objects with typed properties, arrays with typed items, booleans, numbers, strings).",
                "Editable JSON Schema passed directly to json_schema_to_gbnf().",
            ),
        )
        compile_schema = st.button(
            "Compile schema",
            help=two_section_help(
                "Parses the JSON Schema text and runs json_schema_to_gbnf() to produce a grammar string.",
                "Parses the JSON and runs llm.gbnf_compiler.json_schema_to_gbnf().",
            ),
        )
    with output_col:
        if compile_schema:
            try:
                grammar = json_schema_to_gbnf(parse_json(schema_text, {}))
                output_label(
                    "Compiled GBNF",
                    two_section_help(
                        "The GBNF grammar string produced by the compiler. This grammar constrains a language model's output to match your schema exactly.",
                        "Grammar string returned by json_schema_to_gbnf().",
                    ),
                )
                st.code(grammar, language="bnf")
            except Exception as exc:  # noqa: BLE001
                st.error(f"{type(exc).__name__}: {exc}")
        else:
            output_label(
                "Compiled GBNF",
                two_section_help(
                    "Click 'Compile schema' to generate the GBNF grammar for the selected schema.",
                    "Click Compile schema to generate grammar here.",
                ),
            )
            st.info("No compile run yet.")


def _render_structured() -> None:
    section_header(
        "Structured-output validation workflow",
        two_section_help(
            "This shows the three-step pipeline for turning raw model output into validated data. Step 1 tries a strict parse (the whole text must be valid JSON matching the schema). Step 2 extracts the first balanced JSON object from the text (handling markdown fences and prose). Step 3 combines both — tries strict first, then salvages with extraction if strict fails.",
            "Shows the real strict parse, balanced-object extraction, and one-pass salvage path in order.",
        ),
    )
    valid_payload = {
        "error_code": "PACING_ISSUE",
        "offending_text": "synthetic offending text",
        "suggested_fix": "synthetic fix",
        "critic_source": "synthetic_critic",
    }
    raw = st.text_area(
        "Raw model text",
        "```json\n" + json.dumps(valid_payload, indent=2) + "\n```",
        height=220,
        help=two_section_help(
            "Paste the raw text a language model returned. It can be plain JSON, JSON wrapped in markdown code fences (```json ... ```), or prose containing JSON anywhere inside. Each of the three columns below will attempt to parse it differently.",
            "Text to validate as a FailureObject. Fenced or prose-wrapped JSON should fail strict parse but may salvage.",
        ),
    )
    strict_col, extract_col, salvage_col = st.columns(3)
    with strict_col:
        section_header(
            "1. Strict parse",
            two_section_help(
                "Requires the entire input to be a single valid JSON document that matches the FailureObject schema. If the model wrapped its JSON in markdown fences or added extra text, this will fail.",
                "validate_structured_text() requires the entire input to be one valid JSON document.",
            ),
            level=4,
        )
        try:
            st.json(validate_structured_text(raw, FailureObject).model_dump())
        except StructuredOutputError as exc:
            st.error(str(exc))
    with extract_col:
        section_header(
            "2. Extract JSON",
            two_section_help(
                "Finds the first balanced JSON object in the text, handling quoted strings with braces inside them. This can unwrap JSON from markdown fences or prose. Returns null if no balanced object is found.",
                "extract_first_json_object() finds the first balanced object while respecting quoted braces.",
            ),
            level=4,
        )
        extracted = extract_first_json_object(raw)
        if extracted is None:
            st.warning("No balanced object found.")
        else:
            st.code(extracted, language="json")
    with salvage_col:
        section_header(
            "3. Salvage parse",
            two_section_help(
                "Tries strict validation first. If that fails, it extracts the first JSON object and retries validation once. This is the production path — handles both clean and messy model output without infinite retries.",
                "validate_with_salvage() retries validation once against the extracted object.",
            ),
            level=4,
        )
        try:
            st.json(validate_with_salvage(raw, FailureObject).model_dump())
        except StructuredOutputError as exc:
            st.error(str(exc))


def main() -> None:
    st.set_page_config(page_title="M04 Inference", layout="wide")
    render()


if __name__ == "__main__":
    main()