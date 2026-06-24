"""Module: M04 (LLM Inference Boundary)
Synthetic tests for provider-neutral JSON Schema to GBNF compilation.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from llm.gbnf_compiler import _json_schema_to_gbnf, json_schema_to_gbnf


class StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


def test_object_with_required_string_and_integer_fields() -> None:
    class RequiredObject(StrictBase):
        name: str
        count: int

    grammar = json_schema_to_gbnf(RequiredObject.model_json_schema())

    assert grammar.startswith("root ::= ws root_value ws\n")
    assert '"\\"name\\""' in grammar
    assert '"\\"count\\""' in grammar
    assert "string ::=" in grammar
    assert "integer ::=" in grammar


def test_arrays_compile_with_homogeneous_items() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"scores": {"type": "array", "items": {"type": "number"}}},
        "required": ["scores"],
    }

    grammar = json_schema_to_gbnf(schema)

    assert "root_value_scores ::= " in grammar
    assert "number (ws \",\" ws number)*" in grammar


def test_enum_literals_compile_as_exact_json_literals() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {"enum": ["draft", "final"]},
            "ok": {"enum": [True, False]},
        },
        "required": ["status", "ok"],
    }

    grammar = json_schema_to_gbnf(schema)

    assert '"\\"draft\\""' in grammar
    assert '"\\"final\\""' in grammar
    assert '"true" | "false"' in grammar


def test_nullable_union_compiles_anyof() -> None:
    class NullableObject(StrictBase):
        note: str | None

    grammar = json_schema_to_gbnf(NullableObject.model_json_schema())

    assert "root_value_note ::= " in grammar
    assert "string" in grammar
    assert "null" in grammar


def test_defs_nested_objects_from_pydantic_compile() -> None:
    class Child(StrictBase):
        label: str

    class Parent(StrictBase):
        child: Child

    schema = Parent.model_json_schema()
    assert "$defs" in schema or "definitions" in schema

    grammar = json_schema_to_gbnf(schema)

    assert '"\\"child\\""' in grammar
    assert '"\\"label\\""' in grammar
    assert "root_value_child ::= " in grammar


def test_output_is_deterministic_and_private_alias_matches() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"a": {"type": "boolean"}},
        "required": ["a"],
    }

    first = json_schema_to_gbnf(schema)
    second = json_schema_to_gbnf(schema)

    assert first == second
    assert _json_schema_to_gbnf(schema) == first


def test_unsupported_schema_feature_raises_value_error() -> None:
    schema = {"type": "string", "pattern": "^a+$"}

    with pytest.raises(ValueError, match="pattern"):
        json_schema_to_gbnf(schema)
