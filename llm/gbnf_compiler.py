"""Module: M04 (LLM Inference Boundary)
Compile strict JSON Schema fragments into deterministic GBNF grammars.

The compiler is provider-neutral: it only turns a schema into a grammar string.
Endpoint-specific request fields are chosen later by the inference boundary.
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from typing import Any

_ANNOTATION_KEYS = {"description", "examples", "title", "default"}
_ROOT_RULE = "root"
_ROOT_VALUE_RULE = "root_value"


def json_schema_to_gbnf(schema: dict) -> str:
    """Compile a supported JSON Schema subset into a deterministic GBNF string."""
    if not isinstance(schema, dict):
        raise TypeError("schema must be a dictionary")
    return _SchemaCompiler(schema).compile()


def _json_schema_to_gbnf(schema: dict) -> str:
    """Compatibility wrapper for the design's private helper name."""
    return json_schema_to_gbnf(schema)


class _SchemaCompiler:
    def __init__(self, schema: dict[str, Any]) -> None:
        self._schema = schema
        self._rules: dict[str, str] = {}

    def compile(self) -> str:
        value_rule = self._compile_schema(self._schema, [_ROOT_VALUE_RULE])
        lines = [f"{_ROOT_RULE} ::= ws {value_rule} ws"]
        lines.extend(f"{name} ::= {body}" for name, body in self._rules.items())
        lines.extend(_builtin_rules())
        return "\n".join(lines) + "\n"

    def _compile_schema(self, schema: dict[str, Any], path: list[str]) -> str:
        if not isinstance(schema, dict):
            raise ValueError(f"unsupported schema branch at {'.'.join(path)}: not an object")

        if "$ref" in schema:
            self._reject_ref_siblings(schema, path)
            return self._compile_schema(self._resolve_ref(schema["$ref"], path), path)

        if "allOf" in schema:
            raise ValueError("unsupported schema feature: allOf")
        if "not" in schema:
            raise ValueError("unsupported schema feature: not")
        if "prefixItems" in schema:
            raise ValueError("unsupported schema feature: prefixItems")
        if "patternProperties" in schema:
            raise ValueError("unsupported schema feature: patternProperties")

        if "anyOf" in schema or "oneOf" in schema:
            union_key = "anyOf" if "anyOf" in schema else "oneOf"
            self._reject_unknown_keys(schema, {union_key}, path)
            return self._compile_union(schema[union_key], path)

        if "enum" in schema:
            self._reject_unknown_keys(schema, {"enum", "type"}, path)
            return self._compile_enum(schema["enum"], path)

        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            self._reject_unknown_keys(schema, {"type"}, path)
            return self._compile_union([{"type": item} for item in schema_type], path)
        if schema_type == "object" or "properties" in schema:
            return self._compile_object(schema, path)
        if schema_type == "array":
            return self._compile_array(schema, path)
        if schema_type == "string":
            self._reject_unknown_keys(schema, {"type"}, path)
            return "string"
        if schema_type == "integer":
            self._reject_unknown_keys(schema, {"type"}, path)
            return "integer"
        if schema_type == "number":
            self._reject_unknown_keys(schema, {"type"}, path)
            return "number"
        if schema_type == "boolean":
            self._reject_unknown_keys(schema, {"type"}, path)
            return "boolean"
        if schema_type == "null":
            self._reject_unknown_keys(schema, {"type"}, path)
            return "null"

        raise ValueError(f"unsupported schema type at {'.'.join(path)}: {schema_type!r}")

    def _compile_object(self, schema: dict[str, Any], path: list[str]) -> str:
        self._reject_unknown_keys(
            schema,
            {"type", "properties", "required", "additionalProperties"},
            path,
        )
        if schema.get("additionalProperties") is not False:
            raise ValueError("unsupported schema feature: additionalProperties must be false")

        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError("unsupported schema feature: properties must be an object")

        required = schema.get("required", [])
        if not isinstance(required, list):
            raise ValueError("unsupported schema feature: required must be an array")
        required_names = set(required)
        unknown_required = required_names - set(properties)
        if unknown_required:
            unknown = ", ".join(sorted(unknown_required))
            raise ValueError(f"unsupported schema feature: required field missing property: {unknown}")

        pair_rules = []
        for property_name, property_schema in properties.items():
            child_rule = self._compile_schema(
                property_schema, [*path, _sanitize_rule_part(property_name)]
            )
            pair_rules.append((property_name, _json_string_literal(property_name), child_rule))

        object_rule = _rule_name(path)
        alternatives = []
        for included in _property_sequences(pair_rules, required_names):
            if not included:
                alternatives.append(f"{_literal('{')} ws {_literal('}')}")
                continue
            fragments = [
                f"{prop_literal} ws {_literal(':')} ws {child_rule}"
                for _name, prop_literal, child_rule in included
            ]
            body = f" ws {_literal(',')} ws ".join(fragments)
            alternatives.append(f"{_literal('{')} ws {body} ws {_literal('}')}")
        self._rules[object_rule] = " | ".join(alternatives)
        return object_rule

    def _compile_array(self, schema: dict[str, Any], path: list[str]) -> str:
        self._reject_unknown_keys(schema, {"type", "items"}, path)
        if "items" not in schema:
            raise ValueError("unsupported schema feature: array without homogeneous items")

        item_rule = self._compile_schema(schema["items"], [*path, "item"])
        array_rule = _rule_name(path)
        self._rules[array_rule] = (
            f"{_literal('[')} ws ({item_rule} (ws {_literal(',')} ws {item_rule})*)? "
            f"ws {_literal(']')}"
        )
        return array_rule

    def _compile_union(self, branches: Any, path: list[str]) -> str:
        if not isinstance(branches, list) or not branches:
            raise ValueError("unsupported schema feature: empty union")
        union_rule = _rule_name(path)
        branch_rules = [
            self._compile_schema(branch, [*path, f"option_{index}"])
            for index, branch in enumerate(branches)
        ]
        self._rules[union_rule] = " | ".join(branch_rules)
        return union_rule

    def _compile_enum(self, values: Any, path: list[str]) -> str:
        if not isinstance(values, list) or not values:
            raise ValueError("unsupported schema feature: empty enum")
        enum_rule = _rule_name(path)
        self._rules[enum_rule] = " | ".join(_literal(json.dumps(value)) for value in values)
        return enum_rule

    def _resolve_ref(self, ref: Any, path: list[str]) -> dict[str, Any]:
        if not isinstance(ref, str) or not ref.startswith("#/"):
            raise ValueError(f"unsupported schema feature: external $ref at {'.'.join(path)}")
        target: Any = self._schema
        for raw_part in ref[2:].split("/"):
            part = raw_part.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, dict) or part not in target:
                raise ValueError(f"unsupported schema feature: unresolved $ref {ref!r}")
            target = target[part]
        if not isinstance(target, dict):
            raise ValueError(f"unsupported schema feature: non-object $ref target {ref!r}")
        return target

    def _reject_ref_siblings(self, schema: dict[str, Any], path: list[str]) -> None:
        allowed = {"$ref"} | _ANNOTATION_KEYS
        unsupported = set(schema) - allowed
        if unsupported:
            feature = sorted(unsupported)[0]
            raise ValueError(f"unsupported schema feature beside $ref at {'.'.join(path)}: {feature}")

    def _reject_unknown_keys(
        self, schema: dict[str, Any], supported: set[str], path: list[str]
    ) -> None:
        allowed = supported | _ANNOTATION_KEYS | {"$defs", "definitions"}
        unsupported = set(schema) - allowed
        if unsupported:
            feature = sorted(unsupported)[0]
            raise ValueError(f"unsupported schema feature at {'.'.join(path)}: {feature}")


def _property_sequences(
    pair_rules: list[tuple[str, str, str]], required_names: set[str]
) -> list[list[tuple[str, str, str]]]:
    required = [pair for pair in pair_rules if pair[0] in required_names]
    optional = [pair for pair in pair_rules if pair[0] not in required_names]
    sequences = []
    for count in range(len(optional) + 1):
        for optional_subset in combinations(optional, count):
            included_names = {pair[0] for pair in optional_subset} | required_names
            sequences.append([pair for pair in pair_rules if pair[0] in included_names])
    return sequences


def _rule_name(path: list[str]) -> str:
    return "_".join(_sanitize_rule_part(part) for part in path)


def _sanitize_rule_part(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_").lower()
    if not normalized:
        return "value"
    if normalized[0].isdigit():
        return f"field_{normalized}"
    return normalized


def _json_string_literal(value: str) -> str:
    return _literal(json.dumps(value))


def _literal(value: str) -> str:
    return json.dumps(value)


def _builtin_rules() -> list[str]:
    return [
        r'ws ::= [ \t\n\r]*',
        r'string ::= "\"" string_char* "\""',
        r'string_char ::= [^"\\\x00-\x1f] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)',
        r'hex ::= [0-9a-fA-F]',
        r'integer ::= "-"? ("0" | [1-9] [0-9]*)',
        r'number ::= integer ("." [0-9]+)? ([eE] [-+]? [0-9]+)?',
        r'boolean ::= "true" | "false"',
        r'null ::= "null"',
    ]
