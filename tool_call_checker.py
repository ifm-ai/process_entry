"""Strict tool-call checker.

Validates a tool call against a list of declared tool definitions using
default JSON Schema semantics: name in declared set, required args present,
no extra args, types match exactly (with `number` accepting int|float, and
three nullability spellings honored — OpenAPI 3.0 `nullable: true`, `null`
listed in `enum`, and the JSON literal `null` inside a `type` list),
enum membership enforced when declared. Recurses into nested object/array
schemas.

Public API:
    check_tool_call(tool_call, tool_defs) -> (ok: bool, errors: list[str])

Tool definitions follow the Anthropic-style shape used by this corpus:
    {"name": ..., "description": ..., "parameters": {<JSON Schema>}}
"""
from __future__ import annotations

from typing import Any

# Map JSON Schema "type" string -> tuple of acceptable Python types.
_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
    "array": (list,),
    "object": (dict,),
}


def _type_name(value: Any) -> str:
    return type(value).__name__


def _enum_contains(value: Any, choices: list) -> bool:
    """JSON-equality membership check.

    Python treats True == 1 and False == 0 (bool subclasses int), but JSON
    treats `true` and `1` as distinct values. Reject cross-type matches so
    enum membership lines up with what the wire format actually carried.
    """
    for c in choices:
        if isinstance(value, bool) != isinstance(c, bool):
            continue
        if value == c:
            return True
    return False


def _check_enum(value: Any, schema: dict, path: str) -> list[str]:
    """Verify enum membership when the schema declares `enum`.

    Treated as a discrete check from `type` so it can be applied uniformly:
    on typed schemas (after type passes) AND on untyped schemas (where
    `enum` may still be the only constraint).
    """
    if "enum" in schema and not _enum_contains(value, schema["enum"]):
        return [f"{path}: value {value!r} not in enum {schema['enum']!r}"]
    return []


def _check_any_of(value: Any, variants: Any, path: str) -> list[str]:
    """Pass if at least one variant matches. JSON Schema semantics."""
    if not isinstance(variants, list) or not variants:
        return [f"{path}: anyOf must be a non-empty list"]
    for sub in variants:
        if not _check_type(value, sub, path):
            return []
    return [f"{path}: value of type {_type_name(value)} matches no variant of anyOf"]


def _check_one_of(value: Any, variants: Any, path: str) -> list[str]:
    """Pass iff exactly one variant matches. JSON Schema semantics."""
    if not isinstance(variants, list) or not variants:
        return [f"{path}: oneOf must be a non-empty list"]
    matched = 0
    for sub in variants:
        if not _check_type(value, sub, path):
            matched += 1
            if matched > 1:
                break
    if matched == 1:
        return []
    if matched == 0:
        return [f"{path}: value of type {_type_name(value)} matches no variant of oneOf"]
    return [f"{path}: value matches multiple variants of oneOf (must match exactly one)"]


def _check_all_of(value: Any, variants: Any, path: str) -> list[str]:
    """Pass iff every variant matches."""
    if not isinstance(variants, list):
        return [f"{path}: allOf must be a list"]
    errors: list[str] = []
    for i, sub in enumerate(variants):
        errors.extend(_check_type(value, sub, f"{path}/allOf[{i}]"))
    return errors


def _check_type(value: Any, schema: dict, path: str) -> list[str]:
    """Validate value against a JSON Schema fragment. Returns list of errors."""
    if not isinstance(schema, dict):
        return []  # malformed schema — don't pretend to validate

    # Union keywords are independent assertions per JSON Schema spec — all
    # present ones must pass alongside any sibling `type`/`enum` constraints.
    union_errors: list[str] = []
    if "anyOf" in schema:
        union_errors.extend(_check_any_of(value, schema["anyOf"], path))
    if "oneOf" in schema:
        union_errors.extend(_check_one_of(value, schema["oneOf"], path))
    if "allOf" in schema:
        union_errors.extend(_check_all_of(value, schema["allOf"], path))

    expected = schema.get("type")
    if expected is None:
        # Untyped schemas have no type constraint, but enum may still apply.
        return union_errors + _check_enum(value, schema, path)

    # `type` may be a list of allowed types (JSON Schema union).
    expected_list = expected if isinstance(expected, list) else [expected]

    # Honor three nullability spellings: OpenAPI 3.0 `nullable: true`,
    # JSON-Schema-style `null` listed in `enum`, and the JSON literal `null`
    # appearing inside the `type` list (e.g. `{"type": ["string", null]}`).
    # All three add `"null"` to the accepted-types list, which picks up
    # NoneType via _TYPE_MAP["null"].
    nullable = (
        schema.get("nullable") is True
        or None in (schema.get("enum") or [])
        or None in expected_list
    )
    expected_list = [t for t in expected_list if t is not None]
    if nullable:
        expected_list = expected_list + ["null"]

    # Reject schemas whose `type` entries aren't strings (e.g. {"type": [{"$ref": ...}]}).
    # Without this, _TYPE_MAP.get(t) raises TypeError on unhashable values.
    # Null entries are handled above; anything else non-string is genuinely malformed.
    for t in expected_list:
        if not isinstance(t, str):
            return union_errors + [f"{path}: schema.type entry is {_type_name(t)}, expected string"]

    accepted: tuple[type, ...] = ()
    for t in expected_list:
        accepted += _TYPE_MAP.get(t, ())
    if not accepted:
        return union_errors  # unknown type names — be permissive

    # bool is a subclass of int in Python; reject bool when only int/number expected.
    if isinstance(value, bool) and bool not in accepted:
        if not any(t is bool for t in accepted):
            return union_errors + [f"{path}: expected {expected}, got bool"]

    if not isinstance(value, accepted):
        return union_errors + [f"{path}: expected {expected}, got {_type_name(value)}"]

    errors: list[str] = union_errors + _check_enum(value, schema, path)
    if isinstance(value, dict) and "object" in expected_list:
        props = schema.get("properties", {}) or {}
        if not isinstance(props, dict):
            errors.append(f"{path}: schema.properties is {_type_name(props)}, expected dict")
            return errors
        required = schema.get("required", [])
        if required is None:
            required = []
        if not isinstance(required, list):
            errors.append(f"{path}: schema.required is {_type_name(required)}, expected list")
            return errors
        for k in required:
            if k not in value:
                errors.append(f"{path}.{k}: required key missing")
        # Strict-by-default at every level: extras flagged unless the schema
        # explicitly opts out via `additionalProperties: true`.
        additional = schema.get("additionalProperties", False)
        for k, v in value.items():
            if k in props:
                errors.extend(_check_type(v, props[k], f"{path}.{k}"))
            elif additional is False:
                errors.append(f"{path}.{k}: extra key not in schema")
        # if additionalProperties is True or a schema, we don't enforce extras here

    if isinstance(value, list) and "array" in expected_list:
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for i, item in enumerate(value):
                errors.extend(_check_type(item, items_schema, f"{path}[{i}]"))

    return errors


def _normalize_tool_def(tool_def: Any) -> dict | None:
    """Return {'name': str, 'parameters': dict} or None if malformed."""
    if not isinstance(tool_def, dict):
        return None
    # Nemotron-style {"function": {...}} wrapper
    if "function" in tool_def and isinstance(tool_def["function"], dict):
        tool_def = tool_def["function"]
    name = tool_def.get("name")
    params = tool_def.get("parameters", {}) or {}
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(params, dict):
        return None
    return {"name": name, "parameters": params}


def check_tool_call(
    tool_call: dict, tool_defs: list[dict]
) -> tuple[bool, list[str]]:
    """Strict-validate a tool call against declared tools.

    Returns (True, []) when valid, otherwise (False, [error_strings]).
    """
    errors: list[str] = []

    if not isinstance(tool_call, dict):
        return False, [f"tool_call is {_type_name(tool_call)}, expected dict"]

    name = tool_call.get("name")
    args = tool_call.get("arguments")

    if not isinstance(name, str) or not name:
        errors.append("tool_call.name is empty or missing")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        errors.append(f"tool_call.arguments is {_type_name(args)}, expected dict")
        return False, errors  # can't validate args further

    # Build name -> normalized def map
    name_to_def: dict[str, dict] = {}
    for td in tool_defs or []:
        nd = _normalize_tool_def(td)
        if nd is not None:
            name_to_def[nd["name"]] = nd

    if name not in name_to_def:
        errors.append(f"tool name {name!r} not in declared tools")
        return False, errors  # without a schema, further checks are meaningless

    schema = name_to_def[name]["parameters"]
    properties = schema.get("properties", {}) or {}
    if not isinstance(properties, dict):
        errors.append(f"schema.properties is {_type_name(properties)}, expected dict")
        return False, errors
    required = schema.get("required", [])
    if required is None:
        required = []
    if not isinstance(required, list):
        errors.append(f"schema.required is {_type_name(required)}, expected list")
        return False, errors
    # Strict-by-default: extras are flagged unless the schema explicitly opts
    # out via `additionalProperties: true`. Note this diverges from the JSON
    # Schema spec default (true) — this checker is the production-gateway for
    # SFT data, and we'd rather reject than accept silently.
    additional = schema.get("additionalProperties", False)

    for key in required:
        if key not in args:
            errors.append(f"missing required arg: {key}")

    if additional is False:
        for key in args.keys():
            if key not in properties:
                errors.append(f"extra arg not in schema: {key}")

    for key, value in args.items():
        if key in properties:
            errors.extend(_check_type(value, properties[key], key))

    return (len(errors) == 0), errors
