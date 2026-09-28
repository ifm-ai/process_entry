"""Test suite for tool_call_checker.py — focus on union (anyOf/oneOf/allOf) handling.

Pre-existing single-type behavior is already covered indirectly by
test_process_entry.py; this file targets the union keywords that
process_entry routes through check_tool_call.

Run directly:  python test_tool_call_checker.py
"""
from tool_call_checker import check_tool_call


passed = 0
failed = 0


def run_test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}{(': ' + detail) if detail else ''}")
        failed += 1


def call(name, args):
    return {"name": name, "arguments": args}


def tool(name, props, required=None):
    return {
        "name": name,
        "parameters": {
            "properties": props,
            "required": required or [],
        },
    }


# -----------------------------------------------------------------------------
# anyOf
# -----------------------------------------------------------------------------
print("Test group: anyOf")

ANYOF_STRING_NULL = {"anyOf": [{"type": "string"}, {"type": "null"}]}
t = tool("foo", {"x": ANYOF_STRING_NULL})

ok, errs = check_tool_call(call("foo", {"x": "hello"}), [t])
run_test("anyOf[string,null] accepts a string", ok, str(errs))

ok, errs = check_tool_call(call("foo", {"x": None}), [t])
run_test("anyOf[string,null] accepts null", ok, str(errs))

ok, errs = check_tool_call(call("foo", {"x": 42}), [t])
run_test("anyOf[string,null] rejects integer", not ok and any("anyOf" in e for e in errs), str(errs))

ok, errs = check_tool_call(call("foo", {"x": [1, 2]}), [t])
run_test("anyOf[string,null] rejects list", not ok, str(errs))

# anyOf where one variant has an enum
ANYOF_INT_ENUMSTR = {"anyOf": [
    {"type": "integer"},
    {"type": "string", "enum": ["low", "high"]},
]}
t = tool("foo", {"priority": ANYOF_INT_ENUMSTR})

ok, _ = check_tool_call(call("foo", {"priority": 5}), [t])
run_test("anyOf[int, enum-string] accepts integer", ok)

ok, _ = check_tool_call(call("foo", {"priority": "low"}), [t])
run_test("anyOf[int, enum-string] accepts in-enum string", ok)

ok, errs = check_tool_call(call("foo", {"priority": "medium"}), [t])
run_test("anyOf[int, enum-string] rejects out-of-enum string", not ok, str(errs))

# Empty / malformed anyOf
ok, errs = check_tool_call(call("foo", {"x": "hi"}),
                            [tool("foo", {"x": {"anyOf": []}})])
run_test("empty anyOf list is reported", not ok and any("anyOf" in e for e in errs), str(errs))

ok, errs = check_tool_call(call("foo", {"x": "hi"}),
                            [tool("foo", {"x": {"anyOf": "not a list"}})])
run_test("non-list anyOf is reported", not ok, str(errs))

print()

# -----------------------------------------------------------------------------
# oneOf
# -----------------------------------------------------------------------------
print("Test group: oneOf")

# Two disjoint primitive variants — straightforward
ONEOF_STR_INT = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
t = tool("foo", {"x": ONEOF_STR_INT})

ok, _ = check_tool_call(call("foo", {"x": "hi"}), [t])
run_test("oneOf[string|integer] accepts string", ok)

ok, _ = check_tool_call(call("foo", {"x": 7}), [t])
run_test("oneOf[string|integer] accepts integer", ok)

ok, errs = check_tool_call(call("foo", {"x": True}), [t])
# bool is rejected by integer variant (bool not in accepted unless explicit), and not string
run_test("oneOf[string|integer] rejects bool", not ok, str(errs))

# Two overlapping object variants — the case the colleague raised.
# Note: this checker's policy is `additionalProperties=False` by default
# (stricter than the JSON Schema default of True). So {email, phone}
# fails both variants for "extra key" reasons, before oneOf even gets
# to count matches. Below we test both the strict-default policy AND
# the JSON-Schema-default `additionalProperties=true` form so we see the
# "matches multiple" branch fire.
CONTACT_ONEOF_STRICT = {"oneOf": [
    {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]},
    {"type": "object", "properties": {"phone": {"type": "string"}}, "required": ["phone"]},
]}
t = tool("notify", {"contact": CONTACT_ONEOF_STRICT})

ok, _ = check_tool_call(call("notify", {"contact": {"email": "a@x.com"}}), [t])
run_test("oneOf[email|phone] (strict) accepts email-only object", ok)

ok, _ = check_tool_call(call("notify", {"contact": {"phone": "+1-555-0100"}}), [t])
run_test("oneOf[email|phone] (strict) accepts phone-only object", ok)

ok, errs = check_tool_call(
    call("notify", {"contact": {"email": "a@x.com", "phone": "+1-555-0100"}}), [t])
# Under strict additionalProperties=False, each variant rejects the value
# for having an extra key, so oneOf says "no variant matches" (rather
# than "matches multiple"). Either error is a correct rejection.
run_test("oneOf[email|phone] (strict) rejects both-keys object", not ok, str(errs))

ok, errs = check_tool_call(call("notify", {"contact": {"sms": "+1"}}), [t])
run_test("oneOf[email|phone] (strict) rejects object matching NEITHER variant",
         not ok, str(errs))

# Now the JSON-Schema-default-style variants (additionalProperties=true).
# {email, phone} is admissible to both variants, so oneOf-strict
# rejects it as "matches multiple".
CONTACT_ONEOF_PERMISSIVE = {"oneOf": [
    {"type": "object", "properties": {"email": {"type": "string"}},
     "required": ["email"], "additionalProperties": True},
    {"type": "object", "properties": {"phone": {"type": "string"}},
     "required": ["phone"], "additionalProperties": True},
]}
t = tool("notify", {"contact": CONTACT_ONEOF_PERMISSIVE})

ok, errs = check_tool_call(
    call("notify", {"contact": {"email": "a@x.com", "phone": "+1-555-0100"}}), [t])
run_test("oneOf[email|phone] (permissive) rejects both-keys with 'matches multiple'",
         not ok and any("multiple variants" in e for e in errs), str(errs))

ok, _ = check_tool_call(call("notify", {"contact": {"email": "a@x.com"}}), [t])
run_test("oneOf[email|phone] (permissive) still accepts email-only", ok)

print()

# -----------------------------------------------------------------------------
# allOf
# -----------------------------------------------------------------------------
print("Test group: allOf")

# Combine type-string and enum membership via allOf
ALLOF_TYPED_ENUM = {"allOf": [
    {"type": "string"},
    {"enum": ["a", "b", "c"]},
]}
t = tool("foo", {"x": ALLOF_TYPED_ENUM})

ok, _ = check_tool_call(call("foo", {"x": "a"}), [t])
run_test("allOf[string, enum-abc] accepts in-enum string", ok)

ok, errs = check_tool_call(call("foo", {"x": "d"}), [t])
run_test("allOf[string, enum-abc] rejects out-of-enum string", not ok, str(errs))

ok, errs = check_tool_call(call("foo", {"x": 1}), [t])
run_test("allOf[string, enum-abc] rejects wrong type", not ok, str(errs))

# Non-list allOf is malformed
ok, errs = check_tool_call(call("foo", {"x": "hi"}),
                            [tool("foo", {"x": {"allOf": "nope"}})])
run_test("non-list allOf is reported", not ok, str(errs))

print()

# -----------------------------------------------------------------------------
# Union nested in array items
# -----------------------------------------------------------------------------
print("Test group: nested unions in array items")

NESTED = {
    "type": "array",
    "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
}
t = tool("foo", {"vals": NESTED})

ok, _ = check_tool_call(call("foo", {"vals": ["a", 1, "b", 2]}), [t])
run_test("array[anyOf[str|int]] accepts mixed valid items", ok)

ok, errs = check_tool_call(call("foo", {"vals": ["a", 1, True]}), [t])
run_test("array[anyOf[str|int]] rejects bool item", not ok, str(errs))

print()

# -----------------------------------------------------------------------------
# Union + sibling `type` — both must hold (JSON Schema independence)
# -----------------------------------------------------------------------------
print("Test group: union alongside sibling type/enum")

# `type: integer` AND `anyOf: [enum 1..5, enum 10..15]`
SIBLING = {
    "type": "integer",
    "anyOf": [{"enum": [1, 2, 3, 4, 5]}, {"enum": [10, 11, 12, 13, 14, 15]}],
}
t = tool("foo", {"x": SIBLING})

ok, _ = check_tool_call(call("foo", {"x": 3}), [t])
run_test("type:int + anyOf accepts value in first enum", ok)

ok, _ = check_tool_call(call("foo", {"x": 11}), [t])
run_test("type:int + anyOf accepts value in second enum", ok)

ok, errs = check_tool_call(call("foo", {"x": 7}), [t])
run_test("type:int + anyOf rejects value in NEITHER enum", not ok, str(errs))

ok, errs = check_tool_call(call("foo", {"x": "3"}), [t])
run_test("type:int + anyOf rejects wrong base type",
         not ok and any("expected integer" in e for e in errs), str(errs))

print()

# -----------------------------------------------------------------------------
# Regression: existing type-list nullability still works
# -----------------------------------------------------------------------------
print("Test group: regression on type-list (existing handling)")

TYPELIST = {"type": ["string", "null"]}
t = tool("foo", {"x": TYPELIST})

ok, _ = check_tool_call(call("foo", {"x": "hi"}), [t])
run_test("type:[string,null] accepts string", ok)

ok, _ = check_tool_call(call("foo", {"x": None}), [t])
run_test("type:[string,null] accepts null", ok)

ok, errs = check_tool_call(call("foo", {"x": 42}), [t])
run_test("type:[string,null] rejects int (regression)",
         not ok and any("expected" in e and "got int" in e for e in errs), str(errs))

# nullable: true shorthand still works
NULLABLE = {"type": "string", "nullable": True}
t = tool("foo", {"x": NULLABLE})

ok, _ = check_tool_call(call("foo", {"x": None}), [t])
run_test("OpenAPI nullable=true accepts null (regression)", ok)

print()

# -----------------------------------------------------------------------------
# Pre-existing single-type behavior still works (sanity)
# -----------------------------------------------------------------------------
print("Test group: regression on non-union schemas (sanity)")

t = tool("foo", {"x": {"type": "string"}}, required=["x"])

ok, _ = check_tool_call(call("foo", {"x": "hi"}), [t])
run_test("non-union string accepts string", ok)

ok, errs = check_tool_call(call("foo", {"x": 1}), [t])
run_test("non-union string rejects int", not ok and "expected string" in str(errs))

ok, errs = check_tool_call(call("foo", {}), [t])
run_test("missing required arg still reported", not ok and any("required" in e for e in errs))

ok, errs = check_tool_call(call("foo", {"x": "hi", "extra": 1}), [t])
run_test("extra arg still reported (strict-by-default)", not ok and any("extra" in e for e in errs))

print()

# -----------------------------------------------------------------------------
print(f"\nTotal: {passed} passed, {failed} failed")
if failed:
    import sys
    sys.exit(1)
