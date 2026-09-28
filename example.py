"""Run process_entry on a few small records: one that passes and several that do not.

    python example.py
"""
import copy
import json

from process_entry import process_entry

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    },
}

VALID = {
    "conversation": [
        {"role": "system", "content": "You are a helpful assistant.", "tools": [WEATHER_TOOL]},
        {"role": "user", "content": "What's the weather in Abu Dhabi?"},
        {"role": "assistant", "think": "I should look up the current weather.",
         "tool_calls": [{"name": "get_weather", "arguments": {"city": "Abu Dhabi", "unit": "celsius"}}]},
        {"role": "tool", "content": "{\"temperature\": 34, \"condition\": \"sunny\"}"},
        {"role": "assistant", "think": "The tool says 34 degrees and sunny.",
         "content": "It is currently 34°C and sunny in Abu Dhabi."},
    ]
}


def variant(edit):
    record = copy.deepcopy(VALID)
    edit(record["conversation"])
    return record


def undeclared_tool(conv):
    conv[2]["tool_calls"][0]["name"] = "get_forecast"


def extra_argument(conv):
    conv[2]["tool_calls"][0]["arguments"]["days"] = 3


def wrong_enum_value(conv):
    conv[2]["tool_calls"][0]["arguments"]["unit"] = "kelvin"


def identity_leak(conv):
    conv[4]["think"] = "As ChatGPT, I should answer concisely. The tool says 34 degrees and sunny."


def leaked_template_marker(conv):
    conv[4]["content"] = "</think> It is currently 34°C and sunny in Abu Dhabi."


def repeated_tool_call(conv):
    call = conv[2]
    response = conv[3]
    conv[2:4] = [copy.deepcopy(t) for _ in range(20) for t in (call, response)]


EXAMPLES = [
    ("valid record", VALID),
    ("call to an undeclared tool", variant(undeclared_tool)),
    ("argument not in the schema", variant(extra_argument)),
    ("value outside the enum", variant(wrong_enum_value)),
    ("model identity leak in reasoning", variant(identity_leak)),
    ("leaked chat-template marker", variant(leaked_template_marker)),
    ("the same tool call repeated 20 times", variant(repeated_tool_call)),
]

if __name__ == "__main__":
    for name, record in EXAMPLES:
        processed, errors, repairs = process_entry(record)
        print(f"== {name}")
        if processed is not None:
            counts = {k: processed[k] for k in ("token_count", "token_count_answer", "token_count_think")}
            print(f"   ACCEPTED  {json.dumps(counts)}")
        for error in errors:
            print(f"   REJECTED  {error}")
        if repairs:
            print(f"   repairs   {json.dumps(repairs)}")
        print()
