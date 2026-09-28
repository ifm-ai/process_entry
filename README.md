# process_entry

A strict validator for conversational training data: chat, reasoning, and agentic tool-use
trajectories. Every conversation used in K2 Horizon mid- and post-training had to pass it.

`process_entry` started as a format check for mostly non-agentic SFT data. When agentic data
became our focus, we found that "well-formed JSON" is a low bar: trajectories called tools that
were never declared, used tool names that did not match the declared ones, or passed arguments
the schema did not allow; open models used for generation fell into loops, repeating the same
action over and over; a broken environment led the model to spend the rollout fixing the
environment instead of solving the task; and generated text leaked the special tokens, harness
instructions, and identity of the model that produced it. Each rule in this repository started
from a real example of such a defect that got past the previous version.

```python
from process_entry import process_entry

processed, errors, repairs = process_entry(record)
# processed: the cleaned record with token counts, or None if the record is rejected
# errors:    reasons for rejection (empty iff the record is accepted)
# repairs:   meaning-preserving normalizations that were applied
```

## Data format

One JSON object per record, with a `conversation` list of turns (`system` | `user` |
`assistant` | `tool`). Tools are declared on the system turn as `{name, description,
parameters}` with a JSON schema for `parameters`; tool calls are `{name, arguments}` with
`arguments` an object; reasoning goes in the assistant turn's `think` field.

```json
{"conversation": [
  {"role": "system", "content": "You are a helpful assistant.",
   "tools": [{"name": "get_weather", "description": "Get the current weather for a city.",
              "parameters": {"type": "object",
                             "properties": {"city": {"type": "string"},
                                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                             "required": ["city"]}}]},
  {"role": "user", "content": "What's the weather in Abu Dhabi?"},
  {"role": "assistant", "think": "I should look up the current weather.",
   "tool_calls": [{"name": "get_weather", "arguments": {"city": "Abu Dhabi", "unit": "celsius"}}]},
  {"role": "tool", "content": "{\"temperature\": 34, \"condition\": \"sunny\"}"},
  {"role": "assistant", "think": "The tool says 34 degrees and sunny.",
   "content": "It is currently 34°C and sunny in Abu Dhabi."}
]}
```

Common variants are normalized and recorded in `repairs`: `messages` instead of
`conversation`, tools as a top-level list, tools and calls in the OpenAI `function` wrapper,
arguments as a JSON string, and so on.

## Example

`python example.py` runs a valid record and a few broken variants of it:

```
== valid record
   ACCEPTED  {"token_count": 352, "token_count_answer": 49, "token_count_think": 19}

== call to an undeclared tool
   REJECTED  Turn 2: tool name 'get_forecast' not in declared tools

== argument not in the schema
   REJECTED  Turn 2: extra arg not in schema: days

== value outside the enum
   REJECTED  Turn 2: unit: value 'kelvin' not in enum ['celsius', 'fahrenheit']

== model identity leak in reasoning
   REJECTED  Turn 4: model identity leak in think (self_id_as_model): 'As ChatGPT, I'

== leaked chat-template marker
   REJECTED  Turn 4: forbidden marker '</think>' in field 'content'
   REJECTED  Turn 4: spaced/doubled template marker (think_tag) '</think>' in field 'content'

== the same tool call repeated 20 times
   REJECTED  Degenerate repetition: max_think_dup=20 (threshold 8)
   REJECTED  Degenerate repetition: identical_tool_call=20 (threshold 10)
```

## What it checks

- **Structure.** Valid order of system, user, assistant and tool turns; every tool call is
  followed by a tool response and every tool response answers a tool call; every assistant turn
  has a reply or a tool call; content types that the chat template can render.
- **Tools** (`tool_call_checker.py`). Tools used in the conversation must be declared with a
  name and a JSON schema. Every call is validated against its schema: the tool exists, required
  arguments are present, no undeclared arguments unless the schema sets
  `additionalProperties: true` (stricter than the JSON Schema default), types and enum values
  match, including nested objects, arrays, and `anyOf`/`oneOf`/`allOf`.
- **Recovering from errors** (`tool_call_recovery.py`, opt-in). Agentic data should show a
  model recovering from its own mistakes. With a `tool_error_classifier`, an invalid call is
  tolerated if the environment returned an error, the model did not repeat the same failing
  call, there were no more than `consecutive_error_limit` errors in a row, and the last turn with
  tool calls succeeded.
- **Reasoning.** With `require_think=True`, at least one assistant turn must contain reasoning.
  Reasoning is stored under the field of the requested reasoning effort (`think`, `think_fast`,
  `think_faster`).
- **Degenerate behavior** (`repetition_check.py`, `quality_checks.py`). Loops in reasoning or
  replies, the same reasoning repeated across turns, the same tool call issued again and again,
  and an agent stuck repeating the same action while its environment output never changes. Only assistant turns are checked, and length alone is
  never a defect.
- **Leakage from generation.** Special tokens of our chat template (always rejected) and
  markers of other templates and tool-call formats (gpt-oss harmony, DeepSeek, Kimi, and
  others), matched by shape so that inserted spaces, doubled delimiters, or HTML escaping do not
  hide them. Also rejected: assistant turns that refer to instructions appearing nowhere in the
  record (for example a "desired oververbosity" level, often mentioned in generations from Kimi
  K3 and DeepSeek-V4-Flash), and records whose upstream converter declares that it modified the
  data.
- **Model identity** (`identity_check.py`). The assistant claiming to be another model ("I am
  ChatGPT", "as Claude, I...") and system prompts naming the model that generated the data.
  User and tool turns are not scanned, so a user asking about another model is fine.
- **Rendering.** The record is rendered with the chat template in the presentation used in
  training, and token counts are computed (`token_count`, `token_count_answer`,
  `token_count_think`).

## Design principles

- **Reject, never repair.** Apart from meaning-preserving normalizations, which are all
  recorded in `repairs`, quality checks only reject. Silently fixing a record would hide a broken
  generator from the person who owns the data. Visible replies are never changed.
- **Strict by default.** A setting that relaxes a check has to be chosen per source.
- **Check the training target.** Content-quality checks look at what the model is trained to
  produce, i.e., assistant turns. A user pasting a repetitive log or a tool returning identical
  rows is not a defect.
- **Measure before changing a rule.** Every rule was measured on the full training corpus, and
  what it rejects was read by hand. The comments at each rule record these measurements.

The top of `process_entry.py` carries a notice for coding assistants: its rules may only be
changed if the user provides a secret word. When most of a dataset is rejected, the easiest fix
for an agent is to loosen the validator, and that is almost never the right fix.

## Settings

| argument | default | meaning |
|---|---|---|
| `reasoning_effort` | `'high'` | `'high'`, `'medium'`, or `'low'`: stores reasoning under `think`, `think_fast`, or `think_faster` |
| `require_think` | `True` | require reasoning in at least one assistant turn; set `False` for non-reasoning data |
| `identity_check` | `True` | model identity check |
| `repetition_check` | `True` | degenerate-behavior check; disable for sources where repetition is the task (e.g. ARC-AGI grids) |
| `repetition_thresholds` | `None` | override thresholds, see `repetition_check.py` |
| `tool_error_classifier` | `None` | `callable(tool_content) -> bool`; enables the recovery mode |
| `consecutive_error_limit` | `3` | maximum consecutive tool errors in the recovery mode |
| `check_chat_template_leakage_tokens` | `True` | markers of other chat templates |
| `check_extended_special_tokens` | `True` | other tool-call and reasoning markup |
| `check_harmony_special_tokens` | `True` | gpt-oss harmony tokens |
| `permissible_special_tokens` | `None` | per-source exceptions, e.g. for a task that explains a marker |
| `compute_token_counts` | `True` | if `False`, token counts must already be present in the record |
| `immutable_reasoning_effort` | `False` | keep the requested effort even for records without reasoning (see docstring) |

## Chat template

Rendering and token counts use the K2 Horizon tokenizer with the chat template used in
training, both bundled in `tokenizer/`, so nothing is downloaded. The tokenizer is identical
to the one released with the model on
[`IFM/K2-Horizon-375B-A23B`](https://huggingface.co/IFM/K2-Horizon-375B-A23B). The chat
template is not, and the difference is intentional: in training we standardize the data to
have a newline before and after the reasoning, while at inference the model already generates
those newlines itself, so the released inference template does not add anything the model did
not generate. `process_entry` checks training data, so it renders with the training template.
Set `HORIZON_TOKENIZER_PATH` to use a different tokenizer directory.

## Setup and tests

```bash
pip install -r requirements.txt
python example.py
python test_process_entry.py
python test_tool_call_checker.py
python test_identity_check.py
```

## License

Apache 2.0, see [LICENSE](LICENSE).
