"""Optional tool-call error-recovery extension for process_entry.

The strict tool-call check in process_entry rejects every record with a
schema-failing tool call. This module's `evaluate_tool_recovery` lets some
of those failures be tolerated when:
  - the failure is matched by an error tool response (per a caller-supplied
    classifier), AND
  - the trajectory is overall healthy (gates implemented in later tasks).

Public API:
    evaluate_tool_recovery(conversation, schema_failed_calls, classifier,
                           consecutive_error_limit) -> dict

The classifier signature is `Callable[[Any], bool]` — given a tool turn's
`content` field (typically a JSON string, sometimes a parsed dict), return
True iff that content represents an error.
"""
from __future__ import annotations

from typing import Any, Callable


def _tool_response_for(conversation: list, asst_idx: int, call_idx: int):
    """Return the tool turn paired with assistant call (asst_idx, call_idx), or None.

    Positional match: the j-th tool call corresponds to conv[asst_idx + 1 + j]
    when that turn has role=='tool'. Falls back to the first tool turn
    immediately after the assistant turn if the positional pick isn't a tool.
    """
    n = len(conversation)
    cand_idx = asst_idx + 1 + call_idx
    if cand_idx < n:
        cand = conversation[cand_idx]
        if isinstance(cand, dict) and cand.get('role') == 'tool':
            return cand
    if asst_idx + 1 < n:
        cand = conversation[asst_idx + 1]
        if isinstance(cand, dict) and cand.get('role') == 'tool':
            return cand
    return None


def _next_assistant_with_tool_calls(conversation: list, after_idx: int):
    """Return (i, turn) for the next assistant turn after `after_idx` whose tool_calls
    is a non-empty list. Returns (None, None) if none."""
    for i in range(after_idx + 1, len(conversation)):
        t = conversation[i]
        if (isinstance(t, dict) and t.get('role') == 'assistant'
                and isinstance(t.get('tool_calls'), list) and t['tool_calls']):
            return i, t
    return None, None


def _args_keys(call: Any) -> frozenset:
    """Return frozenset of arg keys for a tool call (or empty if args isn't a dict)."""
    if not isinstance(call, dict):
        return frozenset()
    args = call.get('arguments')
    if not isinstance(args, dict):
        return frozenset()
    return frozenset(args.keys())


def _walk_calls_with_responses(conversation: list, classifier: Callable[[Any], bool]):
    """Yield (turn_idx, call_idx, name, status) for each tool call in linear order.

    status is 'error', 'success', or 'missing' (no tool response located).
    """
    for i, turn in enumerate(conversation):
        if not (isinstance(turn, dict) and turn.get('role') == 'assistant'):
            continue
        tcs = turn.get('tool_calls')
        if not isinstance(tcs, list):
            continue
        for ci, tc in enumerate(tcs):
            if not isinstance(tc, dict):
                continue
            name = tc.get('name')
            tr = _tool_response_for(conversation, i, ci)
            if tr is None:
                yield (i, ci, name, 'missing')
                continue
            if classifier(tr.get('content')):
                yield (i, ci, name, 'error')
            else:
                yield (i, ci, name, 'success')


def evaluate_tool_recovery(
    conversation: list,
    schema_failed_calls: list,
    classifier: Callable[[Any], bool],
    consecutive_error_limit: int = 3,
) -> dict:
    """Decide which schema-fails to tolerate and emit trajectory-level errors.

    Args:
        conversation: list of conversation turns.
        schema_failed_calls: list of (turn_idx, call_idx, schema_errors) tuples
            for every schema-failing tool call (collected by the strict checker).
        classifier: Callable[[Any], bool] — True iff content is an error.
        consecutive_error_limit: int — reject if this many consecutive tool-response errors.

    Returns dict:
        {
            'tolerated': set[(turn_idx, call_idx)] — schema-fails that recovery tolerates,
            'errors':    list[str] — trajectory-level rejection reasons,
            'audit':     {'response_errors_count': int, 'tolerated_count': int},
        }
    """
    tolerated: set = set()
    errors: list = []
    audit = {'response_errors_count': 0, 'tolerated_count': 0}

    # Rule 1: per-call tolerance. A schema-fail is tolerable iff its tool response
    # exists and the classifier says it's an error.
    for (i, ci, _) in schema_failed_calls:
        tr = _tool_response_for(conversation, i, ci)
        if tr is not None and classifier(tr.get('content')):
            tolerated.add((i, ci))
    audit['tolerated_count'] = len(tolerated)

    # Rule 2: reject if consecutive_error_limit+ consecutive tool-response errors.
    # Also accumulates response_errors_count audit.
    consecutive = 0
    streak = []
    rule2_emitted = False
    for (i, ci, _name, status) in _walk_calls_with_responses(conversation, classifier):
        if status == 'missing':
            continue  # defensive: doesn't break or extend a run
        if status == 'error':
            audit['response_errors_count'] += 1
            consecutive += 1
            streak.append((i, ci))
            if not rule2_emitted and consecutive >= consecutive_error_limit:
                positions = ', '.join(f'turn {ti} idx {ci2}' for ti, ci2 in streak)
                errors.append(
                    f'tool recovery rejected — {consecutive_error_limit}+ consecutive '
                    f'tool-response errors at calls ({positions})'
                )
                rule2_emitted = True
        else:  # success
            consecutive = 0
            streak = []

    # Rule 3: reject RETRY_SAME_TOOL_UNFIXED — failing call followed in the next
    # asst-with-tool-calls turn by a same-tool call with identical arg-key signature
    # AND that retry ALSO failed schema validation. A schema-passing retry with
    # same keys = the model adjusted a value (FIXED), not parroting.
    schema_failed_set = {(i_, ci_) for (i_, ci_, _) in schema_failed_calls}
    rule3_emitted = False
    for (i, ci, name, status) in _walk_calls_with_responses(conversation, classifier):
        if rule3_emitted or status != 'error' or not isinstance(name, str) or not name:
            continue
        orig_call = conversation[i]['tool_calls'][ci]
        orig_keys = _args_keys(orig_call)
        j, next_turn = _next_assistant_with_tool_calls(conversation, i)
        if j is None:
            continue
        for ci2, c2 in enumerate(next_turn['tool_calls']):
            if not isinstance(c2, dict) or c2.get('name') != name:
                continue
            if _args_keys(c2) != orig_keys:
                continue
            # Same key set. Was the retry's schema check still failing?
            if (j, ci2) not in schema_failed_set:
                continue  # retry schema-passes -> FIXED, not UNFIXED
            errors.append(
                f'Turn {j}: tool recovery rejected — RETRY_SAME_TOOL_UNFIXED '
                f'for tool {name!r} (same arg keys as turn {i})'
            )
            rule3_emitted = True
            break

    # Rule 4: last assistant tool-calling turn must have all-success tool responses.
    last_idx = None
    for i, turn in enumerate(conversation):
        if (isinstance(turn, dict) and turn.get('role') == 'assistant'
                and isinstance(turn.get('tool_calls'), list) and turn['tool_calls']):
            last_idx = i
    if last_idx is not None:
        last_calls = conversation[last_idx]['tool_calls']
        for ci in range(len(last_calls)):
            tr = _tool_response_for(conversation, last_idx, ci)
            if tr is None:
                continue
            if classifier(tr.get('content')):
                errors.append(
                    f'tool recovery rejected — last tool-calling turn '
                    f'(turn {last_idx}) had error responses'
                )
                break

    return {'tolerated': tolerated, 'errors': errors, 'audit': audit}


if __name__ == '__main__':
    import json

    passed = 0
    failed = 0

    def run_test(name, condition):
        global passed, failed
        if condition:
            print(f'  [PASS] {name}')
            passed += 1
        else:
            print(f'  [FAIL] {name}')
            failed += 1

    def err_classifier(content):
        """Test classifier: detects {"error": ...} shape."""
        if isinstance(content, dict):
            obj = content
        elif isinstance(content, str):
            try:
                obj = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                return False
        else:
            return False
        return isinstance(obj, dict) and bool(obj.get('error'))

    # Test H1: _tool_response_for positional match
    print('Test H1: _tool_response_for positional match')
    conv_h1 = [
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'tool_calls': [
            {'name': 'a', 'arguments': {}}, {'name': 'b', 'arguments': {}}]},
        {'role': 'tool', 'name': 'a', 'content': 'A'},
        {'role': 'tool', 'name': 'b', 'content': 'B'},
    ]
    run_test('positional 0 -> A', _tool_response_for(conv_h1, 1, 0) == conv_h1[2])
    run_test('positional 1 -> B', _tool_response_for(conv_h1, 1, 1) == conv_h1[3])
    print()

    # Test H2: _tool_response_for fallback
    print('Test H2: _tool_response_for fallback when positional misses')
    conv_h2 = [
        {'role': 'assistant', 'tool_calls': [
            {'name': 'a', 'arguments': {}}, {'name': 'b', 'arguments': {}}]},
        {'role': 'tool', 'name': 'a', 'content': 'A'},
        {'role': 'user', 'content': 'next'},
    ]
    run_test('positional miss -> first tool turn', _tool_response_for(conv_h2, 0, 1) == conv_h2[1])
    print()

    # Test H3: _tool_response_for returns None when no tool turn
    print('Test H3: _tool_response_for returns None')
    conv_h3 = [
        {'role': 'assistant', 'tool_calls': [{'name': 'a', 'arguments': {}}]},
        {'role': 'user', 'content': 'next'},
    ]
    run_test('no tool turn -> None', _tool_response_for(conv_h3, 0, 0) is None)
    print()

    # Test H4: _next_assistant_with_tool_calls
    print('Test H4: _next_assistant_with_tool_calls')
    conv_h4 = [
        {'role': 'assistant', 'tool_calls': [{'name': 'a', 'arguments': {}}]},
        {'role': 'tool', 'name': 'a', 'content': 'A'},
        {'role': 'assistant', 'content': 'just text'},
        {'role': 'user', 'content': 'go on'},
        {'role': 'assistant', 'tool_calls': [{'name': 'b', 'arguments': {}}]},
    ]
    j, _ = _next_assistant_with_tool_calls(conv_h4, 0)
    run_test('next asst-with-tool-calls = idx 4 (skip text-only)', j == 4)
    j2, _ = _next_assistant_with_tool_calls(conv_h4, 4)
    run_test('no further asst-with-tool-calls -> None', j2 is None)
    print()

    # Test H5: _args_keys
    print('Test H5: _args_keys')
    run_test('dict args', _args_keys({'name': 'x', 'arguments': {'a': 1, 'b': 2}}) == frozenset({'a', 'b'}))
    run_test('non-dict args -> empty', _args_keys({'name': 'x', 'arguments': '[]'}) == frozenset())
    run_test('non-dict call -> empty', _args_keys('not a dict') == frozenset())
    print()

    # Test H6: _walk_calls_with_responses status assignment
    print('Test H6: _walk_calls_with_responses')
    conv_h6 = [
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'tool_calls': [
            {'name': 'a', 'arguments': {}},
            {'name': 'b', 'arguments': {}},
        ]},
        {'role': 'tool', 'name': 'a', 'content': '{"error": "bad"}'},
        {'role': 'tool', 'name': 'b', 'content': '{"result": "ok"}'},
        {'role': 'assistant', 'tool_calls': [
            {'name': 'c', 'arguments': {}},
        ]},
        # missing tool response
    ]
    statuses = list(_walk_calls_with_responses(conv_h6, err_classifier))
    run_test('three calls walked', len(statuses) == 3)
    run_test('first is error', statuses[0][3] == 'error')
    run_test('second is success', statuses[1][3] == 'success')
    run_test('third is missing', statuses[2][3] == 'missing')
    print()

    # Test R1a: schema-fail with error tool response -> tolerated
    print('Test R1a: tolerate schema-fail with error response')
    conv_r1a = [
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'tool_calls': [
            {'name': 'lookup', 'arguments': {'zip': 12345}}]},
        {'role': 'tool', 'name': 'lookup', 'content': '{"error": "zip must be string"}'},
        {'role': 'assistant', 'content': 'sorry'},
    ]
    res = evaluate_tool_recovery(conv_r1a, [(1, 0, ['zip: expected string, got int'])], err_classifier, 3)
    run_test('one tolerated', res['tolerated'] == {(1, 0)})
    run_test('audit tolerated_count = 1', res['audit']['tolerated_count'] == 1)
    # Rule 4 fires because the last (and only) tool-calling turn ended with an
    # error response. Tolerance applies per-call; trajectory health is global.
    run_test('rule-4 fires (trajectory ends with error)',
             any('last tool-calling turn' in e for e in res['errors']))
    print()

    # Test R1b: schema-fail with success-shape response -> NOT tolerated
    print('Test R1b: do not tolerate schema-fail with success response')
    conv_r1b = [
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'tool_calls': [
            {'name': 'lookup', 'arguments': {'zip': 12345}}]},
        {'role': 'tool', 'name': 'lookup', 'content': '{"result": "ok"}'},
        {'role': 'assistant', 'content': 'done'},
    ]
    res = evaluate_tool_recovery(conv_r1b, [(1, 0, ['zip: expected string, got int'])], err_classifier, 3)
    run_test('zero tolerated', res['tolerated'] == set())
    run_test('audit tolerated_count = 0', res['audit']['tolerated_count'] == 0)
    print()

    # Test R1c: schema-fail with no tool response -> NOT tolerated
    print('Test R1c: missing tool response is not tolerable')
    conv_r1c = [
        {'role': 'assistant', 'tool_calls': [{'name': 'lookup', 'arguments': {}}]},
        {'role': 'user', 'content': 'still here'},
    ]
    res = evaluate_tool_recovery(conv_r1c, [(0, 0, ['missing required arg'])], err_classifier, 3)
    run_test('zero tolerated', res['tolerated'] == set())
    print()

    # Test R2a: 3 consecutive errors -> rejected, audit counts errors
    print('Test R2a: 3 consecutive errors rejected')
    conv_r2a = [
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'tool_calls': [{'name': 'a', 'arguments': {}}]},
        {'role': 'tool', 'name': 'a', 'content': '{"error": "x"}'},
        {'role': 'assistant', 'tool_calls': [{'name': 'a', 'arguments': {'k': 1}}]},
        {'role': 'tool', 'name': 'a', 'content': '{"error": "y"}'},
        {'role': 'assistant', 'tool_calls': [{'name': 'a', 'arguments': {'k': 2}}]},
        {'role': 'tool', 'name': 'a', 'content': '{"error": "z"}'},
        {'role': 'assistant', 'content': 'giving up'},
    ]
    res = evaluate_tool_recovery(conv_r2a, [], err_classifier, 3)
    # Both Rule 2 (consecutive errors) and Rule 4 (last tool-calling turn ends in
    # error) fire — independent gates, independent error messages.
    run_test('rule-2 error emitted', any('3+ consecutive' in e for e in res['errors']))
    run_test('rule-4 error emitted', any('last tool-calling turn' in e for e in res['errors']))
    run_test('response_errors_count = 3', res['audit']['response_errors_count'] == 3)
    print()

    # Test R2b: 2 consecutive errors -> NOT rejected by rule 2
    print('Test R2b: 2 consecutive errors not rejected')
    conv_r2b = [
        {'role': 'assistant', 'tool_calls': [{'name': 'a', 'arguments': {}}]},
        {'role': 'tool', 'name': 'a', 'content': '{"error": "x"}'},
        {'role': 'assistant', 'tool_calls': [{'name': 'a', 'arguments': {'k': 1}}]},
        {'role': 'tool', 'name': 'a', 'content': '{"error": "y"}'},
        {'role': 'assistant', 'content': 'fine'},
    ]
    res = evaluate_tool_recovery(conv_r2b, [], err_classifier, 3)
    run_test('no rule-2 error', not any('consecutive' in e for e in res['errors']))
    run_test('response_errors_count = 2', res['audit']['response_errors_count'] == 2)
    print()

    # Test R2c: success in between resets the streak
    print('Test R2c: success resets streak')
    conv_r2c = [
        {'role': 'assistant', 'tool_calls': [{'name': 'a', 'arguments': {}}]},
        {'role': 'tool', 'name': 'a', 'content': '{"error": "x"}'},
        {'role': 'assistant', 'tool_calls': [{'name': 'a', 'arguments': {}}]},
        {'role': 'tool', 'name': 'a', 'content': '{"result": "ok"}'},
        {'role': 'assistant', 'tool_calls': [{'name': 'a', 'arguments': {}}]},
        {'role': 'tool', 'name': 'a', 'content': '{"error": "y"}'},
        {'role': 'assistant', 'tool_calls': [{'name': 'a', 'arguments': {}}]},
        {'role': 'tool', 'name': 'a', 'content': '{"error": "z"}'},
        {'role': 'assistant', 'content': 'last'},
    ]
    res = evaluate_tool_recovery(conv_r2c, [], err_classifier, 3)
    run_test('no rule-2 error (streak interrupted)', not any('consecutive' in e for e in res['errors']))
    run_test('response_errors_count = 3', res['audit']['response_errors_count'] == 3)
    print()

    # Test R2d: limit = 5 -> 4 consecutive errors not rejected
    print('Test R2d: limit param respected')
    conv_r2d = conv_r2a[:]  # 3 consecutive errors
    res = evaluate_tool_recovery(conv_r2d, [], err_classifier, 5)
    run_test('no rule-2 error at limit=5', not any('consecutive' in e for e in res['errors']))
    print()

    # Test R3a: same tool, same arg keys, BOTH schema-fail -> rejected
    print('Test R3a: RETRY_SAME_TOOL_UNFIXED rejected (both retries schema-fail)')
    conv_r3a = [
        {'role': 'assistant', 'tool_calls': [
            {'name': 'lookup', 'arguments': {'zip': 12345}}]},
        {'role': 'tool', 'name': 'lookup', 'content': '{"error": "bad"}'},
        {'role': 'assistant', 'tool_calls': [
            {'name': 'lookup', 'arguments': {'zip': 99999}}]},  # same key set, still int
        {'role': 'tool', 'name': 'lookup', 'content': '{"error": "still bad"}'},
        {'role': 'assistant', 'content': 'oh well'},
    ]
    # Both turns schema-fail in the simulated input.
    schema_fails_r3a = [(0, 0, ['zip: expected string, got int']),
                        (2, 0, ['zip: expected string, got int'])]
    res = evaluate_tool_recovery(conv_r3a, schema_fails_r3a, err_classifier, 3)
    run_test('rule-3 error emitted', any('RETRY_SAME_TOOL_UNFIXED' in e for e in res['errors']))
    run_test('error mentions tool name', any("'lookup'" in e for e in res['errors']))
    print()

    # Test R3b: same tool, same arg keys, retry SCHEMA-PASSES -> NOT rejected (FIXED)
    print('Test R3b: same keys but retry schema-passes -> FIXED, not UNFIXED')
    conv_r3b = [
        {'role': 'assistant', 'tool_calls': [
            {'name': 'lookup', 'arguments': {'zip': 12345}}]},
        {'role': 'tool', 'name': 'lookup', 'content': '{"error": "bad"}'},
        {'role': 'assistant', 'tool_calls': [
            {'name': 'lookup', 'arguments': {'zip': '12345'}}]},  # same keys, fixed type
        {'role': 'tool', 'name': 'lookup', 'content': '{"result": "ok"}'},
        {'role': 'assistant', 'content': 'ok'},
    ]
    # Only the first call schema-fails; retry is schema-clean.
    schema_fails_r3b = [(0, 0, ['zip: expected string, got int'])]
    res = evaluate_tool_recovery(conv_r3b, schema_fails_r3b, err_classifier, 3)
    run_test('no rule-3 error', not any('RETRY_SAME_TOOL_UNFIXED' in e for e in res['errors']))
    print()

    # Test R3c: same tool, different arg keys -> NOT rejected
    print('Test R3c: same tool with different arg keys not rejected')
    conv_r3c = [
        {'role': 'assistant', 'tool_calls': [
            {'name': 'lookup', 'arguments': {'zip': 12345}}]},
        {'role': 'tool', 'name': 'lookup', 'content': '{"error": "bad"}'},
        {'role': 'assistant', 'tool_calls': [
            {'name': 'lookup', 'arguments': {'city': 'NYC'}}]},  # different key
        {'role': 'tool', 'name': 'lookup', 'content': '{"result": "ok"}'},
        {'role': 'assistant', 'content': 'ok'},
    ]
    res = evaluate_tool_recovery(conv_r3c, [], err_classifier, 3)
    run_test('no rule-3 error', not any('RETRY_SAME_TOOL_UNFIXED' in e for e in res['errors']))
    print()

    # Test R3d: pivot to different tool -> NOT rejected by rule 3
    print('Test R3d: pivot to different tool not flagged by rule 3')
    conv_r3d = [
        {'role': 'assistant', 'tool_calls': [
            {'name': 'lookup', 'arguments': {'zip': 12345}}]},
        {'role': 'tool', 'name': 'lookup', 'content': '{"error": "bad"}'},
        {'role': 'assistant', 'tool_calls': [
            {'name': 'search', 'arguments': {'q': 'NYC'}}]},
        {'role': 'tool', 'name': 'search', 'content': '{"result": "ok"}'},
        {'role': 'assistant', 'content': 'ok'},
    ]
    res = evaluate_tool_recovery(conv_r3d, [], err_classifier, 3)
    run_test('no rule-3 error', not any('RETRY_SAME_TOOL_UNFIXED' in e for e in res['errors']))
    print()

    # Test R4a: last tool-calling turn has error -> rejected
    print('Test R4a: last tool-calling turn with error rejected')
    conv_r4a = [
        {'role': 'assistant', 'tool_calls': [{'name': 'a', 'arguments': {}}]},
        {'role': 'tool', 'name': 'a', 'content': '{"result": "ok"}'},
        {'role': 'assistant', 'tool_calls': [{'name': 'a', 'arguments': {'k': 1}}]},
        {'role': 'tool', 'name': 'a', 'content': '{"error": "fail"}'},
        {'role': 'assistant', 'content': 'sorry'},
    ]
    res = evaluate_tool_recovery(conv_r4a, [], err_classifier, 3)
    run_test('rule-4 error emitted', any('last tool-calling turn' in e for e in res['errors']))
    print()

    # Test R4b: last tool-calling turn all-success -> NOT rejected
    print('Test R4b: last tool-calling turn all success not rejected')
    conv_r4b = [
        {'role': 'assistant', 'tool_calls': [{'name': 'a', 'arguments': {}}]},
        {'role': 'tool', 'name': 'a', 'content': '{"error": "early"}'},
        {'role': 'assistant', 'tool_calls': [{'name': 'a', 'arguments': {}}]},
        {'role': 'tool', 'name': 'a', 'content': '{"result": "ok"}'},
        {'role': 'assistant', 'content': 'done'},
    ]
    res = evaluate_tool_recovery(conv_r4b, [], err_classifier, 3)
    run_test('no rule-4 error', not any('last tool-calling turn' in e for e in res['errors']))
    print()

    # Test R4c: last tool-calling turn has multiple calls, one fails -> rejected
    print('Test R4c: last tool-calling turn with one of N failing rejected')
    conv_r4c = [
        {'role': 'assistant', 'tool_calls': [
            {'name': 'a', 'arguments': {}},
            {'name': 'b', 'arguments': {}}]},
        {'role': 'tool', 'name': 'a', 'content': '{"result": "ok"}'},
        {'role': 'tool', 'name': 'b', 'content': '{"error": "bad"}'},
        {'role': 'assistant', 'content': 'one failed'},
    ]
    res = evaluate_tool_recovery(conv_r4c, [], err_classifier, 3)
    run_test('rule-4 error emitted', any('last tool-calling turn' in e for e in res['errors']))
    print()

    # Test R4d: zero tool calls -> rule 4 vacuous
    print('Test R4d: zero tool calls -> rule 4 vacuous')
    conv_r4d = [
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': 'a'},
    ]
    res = evaluate_tool_recovery(conv_r4d, [], err_classifier, 3)
    run_test('no errors at all', res['errors'] == [])
    run_test('audit response_errors_count = 0', res['audit']['response_errors_count'] == 0)
    print()

    total = passed + failed
    print(f'Results: {passed}/{total} passed, {failed}/{total} failed')
    if failed == 0:
        print('All tests passed!')
    else:
        print('Some tests failed!')
        exit(1)
