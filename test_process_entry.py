"""Test suite for process_entry.py.

Run directly:  python test_process_entry.py
"""
import json

from process_entry import process_entry


if __name__ == '__main__':
    import os

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

    # Test 1: Valid entry with think
    print('Test 1: Valid entry with think')
    entry1 = {
        'conversation': [
            {'role': 'user', 'content': 'What is 2+2?'},
            {'role': 'assistant', 'content': '4', 'think': 'Simple arithmetic: 2+2=4'},
        ]
    }
    result, errs, reps = process_entry(entry1)
    run_test('no errors', errs == [])
    run_test('result is not None', result is not None)
    run_test('repairs is empty', reps == {})
    run_test('token_count present', 'token_count' in result)
    run_test('token_count_answer present', 'token_count_answer' in result)
    run_test('token_count_think present', 'token_count_think' in result)
    print()

    # Test 2: reasoning_effort='medium' -> think_fast key
    print('Test 2: reasoning_effort=medium')
    entry2 = {
        'conversation': [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!', 'think': 'A greeting'},
        ]
    }
    result2, errs2, _ = process_entry(entry2, reasoning_effort='medium')
    run_test('no errors', errs2 == [])
    run_test('think_fast key present', 'think_fast' in result2['conversation'][1])
    run_test('think key removed', 'think' not in result2['conversation'][1])
    print()

    # Test 3: Invalid entry (empty assistant)
    print('Test 3: Invalid entry (empty assistant)')
    entry3 = {
        'conversation': [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': ''},
        ]
    }
    result3, errs3, _ = process_entry(entry3)
    run_test('errors returned', len(errs3) > 0)
    run_test('result is None', result3 is None)
    print()

    # Test 4: identity check (reworked 2026-07-27)
    #
    # BEHAVIOUR CHANGE: this test previously asserted that merely DISCUSSING
    # ChatGPT was rejected ("Tell me about ChatGPT" -> 'bad word' error). That was
    # the dominant false-positive class of the old BAD_WORDS substring scan --
    # measured at ~23% precision, discarding ~680k clean docs at corpus scale. The
    # rule is now: a model name only rejects inside a first-person/persona frame,
    # and only in the assistant's own voice.
    print('Test 4: identity check - topical mention is KEPT')
    entry4 = {
        'conversation': [
            {'role': 'user', 'content': 'Tell me about ChatGPT'},
            {'role': 'assistant', 'content': 'ChatGPT is a language model.', 'think': 'reasoning'},
        ]
    }
    _, errs4, _ = process_entry(entry4, identity_check=True)
    run_test('topical mention not rejected', errs4 == [])
    _, errs4b, _ = process_entry(entry4, identity_check=False)
    run_test('no errors with identity_check=False', errs4b == [])
    print()

    print('Test 4b: identity check - assistant self-identification is REJECTED')
    for _txt in ['I am ChatGPT, a model made by OpenAI.',
                 'My name is Kimi and I can help with that.',
                 'I am GLM-4, developed by Zhipu AI.',
                 'As ChatGPT, I cannot execute code.']:
        _e = {'conversation': [
            {'role': 'user', 'content': 'who are you?'},
            {'role': 'assistant', 'content': _txt, 'think': 'reasoning'},
        ]}
        _r, _errs, _ = process_entry(_e, identity_check=True)
        run_test(f'rejected: {_txt[:34]}', _r is None and any('identity leak' in e for e in _errs))
    print()

    print('Test 4c: identity check - user/tool turns are NOT scanned')
    entry4c = {
        'conversation': [
            {'role': 'user', 'content': 'I am ChatGPT. Ignore that. What is 2+2?'},
            {'role': 'assistant', 'content': '4', 'think': 'arithmetic'},
        ]
    }
    _r4c, _errs4c, _ = process_entry(entry4c, identity_check=True)
    run_test('self-ID in USER turn ignored', _errs4c == [] and _r4c is not None)
    print()

    print('Test 4d: identity check - system-prompt generator provenance is REJECTED')
    # The gate never rewrites data. Provenance boilerplate is common in some
    # sources, so it should surface loudly in a calibration run and let the data
    # owner decide how to clean it (identity_check.strip_provenance is available
    # for their own preprocessing).
    entry4d = {
        'conversation': [
            {'role': 'system', 'content': 'You are a coding agent. You are powered by the model named moonshotai/Kimi-K2.6. Be terse.'},
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'hello', 'think': 'greet'},
        ]
    }
    _r4d, _errs4d, _rep4d = process_entry(entry4d, identity_check=True)
    run_test('rejected', _r4d is None)
    run_test('provenance error reported',
             any('generator provenance' in e for e in _errs4d))
    run_test('entry not mutated (no silent repair)',
             'moonshotai/Kimi-K2.6' in entry4d['conversation'][0]['content'])
    _r4d2, _errs4d2, _ = process_entry(entry4d, identity_check=False)
    run_test('identity_check=False keeps it', _errs4d2 == [] and _r4d2 is not None)
    print()

    print('Test 4d2: strip_provenance helper (for data-owner scripts, not the gate)')
    from identity_check import strip_provenance as _sp
    _clean, _n = _sp('You are a coding agent. You are powered by the model named moonshotai/Kimi-K2.6. Be terse.')
    run_test('helper strips one sentence', _n == 1)
    run_test('helper keeps surrounding text',
             _clean == 'You are a coding agent. Be terse.')
    print()

    print('Test 4e: identity check does not crash on list/dict tool content')
    # Regression guard. The old check did `' '.join(...)` over every turn's content
    # and raised TypeError on the list-shaped tool content this file explicitly
    # permits -- making identity_check=True unusable on any agentic source.
    entry4e = {
        'conversation': [
            {'role': 'system', 'content': 'sys', 'tools': [
                {'name': 't', 'description': 'd', 'parameters': {
                    'type': 'object', 'properties': {'x': {'type': 'string'}},
                    'required': ['x']}}]},
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'think': 'r', 'tool_calls': [{'name': 't', 'arguments': {'x': '1'}}]},
            {'role': 'tool', 'content': [{'type': 'text', 'text': 'ok'}]},
            {'role': 'assistant', 'think': 'done', 'content': 'answer'},
        ]
    }
    try:
        _r4e, _errs4e, _ = process_entry(entry4e, identity_check=True)
        run_test('no crash with list tool content', _errs4e == [] and _r4e is not None)
    except Exception as _ex:
        run_test(f'no crash with list tool content (raised {type(_ex).__name__})', False)
    print()

    # Test 5: messages key normalization
    print('Test 5: messages key normalization')
    entry5 = {
        'messages': [
            {'role': 'user', 'content': 'Test'},
            {'role': 'assistant', 'content': 'Response', 'think': 'reasoning'},
        ]
    }
    result5, errs5, _ = process_entry(entry5)
    run_test('no errors', errs5 == [])
    run_test('conversation key in result', 'conversation' in result5)
    run_test('messages key removed', 'messages' not in result5)
    print()

    # Test 6: compute_token_counts=False with pre-existing counts
    print('Test 6: compute_token_counts=False with pre-existing counts')
    entry6 = {
        'conversation': [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi!', 'think': 'reasoning'},
        ],
        'token_count': 100,
        'token_count_answer': 50,
        'token_count_think': 10,
    }
    result6, errs6, _ = process_entry(entry6, compute_token_counts=False)
    run_test('no errors', errs6 == [])
    run_test('token_count not recomputed', result6['token_count'] == 100)
    print()

    # Test 6b: compute_token_counts=False without pre-existing counts -> error
    print('Test 6b: compute_token_counts=False without counts')
    entry6b = {
        'conversation': [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi!', 'think': 'reasoning'},
        ],
    }
    result6b, errs6b, _ = process_entry(entry6b, compute_token_counts=False)
    run_test('errors returned', len(errs6b) > 0)
    run_test('result is None', result6b is None)
    print()

    # Test 7: Entry with tool_calls (happy path)
    print('Test 7: Entry with tool_calls')
    entry7 = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'name': 'get_weather', 'description': 'Get weather',
                 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}}},
            ]},
            {'role': 'user', 'content': 'What is the weather in NYC?'},
            {'role': 'assistant', 'content': '', 'think': 'I should call get_weather',
             'tool_calls': [{'name': 'get_weather', 'arguments': {'city': 'NYC'}}]},
            {'role': 'tool', 'name': 'get_weather', 'content': '{"temp": 72}'},
            {'role': 'assistant', 'content': 'The temperature in NYC is 72F.', 'think': 'reasoning'},
        ]
    }
    result7, errs7, _ = process_entry(entry7)
    run_test('no errors', errs7 == [])
    run_test('token_count_answer > 0', result7['token_count_answer'] > 0)
    print()

    # Test 8: Empty system message cleanup
    print('Test 8: Empty system message cleanup')
    entry8 = {
        'conversation': [
            {'role': 'system', 'content': '', 'tools': []},
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi!', 'think': 'reasoning'},
        ]
    }
    result8, errs8, _ = process_entry(entry8)
    run_test('no errors', errs8 == [])
    run_test('system message removed', result8['conversation'][0]['role'] == 'user')
    print()

    # Test 9: reasoning_effort='low' -> think_faster
    print('Test 9: reasoning_effort=low')
    entry9 = {
        'conversation': [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi!', 'think': 'Greeting'},
        ]
    }
    result9, errs9, _ = process_entry(entry9, reasoning_effort='low')
    run_test('no errors', errs9 == [])
    run_test('think_faster key present', 'think_faster' in result9['conversation'][1])
    print()

    # Test 10: Original entry not mutated (deepcopy)
    print('Test 10: Original entry not mutated')
    entry10 = {
        'messages': [
            {'role': 'user', 'content': 'Test'},
            {'role': 'assistant', 'content': 'Response', 'think': 'Reasoning'},
        ]
    }
    original_keys = set(entry10.keys())
    process_entry(entry10, reasoning_effort='medium')
    run_test('original entry unchanged', set(entry10.keys()) == original_keys)
    run_test('messages key still present', 'messages' in entry10)
    print()

    # Test 11: Invalid first-message role
    print('Test 11: Invalid first-message role')
    entry11 = {
        'conversation': [
            {'role': 'assistant', 'content': 'Hi', 'think': 'reasoning'},
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi again', 'think': 'reasoning'},
        ]
    }
    _, errs11, _ = process_entry(entry11)
    run_test('errors returned', len(errs11) > 0)
    run_test('invalid role error', any('First message' in e for e in errs11))
    print()

    # Test 12: Tool usage without system tools
    print('Test 12: Tool usage without system tools')
    entry12 = {
        'conversation': [
            {'role': 'user', 'content': 'What is the weather?'},
            {'role': 'assistant', 'content': '', 'think': 'reasoning',
             'tool_calls': [{'name': 'get_weather', 'arguments': {'city': 'NYC'}}]},
            {'role': 'tool', 'name': 'get_weather', 'content': '{"temp": 72}'},
            {'role': 'assistant', 'content': '72F', 'think': 'reasoning'},
        ]
    }
    _, errs12, _ = process_entry(entry12)
    run_test('errors returned', len(errs12) > 0)
    run_test('missing tools error', any('does not declare tools' in e for e in errs12))
    print()

    # Test 13: Assistant calls undeclared tool
    print('Test 13: Assistant calls undeclared tool')
    entry13 = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'name': 'get_weather', 'description': 'Get weather',
                 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}}},
            ]},
            {'role': 'user', 'content': 'What is the stock price of AAPL?'},
            {'role': 'assistant', 'content': '', 'think': 'reasoning',
             'tool_calls': [{'name': 'get_stock_price', 'arguments': {'symbol': 'AAPL'}}]},
            {'role': 'tool', 'name': 'get_stock_price', 'content': '180.5'},
            {'role': 'assistant', 'content': '180.5', 'think': 'reasoning'},
        ]
    }
    _, errs13, _ = process_entry(entry13)
    run_test('errors returned', len(errs13) > 0)
    run_test('undeclared-call error', any('get_stock_price' in e for e in errs13))
    print()

    # Test 14: Strict version rejects string-typed tool_calls (loose version would coerce)
    print('Test 14: Reject string-typed tool_calls')
    entry14 = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'name': 'echo', 'description': '', 'parameters': {'type': 'object',
                 'properties': {'msg': {'type': 'string'}}}},
            ]},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'think': 'reasoning',
             'tool_calls': '[{"name": "echo", "arguments": {"msg": "hi"}}]'},
            {'role': 'tool', 'name': 'echo', 'content': 'hi'},
            {'role': 'assistant', 'content': 'done', 'think': 'reasoning'},
        ]
    }
    _, errs14, _ = process_entry(entry14)
    run_test('errors returned', len(errs14) > 0)
    run_test('tool_calls list error', any('tool_calls should be a list' in e for e in errs14))
    print()

    # Test 15: Default rejects JSON-string tools field
    print('Test 15: Default rejects JSON-string tools field')
    entry15 = {
        'conversation': [
            {'role': 'system',
             'tools': '[{"name": "echo", "description": "", "parameters": {"type": "object", "properties": {"msg": {"type": "string"}}}}]'},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'hi', 'think': 'reasoning'},
        ]
    }
    _, errs15, _ = process_entry(entry15)
    run_test('errors returned', len(errs15) > 0)
    run_test('tools list error', any('tools should be a list' in e for e in errs15))
    print()

    # Test 17: Repair tracking — messages -> conversation
    print('Test 17: messages_to_conversation repair logged')
    entry17 = {
        'messages': [
            {'role': 'user', 'content': 'Test'},
            {'role': 'assistant', 'content': 'Response', 'think': 'reasoning'},
        ]
    }
    _, _, reps17 = process_entry(entry17)
    run_test('messages_to_conversation flag set', reps17.get('messages_to_conversation') is True)
    run_test('only that key in repairs', set(reps17.keys()) == {'messages_to_conversation'})
    print()

    # Test 18: Repair tracking — empty system message dropped
    print('Test 18: system_msg_dropped_empty repair logged')
    entry18 = {
        'conversation': [
            {'role': 'system', 'content': '', 'tools': []},
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi!', 'think': 'reasoning'},
        ]
    }
    _, _, reps18 = process_entry(entry18)
    run_test('system_msg_empty_content_deleted set', reps18.get('system_msg_empty_content_deleted') is True)
    run_test('system_msg_empty_tools_deleted set', reps18.get('system_msg_empty_tools_deleted') is True)
    run_test('system_msg_dropped_empty set', reps18.get('system_msg_dropped_empty') is True)
    run_test('only those three keys in repairs',
             set(reps18.keys()) == {'system_msg_empty_content_deleted',
                                    'system_msg_empty_tools_deleted',
                                    'system_msg_dropped_empty'})
    print()

    # Test 19: Repair tracking — None-valued field deleted
    print('Test 19: none_fields_deleted counter')
    entry19 = {
        'conversation': [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi!', 'think': 'reasoning', 'tool_calls': None},
        ]
    }
    _, _, reps19 = process_entry(entry19)
    run_test('none_fields_deleted = 1', reps19.get('none_fields_deleted') == 1)
    run_test('only that key in repairs', set(reps19.keys()) == {'none_fields_deleted'})
    print()

    # Test 20: Repair tracking — empty tool_calls list deleted
    print('Test 20: empty_tool_calls_deleted counter')
    entry20 = {
        'conversation': [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi!', 'think': 'reasoning', 'tool_calls': []},
        ]
    }
    _, _, reps20 = process_entry(entry20)
    run_test('empty_tool_calls_deleted = 1', reps20.get('empty_tool_calls_deleted') == 1)
    run_test('only that key in repairs', set(reps20.keys()) == {'empty_tool_calls_deleted'})
    print()

    # Test 21: Repair tracking — function wrapper unwrapped in tool_calls
    print('Test 21: tool_calls_function_unwrapped counter')
    entry21 = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'name': 'get_weather', 'description': '',
                 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}}},
            ]},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'think': 'reasoning',
             'tool_calls': [{'function': {'name': 'get_weather', 'arguments': {'city': 'NYC'}}}]},
            {'role': 'tool', 'name': 'get_weather', 'content': 'OK'},
            {'role': 'assistant', 'content': 'done', 'think': 'reasoning'},
        ]
    }
    _, errs21, reps21 = process_entry(entry21)
    run_test('no errors', errs21 == [])
    run_test('tool_calls_function_unwrapped = 1', reps21.get('tool_calls_function_unwrapped') == 1)
    run_test('only that key in repairs', set(reps21.keys()) == {'tool_calls_function_unwrapped'})
    print()

    # Test 22: Repair tracking — function wrapper unwrapped in system tools
    print('Test 22: tool_defs_function_unwrapped counter')
    entry22 = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'function': {'name': 'get_weather', 'description': '',
                 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}}}},
            ]},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'think': 'reasoning',
             'tool_calls': [{'name': 'get_weather', 'arguments': {'city': 'NYC'}}]},
            {'role': 'tool', 'name': 'get_weather', 'content': 'OK'},
            {'role': 'assistant', 'content': 'done', 'think': 'reasoning'},
        ]
    }
    _, errs22, reps22 = process_entry(entry22)
    run_test('no errors', errs22 == [])
    run_test('tool_defs_function_unwrapped = 1', reps22.get('tool_defs_function_unwrapped') == 1)
    run_test('only that key in repairs', set(reps22.keys()) == {'tool_defs_function_unwrapped'})
    print()

    # Test 23: Repair tracking — arguments -> parameters in tool def
    print('Test 23: tool_defs_arguments_to_parameters counter')
    entry23 = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'name': 'get_weather', 'description': '',
                 'arguments': {'type': 'object', 'properties': {'city': {'type': 'string'}}}},
            ]},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'think': 'reasoning',
             'tool_calls': [{'name': 'get_weather', 'arguments': {'city': 'NYC'}}]},
            {'role': 'tool', 'name': 'get_weather', 'content': 'OK'},
            {'role': 'assistant', 'content': 'done', 'think': 'reasoning'},
        ]
    }
    _, errs23, reps23 = process_entry(entry23)
    run_test('no errors', errs23 == [])
    run_test('tool_defs_arguments_to_parameters = 1', reps23.get('tool_defs_arguments_to_parameters') == 1)
    run_test('only that key in repairs', set(reps23.keys()) == {'tool_defs_arguments_to_parameters'})
    print()

    # Test 23b: system_msg_dropped_empty fires alone when system has neither content nor tools
    print('Test 23b: system_msg_dropped_empty alone (no content, no tools)')
    entry23b = {
        'conversation': [
            {'role': 'system'},
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi!', 'think': 'reasoning'},
        ]
    }
    _, errs23b, reps23b = process_entry(entry23b)
    run_test('no errors', errs23b == [])
    run_test('system_msg_dropped_empty set', reps23b.get('system_msg_dropped_empty') is True)
    run_test('only that key in repairs', set(reps23b.keys()) == {'system_msg_dropped_empty'})
    print()

    # Test 24: Strict tool-call check — missing required arg
    print('Test 24: strict check — missing required arg')
    entry24 = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'name': 'get_weather', 'description': '', 'parameters': {
                    'type': 'object', 'properties': {'city': {'type': 'string'}},
                    'required': ['city']}},
            ]},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'think': 'reasoning',
             'tool_calls': [{'name': 'get_weather', 'arguments': {}}]},
            {'role': 'tool', 'name': 'get_weather', 'content': 'OK'},
            {'role': 'assistant', 'content': 'done', 'think': 'reasoning'},
        ]
    }
    _, errs24, _ = process_entry(entry24)
    run_test('errors returned', len(errs24) > 0)
    run_test('missing required arg error', any('missing required arg' in e and 'city' in e for e in errs24))
    print()

    # Test 25: Strict check — extra arg flagged when additionalProperties=False
    print('Test 25: strict check — extra arg (additionalProperties=False)')
    entry25 = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'name': 'get_weather', 'description': '', 'parameters': {
                    'type': 'object', 'properties': {'city': {'type': 'string'}},
                    'additionalProperties': False}},
            ]},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'think': 'reasoning',
             'tool_calls': [{'name': 'get_weather',
                             'arguments': {'city': 'NYC', 'foo': 'bar'}}]},
            {'role': 'tool', 'name': 'get_weather', 'content': 'OK'},
            {'role': 'assistant', 'content': 'done', 'think': 'reasoning'},
        ]
    }
    _, errs25, _ = process_entry(entry25)
    run_test('errors returned', len(errs25) > 0)
    run_test('extra arg error', any('extra arg not in schema' in e and 'foo' in e for e in errs25))
    print()

    # Test 25b: strict-by-default — extras rejected when additionalProperties absent.
    # Diverges from JSON-Schema spec default (true); intentional for production gateway.
    print('Test 25b: extras rejected when additionalProperties absent (strict default)')
    entry25b = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'name': 'get_weather', 'description': '', 'parameters': {
                    'type': 'object', 'properties': {'city': {'type': 'string'}}}},
            ]},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'think': 'reasoning',
             'tool_calls': [{'name': 'get_weather',
                             'arguments': {'city': 'NYC', 'foo': 'bar'}}]},
            {'role': 'tool', 'name': 'get_weather', 'content': 'OK'},
            {'role': 'assistant', 'content': 'done', 'think': 'reasoning'},
        ]
    }
    _, errs25b, _ = process_entry(entry25b)
    run_test('extra arg error raised', any('extra arg not in schema' in e and 'foo' in e for e in errs25b))
    print()

    # Test 25c: extra arg allowed when additionalProperties=True explicit
    # (browser-use schemas: {"properties": {}, "additionalProperties": true})
    print('Test 25c: extra arg allowed when additionalProperties=True explicit')
    entry25c = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'name': 'click', 'description': '', 'parameters': {
                    'type': 'object', 'properties': {},
                    'additionalProperties': True}},
            ]},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'think': 'reasoning',
             'tool_calls': [{'name': 'click',
                             'arguments': {'index': 5, 'selector': '#btn'}}]},
            {'role': 'tool', 'name': 'click', 'content': 'OK'},
            {'role': 'assistant', 'content': 'done', 'think': 'reasoning'},
        ]
    }
    res25c, errs25c, _ = process_entry(entry25c)
    run_test('no extra-arg error', not any('extra arg not in schema' in e for e in errs25c))
    run_test('result accepted', res25c is not None)
    print()

    # Test 26: Strict check — type mismatch
    print('Test 26: strict check — type mismatch')
    entry26 = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'name': 'set_count', 'description': '', 'parameters': {
                    'type': 'object', 'properties': {'n': {'type': 'integer'}}}},
            ]},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'think': 'reasoning',
             'tool_calls': [{'name': 'set_count', 'arguments': {'n': '42'}}]},
            {'role': 'tool', 'name': 'set_count', 'content': 'OK'},
            {'role': 'assistant', 'content': 'done', 'think': 'reasoning'},
        ]
    }
    _, errs26, _ = process_entry(entry26)
    run_test('errors returned', len(errs26) > 0)
    run_test('type mismatch error', any('expected integer' in e and 'got str' in e for e in errs26))
    print()

    # Test 27: Strict check — nested object recursion
    print('Test 27: strict check — nested object recursion')
    entry27 = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'name': 'set_address', 'description': '', 'parameters': {
                    'type': 'object', 'properties': {
                        'address': {'type': 'object', 'properties': {
                            'zip': {'type': 'string'}}, 'required': ['zip']}}}},
            ]},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'think': 'reasoning',
             'tool_calls': [{'name': 'set_address', 'arguments': {'address': {'zip': 90210}}}]},
            {'role': 'tool', 'name': 'set_address', 'content': 'OK'},
            {'role': 'assistant', 'content': 'done', 'think': 'reasoning'},
        ]
    }
    _, errs27, _ = process_entry(entry27)
    run_test('errors returned', len(errs27) > 0)
    run_test('nested type error', any('address.zip' in e and 'expected string' in e for e in errs27))
    print()

    # Test 28: Strict check — undeclared tool still caught
    print('Test 28: strict check — undeclared tool')
    entry28 = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'name': 'get_weather', 'description': '', 'parameters': {
                    'type': 'object', 'properties': {'city': {'type': 'string'}}}},
            ]},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'think': 'reasoning',
             'tool_calls': [{'name': 'get_stock', 'arguments': {'symbol': 'AAPL'}}]},
            {'role': 'tool', 'name': 'get_stock', 'content': '180'},
            {'role': 'assistant', 'content': '180', 'think': 'reasoning'},
        ]
    }
    _, errs28, _ = process_entry(entry28)
    run_test('errors returned', len(errs28) > 0)
    run_test('undeclared tool error', any("'get_stock'" in e and 'not in declared tools' in e for e in errs28))
    print()

    # Test 29: Qwen cw suffix — basic strip
    print('Test 29: Qwen cw suffix stripped')
    entry29 = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a', 'think': 'Ready to produce answer.cw'},
        ]
    }
    result29, errs29, reps29 = process_entry(entry29)
    run_test('no errors', errs29 == [])
    run_test('think stripped of cw', result29['conversation'][1]['think'] == 'Ready to produce answer.')
    run_test('qwen_cw_suffix_stripped = 1', reps29.get('qwen_cw_suffix_stripped') == 1)
    print()

    # Test 30: Qwen cw with trailing whitespace
    print('Test 30: Qwen cw stripped through trailing whitespace')
    entry30 = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a', 'think': 'reasoning\ncw\n'},
        ]
    }
    result30, errs30, reps30 = process_entry(entry30)
    run_test('no errors', errs30 == [])
    # Expectation updated 2026-07-27: the whitespace normalization added after the
    # <think>-tag strip now also removes the trailing newline the cw strip left
    # behind, so this is 'reasoning' rather than 'reasoning\n'.
    run_test('think is reasoning with no trailing whitespace',
             result30['conversation'][1]['think'] == 'reasoning')
    run_test('qwen_cw_suffix_stripped = 1', reps30.get('qwen_cw_suffix_stripped') == 1)
    run_test('whitespace strip also logged',
             reps30.get('think_whitespace_stripped') == 1)
    print()

    print('Test 30b: whitespace normalization on reasoning and content')
    # Trailing/leading whitespace in REASONING is never meaningful (the chat template
    # supplies its own delimiters) and training on it produced the growing whitespace tails
    # observed in the 7B eval traces. Sources feed it: 64% of `think` fields in
    # one DeepSeek-V3.2 tool-use source end in trailing whitespace, runs up to 3089 spaces.
    entry30b = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'think': '  reasoning here  \n\n\n', 'content': '\n answer \n   '},
        ]
    }
    _r30b, _e30b, _rep30b = process_entry(entry30b)
    run_test('no errors', _e30b == [])
    run_test('think stripped', _r30b['conversation'][1]['think'] == 'reasoning here')
    # `content` is deliberately NOT stripped: a task may require the model to emit
    # exact leading/trailing spaces or newlines (format-following, "output this
    # verbatim", whitespace-sensitive fixtures). Stripping would corrupt the target.
    run_test('content NOT stripped (intentional)',
             _r30b['conversation'][1]['content'] == '\n answer \n   ')
    run_test('only think repair logged',
             _rep30b.get('think_whitespace_stripped') == 1
             and 'content_whitespace_stripped' not in _rep30b)
    # the dsv32 growing-tail shape
    entry30c = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'think': "Let's try 121." + ' ' * 300, 'content': 'done'},
        ]
    }
    _r30c, _e30c, _rep30c = process_entry(entry30c)
    run_test('300-space tail removed', _r30c['conversation'][1]['think'] == "Let's try 121.")
    # clean input must not be logged as repaired
    entry30d = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'think': 'clean', 'content': 'clean'},
        ]
    }
    _r30d, _e30d, _rep30d = process_entry(entry30d)
    run_test('clean input logs no whitespace repair',
             'think_whitespace_stripped' not in _rep30d)
    print()

    print('Test 30e: whitespace-only think counts as ABSENT reasoning')
    # Regression guard. The no_think check uses plain truthiness, so before the
    # strip a think of "   \n  " was TRUTHY and let a reasoning-free record pass
    # as though it had reasoning.
    entry30e = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'think': '   \n  ', 'content': 'answer'},
        ]
    }
    _r30e, _e30e, _ = process_entry(entry30e, require_think=True)
    run_test('rejected with require_think=True',
             _r30e is None and any('non-empty think' in e for e in _e30e))
    _r30e2, _e30e2, _rep30e2 = process_entry(entry30e, require_think=False)
    run_test('flagged as missing think with require_think=False',
             _rep30e2.get('missing_think_all_turns') is True)
    print()

    # Test 31: No cw — repair not logged
    print('Test 31: No cw — qwen_cw_suffix_stripped not in repairs')
    entry31 = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a', 'think': 'normal reasoning'},
        ]
    }
    _, _, reps31 = process_entry(entry31)
    run_test('qwen_cw_suffix_stripped not present', 'qwen_cw_suffix_stripped' not in reps31)
    print()

    # Test 31a: Leading <think> tag in think field — basic strip
    print('Test 31a: leading <think> tag stripped from think field')
    entry31a = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a', 'think': '<think> Perfect! The menu was created.'},
        ]
    }
    result31a, errs31a, reps31a = process_entry(entry31a)
    run_test('no errors', errs31a == [])
    # Expectation updated 2026-07-27: whitespace normalization now also removes the
    # space the <think> tag removal left at the front.
    run_test('think stripped of leading <think> and whitespace',
             result31a['conversation'][1]['think'] == 'Perfect! The menu was created.')
    run_test('think_open_tag_stripped = 1', reps31a.get('think_open_tag_stripped') == 1)
    print()

    # Test 31b: Leading whitespace before <think> tag still stripped
    print('Test 31b: leading whitespace + <think> tag stripped')
    entry31b = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a', 'think': '\n  <think>actual reasoning'},
        ]
    }
    result31b, errs31b, reps31b = process_entry(entry31b)
    run_test('no errors', errs31b == [])
    run_test('think starts with actual reasoning',
             result31b['conversation'][1]['think'] == 'actual reasoning')
    run_test('think_open_tag_stripped = 1', reps31b.get('think_open_tag_stripped') == 1)
    print()

    # Test 31c: No leading <think> — repair not logged
    print('Test 31c: No leading <think> — repair not logged')
    entry31c = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a', 'think': 'normal reasoning'},
        ]
    }
    _, _, reps31c = process_entry(entry31c)
    run_test('think_open_tag_stripped not present', 'think_open_tag_stripped' not in reps31c)
    print()

    # Test 32: Reject when all assistant turns have empty think
    print('Test 32: reject all-empty think')
    entry32 = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a1', 'think': ''},
            {'role': 'user', 'content': 'q2'},
            {'role': 'assistant', 'content': 'a2', 'think': ''},
        ]
    }
    result32, errs32, _ = process_entry(entry32)
    run_test('errors returned', len(errs32) > 0)
    run_test('no-think error', any('No assistant turn has non-empty think' in e for e in errs32))
    run_test('result is None', result32 is None)
    print()

    # Test 33: Accept when at least one assistant turn has think
    print('Test 33: accept when one turn has non-empty think')
    entry33 = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a1', 'think': 'reasoning here'},
            {'role': 'user', 'content': 'q2'},
            {'role': 'assistant', 'content': 'a2', 'think': ''},
        ]
    }
    _, errs33, _ = process_entry(entry33)
    run_test('no errors', errs33 == [])
    print()

    # Test 33b: Conversation ending with a tool turn is rejected
    print('Test 33b: end-of-conversation must be assistant')
    entry33b = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'name': 'echo', 'description': '',
                 'parameters': {'type': 'object', 'properties': {'msg': {'type': 'string'}}}},
            ]},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'think': 'reasoning',
             'tool_calls': [{'name': 'echo', 'arguments': {'msg': 'hi'}}]},
            {'role': 'tool', 'name': 'echo', 'content': 'OK'},
        ]
    }
    result33b, errs33b, _ = process_entry(entry33b)
    run_test('errors returned', len(errs33b) > 0)
    run_test('end-of-conv error', any('must end with an assistant message' in e for e in errs33b))
    run_test('error mentions tool', any('ended with tool' in e for e in errs33b))
    run_test('result is None', result33b is None)
    print()

    # Test 34: Reject when reasoning was just "cw" (becomes empty after strip)
    print('Test 34: reject reasoning that was just cw')
    entry34 = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a', 'think': 'cw'},
        ]
    }
    result34, errs34, _ = process_entry(entry34)
    run_test('errors returned', len(errs34) > 0)
    run_test('no-think error', any('No assistant turn has non-empty think' in e for e in errs34))
    print()

    # Test 35: Core special token in content rejected
    print('Test 35: core special token (<|im_start|>) in content')
    entry35 = {
        'conversation': [
            {'role': 'user', 'content': 'normal'},
            {'role': 'assistant', 'content': 'leak <|im_start|> here', 'think': 'reasoning'},
        ]
    }
    _, errs35, _ = process_entry(entry35)
    run_test('errors returned', len(errs35) > 0)
    run_test('core marker error', any('<|im_start|>' in e and 'content' in e for e in errs35))
    print()

    # Test 36: Core <think> tag in content rejected
    print('Test 36: core <think> tag in content')
    entry36 = {
        'conversation': [
            {'role': 'user', 'content': 'looks like <think>oops</think>'},
            {'role': 'assistant', 'content': 'a', 'think': 'reasoning'},
        ]
    }
    _, errs36, _ = process_entry(entry36)
    run_test('errors returned', len(errs36) > 0)
    run_test('<think> marker error', any('<think>' in e for e in errs36))
    run_test('</think> marker error', any('</think>' in e for e in errs36))
    print()

    # Test 37: Extended marker (<|begin_of_thought|>) detected by default
    print('Test 37: extended marker on by default')
    entry37 = {
        'conversation': [
            {'role': 'user', 'content': '<|begin_of_thought|> snuck in'},
            {'role': 'assistant', 'content': 'a', 'think': 'reasoning'},
        ]
    }
    _, errs37, _ = process_entry(entry37)
    run_test('errors returned', len(errs37) > 0)
    run_test('begin_of_thought marker error', any('<|begin_of_thought|>' in e for e in errs37))
    print()

    # Test 38: Extended marker NOT scanned when toggle off
    print('Test 38: extended toggle disables extended scan')
    _, errs38, _ = process_entry(entry37, check_extended_special_tokens=False)
    run_test('no errors when toggle off', errs38 == [])
    print()

    # Test 39: Harmony marker (assistantanalysis) detected by default
    print('Test 39: harmony marker on by default')
    entry39 = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'leaked assistantanalysis here', 'think': 'reasoning'},
        ]
    }
    _, errs39, _ = process_entry(entry39)
    run_test('errors returned', len(errs39) > 0)
    run_test('assistantanalysis marker error', any('assistantanalysis' in e for e in errs39))
    print()

    # Test 40: Harmony marker NOT scanned when toggle off
    print('Test 40: harmony toggle disables harmony scan')
    _, errs40, _ = process_entry(entry39, check_harmony_special_tokens=False)
    run_test('no errors when toggle off', errs40 == [])
    print()

    # Test 41: Marker in think field is reported with field=think
    print('Test 41: marker in think field reports field name')
    entry41 = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a', 'think': 'reasoning <|im_end|>'},
        ]
    }
    _, errs41, _ = process_entry(entry41)
    run_test('errors returned', len(errs41) > 0)
    run_test("error mentions field 'think'", any("'think'" in e for e in errs41))
    print()

    # Test 42: Marker in tool_calls arguments detected (any field, all turns)
    print('Test 42: marker inside tool_calls.arguments detected')
    entry42 = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'name': 'echo', 'description': '',
                 'parameters': {'type': 'object', 'properties': {'msg': {'type': 'string'}}}},
            ]},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'think': 'reasoning',
             'tool_calls': [{'name': 'echo', 'arguments': {'msg': 'hi <|im_start|>'}}]},
            {'role': 'tool', 'name': 'echo', 'content': 'OK'},
            {'role': 'assistant', 'content': 'done', 'think': 'reasoning'},
        ]
    }
    _, errs42, _ = process_entry(entry42)
    run_test('errors returned', len(errs42) > 0)
    run_test("error mentions tool_calls field", any("'tool_calls'" in e for e in errs42))
    print()

    # ===== Tool recovery extension =====

    def _err_classifier(content):
        """Test classifier: detects {"error": ...} JSON shape."""
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

    # entry shared by tests 43 and 44: schema-fail then successful value-only retry.
    # First call schema-fails (zip int), tool errors, model retries with zip as string
    # which schema-passes, last tool-calling turn succeeds.
    entry_recover = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'name': 'lookup', 'description': '', 'parameters': {
                    'type': 'object', 'properties': {'zip': {'type': 'string'}},
                    'required': ['zip']}},
            ]},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'think': 'r',
             'tool_calls': [{'name': 'lookup', 'arguments': {'zip': 12345}}]},
            {'role': 'tool', 'name': 'lookup', 'content': '{"error": "zip must be string"}'},
            {'role': 'assistant', 'content': '', 'think': 'r',
             'tool_calls': [{'name': 'lookup', 'arguments': {'zip': '12345'}}]},
            {'role': 'tool', 'name': 'lookup', 'content': '{"result": "ok"}'},
            {'role': 'assistant', 'content': 'all set', 'think': 'r2'},
        ]
    }

    # Test 43: classifier=None -> strict behavior unchanged (schema-fail surfaces)
    print('Test 43: classifier=None -> strict path unchanged')
    _, errs43, reps43 = process_entry(entry_recover)  # strict
    run_test('errors returned', len(errs43) > 0)
    run_test('strict-mode repairs lacks recovery keys',
             not any(k.startswith('tool_recovery') or k == 'tool_response_errors_count' for k in reps43))
    print()

    # Test 44: classifier provided -> tolerated when error response matches and trajectory recovers
    print('Test 44: tolerate schema-fail when matched by error response (value-only retry)')
    res44, errs44, reps44 = process_entry(entry_recover, tool_error_classifier=_err_classifier)
    run_test('no errors', errs44 == [])
    run_test('result returned', res44 is not None)
    run_test('tool_recovery_applied = True', reps44.get('tool_recovery_applied') is True)
    run_test('tool_recovery_tolerated_count = 1', reps44.get('tool_recovery_tolerated_count') == 1)
    run_test('tool_response_errors_count = 1', reps44.get('tool_response_errors_count') == 1)
    print()

    # Test 45: classifier provided + schema-fail with success-shape response -> still rejected
    print('Test 45: schema-fail without matching error response is rejected')
    entry45 = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'name': 'lookup', 'description': '', 'parameters': {
                    'type': 'object', 'properties': {'zip': {'type': 'string'}},
                    'required': ['zip']}},
            ]},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'think': 'r',
             'tool_calls': [{'name': 'lookup', 'arguments': {'zip': 12345}}]},
            {'role': 'tool', 'name': 'lookup', 'content': '{"result": "ok"}'},
            {'role': 'assistant', 'content': 'done', 'think': 'r2'},
        ]
    }
    _, errs45, _ = process_entry(entry45, tool_error_classifier=_err_classifier)
    run_test('errors returned', len(errs45) > 0)
    run_test('schema error still surfaced',
             any('expected string' in e and 'got int' in e for e in errs45))
    print()

    # Test 46: 3+ consecutive errors -> rejected even if no schema-fails
    print('Test 46: 3 consecutive tool-response errors rejected')
    entry46 = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'name': 'a', 'description': '', 'parameters': {'type': 'object',
                 'properties': {'k': {'type': 'integer'}}}},
            ]},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'think': 'r',
             'tool_calls': [{'name': 'a', 'arguments': {'k': 1}}]},
            {'role': 'tool', 'name': 'a', 'content': '{"error": "x"}'},
            {'role': 'assistant', 'content': '', 'think': 'r',
             'tool_calls': [{'name': 'a', 'arguments': {'k': 2}}]},
            {'role': 'tool', 'name': 'a', 'content': '{"error": "y"}'},
            {'role': 'assistant', 'content': '', 'think': 'r',
             'tool_calls': [{'name': 'a', 'arguments': {'k': 3}}]},
            {'role': 'tool', 'name': 'a', 'content': '{"error": "z"}'},
            {'role': 'assistant', 'content': 'giving up', 'think': 'r2'},
        ]
    }
    _, errs46, _ = process_entry(entry46, tool_error_classifier=_err_classifier)
    run_test('errors returned', len(errs46) > 0)
    run_test('rule-2 error mentions consecutive', any('consecutive' in e for e in errs46))
    print()

    # Test 47: RETRY_SAME_TOOL_UNFIXED -> rejected (same keys, both schema-fail)
    print('Test 47: parrot rejected (both calls schema-fail)')
    entry47 = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'name': 'lookup', 'description': '', 'parameters': {'type': 'object',
                 'properties': {'zip': {'type': 'string'}},
                 'required': ['zip']}},
            ]},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'think': 'r',
             'tool_calls': [{'name': 'lookup', 'arguments': {'zip': 12345}}]},  # int -> schema-fail
            {'role': 'tool', 'name': 'lookup', 'content': '{"error": "bad"}'},
            {'role': 'assistant', 'content': '', 'think': 'r',
             'tool_calls': [{'name': 'lookup', 'arguments': {'zip': 67890}}]},  # still int -> schema-fail
            {'role': 'tool', 'name': 'lookup', 'content': '{"error": "still bad"}'},
            {'role': 'assistant', 'content': 'oh well', 'think': 'r2'},
        ]
    }
    _, errs47, _ = process_entry(entry47, tool_error_classifier=_err_classifier)
    run_test('errors returned', len(errs47) > 0)
    run_test('rule-3 error mentions UNFIXED', any('RETRY_SAME_TOOL_UNFIXED' in e for e in errs47))
    print()

    # Test 48: last tool-calling turn ends in error -> rejected
    print('Test 48: trajectory ending in tool error rejected')
    entry48 = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'name': 'a', 'description': '', 'parameters': {'type': 'object',
                 'properties': {'k': {'type': 'integer'}}}},
            ]},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'think': 'r',
             'tool_calls': [{'name': 'a', 'arguments': {'k': 1}}]},
            {'role': 'tool', 'name': 'a', 'content': '{"error": "fail"}'},
            {'role': 'assistant', 'content': 'sorry', 'think': 'r2'},
        ]
    }
    _, errs48, _ = process_entry(entry48, tool_error_classifier=_err_classifier)
    run_test('errors returned', len(errs48) > 0)
    run_test('rule-4 error mentions last tool-calling turn',
             any('last tool-calling turn' in e for e in errs48))
    print()

    # Test 49: consecutive_error_limit=5 -> 3 consecutive errors NOT rejected by rule 2
    print('Test 49: consecutive_error_limit param respected')
    _, errs49, _ = process_entry(entry46, tool_error_classifier=_err_classifier,
                                  consecutive_error_limit=5)
    # Note: rule 4 may still reject because last tool turn has error.
    run_test('rule-2 NOT triggered at limit=5', not any('consecutive' in e for e in errs49))
    print()

    # Test 50: full recovery scenario passes
    # First call schema-fails (missing required 'zip'); retry adds it -> different
    # arg-key set, retry schema-passes, last tool-calling turn succeeds.
    print('Test 50: full recovery scenario')
    entry50 = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'name': 'lookup', 'description': '', 'parameters': {
                    'type': 'object', 'properties': {'zip': {'type': 'string'}},
                    'required': ['zip']}},
            ]},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'think': 'r',
             'tool_calls': [{'name': 'lookup', 'arguments': {}}]},
            {'role': 'tool', 'name': 'lookup', 'content': '{"error": "missing required arg zip"}'},
            {'role': 'assistant', 'content': 'let me try again', 'think': 'r',
             'tool_calls': [{'name': 'lookup', 'arguments': {'zip': '12345'}}]},
            {'role': 'tool', 'name': 'lookup', 'content': '{"result": "ok"}'},
            {'role': 'assistant', 'content': 'all set', 'think': 'r2'},
        ]
    }
    res50, errs50, reps50 = process_entry(entry50, tool_error_classifier=_err_classifier)
    run_test('no errors', errs50 == [])
    run_test('result returned', res50 is not None)
    run_test('tool_recovery_applied = True', reps50.get('tool_recovery_applied') is True)
    run_test('tool_response_errors_count = 1', reps50.get('tool_response_errors_count') == 1)
    print()

    # Test 51: classifier provided + no tool calls -> tool_response_errors_count = 0 in repairs
    print('Test 51: classifier + no tool calls -> audit lands as 0')
    entry51 = {
        'conversation': [
            {'role': 'user', 'content': 'hello'},
            {'role': 'assistant', 'content': 'hi there', 'think': 'r'},
        ]
    }
    res51, errs51, reps51 = process_entry(entry51, tool_error_classifier=_err_classifier)
    run_test('no errors', errs51 == [])
    run_test('result returned', res51 is not None)
    run_test('tool_response_errors_count = 0', reps51.get('tool_response_errors_count') == 0)
    run_test('tool_recovery_applied not present',
             'tool_recovery_applied' not in reps51)
    print()

    # Test 52: require_think=False disables the no-think gate
    print('Test 52: require_think=False allows empty-think records')
    entry52 = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a', 'think': ''},
        ]
    }
    # Default: rejected
    _, errs52a, _ = process_entry(entry52)
    run_test('default rejects empty-think', any('No assistant turn has non-empty think' in e for e in errs52a))
    # Disabled: passes
    res52b, errs52b, reps52b = process_entry(entry52, require_think=False)
    run_test('require_think=False -> no errors', errs52b == [])
    run_test('require_think=False -> result returned', res52b is not None)
    run_test('missing_think_all_turns logged', reps52b.get('missing_think_all_turns') is True)
    # Sanity: when think is present, repair NOT logged
    entry52c = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a', 'think': 'reasoning'},
        ]
    }
    _, _, reps52c = process_entry(entry52c, require_think=False)
    run_test('missing_think_all_turns NOT logged when think present',
             'missing_think_all_turns' not in reps52c)
    print()

    # Test 53: drop extra reasoning keys when desired key is `think`
    print('Test 53: extra reasoning keys dropped when desired=think')
    entry53 = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a',
             'think': 'detailed', 'think_fast': 'condensed',
             'reasoning': 'r', 'reasoning_content': 'rc'},
        ]
    }
    res53, errs53, reps53 = process_entry(entry53, reasoning_effort='high')
    run_test('no errors', errs53 == [])
    run_test('think preserved', res53['conversation'][1].get('think') == 'detailed')
    run_test('think_fast dropped', 'think_fast' not in res53['conversation'][1])
    run_test('reasoning dropped', 'reasoning' not in res53['conversation'][1])
    run_test('reasoning_content dropped', 'reasoning_content' not in res53['conversation'][1])
    run_test('extra_reasoning_keys_dropped = 3', reps53.get('extra_reasoning_keys_dropped') == 3)
    print()

    # Test 54: existing think_fast preserved when desired=think_fast (NOT overwritten by think)
    print('Test 54: desired=think_fast preserves existing think_fast (not overwritten)')
    entry54 = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a',
             'think': 'detailed_long_version',
             'think_fast': 'condensed_version'},
        ]
    }
    res54, errs54, reps54 = process_entry(entry54, reasoning_effort='medium')
    run_test('no errors', errs54 == [])
    run_test('think_fast preserved (not overwritten)',
             res54['conversation'][1].get('think_fast') == 'condensed_version')
    run_test('think dropped', 'think' not in res54['conversation'][1])
    run_test('extra_reasoning_keys_dropped = 1', reps54.get('extra_reasoning_keys_dropped') == 1)
    print()

    # Test 55: desired=think_fast, only think present -> rename think to think_fast
    print('Test 55: desired=think_fast with only think -> rename')
    entry55 = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a', 'think': 'reasoning'},
        ]
    }
    res55, errs55, reps55 = process_entry(entry55, reasoning_effort='medium')
    run_test('no errors', errs55 == [])
    run_test('think_fast = original think', res55['conversation'][1].get('think_fast') == 'reasoning')
    run_test('think dropped', 'think' not in res55['conversation'][1])
    run_test('extra_reasoning_keys_dropped not logged (just a rename)',
             'extra_reasoning_keys_dropped' not in reps55)
    print()

    # Test 56: empty user content rejected
    print('Test 56: empty user content rejected')
    entry56 = {
        'conversation': [
            {'role': 'user', 'content': ''},
            {'role': 'assistant', 'content': 'hi', 'think': 'r'},
        ]
    }
    res56, errs56, _ = process_entry(entry56)
    run_test('errors returned', len(errs56) > 0)
    run_test('empty user content error', any('empty or missing content' in e and 'User message' in e for e in errs56))
    run_test('result is None', res56 is None)
    print()

    # Test 57: missing user content also rejected
    print('Test 57: missing user content rejected')
    entry57 = {
        'conversation': [
            {'role': 'user'},  # no content key at all
            {'role': 'assistant', 'content': 'hi', 'think': 'r'},
        ]
    }
    _, errs57, _ = process_entry(entry57)
    run_test('errors returned', len(errs57) > 0)
    run_test('empty user content error', any('empty or missing content' in e for e in errs57))
    print()

    # Test 58: empty-user error surfaces alongside other per-turn errors
    print('Test 58: empty user content reported alongside other per-turn errors')
    entry58 = {
        'conversation': [
            {'role': 'user', 'content': ''},
            {'role': 'assistant'},  # also fails: missing both content and tool_calls
        ]
    }
    _, errs58, _ = process_entry(entry58)
    run_test('empty-user error present', any('empty or missing content' in e for e in errs58))
    run_test('assistant-missing-both error also present',
             any('Assistant response must have either tools or content' in e for e in errs58))
    print()

    # Test 59: extra reasoning fields stripped BEFORE special-token scan
    print('Test 59: special token in dropped reasoning field is NOT flagged')
    entry59 = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a',
             'think': 'clean reasoning here',
             'think_fast': 'has <|im_start|> leak'},  # would fail special-token check if scanned
        ]
    }
    res59, errs59, reps59 = process_entry(entry59, reasoning_effort='high')
    run_test('no errors (think_fast dropped before scan)', errs59 == [])
    run_test('result returned', res59 is not None)
    run_test('extra_reasoning_keys_dropped logged', reps59.get('extra_reasoning_keys_dropped') == 1)
    print()

    # Test 60: same scenario but desired key is the leaky one -> SHOULD be rejected
    print('Test 60: special token in chosen reasoning key is rejected')
    _, errs60, _ = process_entry(entry59, reasoning_effort='medium')
    run_test('errors returned', len(errs60) > 0)
    run_test('special token error',
             any('<|im_start|>' in e and 'think_fast' in e for e in errs60))
    print()

    # Test 61: all tool responses in conversation are empty -> reject
    print('Test 61: all tool responses empty -> reject')
    entry61 = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'name': 'echo', 'description': '',
                 'parameters': {'type': 'object', 'properties': {'msg': {'type': 'string'}}}},
            ]},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'think': 'r',
             'tool_calls': [{'name': 'echo', 'arguments': {'msg': 'a'}}]},
            {'role': 'tool', 'name': 'echo', 'content': ''},
            {'role': 'assistant', 'content': '', 'think': 'r',
             'tool_calls': [{'name': 'echo', 'arguments': {'msg': 'b'}}]},
            {'role': 'tool', 'name': 'echo', 'content': ''},
            {'role': 'assistant', 'content': 'done', 'think': 'r'},
        ]
    }
    _, errs61, _ = process_entry(entry61)
    run_test('errors returned', len(errs61) > 0)
    run_test('all-tool-responses-empty error',
             any('All tool responses in conversation are empty' in e for e in errs61))
    print()

    # Test 62: at least one non-empty tool response -> NOT rejected by this rule
    print('Test 62: at least one non-empty tool response -> not rejected')
    entry62 = {
        'conversation': [
            {'role': 'system', 'tools': [
                {'name': 'echo', 'description': '',
                 'parameters': {'type': 'object', 'properties': {'msg': {'type': 'string'}}}},
            ]},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'think': 'r',
             'tool_calls': [{'name': 'echo', 'arguments': {'msg': 'a'}}]},
            {'role': 'tool', 'name': 'echo', 'content': ''},
            {'role': 'assistant', 'content': '', 'think': 'r',
             'tool_calls': [{'name': 'echo', 'arguments': {'msg': 'b'}}]},
            {'role': 'tool', 'name': 'echo', 'content': 'OK'},
            {'role': 'assistant', 'content': 'done', 'think': 'r'},
        ]
    }
    _, errs62, _ = process_entry(entry62)
    run_test('all-empty error NOT raised',
             not any('All tool responses in conversation are empty' in e for e in errs62))
    print()

    # Test 63: no tool turns at all -> rule doesn't fire
    print('Test 63: zero tool turns -> rule does not fire')
    entry63 = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a', 'think': 'r'},
        ]
    }
    _, errs63, _ = process_entry(entry63)
    run_test('all-empty error NOT raised',
             not any('All tool responses in conversation are empty' in e for e in errs63))
    print()

    # ----- nullable + enum tests (PR #48) -----
    def _conv_with_call(tool_def, args):
        """Build a minimal valid conversation that exercises a single tool call."""
        return {'conversation': [
            {'role': 'system', 'tools': [tool_def]},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'think': 'r',
             'tool_calls': [{'name': tool_def['name'], 'arguments': args}]},
            {'role': 'tool', 'name': tool_def['name'], 'content': 'OK'},
            {'role': 'assistant', 'content': 'done', 'think': 'r'},
        ]}

    # Test 64: nullable: true accepts null
    print('Test 64: nullable: true accepts null')
    tdef64 = {'name': 'fs', 'parameters': {'type': 'object', 'properties': {
        'output_mode': {'type': 'string', 'nullable': True}}}}
    res64, errs64, _ = process_entry(_conv_with_call(tdef64, {'output_mode': None}))
    run_test('null accepted', errs64 == [])
    run_test('result returned', res64 is not None)
    # And a real string still passes:
    _, errs64b, _ = process_entry(_conv_with_call(tdef64, {'output_mode': 'content'}))
    run_test('string still accepted', errs64b == [])
    # An int still rejected:
    _, errs64c, _ = process_entry(_conv_with_call(tdef64, {'output_mode': 42}))
    run_test('int rejected', any('expected string' in e and 'got int' in e for e in errs64c))
    print()

    # Test 65: null in enum accepts null (without explicit nullable)
    print('Test 65: null in enum accepts null')
    tdef65 = {'name': 'fs', 'parameters': {'type': 'object', 'properties': {
        'mode': {'type': 'string', 'enum': ['a', 'b', None]}}}}
    res65, errs65, _ = process_entry(_conv_with_call(tdef65, {'mode': None}))
    run_test('null accepted via enum', errs65 == [])
    run_test('result returned', res65 is not None)
    print()

    # Test 66: enum membership enforced (typed)
    print('Test 66: enum membership enforced (typed)')
    tdef66 = {'name': 'fs', 'parameters': {'type': 'object', 'properties': {
        'mode': {'type': 'string', 'enum': ['a', 'b']}}}}
    _, errs66a, _ = process_entry(_conv_with_call(tdef66, {'mode': 'c'}))
    run_test('out-of-enum rejected', any('not in enum' in e and "'c'" in e for e in errs66a))
    res66b, errs66b, _ = process_entry(_conv_with_call(tdef66, {'mode': 'a'}))
    run_test('in-enum accepted', errs66b == [])
    run_test('result returned', res66b is not None)
    print()

    # Test 67: enum membership enforced (untyped — enum is the only constraint)
    print('Test 67: enum membership enforced on untyped schema')
    tdef67 = {'name': 'fs', 'parameters': {'type': 'object', 'properties': {
        'mode': {'enum': ['a', 'b']}}}}
    _, errs67, _ = process_entry(_conv_with_call(tdef67, {'mode': 'c'}))
    run_test('out-of-enum rejected', any('not in enum' in e for e in errs67))
    print()

    # Test 68: enum enforced inside nested object
    print('Test 68: enum enforced inside nested object')
    tdef68 = {'name': 'fs', 'parameters': {'type': 'object', 'properties': {
        'opts': {'type': 'object', 'properties': {
            'mode': {'type': 'string', 'enum': ['a', 'b']}}}}}}
    _, errs68, _ = process_entry(_conv_with_call(tdef68, {'opts': {'mode': 'c'}}))
    run_test('nested out-of-enum rejected',
             any('opts.mode' in e and 'not in enum' in e for e in errs68))
    print()

    # Test 69: bool/int enum distinction (Python True == 1, but JSON treats them as distinct)
    print('Test 69: bool/int enum distinction (untyped)')
    tdef69_int = {'name': 'fs', 'parameters': {'type': 'object', 'properties': {
        'x': {'enum': [0, 1, 2]}}}}
    _, errs69a, _ = process_entry(_conv_with_call(tdef69_int, {'x': True}))
    run_test('True NOT in int-enum [0,1,2]',
             any('not in enum' in e and 'True' in e for e in errs69a))
    _, errs69b, _ = process_entry(_conv_with_call(tdef69_int, {'x': 1}))
    run_test('1 IS in int-enum', errs69b == [])

    tdef69_bool = {'name': 'fs', 'parameters': {'type': 'object', 'properties': {
        'x': {'enum': [True, False]}}}}
    _, errs69c, _ = process_entry(_conv_with_call(tdef69_bool, {'x': 1}))
    run_test('1 NOT in bool-enum [True,False]',
             any('not in enum' in e for e in errs69c))
    _, errs69d, _ = process_entry(_conv_with_call(tdef69_bool, {'x': True}))
    run_test('True IS in bool-enum', errs69d == [])
    print()

    # ----- Crash hardening: malformed input must return errors, never raise -----
    print('Test 70: crash hardening — malformed input returns errors (no exceptions)')

    def _no_crash(name, expected_substr, *args, **kwargs):
        try:
            _, errs, _ = process_entry(*args, **kwargs)
        except Exception as ex:
            run_test(f'{name} (crashed: {type(ex).__name__}: {ex})', False)
            return
        run_test(name, any(expected_substr in e for e in errs))

    _no_crash('empty conversation', 'empty or not a list', {'conversation': []})
    _no_crash('non-list conversation', 'empty or not a list', {'conversation': 'not a list'})
    _no_crash('non-dict turn', 'not a dict', {'conversation': ['just a string']})
    _no_crash('turn missing role', 'missing role', {'conversation': [{'content': 'no role'}]})
    _no_crash('invalid reasoning_effort', 'Invalid reasoning_effort',
        {'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a', 'think': 'r'},
        ]}, reasoning_effort='ultra')
    _no_crash('non-JSON-serializable field', 'could not be serialized',
        {'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a', 'think': 'r', 'meta': {1, 2, 3}},
        ]})
    print()

    # Test 71: assistant with tool_calls must be followed by a tool turn
    print('Test 71: assistant with tool_calls must be followed by tool turn')
    tdef71 = {'name': 'echo', 'parameters': {
        'type': 'object', 'properties': {'msg': {'type': 'string'}}}}
    # 71a: tool_calls then user (not tool) -> reject
    entry71a = {'conversation': [
        {'role': 'system', 'tools': [tdef71]},
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': '', 'think': 'r',
         'tool_calls': [{'name': 'echo', 'arguments': {'msg': 'hi'}}]},
        {'role': 'user', 'content': 'q2'},
        {'role': 'assistant', 'content': 'done', 'think': 'r'},
    ]}
    _, errs71a, _ = process_entry(entry71a)
    run_test('tool_calls then user -> rejected',
             any('expected \'tool\'' in e for e in errs71a))

    # 71b: tool_calls and conversation ends -> ACCEPTED (single-turn tool calling data)
    entry71b = {'conversation': [
        {'role': 'system', 'tools': [tdef71]},
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': '', 'think': 'r',
         'tool_calls': [{'name': 'echo', 'arguments': {'msg': 'hi'}}]},
    ]}
    res71b, errs71b, _ = process_entry(entry71b)
    run_test('tool_calls at end -> accepted (single-turn tool calling)', errs71b == [])
    run_test('result returned', res71b is not None)

    # 71c: tool_calls then tool -> accepted (sanity)
    entry71c = {'conversation': [
        {'role': 'system', 'tools': [tdef71]},
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': '', 'think': 'r',
         'tool_calls': [{'name': 'echo', 'arguments': {'msg': 'hi'}}]},
        {'role': 'tool', 'name': 'echo', 'content': 'OK'},
        {'role': 'assistant', 'content': 'done', 'think': 'r'},
    ]}
    res71c, errs71c, _ = process_entry(entry71c)
    run_test('tool_calls then tool -> accepted', errs71c == [])
    run_test('result returned', res71c is not None)
    print()

    # Test 72: whitespace-only content rejected (assistant + user)
    print('Test 72: whitespace-only content rejected')
    # 72a: assistant with whitespace-only content + no tool_calls -> reject
    entry72a = {'conversation': [
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': '   \n\t', 'think': 'r'},
    ]}
    _, errs72a, _ = process_entry(entry72a)
    run_test('assistant whitespace-only -> rejected',
             any('MISSING BOTH' in e for e in errs72a))

    # 72b: user with whitespace-only content -> reject
    entry72b = {'conversation': [
        {'role': 'user', 'content': '   '},
        {'role': 'assistant', 'content': 'a', 'think': 'r'},
    ]}
    _, errs72b, _ = process_entry(entry72b)
    run_test('user whitespace-only -> rejected',
             any('empty or missing content' in e for e in errs72b))

    # 72c: assistant with whitespace content but real tool_calls -> accepted
    tdef72 = {'name': 'echo', 'parameters': {
        'type': 'object', 'properties': {'msg': {'type': 'string'}}}}
    entry72c = {'conversation': [
        {'role': 'system', 'tools': [tdef72]},
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': '   ', 'think': 'r',
         'tool_calls': [{'name': 'echo', 'arguments': {'msg': 'hi'}}]},
        {'role': 'tool', 'name': 'echo', 'content': 'OK'},
        {'role': 'assistant', 'content': 'done', 'think': 'r'},
    ]}
    res72c, errs72c, _ = process_entry(entry72c)
    run_test('whitespace content + tool_calls -> accepted', errs72c == [])
    run_test('result returned', res72c is not None)
    print()

    # Test 73: malformed schema.required (bool instead of list) is reported, not crashed
    print('Test 73: schema.required as bool — strict reject (no crash)')
    bad_tdef = {
        'name': 'echo', 'description': '',
        'parameters': {
            'type': 'object',
            'properties': {'msg': {'type': 'string'}},
            'required': True,  # bug: should be a list
        },
    }
    entry73 = {'conversation': [
        {'role': 'system', 'tools': [bad_tdef]},
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': '', 'think': 'r',
         'tool_calls': [{'name': 'echo', 'arguments': {'msg': 'hi'}}]},
        {'role': 'tool', 'name': 'echo', 'content': 'OK'},
        {'role': 'assistant', 'content': 'done', 'think': 'r'},
    ]}
    _, errs73, _ = process_entry(entry73)
    run_test('errors returned (no crash)', len(errs73) > 0)
    run_test('schema.required error surfaced',
             any('schema.required is bool' in e for e in errs73))
    print()

    # Test 74: malformed nested schema.required (bool) at object property
    print('Test 74: nested schema.required as bool — strict reject (no crash)')
    nested_bad_tdef = {
        'name': 'set_addr', 'description': '',
        'parameters': {
            'type': 'object',
            'properties': {
                'addr': {
                    'type': 'object',
                    'properties': {'zip': {'type': 'string'}},
                    'required': True,  # bug at nested level
                },
            },
        },
    }
    entry74 = {'conversation': [
        {'role': 'system', 'tools': [nested_bad_tdef]},
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': '', 'think': 'r',
         'tool_calls': [{'name': 'set_addr', 'arguments': {'addr': {'zip': '12345'}}}]},
        {'role': 'tool', 'name': 'set_addr', 'content': 'OK'},
        {'role': 'assistant', 'content': 'done', 'think': 'r'},
    ]}
    _, errs74, _ = process_entry(entry74)
    run_test('errors returned (no crash)', len(errs74) > 0)
    run_test('nested schema.required error surfaced',
             any('schema.required is bool' in e for e in errs74))
    print()

    # Test 75: malformed schema.properties (list instead of dict) is reported, not crashed
    print('Test 75: schema.properties as list — strict reject (no crash)')
    bad_props_tdef = {
        'name': 'echo', 'description': '',
        'parameters': {
            'type': 'object',
            'properties': ['msg'],  # bug: should be a dict
        },
    }
    entry75 = {'conversation': [
        {'role': 'system', 'tools': [bad_props_tdef]},
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': '', 'think': 'r',
         'tool_calls': [{'name': 'echo', 'arguments': {'msg': 'hi'}}]},
        {'role': 'tool', 'name': 'echo', 'content': 'OK'},
        {'role': 'assistant', 'content': 'done', 'think': 'r'},
    ]}
    _, errs75, _ = process_entry(entry75)
    run_test('errors returned (no crash)', len(errs75) > 0)
    run_test('schema.properties error surfaced',
             any('schema.properties is list' in e for e in errs75))
    print()

    # Test 76: malformed schema.type entry (dict in type list) — strict reject (no crash)
    print('Test 76: schema.type list contains non-string — strict reject (no crash)')
    bad_type_tdef = {
        'name': 'echo', 'description': '',
        'parameters': {
            'type': 'object',
            'properties': {
                'msg': {'type': ['string', {'nested': 'thing'}]},  # bug: dict in type list
            },
        },
    }
    entry76 = {'conversation': [
        {'role': 'system', 'tools': [bad_type_tdef]},
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': '', 'think': 'r',
         'tool_calls': [{'name': 'echo', 'arguments': {'msg': 'hi'}}]},
        {'role': 'tool', 'name': 'echo', 'content': 'OK'},
        {'role': 'assistant', 'content': 'done', 'think': 'r'},
    ]}
    _, errs76, _ = process_entry(entry76)
    run_test('errors returned (no crash)', len(errs76) > 0)
    run_test('schema.type entry error surfaced',
             any('schema.type entry is dict' in e for e in errs76))
    print()

    # Test 76b: `null` literal inside a `type` list is the third nullability
    # spelling — equivalent to `nullable: true` and `null in enum`. This is
    # the shape emitted by zod/pydantic-v1/typescript-json-schema generators
    # and is widespread in MCP server tool definitions.
    print('Test 76b: null in type list accepts null value')
    tdef76b = {'name': 'fs', 'parameters': {'type': 'object', 'properties': {
        'hex': {'type': ['string', None]}}}}
    res76b, errs76b, _ = process_entry(_conv_with_call(tdef76b, {'hex': None}))
    run_test('null accepted via type-list', errs76b == [])
    run_test('result returned', res76b is not None)
    print()

    # Test 76c: same schema still type-checks non-null values strictly
    print('Test 76c: null in type list — non-null still type-checked')
    _, errs76c_str, _ = process_entry(_conv_with_call(tdef76b, {'hex': '#ff0000'}))
    run_test('string accepted', errs76c_str == [])
    _, errs76c_int, _ = process_entry(_conv_with_call(tdef76b, {'hex': 42}))
    run_test('int rejected', any('expected' in e and 'got int' in e for e in errs76c_int))
    print()

    # Test 76d: null in type list combined with null in enum — both nullability
    # spellings present together should not break anything (idempotence).
    print('Test 76d: type-list null + enum null are consistent')
    tdef76d = {'name': 'fs', 'parameters': {'type': 'object', 'properties': {
        'mode': {'type': ['string', None], 'enum': ['a', 'b', None]}}}}
    _, errs76d_null, _ = process_entry(_conv_with_call(tdef76d, {'mode': None}))
    run_test('null accepted', errs76d_null == [])
    _, errs76d_in, _ = process_entry(_conv_with_call(tdef76d, {'mode': 'a'}))
    run_test('in-enum string accepted', errs76d_in == [])
    _, errs76d_out, _ = process_entry(_conv_with_call(tdef76d, {'mode': 'c'}))
    run_test('out-of-enum string rejected',
             any('not in enum' in e for e in errs76d_out))
    print()

    # Test 77: immutable_reasoning_effort=False (default) downgrades no-think
    # trajectories to 'low' (think_faster) regardless of requested effort.
    print("Test 77: no-think + require_think=False + mutable -> downgraded to think_faster")
    entry77 = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a'},
        ]
    }
    res77, errs77, reps77 = process_entry(
        entry77, reasoning_effort='high', require_think=False,
    )
    run_test('result returned', res77 is not None and errs77 == [])
    run_test('effort_downgraded_to_low_no_think logged',
             reps77.get('effort_downgraded_to_low_no_think') is True)
    run_test('assistant turn carries think_faster (not think)',
             'think_faster' in res77['conversation'][1] and 'think' not in res77['conversation'][1])
    # And requesting medium downgrades just the same
    entry77b = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a'},
        ]
    }
    res77b, _, reps77b = process_entry(
        entry77b, reasoning_effort='medium', require_think=False,
    )
    run_test('medium also downgraded',
             reps77b.get('effort_downgraded_to_low_no_think') is True
             and 'think_faster' in res77b['conversation'][1])
    print()

    # Test 78: immutable_reasoning_effort=True suppresses the downgrade so each
    # 3-effort fan-out keeps its own tag even on no-think trajectories.
    print("Test 78: no-think + require_think=False + immutable -> kept at requested effort")
    entry78 = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a'},
        ]
    }
    res78, errs78, reps78 = process_entry(
        entry78, reasoning_effort='high', require_think=False,
        immutable_reasoning_effort=True,
    )
    run_test('result returned', res78 is not None and errs78 == [])
    run_test('effort_downgraded_to_low_no_think NOT logged',
             'effort_downgraded_to_low_no_think' not in reps78)
    run_test('assistant turn carries think (requested effort preserved)',
             'think' in res78['conversation'][1] and 'think_faster' not in res78['conversation'][1])
    run_test('missing_think_all_turns still logged',
             reps78.get('missing_think_all_turns') is True)
    print()

    # Test 79: trajectory WITH think content -> no downgrade regardless of flag
    print('Test 79: has-think trajectory -> no downgrade')
    entry79 = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a', 'think': 'pondering'},
        ]
    }
    res79, errs79, reps79 = process_entry(
        entry79, reasoning_effort='high', require_think=False,
    )
    run_test('result returned', res79 is not None and errs79 == [])
    run_test('downgrade NOT logged',
             'effort_downgraded_to_low_no_think' not in reps79)
    run_test('assistant turn keeps think with content',
             res79['conversation'][1].get('think') == 'pondering')
    print()

    # Test 80: requested effort already 'low' -> no downgrade-repair noise
    print("Test 80: no-think + require_think=False + requested 'low' -> no repair noise")
    entry80 = {
        'conversation': [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a'},
        ]
    }
    res80, _, reps80 = process_entry(
        entry80, reasoning_effort='low', require_think=False,
    )
    run_test('result returned', res80 is not None)
    run_test('downgrade NOT logged when already low',
             'effort_downgraded_to_low_no_think' not in reps80)
    run_test('think_faster present', 'think_faster' in res80['conversation'][1])
    print()


    # ------------------------------------------------------------------
    # Tests 81-88: degenerate-repetition gate (repetition_check.py)
    # ------------------------------------------------------------------
    def _conv(think='thinking', content='answer', n=1):
        c = []
        for i in range(n):
            c.append({'role': 'user', 'content': f'q{i}'})
            c.append({'role': 'assistant', 'think': think, 'content': content})
        return {'conversation': c}

    # Test 81: a healthy record is untouched by the gate
    print('Test 81: repetition gate passes healthy reasoning')
    res81, err81, _ = process_entry(_conv(
        think='Let me read the file, then check the failing assertion.',
        content='The bug is an off-by-one in the loop bound.'))
    run_test('healthy record accepted', res81 is not None and not err81)
    print()

    # Test 82: the whitespace gate is TRAILING-only, and this is the whole point.
    # A run that terminates a line is the dsv32 emission wall. A run with
    # printable content on BOTH sides is column padding in a pandas/xlsx render the
    # model quoted -- measured 0 true positives and 16 false positives on the hand
    # labels, so max_space_run was replaced by max_trailing_space_run (precision
    # 0.20 -> 0.80, 394 -> 133 corpus docs).
    print('Test 82: whitespace gate is trailing-only')
    _, err82, _ = process_entry(_conv(
        think='Line one.' + ' ' * 300 + '\nLine two, so the run is not at the end.'))
    run_test('trailing run rejected',
             any('think_max_trailing_space_run' in e for e in err82))
    res82b, err82b, _ = process_entry(_conv(
        think='col_a' + ' ' * 3000 + 'col_b   NaN   NaN'))
    run_test('interior run (quoted table padding) accepted',
             res82b is not None and not err82b)
    print()

    # Test 83: TRAILING whitespace is REPAIRED, not rejected. This is the
    # contract between the strip and the gate: the strip runs first, so an
    # unambiguous tail is fixed rather than costing a record. Verified against
    # 8 real records from our regression corpus -- all repair to space_run 0.
    print('Test 83: trailing whitespace repaired, not rejected')
    res83, err83, rep83 = process_entry(_conv(think='Done.' + ' ' * 3000))
    run_test('accepted, not rejected', res83 is not None and not err83)
    run_test('repair logged', rep83.get('think_whitespace_stripped'))
    run_test('tail actually gone', res83['conversation'][1]['think'] == 'Done.')
    print()

    # Test 84: cross-turn duplicate reasoning -- the canonical loop. Every
    # per-turn metric reads ~1 here; only max_think_dup sees it.
    print('Test 84: cross-turn duplicate reasoning rejected')
    _, err84, _ = process_entry(_conv(
        think="We are stuck. Let's think if there is any other tool or data "
              'source we have not considered.', n=15))
    run_test('rejected by max_think_dup',
             any('max_think_dup' in e for e in err84))
    print()

    # Test 85: user turns and tool responses are NEVER gated
    print('Test 85: repetitive user/tool text does not reject')
    entry85 = {'conversation': [
        {'role': 'user', 'content': 'log:\n' + 'same line\n' * 400},
        {'role': 'assistant', 'think': 'A repetitive log.', 'content': 'ok',
         'tool_calls': [{'name': 't', 'arguments': {}}]},
        {'role': 'tool', 'content': 'row\n' * 2000},
        {'role': 'assistant', 'think': 'Now I can answer.', 'content': 'done'},
    ], 'system': None}
    entry85['conversation'].insert(0, {'role': 'system', 'content': 'sys',
                                       'tools': [{'name': 't', 'description': 'd',
                                                  'parameters': {'type': 'object',
                                                                 'properties': {}}}]})
    entry85.pop('system')
    res85, err85, _ = process_entry(entry85)
    run_test('accepted despite repetitive user + tool text',
             res85 is not None and not err85)
    print()

    # Test 86: content is gated too, on its own metrics at looser thresholds.
    # Content still uses the RAW char/token runs -- they measured 97.9% and 100%
    # on full censuses there, so there was nothing to reformulate.
    print('Test 86: content gated at its own looser threshold')
    _, err86, _ = process_entry(_conv(content='x' * 2000))
    run_test('content char run rejected',
             any('content_max_char_run' in e for e in err86))
    res86b, err86b, _ = process_entry(_conv(content='x' * 700))
    run_test('700 passes content', res86b is not None and not err86b)
    # reasoning uses glyph_run at 1000: same 700 chars is fine, 1200 is not
    res86c, err86c, _ = process_entry(_conv(think='x' * 700))
    run_test('700 passes reasoning glyph gate', res86c is not None and not err86c)
    _, err86d, _ = process_entry(_conv(think='x' * 1200))
    run_test('1200 fails reasoning glyph gate',
             any('think_max_glyph_run' in e for e in err86d))
    _, err86e, _ = process_entry(_conv(think='=' * 5000))
    run_test('rule glyphs never fire (a `# ====` banner is typography)',
             not any('glyph' in e for e in err86e))
    print()

    # Test 87: the gate can be turned off, and thresholds overridden per source.
    # ARC-AGI needs this -- its reasoning quotes puzzle grids verbatim.
    print('Test 87: repetition_check=False and per-source threshold override')
    _wall = 'Line one.' + ' ' * 3000 + '\nLine two.'
    res87, err87, _ = process_entry(_conv(think=_wall), repetition_check=False)
    run_test('gate off -> accepted', res87 is not None and not err87)
    res87b, err87b, _ = process_entry(
        _conv(think=_wall),
        repetition_thresholds={'reasoning': {'think_max_trailing_space_run': 5000}})
    run_test('raised threshold -> accepted', res87b is not None and not err87b)
    print()

    # Test 88: identical tool call is a COMBINED rule. Many identical calls in a
    # LONG healthy trajectory must not reject -- only when they dominate it.
    print('Test 88: identical tool call needs count AND fraction')
    tools = [{'name': 'read', 'description': 'd',
              'parameters': {'type': 'object',
                             'properties': {'path': {'type': 'string'}}}}]
    def _traj(n_same, n_varied, interleave=False):
        # interleave=True models legitimate iterate-and-check: the identical call
        # recurs, but a DIFFERENT call always happens in between, so no consecutive
        # run forms. That distinction is the whole point of the consecutive rule
        # (test 91) -- a back-to-back run of identical calls is a loop, an
        # interleaved one is a trajectory making progress.
        c = [{'role': 'system', 'content': 's', 'tools': tools}]
        seq = []
        if interleave:
            for i in range(max(n_same, n_varied)):
                if i < n_varied:
                    seq.append({'path': f'f{i}.py'})
                if i < n_same:
                    seq.append({})
        else:
            seq = [{}] * n_same + [{'path': f'f{i}.py'} for i in range(n_varied)]
        for i, args in enumerate(seq):
            c.append({'role': 'user', 'content': f'q{i}'})
            c.append({'role': 'assistant', 'think': f'step {i}', 'content': '',
                      'tool_calls': [{'name': 'read', 'arguments': args}]})
            c.append({'role': 'tool', 'content': 'ok'})
        c.append({'role': 'assistant', 'think': 'Done.', 'content': 'finished'})
        return {'conversation': c}
    res88a, err88a, _ = process_entry(_traj(30, 0))
    run_test('30 identical calls dominating -> rejected',
             any('identical_tool_call' in e for e in err88a))
    res88b, err88b, _ = process_entry(_traj(12, 60, interleave=True))
    run_test('12 identical INTERLEAVED in a 72-call trajectory -> accepted',
             res88b is not None and not err88b)
    print()


    # Test 89: the scan cap must not hide a defect that sits late in a long field.
    # quality_checks.SCAN_CAP_CHARS was 400_000, which made EVERY class of degeneracy
    # invisible past that offset -- measured, all five below were accepted. It is now
    # None (no truncation); removing it cost 1.02x over 306,090 real records, and newly
    # caught 51 genuine defects in one long-proof source (single-turn records with 600k-1.2M
    # char reasoning that loops and never terminates).
    print('Test 89: defects past 400k chars are still caught')
    import random as _rnd
    from quality_checks import SCAN_CAP_CHARS as _CAP
    run_test('scan cap disabled in production', _CAP is None)
    _r = _rnd.Random(0)
    _w = ['inspect', 'module', 'failing', 'assertion', 'handler', 'buffer', 'request',
          'parser', 'index', 'cache', 'thread', 'socket', 'commit', 'branch', 'schema',
          'because', 'however', 'therefore', 'initial', 'result', 'value', 'offset']
    _pad, _n = [], 0
    while _n < 420_000:                      # non-repetitive, so the padding is clean
        _s = ' '.join(_r.choice(_w) for _ in range(_r.randint(9, 18))) + '. '
        _pad.append(_s); _n += len(_s)
    _pad = ''.join(_pad)
    _clean, _errs_clean, _ = process_entry(_conv(think=_pad))
    run_test('420k of clean prose is accepted', _clean is not None and not _errs_clean)
    for _label, _tail, _rule in (
            ('trailing-space wall', '\nStuck.' + ' ' * 4000 + '\nx', 'think_max_trailing_space_run'),
            ('glyph wall',          '\n' + '\u6816' * 5000,            'think_max_glyph_run'),
            ('token wall',          '\n' + 'boom ' * 3000,             'think_max_unfenced_token_run'),
            ('looped sentence',     '\n' + 'We are stuck, let us reconsider the available tools now. ' * 300,
                                                                       'think_max_prose_sentence_repeat')):
        _, _e89, _ = process_entry(_conv(think=_pad + _tail))
        run_test(f'{_label} past 400k rejected by {_rule}',
                 any(_rule in _x for _x in _e89))
    print()


    # Test 90: delimiter-free repetition (shingle) + consecutive identical calls.
    print('Test 90: shingle catches walls with no delimiter')
    # The motivating record: 817,794 chars of "DataGridViewTextBoxColumn" x32,630 with
    # no whitespace. Every delimiter-based detector read ~0 (char_run 68, token_run 1,
    # prose_sentence 0, wordy_line 2) and scan_repetition returned []. Census over
    # 218,637 non-ARC records: the rule fires on 0.021%, and 99.77% of records score
    # <= 0.2, so 0.90 sits in an empty band.
    _wall = 'DataGridViewTextBoxColumn' * 2000
    _, _e90, _ = process_entry(_conv(content=_wall))
    run_test('no-delimiter wall rejected',
             any('content_max_shingle_dup_frac' in _x for _x in _e90))
    _, _e90b, _ = process_entry(_conv(think=_wall))
    run_test('same in reasoning',
             any('think_max_shingle_dup_frac' in _x for _x in _e90b))
    # a 100%-whitespace answer -- real: a browser-agent record, 18,607 chars
    # over 2,060 lines with ZERO non-blank. Non-empty, and content is never stripped,
    # so nothing else catches it.
    _, _e90c, _ = process_entry(_conv(content='   \n' * 4000))
    run_test('all-whitespace answer rejected', bool(_e90c))
    # must NOT fire on ordinary long prose
    import random as _r2
    _rr = _r2.Random(7)
    _vocab = ['module', 'assertion', 'buffer', 'parser', 'commit', 'schema', 'tensor',
              'because', 'however', 'result', 'offset', 'segment', 'handler', 'index']
    _long = ' '.join(_rr.choice(_vocab) for _ in range(4000))
    _r90, _e90d, _ = process_entry(_conv(think=_long))
    run_test('ordinary long prose accepted', _r90 is not None and not _e90d)
    print()

    # Test 91: consecutive identical tool calls. Measured on 139 hand labels:
    #   current rule (count>=10 AND frac>=0.30)          precision 0.646
    #   + consec>=3 conjunct                             precision 0.859
    #   consec>=8 standalone                             precision 0.978 (44/45)
    # The standalone rule catches what the fraction rule structurally cannot: a long
    # trajectory with a long stuck run (a coding-agent record with 30
    # consecutive identical failing `npm run build`, frac 0.280, misses by 0.02).
    print('Test 91: consecutive identical calls')
    _tools91 = [{'name': 'build', 'description': 'd',
                 'parameters': {'type': 'object', 'properties': {'p': {'type': 'string'}}}}]
    def _traj91(n_consec, n_other):
        c = [{'role': 'system', 'content': 's', 'tools': _tools91}]
        for i in range(n_consec):
            c.append({'role': 'assistant', 'think': f'step {i}', 'content': '',
                      'tool_calls': [{'name': 'build', 'arguments': {}}]})
            c.append({'role': 'tool', 'content': 'fail'})
        for i in range(n_other):
            c.append({'role': 'assistant', 'think': f'other {i}', 'content': '',
                      'tool_calls': [{'name': 'build', 'arguments': {'p': f'f{i}'}}]})
            c.append({'role': 'tool', 'content': 'ok'})
        c.append({'role': 'assistant', 'think': 'Done.', 'content': 'finished'})
        return {'conversation': c}
    # 12 consecutive inside a 100-call trajectory: frac is only 0.12, so the combined
    # rule cannot see it. The standalone consecutive rule must.
    _, _e91, _ = process_entry(_traj91(12, 88))
    run_test('long stuck run in a long trajectory rejected',
             any('consecutive_identical_tool_call' in _x for _x in _e91))
    # interleaved observe-step: many identical calls, never consecutive -> accepted.
    # This is the browser_get_state false-positive class the conjunct removes.
    _c91 = [{'role': 'system', 'content': 's', 'tools': _tools91}]
    for i in range(14):
        _c91 += [{'role': 'assistant', 'think': f'act {i}', 'content': '',
                  'tool_calls': [{'name': 'build', 'arguments': {'p': f'page{i}'}}]},
                 {'role': 'tool', 'content': 'ok'},
                 {'role': 'assistant', 'think': f'observe {i}', 'content': '',
                  'tool_calls': [{'name': 'build', 'arguments': {}}]},
                 {'role': 'tool', 'content': 'state'}]
    _c91.append({'role': 'assistant', 'think': 'Done.', 'content': 'answer'})
    _r91b, _e91b, _ = process_entry({'conversation': _c91})
    run_test('interleaved observe step accepted (never consecutive)',
             _r91b is not None and not _e91b)
    print()


    # Test 92: spaced / doubled template markers. The substring lists are exact, so a
    # single inserted space or a doubled fullwidth pipe walked straight past them.
    # All four of these were reported from the field and reproduced as BYPASSES.
    print('Test 92: spaced/doubled template markers rejected')
    for _m in ('<\uff5c\uff5cDSML\uff5c\uff5ctool_calls>', '< \uff5cDSML\uff5cparameter name=',
               '< /think>', '</ think>', '<  think >'):
        _, _e92, _ = process_entry(_conv(think=f'reasoning {_m} more'))
        run_test(f'rejected {_m!r}', bool(_e92))
    # and the shape regexes must not fire on ordinary prose or code
    for _ok in ('normal text with <angle> brackets', 'if (a<b) { think(); }',
                'I think this is right.', 'x < 5 / think about it'):
        _r92, _e92b, _ = process_entry(_conv(think=_ok))
        run_test(f'accepted {_ok[:26]!r}', _r92 is not None and not _e92b)
    print()

    # Test 93: Kimi K3 special tokens (one of the two generation models).
    print('Test 93: Kimi K3 markers rejected')
    for _m in ('<|end_of_msg|>', '<osagent_mode>', '<|media_begin|>', '<|sep|>'):
        _, _e93, _ = process_entry(_conv(think=f'text {_m} text'))
        run_test(f'rejected {_m}', bool(_e93))
    print()

    # Test 94: generation-harness preamble leaking into assistant output. DS-Flash at
    # max effort is given a "Reasoning Effort: ..." system message that is not part of
    # the trajectory; it leaks when the model talks about it. The model PARAPHRASES it
    # ("Absolute maximum" for "Beyond maximum"), so this keys on distinctive fragments.
    print('Test 94: harness preamble leak rejected')
    _leak = ('Actually in the prompt we see: Reasoning Effort: Absolute maximum, '
             'so I must be thorough.')
    _, _e94, _ = process_entry(_conv(think=_leak))
    run_test('paraphrased leak in think rejected',
             any('harness preamble' in _x for _x in _e94))
    for _frag in ('exhaustive, relentless, and uncompromising',
                  'leaving absolutely nothing to chance',
                  'no assumption remains unchecked'):
        _, _e94b, _ = process_entry(_conv(think=f'The system said: {_frag}.'))
        run_test(f'fragment rejected: {_frag[:30]}', bool(_e94b))
    # must NOT fire on ordinary careful reasoning, nor on a USER pasting the text
    _r94, _e94c, _ = process_entry(_conv(
        think='I will reason carefully and verify the solution from several angles.'))
    run_test('ordinary careful reasoning accepted', _r94 is not None and not _e94c)
    _u94 = {'conversation': [{'role': 'user', 'content': _leak},
                             {'role': 'assistant', 'think': 't', 'content': 'a'}]}
    _r94b, _e94d, _ = process_entry(_u94)
    run_test('same text in a USER turn accepted (role scoping)',
             _r94b is not None and not _e94d)
    print()

    # Test 95: the PRODUCTION render. Token counting uses json+xml_typed (an upper
    # bound), but training renders markdown+xml, and records were observed passing the
    # first and failing the second. Both must now render. A markdown render that
    # silently falls back to verbatim JSON for a tool is a WARNING, not a rejection.
    print('Test 95: markdown/xml production render + json-fallback warning')
    _tools95 = [{'name': 'ok', 'description': 'd',
                 'parameters': {'type': 'object',
                                'properties': {'p': {'type': 'string'}}}}]
    _c95 = {'conversation': [
        {'role': 'system', 'content': 's', 'tools': _tools95},
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'think': 'reasoning', 'content': 'answer'}]}
    _r95, _e95, _rep95 = process_entry(json.loads(json.dumps(_c95)))
    run_test('renderable tools accepted', _r95 is not None and not _e95)
    run_test('no fallback warning when fully renderable',
             'tool_presentation_json_fallback_triggered' not in _rep95)
    # an unknown container-valued key at the FUNCTION level forces verbatim JSON
    _c95b = json.loads(json.dumps(_c95))
    _c95b['conversation'][0]['tools'][0]['x_meta'] = {'a': 1}
    _r95b, _e95b, _rep95b = process_entry(_c95b)
    run_test('fallback record still ACCEPTED (warning, not rejection)',
             _r95b is not None and not _e95b)
    run_test('fallback warning recorded in repairs',
             _rep95b.get('tool_presentation_json_fallback_triggered') is True)
    print()


    # Test 96: laundered Anthropic tool-call scaffold. An UPSTREAM "neutralizer"
    # repairs these markers by inserting a space, which walks them past a substring
    # filter -- records even carry a `markers_neutralized` field recording it. In the
    # shipped corpus the mutated forms outnumber the clean ones 746 to 37, so the
    # exact list was catching the 5% that had not been "fixed".
    print('Test 96: laundered invoke/antml scaffold rejected')
    for _m in ('< invoke name="bash">', '< /invoke>', 'antml:invoke', 'antml:parameter',
               '</function_calls>', '< function_calls >'):
        _, _e96, _ = process_entry(_conv(think=f'reasoning {_m} more'))
        run_test(f'rejected {_m!r}', bool(_e96))
    # These must stay CLEAN. `<function=`/`</function>` and `<parameter ` were
    # DELIBERATELY dropped earlier because they collide with the active template's own
    # XML tool-spec rendering; adding the invoke/antml family must not resurrect them.
    for _ok in ('<function name="f">x</function>', '<parameter name="p">v</parameter>',
                'We invoke the function here.', 'the invoke() method',
                'x = function_calls_total + 1', 'if (a<b) invoke;'):
        _r96, _e96b, _ = process_entry(_conv(think=_ok))
        run_test(f'accepted {_ok[:30]!r}', _r96 is not None and not _e96b)
    print()


    # Test 99: the shape patterns are CASE-SENSITIVE, and that is load-bearing.
    # They were written with re.I, which generated false positives on legitimate
    # content while buying no recall -- the models emit template markers in exact
    # lowercase, and the substring lists these patterns backstop are themselves
    # case-sensitive. Every string below was FLAGGED under re.I and is real content
    # from the shipped corpus.
    print('Test 99: leakage patterns are case-sensitive')
    for _label, _txt in (
            ('C++ header doc comment',
             'bool (idaapi *run)(size_t arg); ///< Invoke plugin.'),
            ('InterSystems XML config',
             '<Invokes>\n  <Invoke Class="fhirtemplate.Setup" Method="X"/>'),
            ('arXiv title in a search result',
             'Entropy After </Think> for reasoning model early exiting'),
            ('a task instructing the model to use THINK blocks',
             'Wrap internal planning in `<THINK>` blocks. These are for you.'),
            ('.NET stack trace',
             '.<InvokeAsync>g__Awaited|17_0 (ResourceInvoker invoker)')):
        _r99, _e99, _ = process_entry(_conv(think=f'output: {_txt}'))
        run_test(f'accepted: {_label}', _r99 is not None and not _e99)
    # the lowercase leakage these backstop must still reject
    for _m in ('< invoke name="x">', '< /invoke>', '</function_calls>',
               'antml:parameter', '<\uff5c\uff5cDSML\uff5c\uff5ctool_calls>', '< /think>'):
        _, _e99b, _ = process_entry(_conv(think=f'text {_m} text'))
        run_test(f'rejected {_m!r}', bool(_e99b))
    print()


    # Test 100: typed tool fields, named. The production render already rejects these,
    # but as a raw Jinja 'can only concatenate str (not "list") to str', which does not
    # say which tool or which field. Real: one agentic-coding source ships a
    # LIST-valued description on the `write` tool; those records passed every rule and
    # the json render, then produced NO training text because the markdown render
    # raised and apply_chat_template_randomized returns False on that path.
    print('Test 100: tool field types are named')
    def _tooled(tool):
        return {'conversation': [
            {'role': 'system', 'content': 's', 'tools': [tool]},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'think': 'r', 'content': 'a'}]}
    _base = {'name': 'write', 'description': 'writes a file',
             'parameters': {'type': 'object', 'properties': {'p': {'type': 'string'}}}}
    _r100, _e100, _ = process_entry(json.loads(json.dumps(_tooled(_base))))
    run_test('well-typed tool accepted', _r100 is not None and not _e100)
    _lst = dict(_base, description=['writes a file', 'line two'])
    _, _e100b, _ = process_entry(_tooled(_lst))
    run_test('list description rejected',
             any("field 'description' must be a string" in _x for _x in _e100b))
    run_test('error names the tool', any("'write'" in _x for _x in _e100b))
    # `name` and `parameters` are already caught by EARLIER checks that return before
    # this one ('Declared tool has empty or missing name', and the template's own
    # parameters-must-be-a-dict guard). The typed check keeps those branches as
    # defence-in-depth if the ordering ever changes, but the only branch it is
    # actually reachable for -- and the only one seen in real data -- is description.
    _, _e100c, _ = process_entry(_tooled(dict(_base, name=['write'])))
    run_test('list name rejected (by the earlier name check)', bool(_e100c))
    _, _e100d, _ = process_entry(_tooled(dict(_base, parameters='not-a-dict')))
    run_test('non-dict parameters rejected (by the template guard)', bool(_e100d))
    print()

    # Test 101: a converter that self-declares it mutated the training target.
    # An upstream converter rewrites exactly the strings on the forbidden list
    # ('<invoke ' -> '< invoke') to launder them past a substring filter, and records
    # it in a top-level `markers_neutralized` field. Nothing read top-level entry
    # fields, so the admission was invisible. Rejecting matters independently of
    # detectability: the rewrite changes the TARGET (one record had a Python module
    # the agent was editing rewritten, shipping `THINK_START = "< think>"`).
    print('Test 101: self-declared converter mutation rejected')
    _c101 = {'conversation': [{'role': 'user', 'content': 'q'},
                              {'role': 'assistant', 'think': 't', 'content': 'a'}]}
    _r101, _e101, _ = process_entry(json.loads(json.dumps(_c101)))
    run_test('clean record accepted', _r101 is not None and not _e101)
    for _fld in ('markers_neutralized', 'schema_widened'):
        _m = json.loads(json.dumps(_c101)); _m[_fld] = ['</invoke>']
        _, _e101b, _ = process_entry(_m)
        run_test(f'{_fld} rejected',
                 any('declared it mutated' in _x for _x in _e101b))
    # an EMPTY marker list means nothing was mutated -- must not reject
    _e = json.loads(json.dumps(_c101)); _e['markers_neutralized'] = []
    _r101c, _e101c, _ = process_entry(_e)
    run_test('empty marker list accepted', _r101c is not None and not _e101c)
    print()


    # Test 102: the phantom-harness family, and the exoneration that keeps it safe
    #
    # This rule rejects an assistant turn that reasons about a harness instruction
    # appearing NOWHERE in the record -- the model citing a directive the trained
    # model will never see. It is the highest-volume rule in the gate (366,929
    # records / 9.1% of the 4,021,234-record mid corpus on `oververbosity` alone),
    # and before this test all four patterns could be deleted together with every
    # suite still green.
    #
    # The second half matters more than the first. DETECT is deliberately tight and
    # EXCLUDE deliberately LOOSER, searched over every non-assistant turn: if the
    # directive really is in the record, the record must PASS. Breaking that half is
    # the one unrecoverable error this rule can make -- it destroys clean data
    # silently -- so each family is tested in both directions.
    #
    # NOTE ON THE FIXTURES: every "exonerated" case asserts the record is KEPT with
    # NO errors at all, not merely that the phantom error is absent. The first draft
    # of this test asserted only the latter and passed vacuously: its system turn sat
    # AFTER the user turn, so the record was rejected on turn order before the
    # phantom check ever ran, and a mutant that disabled the exoneration entirely
    # still showed 348/348 green. Assert the whole verdict, or the test proves
    # nothing.
    print('Test 102: phantom harness -- detection AND the present-in-record exoneration')

    def _detects(conv):
        """(kept, errors, phantom_errors) for one conversation."""
        _r, _e, _ = process_entry({'conversation': conv})
        return _r, _e, [x for x in _e if 'appears nowhere in this record' in x]

    def _leaky(think, field='think'):
        """user + assistant, the assistant citing a directive that is not present."""
        asst = {'role': 'assistant', 'content': 'Here is the answer.',
                'think': 'Reasoning about the problem.'}
        asst[field] = think
        return [{'role': 'user', 'content': 'Solve this.'}, asst]

    def _with_system(directive, think, field='think'):
        """system FIRST (any other order is a turn-order rejection), then user."""
        asst = {'role': 'assistant', 'content': 'Here is the answer.',
                'think': 'Reasoning about the problem.'}
        asst[field] = think
        return [{'role': 'system', 'content': directive},
                {'role': 'user', 'content': 'Solve this.'}, asst]

    def _with_user(directive, think, field='think'):
        asst = {'role': 'assistant', 'content': 'Here is the answer.',
                'think': 'Reasoning about the problem.'}
        asst[field] = think
        return [{'role': 'user', 'content': 'Solve this. ' + directive}, asst]

    def _with_tool(directive, think, field='think'):
        """A real tool round-trip, so the directive sits in a TOOL turn."""
        asst = {'role': 'assistant', 'content': 'Here is the answer.',
                'think': 'Reasoning about the problem.'}
        asst[field] = think
        return [
            {'role': 'system', 'content': 'sys', 'tools': [
                {'name': 't', 'description': 'd', 'parameters': {
                    'type': 'object', 'properties': {'x': {'type': 'string'}},
                    'required': ['x']}}]},
            {'role': 'user', 'content': 'Solve this.'},
            {'role': 'assistant', 'think': 'call it',
             'tool_calls': [{'name': 't', 'arguments': {'x': '1'}}]},
            {'role': 'tool', 'content': directive},
            asst,
        ]

    # Every fixture shape must be clean when it carries no directive at all --
    # otherwise an "exonerated" PASS below could come from the shape, not the rule.
    for _shape, _fx in (('user+assistant', _leaky('Plain reasoning.')),
                        ('system first', _with_system('sys', 'Plain reasoning.')),
                        ('directive in user', _with_user('', 'Plain reasoning.')),
                        ('tool round-trip', _with_tool('ok', 'Plain reasoning.'))):
        _r, _e, _p = _detects(_fx)
        run_test(f'fixture control: {_shape} is clean on its own',
                 _r is not None and _e == [])

    # --- oververbosity: the volume driver -------------------------------------
    _OV = 'Desired oververbosity 5, so keep it moderate.'
    _r, _e, _p = _detects(_leaky(_OV))
    run_test('oververbosity: phantom directive is REJECTED', _r is None and bool(_p))
    _r, _e, _p = _detects(_with_system('Desired oververbosity 5.', _OV))
    run_test('oververbosity: KEPT when the system turn carries it',
             _r is not None and _e == [])
    _r, _e, _p = _detects(_with_user('Note: desired OVERVERBOSITY 5', _OV))
    run_test('oververbosity: KEPT via a USER turn, case-insensitively',
             _r is not None and _e == [])
    run_test('oververbosity: ordinary prose about verbosity does NOT fire',
             _detects(_leaky('The user asked for a verbose answer; I will be '
                             'thorough.'))[0] is not None)

    # --- no_tools_directive --------------------------------------------------
    _NT = ('The system says I MUST NOT call any tools in the next message, so I '
           'will answer directly.')
    _r, _e, _p = _detects(_leaky(_NT))
    run_test('no_tools_directive: phantom directive is REJECTED',
             _r is None and bool(_p))
    _r, _e, _p = _detects(_with_system(
        'You must not call any tools in the next message.', _NT))
    run_test('no_tools_directive: KEPT when the record carries it',
             _r is not None and _e == [])

    # --- tool_choice_none ----------------------------------------------------
    _TC = 'Given tool_choice=none I should reply in plain text.'
    _r, _e, _p = _detects(_leaky(_TC))
    run_test('tool_choice_none: phantom directive is REJECTED',
             _r is None and bool(_p))
    _r, _e, _p = _detects(_with_user('Request sent with tool_choice: none', _TC))
    run_test('tool_choice_none: KEPT when the record carries it',
             _r is not None and _e == [])
    # Measured on real data: an agent implementing an OpenAI-compatible API reasons
    # about its own `tool_choice: "none"` field while its own program prints the
    # same string into the tool output. One agentic-coding record is exactly this.
    # It is the ONE record in 4,021,234 where the looseness of EXCLUDE changes a
    # verdict -- and it changes it correctly, so tightening EXCLUDE must not be
    # done without replacing this record's protection.
    _r, _e, _p = _detects(_with_tool(
        'tools=None tool_choice=None stream=False',
        'The hidden test may check that for a stream request with '
        'tool_choice: "none" the stream has content deltas.'))
    run_test('tool_choice_none: an agent discussing its own API field is KEPT',
             _r is not None and _e == [])

    # --- round_end_notice: the harness iteration-cap banner -------------------
    _BANNER = ('\u672c\u8f6e\u8c03\u7528\u5df2\u8fbe\u6700\u5927\u6b21\u6570'
               '\uff0c\u56de\u590d\u5df2\u7ec8\u6b62\uff0c\u8bf7\u7ee7\u7eed'
               '\u8f93\u5165\u3002\nMaximum iterations reached for this round. '
               'Please send a new message to continue.')
    _r, _e, _p = _detects(_leaky('Work done.\n\n---\n' + _BANNER, field='content'))
    run_test('round_end_notice: banner in assistant CONTENT is REJECTED',
             _r is None and bool(_p))
    _r, _e, _p = _detects(_with_tool(_BANNER,
                                     'Work done.\n\n---\n' + _BANNER,
                                     field='content'))
    run_test('round_end_notice: KEPT when a TOOL turn carries it',
             _r is not None and _e == [])

    # --- the invariant, stated once ------------------------------------------
    _r, _e, _p = _detects(_leaky('Desired oververbosity 5.'))
    run_test('a phantom reference is the ONLY reason such a record is rejected',
             _r is None and _e and all('appears nowhere in this record' in x
                                       for x in _e))
    print()


    total = passed + failed
    print(f'Results: {passed}/{total} passed, {failed}/{total} failed')
    if failed == 0:
        print('All tests passed!')
    else:
        print('Some tests failed!')
        exit(1)
