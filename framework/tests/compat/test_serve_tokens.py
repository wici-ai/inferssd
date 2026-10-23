"""The adaptive front end must count the tokens used by chat inference."""
import ast
import io
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

SERVE = Path(__file__).resolve().parents[1] / 'fixtures/adaptive/serve.py'


def load_counting(post):
    # serve.py starts its controller and HTTP server at import. Load the actual
    # counting and request-handler code without starting either service.
    tree = ast.parse(SERVE.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in ('prompt_tokens', 'Handler')]
    namespace = {'post': post, 'port': 18250, 'json': json, 'time': time,
                 'BaseHTTPRequestHandler': object, 'lock': threading.Lock(),
                 'GENERATE': {'/completion', '/v1/chat/completions'}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SERVE), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('text,tokens', [('abc', [1, 100, 101, 102]),
                                      ('abc<s>def', [1, 100, 101, 102, 63, 118, 65, 103, 104, 105])])
def test_chat_uses_authoritative_tokens_and_preserves_request(text, tokens):
    # API receipts: the literal <s> takes three ordinary IDs, whereas a second
    # /tokenize on the rendered text would turn it into a single BOS ID.
    body = {'messages': [{'role': 'user', 'content': text}],
            'enable_thinking': True, 'chat_template_kwargs': {'custom': 'retained'}}
    calls = []

    def post(port, path, request, timeout):
        calls.append((port, path, request, timeout))
        return {'prompt': '<s>' + text, 'tokens': tokens}

    count = load_counting(post)['prompt_tokens']('/v1/chat/completions', body)
    assert count == len(tokens)
    assert calls == [(18250, '/apply-template', body, 60)]
    assert calls[0][2] is body


@pytest.mark.parametrize('response', [None, [], {}, {'tokens': None}, {'tokens': '123'},
                                      {'tokens': [True]}, {'tokens': [-1]}, {'tokens': [1.0]}])
def test_missing_or_invalid_authoritative_tokens_have_no_flatten_fallback(response):
    calls = []

    def post(port, path, request, timeout):
        calls.append(path)
        return response

    with pytest.raises((KeyError, ValueError)):
        load_counting(post)['prompt_tokens']('/v1/chat/completions', {'messages': []})
    assert calls == ['/apply-template']


def test_native_completion_keeps_its_tokenizer_contract():
    calls = []

    def post(port, path, request, timeout):
        calls.append((path, request))
        return {'tokens': [1, 2, 3]}

    count = load_counting(post)['prompt_tokens']
    assert count('/completion', {'prompt': [4, 5]}) == 2
    assert calls == []
    assert count('/completion', {'prompt': 'abc'}) == 3
    assert calls == [('/tokenize', {'content': 'abc', 'add_special': True})]


@pytest.mark.parametrize('template_response', [{'tokens': list(range(10))}, {}, None, []])
def test_handler_uses_count_for_planner_and_preserves_forwarded_chat(template_response):
    body = {'messages': [{'role': 'user', 'content': 'abc<s>def'}],
            'enable_thinking': True, 'stream': False}
    raw = json.dumps(body).encode()
    valid = isinstance(template_response, dict) and 'tokens' in template_response
    namespace = load_counting(lambda *args, **kwargs: template_response)
    before, after, forwarded, ensured, prepared = [], [], [], [], []
    namespace['ctl'] = SimpleNamespace(context=4, before=before.append,
        after=lambda count, record: after.append(count), ensure=lambda: ensured.append(True),
        prepare=lambda: prepared.append(True))
    response = SimpleNamespace(status=200, read=lambda: b'{"timings":{"prompt_n":10}}',
                               getheaders=lambda: [])

    def upstream(method, payload):
        forwarded.append((method, json.loads(payload)))
        return SimpleNamespace(close=lambda: None), response

    request = SimpleNamespace(rfile=io.BytesIO(raw), headers={'Content-Length': str(len(raw))},
                              path='/v1/chat/completions', _upstream=upstream,
                              _reply=lambda *args: None)
    namespace['Handler'].do_POST(request)
    assert before == ([6] if valid else [])
    assert ensured == ([] if valid else [True])
    assert after == ([6] if valid else [])
    assert prepared == ([True] if valid else [])
    assert forwarded == [('POST', body)]
