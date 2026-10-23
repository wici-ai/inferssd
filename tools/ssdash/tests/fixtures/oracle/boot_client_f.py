#!/usr/bin/env python3
"""Bootstrap client.  Request 1: the bootstrap text as a chat turn (cold prefill, then a streamed decode of
--max-tokens tokens with the time of every token).  Request 2: the same conversation, prompt plus the model's own
reply, prefilled again from scratch with the prompt cache off, now with a warm GPU tier.  Two prefill points
(cold, warm) and the decode token times, all on one server.  With --drive, also the bytes the drives that hold the
expert files read from the first decode token to the last (/proc/diskstats): the sink's reads and any other reader's.
With --graph, for a server whose every prefill takes the GPU graph (threshold 0), the graph's prefill instead: the
longest cut of the text at a sentence end that the decode trace covers (a prompt of at most --max-tokens tokens),
then the whole text, each prefilled from scratch with one token of decode.
usage: boot_client.py --port P --script S.json --max-tokens D --out client.jsonl [--drive EXPERT_SOURCE/experts.json]
       boot_client.py --graph --port P --script S.json --max-tokens D --out client.jsonl"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import turn  # noqa: E402


def drive_bytes(devices):
    """Bytes read so far by the block devices (major, minor), from /proc/diskstats (sectors of 512 bytes)."""
    total = 0
    for line in open('/proc/diskstats'):
        f = line.split()
        if (int(f[0]), int(f[1])) in devices:
            total += int(f[5]) * 512
    return total


def post(port, path, body):
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', json.dumps(body, ensure_ascii=False).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.loads(r.read())


ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, required=True)
ap.add_argument('--script', required=True)
ap.add_argument('--max-tokens', type=int, default=512)
ap.add_argument('--out', required=True)
ap.add_argument('--tokenize', help='also count the new prompt tokens of each turn of this script, one conversation '
                '(the workload)')
ap.add_argument('--tokenize-out')
ap.add_argument('--drive', help="the expert source's experts.json: count what the drives of its files read in decode")
ap.add_argument('--graph', action='store_true', help="the GPU graph's prefill of a cut of the text and of the whole")
a = ap.parse_args()
text = json.load(open(a.script, encoding='utf-8'))[0]['text']
msgs = [{'role': 'user', 'content': text}]


def prompt_of(t):
    return post(a.port, '/apply-template', {'messages': [{'role': 'user', 'content': t}]})['prompt']


def graph_prefills():
    """--graph: the cut, then the whole text, each prefilled from scratch (one token of decode); the whole text
    alone when the trace covers it."""
    def tokens(t):
        return len(post(a.port, '/tokenize', {'content': prompt_of(t)})['tokens'])
    ends = [i + 1 for i in range(len(text) - 1) if text[i] in '.?!' and text[i + 1].isspace()]
    # the cut at ends[lo] fits the trace (none: -1), at ends[hi] it does not; none when the whole text fits
    lo, hi = -1, 0 if tokens(text) <= a.max_tokens else len(ends)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if tokens(text[:ends[mid]]) <= a.max_tokens else (lo, mid)
    texts = ([text[:ends[lo]]] if lo >= 0 else []) + [text]
    with open(a.out, 'w', encoding='utf-8') as out:
        for i, t in enumerate(texts):
            t0 = time.time()
            r = post(a.port, '/completion', {'prompt': prompt_of(t), 'n_predict': 1, 'cache_prompt': False,
                                              'temperature': 0, 'top_k': 1})
            rec = {'turn': i, 'kind': 'graph_prefill', 'timings': r.get('timings', {}), 'wall_s': time.time() - t0}
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out.flush()
            print(f"graph prefill: prompt_n={rec['timings'].get('prompt_n')} "
                  f"prompt_ms={rec['timings'].get('prompt_ms')}", flush=True)


if a.graph:
    graph_prefills()
    sys.exit(0)
# the drives that hold the expert files: a file on a block device's filesystem has that device's numbers (none for
# an overlay or a network filesystem, which /proc/diskstats does not list)
devices = set()
if a.drive:
    for p in json.load(open(a.drive))['files']:
        d = os.stat(p).st_dev
        devices.add((os.major(d), os.minor(d)))
    with open('/proc/diskstats') as f:
        devices &= {(int(x.split()[0]), int(x.split()[1])) for x in f}
drive = {}


def on_token(n):
    if devices and n in (1, a.max_tokens):
        drive[n] = drive_bytes(devices)


with open(a.out, 'w', encoding='utf-8') as out:
    rec = {'turn': 0, 'kind': 'chat', **turn(a.port, msgs, a.max_tokens, on_token)}
    if len(drive) == 2:
        rec['drive_read_bytes'] = drive[a.max_tokens] - drive[1]
    out.write(json.dumps(rec, ensure_ascii=False) + '\n')
    out.flush()
    t = rec['timings']
    print(f"chat: prompt_n={t.get('prompt_n')} prompt_ms={t.get('prompt_ms')} predicted_n={t.get('predicted_n')}", flush=True)
    prompt = post(a.port, '/apply-template', {'messages': msgs})['prompt'] + rec['reply']
    t0 = time.time()
    r = post(a.port, '/completion', {'prompt': prompt, 'n_predict': 1, 'cache_prompt': False, 'temperature': 0,
                                      'top_k': 1})
    rec2 = {'turn': 1, 'kind': 'warm_prefill', 'timings': r.get('timings', {}), 'wall_s': time.time() - t0}
    out.write(json.dumps(rec2, ensure_ascii=False) + '\n')
    t = rec2['timings']
    print(f"warm prefill: prompt_n={t.get('prompt_n')} prompt_ms={t.get('prompt_ms')}", flush=True)
if a.tokenize:
    # a turn's new prompt tokens: the conversation rendered through it less the conversation through the turn
    # before, the replies empty (a reply's tokens are decode, and the server keeps them)
    counts, hist, total = [], [], 0
    for entry in json.load(open(a.tokenize, encoding='utf-8')):
        hist.append({'role': 'user', 'content': entry['text']})
        p = post(a.port, '/apply-template', {'messages': hist})['prompt']
        n = len(post(a.port, '/tokenize', {'content': p})['tokens'])
        counts.append(n - total)
        total = n
        hist.append({'role': 'assistant', 'content': ''})
    json.dump(counts, open(a.tokenize_out, 'w'))
    print(f'workload prompt tokens per turn: {counts}', flush=True)
