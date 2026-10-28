#!/usr/bin/env python3
"""Validation driver: one conversation (a real-text script, one user turn per entry) served turn by turn through a
Controller (controller.py) on a fresh server, under the closed loop or static in its start configuration.
--workload known: the controller sees every remaining turn's prompt length (the script's); learned: only the next
request's, on its arrival, and expects the rest from the turns served so far (--horizon turns ahead).
Replies are --decode tokens long (ignore_eos), streamed, with the arrival time of every token.
usage: converse.py --boot RUN_DIR --script W.json --tokens W.tokens.json --decode 512 --window 129-512
                   --mode static|loop --start TIER,CPUS --out DIR [--workload known|learned --horizon 3]
env:   PORT SERVER_ARGS (+ controller.py's)"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import turn  # noqa: E402
from controller import Controller  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--boot', required=True)
ap.add_argument('--script', required=True)
ap.add_argument('--tokens', required=True)
ap.add_argument('--decode', type=int, default=512)
ap.add_argument('--window', default='129-512')
ap.add_argument('--mode', choices=('static', 'loop'), required=True)
ap.add_argument('--start', required=True, help='TIER,CPUS of the first server')
ap.add_argument('--out', required=True)
ap.add_argument('--workload', choices=('known', 'learned'), default='known')
ap.add_argument('--horizon', type=int, default=3)
a = ap.parse_args()
texts = [e['text'] for e in json.load(open(a.script, encoding='utf-8'))]
ntok = json.load(open(a.tokens))
tier, cpus = a.start.split(',', 1)
port = int(os.environ['PORT'])
ctl = Controller(a.boot, a.out, port, (int(tier), cpus), os.environ.get('SERVER_ARGS', '').split(),
                 loop=a.mode == 'loop', window=tuple(int(x) for x in a.window.split('-')), decode=a.decode,
                 horizon=a.horizon)
ctl.note({'event': 'conversation', 'mode': a.mode, 'start': a.start, 'turns': len(texts), 'tokens': ntok,
          'workload': a.workload, 'horizon': a.horizon if a.workload == 'learned' else None})
history, t_conv = [], time.time()
for i, text in enumerate(texts):
    ctl.before(ntok[i], ntok[i:] if a.workload == 'known' else None)
    history.append({'role': 'user', 'content': text})
    t0 = time.time()
    try:
        rec = turn(port, history, a.decode)
    except OSError as exc:
        ctl.note({'event': 'turn', 'turn': i, 'error': repr(exc)})
        break
    history.append({'role': 'assistant', 'content': rec['reply']})
    ctl.after(ntok[i], {'wall_s': time.time() - t0, **rec})
ctl.close()
ctl.note({'event': 'done', 'total_s': round(time.time() - t_conv, 1)})
