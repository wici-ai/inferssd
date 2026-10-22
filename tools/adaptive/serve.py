#!/usr/bin/env python3
"""Serve through the closed loop: an HTTP front end on SERVE_HOST:SERVE_PORT that passes every request on to
llama-server (on PORT) and, for each generation request, has the Controller (controller.py) re-plan the rest of the
conversation when it arrives, switching the configuration first when that pays, and record it when it is done.
The server starts in the plan's configuration (RUN_DIR/plan.json).
One conversation at a time: generation requests are served one after another, as the engine's single slot would
anyway; a request's new prompt tokens are its prompt's tokens beyond the context the last one left.  A streamed
request is relayed event by event with the server's per-token timings on (they give every decode token's arrival
time, so its decode rate counts in the feedback); one that is not streamed counts with its prefill only.
usage: serve.py RUN_DIR [llama-server args]
env:   SERVE_PORT (8080) SERVE_HOST (127.0.0.1) PORT (the engine's) HORIZON (3) (+ controller.py's)"""
import http.client
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from controller import Controller, post  # noqa: E402

GENERATE = {'/completion', '/completions', '/v1/completions', '/chat/completions', '/v1/chat/completions'}
run = sys.argv[1]
plan = json.load(open(f'{run}/plan.json'))['plan']
port = int(os.environ['PORT'])
ctl = Controller(run, f'{run}/serve-{time.strftime("%Y%m%d-%H%M%S")}', port, (plan['tier'], plan['cpus']), sys.argv[2:],
                 horizon=int(os.environ.get('HORIZON', '3')))
lock = threading.Lock()


def prompt_tokens(path, body):
    """Tokens of the prompt this request will run, by the engine's own template and tokenizer."""
    if 'chat' in path:
        prompt = post(port, '/apply-template', body, timeout=60)['prompt']
    else:
        prompt = body.get('prompt', '')
        if isinstance(prompt, list):
            if prompt and all(isinstance(x, int) for x in prompt):
                return len(prompt)
            prompt = ''.join(x for x in prompt if isinstance(x, str))
    return len(post(port, '/tokenize', {'content': prompt, 'add_special': True}, timeout=60)['tokens'])


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        pass

    def _upstream(self, method, body=None):
        c = http.client.HTTPConnection('127.0.0.1', port, timeout=7200)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ('host', 'content-length', 'connection')}
        c.request(method, self.path, body=body, headers=headers)
        return c, c.getresponse()

    def _reply(self, status, headers, data):
        self.send_response(status)
        for k, v in headers:
            if k.lower() not in ('content-length', 'transfer-encoding', 'connection'):
                self.send_header(k, v)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _relay(self, method, body=None):
        try:
            c, r = self._upstream(method, body)
            data = r.read()
            c.close()
            self._reply(r.status, r.getheaders(), data)
        except OSError as exc:
            self._reply(502, [('Content-Type', 'text/plain')], f'engine unavailable: {exc}\n'.encode())

    def do_GET(self):
        self._relay('GET')

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        path = self.path.split('?')[0]
        if path not in GENERATE:
            self._relay('POST', raw)
            return
        body = json.loads(raw or b'{}')
        stream = bool(body.get('stream'))
        if stream:
            body['timings_per_token'] = True
        with lock:
            try:
                n_total = prompt_tokens(path, body)
            except (OSError, ValueError, KeyError):
                n_total = None                              # the engine will answer with the error itself
            n_new = n_total - ctl.context if n_total and n_total > ctl.context else n_total or 0
            if n_total is None:
                ctl.ensure()
            else:
                ctl.before(n_new)
            t0 = time.time()
            timings, tok = {}, {}
            try:
                c, r = self._upstream('POST', json.dumps(body).encode())
            except OSError as exc:
                self._reply(502, [('Content-Type', 'text/plain')], f'engine unavailable: {exc}\n'.encode())
                return
            if not stream or r.status != 200:
                data = r.read()
                c.close()
                self._reply(r.status, r.getheaders(), data)
                if r.status == 200:
                    timings = json.loads(data).get('timings') or {}
            else:
                self.send_response(200)
                for k, v in r.getheaders():
                    if k.lower() not in ('content-length', 'transfer-encoding', 'connection'):
                        self.send_header(k, v)
                self.send_header('Transfer-Encoding', 'chunked')
                self.end_headers()
                try:
                    for line in r:
                        if line.startswith(b'data: ') and line[6:].strip() != b'[DONE]':
                            try:
                                t = json.loads(line[6:]).get('timings')
                            except ValueError:
                                t = None
                            if t:
                                timings = t
                                n = t.get('predicted_n')
                                if isinstance(n, int) and n >= 1:
                                    tok.setdefault(n, time.time() - t0)
                        self.wfile.write(b'%x\r\n%s\r\n' % (len(line), line))
                        self.wfile.flush()
                    self.wfile.write(b'0\r\n\r\n')
                except OSError:
                    timings = {}                            # the client went away: the engine stops, nothing to record
                c.close()
            if timings:
                n = max(tok) if tok else 0
                ctl.after(n_new, {'timings': timings, 'wall_s': time.time() - t0,
                                  'tok_s': [round(tok[i], 6) if i in tok else None for i in range(1, n + 1)]})
                ctl.prepare()


srv = ThreadingHTTPServer((os.environ.get('SERVE_HOST', '127.0.0.1'), int(os.environ.get('SERVE_PORT', '8080'))), Handler)
ctl.note({'event': 'serving', 'host': srv.server_address[0], 'port': srv.server_address[1], 'plan': [plan['tier'], plan['cpus']]})


def stop(*_):
    raise KeyboardInterrupt


signal.signal(signal.SIGTERM, stop)
try:
    srv.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    ctl.close()
