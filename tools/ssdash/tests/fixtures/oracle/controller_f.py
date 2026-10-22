"""The closed loop around llama-server: a Controller runs the engine in one configuration and, when a request
arrives, re-plans the rest of the conversation (online.py) and switches the configuration when that pays.
A switch displaces the conversation into the engine's disk prompt cache (a one-token request with another prompt),
stops the server and starts it in the new configuration; the request then restores the history from that cache (an
engine without one prefills it again, which the decision counts).  If the server has died it is started again in
the same configuration.
Each server instance gets a directory OUT/inst<i> (server.log, gzipped when it stops; turns.jsonl); OUT/events.jsonl
records every start, decision, switch and turn with wall times.  converse.py (validation) and serve.py (the HTTP
front end) drive it.
env: LAUNCH KNOBS_VIA SWITCH_KEEPS_HISTORY WATCHDOG_MIN_AVAIL_MB TRACE_ENV (+ the launcher's)"""
import gzip
import json
import os
import shutil
import signal
import socket
import statistics
import subprocess
import threading
import time
import urllib.request

import online
from machine import Machine

# the engine's per-token trace and sink profile: the loop reads the routing and the decode costs from them
TRACE_ENV = 'GGML_KIMI_CPU_SINK_TRACE=1 GGML_KIMI_CPU_SINK_PROFILE=1'


def post(port, path, body, timeout=7200):
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', json.dumps(body, ensure_ascii=False).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Engine:
    """One llama-server in one configuration, started by the model's launcher; its output goes to DIR/server.log."""

    def __init__(self, d, port, tier, cpus, envs, args):
        self.dir, self.port, self.tier, self.cpus, self.envs = d, port, tier, cpus, envs
        os.makedirs(d, exist_ok=True)
        env = dict(os.environ, LLAMA_SERVER_PORT=str(port), PORT=str(port))
        knobs = (os.environ.get('TRACE_ENV', TRACE_ENV) + ' ' + envs).split()
        for kv in knobs:
            k, v = kv.split('=', 1)
            env[k] = v
        if os.environ.get('KNOBS_VIA', 'env') == 'extra_env':     # the launcher appends EXTRA_ENV after its own
            env['EXTRA_ENV'] = ' '.join(knobs)
        t0 = time.time()
        self.log = open(f'{d}/server.log', 'w')
        self.p = subprocess.Popen(['bash', os.environ['LAUNCH']] + list(args), stdout=self.log,
                                  stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
        self.ok = False
        for _ in range(900):
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as r:
                    if b'"ok"' in r.read():
                        self.ok = True
                        break
            except OSError:
                pass
            if self.p.poll() is not None:
                break
            time.sleep(1)
        self.start_s = time.time() - t0
        self.stopping = False
        threading.Thread(target=self._watchdog, daemon=True).start()

    def alive(self):
        return self.p.poll() is None

    def _watchdog(self):
        """Kill the server before the machine runs out of memory (the RAM expert cache must give way first)."""
        lim = int(os.environ.get('WATCHDOG_MIN_AVAIL_MB', '256'))
        while not self.stopping and self.alive():
            with open('/proc/meminfo') as f:
                av = next(int(x.split()[1]) // 1024 for x in f if x.startswith('MemAvailable:'))
            if av < lim:
                with open(f'{self.dir}/watchdog.txt', 'a') as w:
                    w.write(f'{time.time()} MemAvailable {av} MiB: killed\n')
                self.p.kill()
                break
            time.sleep(0.5)

    def stop(self):
        self.stopping = True
        if self.alive():
            self.p.send_signal(signal.SIGTERM)
            try:
                self.p.wait(60)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        self.log.close()
        with open(f'{self.dir}/server.log', 'rb') as f, gzip.open(f'{self.dir}/server.log.gz', 'wb', 1) as g:
            shutil.copyfileobj(f, g)
        os.remove(f'{self.dir}/server.log')
        for _ in range(60):                                  # the port is free before the next server
            with socket.socket() as s:
                if s.connect_ex(('127.0.0.1', self.port)) != 0:
                    return
            time.sleep(1)
        subprocess.run(['fuser', '-k', f'{self.port}/tcp'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Controller:
    """run: the bootstrap's directory (the machine model); out: this controller's directory; start: (tier, cpus).
    loop False keeps the start configuration (a static baseline).  horizon: turns the re-plan looks ahead when the
    rest of the conversation is not known; decode: reply length assumed before any reply was served."""

    def __init__(self, run, out, port, start, args, loop=True, window=(129, 512), decode=512, horizon=3):
        self.m = Machine(run)
        self.out, self.port, self.args, self.loop = out, port, list(args), loop
        self.w0, self.w1 = window
        self.decode, self.horizon = decode, horizon
        self.keeps_history = os.environ.get('SWITCH_KEEPS_HISTORY', '1') == '1'
        os.makedirs(out, exist_ok=True)
        self.events = open(f'{out}/events.jsonl', 'a')
        self.insts, self.asked, self.n_turns, self.switch_s = [], [], 0, []
        self.engine, self.seen = None, None
        self._start(*start)
        self.prepare()

    def note(self, rec):
        rec['wall'] = time.time()
        self.events.write(json.dumps(rec, ensure_ascii=False) + '\n')
        self.events.flush()
        print(json.dumps({k: v for k, v in rec.items() if k != 'decision'}), flush=True)

    def _start(self, tier, cpus):
        d = f'{self.out}/inst{len(self.insts)}'
        self.engine = Engine(d, self.port, tier, cpus, self.m.env(tier, cpus), self.args)
        self.insts.append(d)
        self.note({'event': 'start', 'inst': len(self.insts) - 1, 'tier': tier, 'cpus': cpus, 'env': self.engine.envs,
                   'ok': self.engine.ok, 'start_s': round(self.engine.start_s, 1)})
        if not self.engine.ok:
            self.engine.stop()
            raise RuntimeError(f'llama-server did not come up in {tier},{cpus} (log: {d}/server.log.gz)')

    @property
    def config(self):
        return self.engine.tier, self.engine.cpus

    def before(self, n_new, remaining=None):
        """A request with n_new new prompt tokens arrives.  remaining: every turn's new prompt tokens from this one on
        when the conversation is known (validation), else the rest is expected from the turns so far.  Returns the
        decision, None before the first turn has been served or without the loop."""
        self.ensure()
        if not self.loop or not self.n_turns:
            return None
        turns, segs = self.prepare()
        if remaining is None:
            remaining, dec = online.expect(self.asked, n_new, self.horizon, turns, self.decode)
        else:
            dec = self.decode
        # a switch costs what earlier switches took, else the first start (a cold one: a restart finds the weights in
        # the page cache)
        cost = statistics.mean(self.switch_s) if self.switch_s else self.engine.start_s
        d = online.decide(self.m, turns, segs, self.config, remaining, dec, self.w0, self.w1, cost, self.keeps_history)
        self.note({'event': 'decision', 'before_turn': self.n_turns, 'switch': d['switch'], 'to': d['to'],
                   'feedback': d['feedback'], 'switch_cost_s': round(cost, 1), 'decision': d})
        if d['switch']:
            t0 = time.time()
            if self.keeps_history:
                try:                                        # displace the conversation into the disk prompt cache
                    post(self.port, '/completion', {'prompt': 'OK', 'n_predict': 1, 'cache_prompt': True,
                                                    'temperature': 0, 'top_k': 1})
                except OSError as exc:
                    self.note({'event': 'displace_failed', 'error': repr(exc)})
            self.engine.stop()
            self._start(*d['to'])
            self.switch_s.append(time.time() - t0)
            self.note({'event': 'switched', 'to': d['to'], 'switch_s': round(self.switch_s[-1], 1)})
        return d

    def ensure(self):
        """Start the server again in its configuration if it has died."""
        if not self.engine.alive():
            self.note({'event': 'engine_died', 'inst': len(self.insts) - 1})
            self.engine.stop()
            self._start(*self.config)

    def prepare(self):
        """Read what the served turns showed (online.served; the slope outside the sink refitted) and price decode in
        every configuration with the routing so far.  Once per served turn: serve.py calls it when a response is out,
        so a request's arrival only has the decision to make (the engine logs a token before it sends it)."""
        if self.seen is None or self.seen[0] != self.n_turns:
            turns, segs, pts = online.served(self.m, self.insts, self.w0, self.w1)
            online.fit_slope(self.m, pts)
            for tier, p in self.m.configs:                  # decode's price depends on the routing only
                self.m.sink(segs or self.m.routing, tier, p)
            self.seen = (self.n_turns, turns, segs)
        return self.seen[1], self.seen[2]

    @property
    def context(self):
        """Context position after the last served turn (0 before any)."""
        return self.seen[1][-1]['end_pos'] if self.seen and self.seen[1] else 0

    def after(self, n_new, rec):
        """The request has been served: rec has its 'timings' (llama-server's) and 'tok_s' (arrival time of every
        decode token, s after the request; empty when not streamed)."""
        tier, cpus = self.config
        rec = {'turn': self.n_turns, 'tier': tier, 'cpus': cpus, **rec}
        with open(f'{self.engine.dir}/turns.jsonl', 'a') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        t = rec.get('timings') or {}
        self.note({'event': 'turn', 'turn': self.n_turns, 'inst': len(self.insts) - 1, 'tier': tier, 'cpus': cpus,
                   'prompt_n': t.get('prompt_n'), 'cache_n': t.get('cache_n'),
                   'prefill_s': round(t.get('prompt_ms', 0) / 1e3, 2), 'predicted_n': t.get('predicted_n'),
                   'decode_tps': online.window_rate(rec.get('tok_s'), self.w0, self.w1),
                   'wall_s': round(rec.get('wall_s', 0), 2)})
        self.asked.append(n_new)
        self.n_turns += 1

    def close(self):
        if self.engine:
            self.engine.stop()
