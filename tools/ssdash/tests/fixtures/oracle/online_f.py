#!/usr/bin/env python3
"""The closed loop's model side: what the served turns of a conversation show, and the re-plan of its rest.
  served    every turn so far, from each server instance's directory (its log and turns.jsonl): the configuration,
            prefill (tokens, cold tier or not, seconds, bytes streamed from the SSD and the CPU compute) and decode
            (rate of the window, the sink's time and CPU part, the time outside the sink) measured, and decode as the
            machine model prices the routing that turn served
  feedback  measured over modelled, per part (machine.py): the CPU compute and the outside level per placement,
            from that placement's turns; the rest of the sink and the time of prefill's reads as the drive's state,
            which every placement shares, from the latest turn whichever placement served it; the bytes a prefill
            reads (prefill.feedback)
  decide    every configuration's predicted rest of the conversation, from the conversation's own routing so far
            (it replaces the bootstrap's sample) and the feedback.  Another configuration costs a restart (measured)
            and meets a cold tier; with the engine's disk prompt cache the conversation's history comes back from it,
            without one it is prefilled again.  Switch when the best beats staying by the margin.
  expect    the rest of a conversation whose future is not known: the next request (known on its arrival), then
            turns like the ones served so far
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prefill  # noqa: E402
from logparse import ArmParser, read_arm, window_rate  # noqa: E402

_ARMS, _LIVE = {}, {}


def _read_arm_cached(d, skip):
    """read_arm without reading a log twice: a stopped server's log.gz is parsed once; a live server's plain log is
    read on from where the last call stopped."""
    gz = f'{d}/server.log.gz'
    if os.path.exists(gz):
        st = os.stat(gz)
        key = (d, skip, st.st_size, st.st_mtime_ns)
        if key not in _ARMS:
            _ARMS[key] = read_arm(d, skip)
        return _ARMS[key]
    path = f'{d}/server.log'
    parser, offset = _LIVE.get(path, (None, 0))
    if parser is None:
        parser = ArmParser()
    with open(path, 'rb') as f:
        f.seek(offset)
        data = f.read()
    cut = data.rfind(b'\n') + 1                          # whole lines only; a partial last line waits
    for line in data[:cut].decode('utf-8', 'replace').splitlines(True):
        parser.feed(line)
    _LIVE[path] = (parser, offset + cut)
    return read_arm(d, skip, parser)


def served(m, insts, w0, w1):
    """Turns served so far, in order, from the server instances' directories; plus every decode segment's routing
    (for the replay) and the outside-the-sink points (position, ms, placement) for the slope.  A request's decode
    is the next trace segment that starts where its prompt ends (cache_n + prompt_n); its prefill the next prefill
    graph of its prompt's size."""
    turns, segs, pts = [], [], []
    for d in insts:
        recs = [json.loads(x) for x in open(f'{d}/turns.jsonl')] if os.path.exists(f'{d}/turns.jsonl') else []
        if not recs:
            continue
        a = _read_arm_cached(d, w0 - 1)
        groups, graphs, gk, gi, mine = a['groups'], a['pf_graphs'], 0, 0, []
        for i, r in enumerate(recs):
            t = r['timings']
            n = t.get('prompt_n') or 0
            start = (t.get('cache_n') or 0) + n
            j = next((j for j in range(gi, len(graphs)) if graphs[j]['n_tokens'] == n), None)
            pg = graphs[j] if j is not None else None
            gi = j + 1 if j is not None else gi
            q = next((q for q in range(gk, len(groups)) if groups[q][0][1] == start), None)
            g = groups[q] if q is not None else []
            gk = q + 1 if q is not None else gk
            tr = {'turn': r['turn'], 'tier': r['tier'], 'cpus': r['cpus'], 'first': i == 0, 'n_prompt': n,
                  'prefill_s': t.get('prompt_ms', 0) / 1e3,
                  'read_bytes': pg['sink_bytes'] + pg['graph_bytes'] if pg else None,
                  'cpu_s': pg['cpu_s'] if pg else None, 'n_decode': t.get('predicted_n'),
                  'end_pos': g[-1][1] + 1 if g else start + max(0, (t.get('predicted_n') or 0) - 1)}
            if g:
                L = len(a['trace'][g[0]])
                seg = [[a['trace'][k][l][0] for l in sorted(a['trace'][k])] for k in g if len(a['trace'][k]) == L]
                segs.append(seg)
                mine.append(seg)
                rp = m.replay(('served', d, r['tier']), mine, r['tier'])
                a0, a1 = rp.seg_rows[len(mine) - 1]
                tr['model_sink_ms'], tr['model_cpu_ms'] = m.price(rp.rows[a0:a1] or rp.rows[max(0, a1 - 1):a1], r['cpus'])
                ts = r.get('tok_s') or []
                for j, k in enumerate(g):
                    if j >= 16 and j + 1 < len(ts) and ts[j] is not None and ts[j + 1] is not None and k in a['prof']:
                        pts.append((float(k[1]), (ts[j + 1] - ts[j]) * 1e3 - a['prof'][k][0] / 1e3, r['cpus']))
                win = [k for k in g[w0 - 1:w1] if k in a['prof']]
                rate = window_rate(ts, w0, w1)
                if win and rate:
                    tr['rate'] = rate
                    tr['pos'] = statistics.mean(k[1] for k in win)
                    tr['sink_ms'] = statistics.mean(a['prof'][k][0] for k in win) / 1e3
                    tr['cpu_ms'] = statistics.mean(a['prof'][k][2] for k in win) / 1e3
                    tr['out_ms'] = 1000.0 / rate - tr['sink_ms']
            turns.append(tr)
    return turns, segs, pts


def fit_slope(m, pts, span=2000):
    """Slope of the time outside the sink against position from the served tokens (x, y, placement), within each
    placement (their levels differ), once the positions span `span`."""
    if not pts or max(x for x, _, _ in pts) - min(x for x, _, _ in pts) < span:
        return
    by = {}
    for x, y, p in pts:
        by.setdefault(p, []).append((x, y))
    sxy = sxx = 0.0
    for v in by.values():
        mx, my = statistics.mean(x for x, _ in v), statistics.mean(y for _, y in v)
        sxy += sum((x - mx) * (y - my) for x, y in v)
        sxx += sum((x - mx) ** 2 for x, _ in v)
    if sxx:
        m.slope = max(0.0, sxy / sxx)


def feedback(m, turns, recent=1):
    """Measured over modelled per part (the module doc); the drive's state from the last `recent` turns."""
    fb = {'k_cpu': {}, 'out0': {}, 'k_drive': 1.0, 'k_bytes': 1.0, 'pf_cpu': {}, 'pf_drive': 1.0}
    dec = [t for t in turns if 'sink_ms' in t]
    for p in m.P:
        mine = [t for t in dec if t['cpus'] == p]
        if mine:
            fb['k_cpu'][p] = statistics.mean(t['cpu_ms'] / t['model_cpu_ms'] for t in mine)
            fb['out0'][p] = statistics.mean(t['out_ms'] - m.slope * t['pos'] for t in mine)   # level at position 0
    if dec:
        fb['k_drive'] = statistics.mean((t['sink_ms'] - t['cpu_ms']) / (t['model_sink_ms'] - t['model_cpu_ms'])
                                        for t in dec[-recent:])
    fb.update(prefill.feedback(m.PF, [t for t in turns if t['n_prompt']], lambda p: (m.route(p), m.ubatch(p)), recent))
    return fb


def expect(asked, next_n, horizon, turns, decode):
    """The rest of a conversation whose future is not known: the next request (its new prompt tokens are known when it
    arrives), then horizon - 1 more turns of the mean new prompt tokens of the requests so far (`asked`) and this one.
    Replies: the served turns' mean length, `decode` before any."""
    ns = list(asked) + [next_n]
    ds = [t['n_decode'] for t in turns if t.get('n_decode')]
    return [next_n] + [round(statistics.mean(ns))] * (horizon - 1), round(statistics.mean(ds)) if ds else decode


def decide(m, turns, segs, cur, remaining, decode, w0, w1, start_s, keeps_history=True, margin_frac=0.05,
           margin_s=15.0, recent=1):
    """Every configuration's predicted rest of the conversation; switch when the best beats staying by the margin.
    keeps_history: a restarted server restores the conversation from the engine's disk prompt cache; without one its
    first turn prefills the whole history again."""
    fb = feedback(m, turns, recent)
    ctx = turns[-1]['end_pos'] if turns else 0
    rows = []
    for tier, p in m.configs:
        stay = (tier, p) == tuple(cur)
        fresh = not stay and not keeps_history and remaining
        rem = [remaining[0] + ctx] + list(remaining[1:]) if fresh else remaining
        detail = m.predict(segs or m.routing, tier, p, 0 if fresh else ctx, rem, decode, w0, w1, not stay, fb)
        total = (0.0 if stay else start_s) + sum(t['prefill_s'] + decode / t['decode_tps'] for t in detail)
        rows.append({'tier': tier, 'cpus': p, 'stay': stay, 'rest_s': round(total, 1),
                     'switch_cost_s': 0.0 if stay else round(start_s, 1),
                     'turns': [{'prefill_s': round(t['prefill_s'], 1), 'decode_tps': round(t['decode_tps'], 3),
                                'hit_v': round(t['hit_v'], 4)} for t in detail]})
    rows.sort(key=lambda r: r['rest_s'])
    here = next(r for r in rows if r['stay'])
    best = rows[0]
    margin = max(margin_frac * here['rest_s'], margin_s)
    switch = (not best['stay']) and here['rest_s'] - best['rest_s'] > margin
    rnd = {k: (round(v, 4) if isinstance(v, float) else {kk: round(vv, 4) for kk, vv in v.items()}) for k, v in fb.items()}
    return {'switch': switch, 'to': [best['tier'], best['cpus']] if switch else list(cur), 'margin_s': round(margin, 1),
            'feedback': rnd, 'slope_ms_per_ktok': round(m.slope * 1e3, 3), 'context': ctx, 'remaining': remaining,
            'decode': decode, 'candidates': rows}
