"""Prefill time of N new prompt tokens on a configuration, pass by pass on the route the engine gives each (design
section 3).  The engine cuts a prompt into passes of its ubatch; a pass of n tokens runs on the CPU sink when n is at
most the prefill threshold T (GGML_KIMI_CPU_PREFILL_MAX_TOKENS; none: every pass) or below the GPU's offload minimum
(OFFLOAD_MIN), else on the ggml graph on the GPU.
  bytes  a pass reads the experts its tokens touch, L * U(n) * B_e: U(n) distinct experts per layer of n prompt
         tokens, through the bootstrap's prefill passes (the sink's distinct experts, the graph's copied bytes) and
         below the smallest of them the decode trace's n consecutive tokens, scaled to it (decode tokens route more
         narrowly than prompt tokens); at most what uniform routing would touch; B_e an expert's bytes as the engine
         reads them
  sink   the larger of its reads (less what the GPU tier holds) at the drive's rate and its CPU compute
         n * r * c, plus n * d: r the routes a prompt token takes, c the sink's compute per route and d its time per
         token besides (attention, dense layers, what its reads and compute do not overlap), all from the
         bootstrap's prefill in the same tier state (cold: a server's first prefill; warm: after decode, the state a
         serving server is in)
  graph  its reads at the rate its expert copy read in the bootstrap's graph prefills, plus a + n * d_g, the time
         of those prefills beyond their reads, a per pass and d_g per token (a cut of the text and the whole text
         give both).  Without graph prefills (provisional): the reads at the rate the GPU gets them (the drive's, or
         the host link's where that is slower), plus the sink's n * d
break_even() is the largest pass the sink takes no longer than the graph, the tier states weighed; threshold() is T
for a workload: the least time for the workload's passes as the server prefills them (served(): a follow-up may
prefill the reply before it again), and between their sizes the break-even over the states those passes meet.  The states differ where the sink computes the tier's experts on the GPU and does not read them
(MXFP4); the graph copies every expert a pass touches, whatever the tier holds.  graph_pass_max() is the largest
graph pass the GPU's memory admits: the graph holds one role tensor of a layer's experts at a time, so its compute
buffers are that copy plus a slope a token, measured on a graph pass's reserve (graph_line()).  Without the sink's
compute (an engine without the prefill profile) c is 0 and d carries it; without the drive's rate the reads take no
time of their own; without both rates there is no threshold.
Feedback from served turns (feedback(), online.py) scales the parts: the CPU compute per placement, the bytes, and
the reads' time by the drive's state, which every placement shares.
"""
import bisect
import math
import statistics

GiB = 2 ** 30
# the scheduler offloads MUL_MAT_ID to the GPU from this many tokens (GGML_OP_OFFLOAD_MIN_BATCH's default): a smaller
# pass takes the sink whatever the threshold
OFFLOAD_MIN = 32


def union_points(segments):
    """U(n) of decode routing: segments of tokens (one run's requests in order; a token: the selected experts of each
    layer), windows of n consecutive tokens across the run's requests (one conversation): [n, mean distinct
    experts per layer] for n = 1, 2, 4, ... and all the tokens there are."""
    seq = [t for s in segments for t in s]
    if not seq:
        return []
    N, L = len(seq), len(seq[0])
    out = []
    for n in sorted({1 << i for i in range(N.bit_length()) if 1 << i <= N} | {N}):
        tot = cnt = 0
        for st in range(0, N - n + 1, max(1, n // 4)):
            win = seq[st:st + n]
            for li in range(L):
                tot += len({e for tok in win for e in tok[li]})
                cnt += 1
        out.append([n, round(tot / cnt, 2)])
    return out


def rest_fit(points):
    """(a, d) of rest = a * passes + d * tokens by least squares over (passes, tokens, rest) points; d alone
    (a = 0) from one point, prefills in proportion, or a fit with a < 0; a alone when d < 0."""
    sp, sn, sr = (sum(x[i] for x in points) for i in range(3))
    spp, snn, spn = sum(p * p for p, _, _ in points), sum(n * n for _, n, _ in points), sum(p * n for p, n, _ in points)
    spr, snr = sum(p * r for p, _, r in points), sum(n * r for _, n, r in points)
    det = spp * snn - spn * spn
    a, d = ((spr * snn - snr * spn) / det, (snr * spp - spr * spn) / det) if det > 1e-9 * spp * snn else (-1.0, 0.0)
    if a < 0:
        return 0.0, sr / sn
    if d < 0:
        return sr / sp, 0.0
    return a, d


def build(ins, dev, routing=()):
    """The model from insitu.json (every placement's bootstrap prefills, the graph's prefills), probe/device.json
    and, for a run without insitu's union, the decode routing of model.json."""
    P, ref = ins['placements'], ins['reference']
    E, L = ins['n_experts'], ins['layers']
    k = ins.get('topk') or len(routing[0][0][0])
    pts = sorted(P[ref]['prefill'], key=lambda x: x['tier'])
    ssd, knee, gpu = dev.get('ssd') or {}, dev.get('ssd_knee_qd'), dev.get('gpu') or {}
    r_d = ssd[str(knee)]['gibs'] * GiB if str(knee) in ssd else None
    link = gpu['h2d_pinned_gbs'] * 1e9 if gpu.get('h2d_pinned_gbs') and not gpu.get('integrated') else None
    rates = [r for r in (r_d, link) if r]
    b_e = dev.get('expert_read_bytes') or statistics.median(
        x['ssd_gb'] * 1e9 / (x['unique'] - x['tier']) for x in pts if x['unique'] > x['tier'])
    # the graph's copy target: one role tensor of a layer's experts, the largest (the header's role slices; without
    # them a whole expert's bytes bound it)
    role = E * max((dev.get('bytes_per_role') or {}).values() or [b_e])
    g = ins.get('graph') or {}
    gp = [x for x in g.get('passes') or [] if x['bytes'] and x['read_s']]
    # U(n): the prefill passes (the sink's distinct experts over the layers they report, the graph's copied bytes per
    # pass); below the smallest, the decode trace scaled to it at its size, when the trace reaches it within one
    # doubling (the spacing of its points), else as it is
    dec = {n: v for n, v in (ins.get('union') or union_points(routing[:1]))}
    prompt = {x['n_tokens']: x['unique'] / len(x['unique_per_layer']) for x in pts}
    for x in gp:
        prompt[round(x['n_tokens'] / x['passes'])] = x['bytes'] / (x['passes'] * L * b_e)
    u, scale = dict(dec), None
    if prompt and dec and min(prompt) <= 2 * max(dec):
        n0 = min(prompt)
        scale = [n0, round(prompt[n0] / coverage({'E': E, 'k': k, 'union': sorted(map(list, dec.items()))}, n0), 4)]
        u = {n: v * scale[1] for n, v in dec.items() if n < n0}
    u.update(prompt)
    union, top = [], 0.0
    for n in sorted(u):
        top = max(top, u[n])
        union.append([n, round(top, 2)])
    graph = None
    if gp:
        a, d = rest_fit([(x['passes'], x['n_tokens'], max(0.0, x['prompt_s'] - x['read_s'])) for x in gp])
        graph = {'R': sum(x['bytes'] for x in gp) / sum(x['read_s'] for x in gp), 'a': a, 'd': d,
                 'n': [x['n_tokens'] for x in gp]}

    def state(x):
        # r: the routes a prompt token takes, as the sink counted them: k L, less where a layer computes fewer tokens
        # (Muse's last layer only the output's)
        cpu = x.get('cpu_s')
        c = cpu / x['routes'] if cpu and x.get('routes') else 0.0
        r = x['routes'] / x['n_tokens'] if x.get('routes') else k * L
        read = x['ssd_gb'] * 1e9 / r_d if r_d else 0.0
        return {'n': x['n_tokens'], 'c': c, 'r': r,
                'd': max(0.0, (x['prompt_ms'] / 1e3 - max(read, cpu or 0.0)) / x['n_tokens'])}

    passes = {}
    for p in P:
        mine = sorted(P[p]['prefill'], key=lambda x: x['tier'])
        passes[p] = {'cold': state(mine[0]), 'warm': state(mine[-1])}
    return {'E': E, 'k': k, 'L': L, 'B_e': b_e, 'R_d': r_d, 'R_io': min(rates) if rates else None, 'union': union,
            'union_scale': scale, 'graph': graph, 'graph_memory': g.get('memory') if gp else None,
            'role_mib': role / 2 ** 20,
            'integrated': bool(gpu.get('integrated')),
            'cov_warm': pts[-1]['tier'] / pts[-1]['unique'], 'tier_at_warm': P[ref]['tier_slots'], 'pass': passes,
            'memory': {p: P[p].get('memory') or {} for p in P},
            'ubatch': ins.get('prefill_ubatch'),
            'ubatch_p': {p: P[p].get('prefill_ubatch') or ins.get('prefill_ubatch') for p in P}}


def coverage(m, n):
    """U(n): through the model's points in log-log, past the last on the slope of the last two, below the first on
    uniform routing's shape; at most what uniform routing touches, E (1 - (1 - k/E)^n)."""
    if n <= 0:
        return 0.0
    E, k, pts = m['E'], m['k'], m['union']
    cap = E * (1 - (1 - k / E) ** n)
    if not pts:
        return cap
    ns = [x for x, _ in pts]
    i = bisect.bisect_left(ns, n)
    if i < len(ns) and ns[i] == n:
        u = pts[i][1]
    elif i == 0 or len(pts) < 2:
        n0, u0 = pts[0] if i == 0 else pts[-1]
        u = u0 * cap / (E * (1 - (1 - k / E) ** n0))
    else:
        (n0, u0), (n1, u1) = pts[min(i, len(pts) - 1) - 1], pts[min(i, len(pts) - 1)]
        u = u0 * (n / n0) ** (math.log(u1 / u0) / math.log(n1 / n0))
    return min(cap, u)


def pass_seconds(m, p, n, graph, first, tier, k_bytes=1.0, k_cpu=1.0, k_drive=1.0):
    """(seconds, CPU compute seconds, bytes read) of one pass of n tokens; graph: on the ggml graph, else the sink."""
    st = m['pass'][p]['cold' if first else 'warm']
    b = k_bytes * m['L'] * coverage(m, n) * m['B_e']
    if graph:
        g = m.get('graph') or {'R': m['R_io'], 'a': 0.0, 'd': st['d']}
        return k_drive * b / g['R'] + g['a'] + n * g['d'], 0.0, b
    cov = 0.0 if first else min(1.0, m['cov_warm'] * tier / m['tier_at_warm'])
    cpu = k_cpu * n * st['r'] * st['c']
    read = b * (1 - cov)
    return max(k_drive * read / m['R_d'] if m['R_d'] else 0.0, cpu) + n * st['d'], cpu, read


def cut(m, p, n, ub=None):
    """The passes of a prompt of n new tokens: the ubatch ub (default the placement's) and the rest."""
    if n <= 0:
        return []
    ub = ub or (m.get('ubatch_p') or {}).get(p) or m.get('ubatch') or n
    return [ub] * (n // ub) + ([n % ub] if n % ub else [])


def seconds(m, p, n, tier, first, k_bytes=1.0, k_cpu=1.0, k_drive=1.0, threshold=None, ub=None):
    """(seconds, CPU compute seconds, bytes read) of N new prompt tokens on placement p: passes of the ubatch ub
    (default the placement's), each on the sink up to the threshold (None: every pass) or below the GPU's offload
    minimum, and on the graph otherwise; first: the server's first turn, on a cold tier; k_*: feedback from served
    turns where there is some."""
    tot = cpu = rb = 0.0
    for q in cut(m, p, n, ub):
        graph = threshold is not None and q > threshold and q >= OFFLOAD_MIN
        s, c, b = pass_seconds(m, p, q, graph, first, tier, k_bytes, k_cpu, k_drive)
        tot, cpu, rb = tot + s, cpu + c, rb + b
    return tot, cpu, rb


def break_even(m, p, cold=0.0):
    """The largest pass the sink takes no longer than the graph, its time weighed over the two tier states, cold
    (a server's first prefill) and 1 - cold (warm, after decode); by bisection over the pass size.  The graph is
    certainly faster once the sink's time a token beyond the graph's (r c + d - d_g, weighed) times n exceeds a whole
    union's read and the graph's time a pass, so the bisection runs below that; infinity when that slope is not
    positive (d_g at least the sink's time a token).  The difference of the two routes need not be convex in n:
    where the graph's copy reads faster than the drive's knee (DeepSeek on the Spark, 12.5 against 12.0 GB/s), a
    read-bound sink's difference is concave.  The bisection finds the crossing where the sign changes once, as on
    every stored run (the unit tests scan for it).  None without the inputs: the sink's compute, the drive's rate
    (and the link's on a discrete GPU), the bytes of an expert."""
    states = [(w, s == 'cold') for w, s in ((cold, 'cold'), (1.0 - cold, 'warm')) if w > 0]
    sts = [(w, m['pass'][p]['cold' if first else 'warm']) for w, first in states]
    if not (all(st['c'] > 0 for _, st in sts) and m['R_d'] and m['R_io'] and m['B_e']):
        return None
    g = m.get('graph')
    R, a = (g['R'], g['a']) if g else (m['R_io'], 0.0)
    slope = sum(w * (st['r'] * st['c'] + st['d'] - (g['d'] if g else st['d'])) for w, st in sts)
    if slope <= 0:
        return math.inf

    def graph_faster(n):
        return sum(w * (pass_seconds(m, p, n, False, first, m['tier_at_warm'])[0] -
                        pass_seconds(m, p, n, True, first, m['tier_at_warm'])[0]) for w, first in states) > 0

    lo, hi = 0, int((m['E'] * m['L'] * m['B_e'] / R + a) / slope) + 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if graph_faster(mid):
            hi = mid
        else:
            lo = mid
    return lo


def served(turns, reply=0):
    """The workload's turns as the server prefills them: (new prompt tokens, on a cold tier, weight).  The first turn
    once, on a cold tier.  With replies of `reply` tokens, each follow-up in two forms of weight 1/2, both on the warm
    tier: its own new tokens, when the server's cache holds the reply before it; and that reply again with them
    (reply + n), when the reply's text does not give back the tokens the server generated (the history renders it
    in other tokens): the cache then parts inside the reply, and the server prefills again from its latest
    checkpoint, the end of the previous prompt (a cache that can drop its tail, from where they part: at most that).
    Which form a follow-up takes is not known before its reply exists (DeepSeek-V4.1-Flash-1017 on the Spark ysy9:
    9 of 15 follow-ups prefilled the reply again; Muse on spark-b9a5: none of 3).  Without replies (reply 0) every
    turn once, as it is counted."""
    out = []
    for i, n in enumerate(turns):
        if i == 0 or not reply:
            out.append((n, i == 0, 1.0))
        else:
            out += [(n, False, 0.5), (reply + n, False, 0.5)]
    return out


def offloaded(m, p, turns, ub=None, reply=0):
    """The workload's passes the GPU can take, (tokens, on a cold tier, weight): each turn as the server prefills it
    (served()) in passes of ub, the passes below the offload minimum left out."""
    return [(q, first, w) for n, first, w in served(turns, reply) for q in cut(m, p, n, ub) if q >= OFFLOAD_MIN]


def cold_share(ps):
    """The weight of the passes on a cold tier among the passes ps (offloaded()), 0 when there are none."""
    total = sum(w for _, _, w in ps)
    return sum(w for _, f, w in ps if f) / total if total else 0.0


def threshold(m, p, turns=None, ub=None, reply=0):
    """T, the prefill threshold of placement p: one value the engine applies to every pass of every request.
    Without a workload, the break-even on the warm tier, the state of a serving server.  turns: the workload's turns
    (each turn's new prompt tokens, in passes of ub), one conversation on a freshly started server as the plan
    prices it, the first turn on a cold tier and the others on the warm one; reply: the tokens of each reply, so
    that a follow-up is priced in both forms the server may prefill it in (served()).  T then sends each of the
    workload's passes the way that gives the workload its least time.  Where that leaves T free (between two of the
    workload's pass sizes), T is the break-even weighed over the states the workload's passes the GPU can take meet
    (all warm when there are none).  So a long first prompt with follow-ups the GPU does not take, and no replies,
    gets the cold break-even; replies add a warm pass of the reply's size for each follow-up, which pulls T up from
    it (DeepSeek on the Spark: a follow-up that prefills a 512-token reply again takes 7 s on the warm sink and
    17 s on the graph); turns of one size weigh the states 1 : turns - 1."""
    if not turns:
        return break_even(m, p)
    ps = offloaded(m, p, turns, ub, reply)
    t = break_even(m, p, cold_share(ps))
    if t is None or not ps:
        return t
    # the workload's time changes only where T crosses one of its pass sizes: the break-even and both sides of each
    # size cover every choice; among equal times, the one nearest the break-even
    cand = {t} | {q for q, _, _ in ps} | {q - 1 for q, _, _ in ps}
    forms = served(turns, reply)
    time = {T: sum(w * seconds(m, p, n, m['tier_at_warm'], first, threshold=T, ub=ub)[0] for n, first, w in forms)
            for T in cand}
    best = min(time.values())
    return min((T for T in cand if time[T] == best), key=lambda T: (0 if T == t else abs(T - t), -T))


def graph_line(m, p):
    """The graph route's compute buffers at a pass of n tokens, role + s n MiB on the device and h n on the host:
    role the largest role copy (one role tensor of a layer's experts, model.role_mib), s and h from the largest
    reserve of a graph pass that held it (more than role MiB at n >= OFFLOAD_MIN tokens): the graph's prefills'
    (every pass theirs), else the bootstrap's of placement p (the reserves of its graph passes, logparse).  None
    when no such reserve was measured: the sink's buffers say nothing of the graph's, so no graph pass is admitted."""
    role = m['role_mib']
    g = m.get('graph_memory')
    for src, mine in ((g, lambda r: True), (m['memory'].get(p), lambda r: r[3])):
        rs = [r for r in (src or {}).get('reserves') or [] if mine(r) and r[0] >= OFFLOAD_MIN and r[1] > role]
        if src is g and g and 'reserves' not in g and g.get('compute_mib', 0) > role and g.get('n_ubatch'):
            # the graph's prefills before reserves were read: its reserve at load is a graph pass's
            rs = [[g['n_ubatch'], g['compute_mib'], g.get('host_compute_mib', 0.0), 1]]
        if rs:
            n, dev, host, _ = max(rs)
            return {'role': role, 'slope': (dev - role) / n, 'host': host / n, 'n': n}
    return None


def graph_pass_max(m, p, limit):
    """The largest graph pass up to `limit` tokens the GPU's memory admits, until a memory plan exists: what the
    device had free at load less the dense weights, the KV cache and 1 GiB for what these leave out (the CUDA
    context, workspaces) must hold the graph's compute buffers at the pass (graph_line(); with the host's on unified
    memory, the same DRAM).  0 when no pass fits, no server reported its memory or the graph's was not measured."""
    mem = m.get('graph_memory') or m['memory'].get(p) or {}
    g = graph_line(m, p)
    if not (g and mem.get('free_mib') and mem.get('dense_mib')):
        return 0
    room = mem['free_mib'] - mem['dense_mib'] - mem.get('kv_mib', 0.0) - 1024 - g['role']
    per_token = g['slope'] + (g['host'] if m.get('integrated') else 0.0)
    return max(0, min(limit, int(room / per_token))) if room > 0 else 0


def drive_factor(time_of, measured, lo=1 / 16, hi=16.0):
    """The factor f on the reads' time with time_of(f) = measured, by bisection in log f; None when the time does
    not depend on it (a compute-bound sink) or the measured time is below what the rest alone takes."""
    if time_of(hi) <= time_of(lo) or measured <= time_of(lo):
        return None
    if measured >= time_of(hi):
        return hi
    for _ in range(40):
        f = math.sqrt(lo * hi)
        lo, hi = (f, hi) if time_of(f) < measured else (lo, f)
    return math.sqrt(lo * hi)


def feedback(m, turns, how, recent=1):
    """Measured over modelled for served prefills (turns: online.served's, with a prompt): pf_cpu, the sink's CPU
    compute per placement; k_bytes, the bytes read; pf_drive, the drive's state from the latest `recent` turns, the
    factor on the reads' time that makes the model's time the measured one.  how(p): the threshold and ubatch
    placement p runs with."""
    def model(t, **kw):
        T, ub = how(t['cpus'])
        return seconds(m, t['cpus'], t['n_prompt'], t['tier'], t['first'], threshold=T, ub=ub, **kw)

    out = {'pf_cpu': {}, 'k_bytes': 1.0, 'pf_drive': 1.0}
    for p in sorted({t['cpus'] for t in turns}):
        pc = [(t['cpu_s'], model(t)[1]) for t in turns if t['cpus'] == p and t['cpu_s']]
        if pc and sum(c for _, c in pc) > 0:
            out['pf_cpu'][p] = sum(x for x, _ in pc) / sum(c for _, c in pc)
    rd = [(t['read_bytes'], model(t)[2]) for t in turns if t['read_bytes']]
    rd = [x / b for x, b in rd if b > 0]
    if rd:
        out['k_bytes'] = statistics.mean(rd)
    fs = []
    for t in turns[-recent:]:
        kw = {'k_bytes': out['k_bytes'], 'k_cpu': out['pf_cpu'].get(t['cpus'], 1.0)}
        f = drive_factor(lambda f: model(t, k_drive=f, **kw)[0], t['prefill_s'])
        if f:
            fs.append(f)
    if fs:
        out['pf_drive'] = statistics.mean(fs)
    return out
