#!/usr/bin/env python3
"""In-situ costs (adaptive.sh step 3): what the bootstrap runs measured, per CPU placement.

Each bootstrap run (boot_client.py: cold prefill of the text, decode, warm prefill of prompt + reply) gives, for its
placement:
  decode   per-layer mean time by (tier hits, prefetched, read) class with its CPU compute part; share of non-tier
           experts the prefetch had ready; time outside the MoE sink fitted as a + b * position; the tier size the
           machine's memory allowed; the admission point the engine settled on; measured hits and decode rate
  prefill  two points (cold tier, warm tier): tokens, distinct experts touched, how many were tier-resident, bytes
           read, the sink's CPU compute, and the server's prompt time; the GPU memory the server had at load
  prefetch the depth the engine ran the next-layer prefetch at, its hint statistics per rank and the drive's idle
           window (its last report), what other readers took from the drives in decode (what they read less the
           sink's own reads), the slots of the RAM expert cache, its storage at load and the new slots decode
           committed (first touches) in tokens 1-128 and after, and the server's load time (arm.sh)
The graph arm (--graph: boot_client.py --graph on a server at the threshold 0) gives the GPU graph's prefills: per
prefill its tokens, passes, the bytes and read time of the graph route's expert copy and the server's prompt time,
and the GPU memory at load with the graph route's compute buffers.
The model profile (decode routing sample, W(tau), online LRU curve) comes from the decode of all runs, the experts
n consecutive decode tokens touch (prefill.py's U(n)) from the reference's.
usage: insitu.py --run all=BOOT_A [--run pin=BOOT_B] [--graph GRAPH_ARM] --skip 128 -o insitu.json --profile model.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prefill  # noqa: E402
from logparse import hits, median, read_arm  # noqa: E402
from sim import lstsq  # noqa: E402


def measure(run, skip):
    arm = read_arm(run, skip)
    chats = [c for c in arm['client'] if c.get('kind', 'chat') == 'chat' and 'error' not in c]
    # the depth the bootstrap's engine ran (it prints none below depth 1)
    pf = arm['prefetch'] or {}
    pf_k = int(pf.get('topk') or 0)
    # other readers of the drives in decode: what the drives read from the first decode token to the last (the
    # client) less the sink's own reads, its layers' and its prefetch's (the decode of the run is the first request's)
    seen = [c['drive_read_bytes'] for c in chats if c.get('drive_read_bytes') is not None]
    own = sum(b for _, b in arm['lio'].values()) + (pf.get('reader_bytes') or 0.0)
    other = max(0.0, seen[0] - own) if seen and arm['lio'] else None
    groups = [g for g in arm['groups'] if len(g) >= 64]
    window = [k for g in groups for k in g[skip:]]
    h = hits(arm, window)
    pre = cap = nxs = 0
    ready = defaultdict(lambda: [0] * 17)      # nx -> how often nr of them were host-ready (prefetched or RAM cache)
    cls = defaultdict(lambda: [0, 0.0, 0.0])
    X, Y = [], []
    for k in window:
        for layer, (sel, vh, ch) in arm['trace'][k].items():
            nx = sum(1 for x in vh if not x)
            nr = sum(1 for x, y in zip(vh, ch) if not x and y)
            pre += nr
            cap += min(nx, pf_k)
            nxs += nx
            ready[nx][nr] += 1
            lp = arm['lprof'].get((k[0], k[1], layer))
            if not lp:
                continue
            nv, nm = len(vh) - nx, nx - nr
            c = cls[(nv, nr, nm)]
            c[0] += 1
            c[1] += lp[0] / 1e3
            c[2] += lp[1] / 1e3
            X.append([1.0, nv, nr, nm, 1.0 if nm else 0.0])
            Y.append(lp[0] / 1e3)
    win = [k for k in window if k in arm['prof']]
    # the drive's state while this bootstrap decoded: its service time per MB read (placement-free: on a Gen4 x4
    # machine the rest of the sink over the model's went with it to the power 1.2, R^2 0.98, both placements alike)
    io = [arm['lio'][k] for k in win if k in arm.get('lio', {})]
    drive_us_per_mb = sum(x[0] for x in io) / (sum(x[1] for x in io) / 1e6) if io and sum(x[1] for x in io) else None
    sink = {f: sum(arm['prof'][k][i] for k in win) / len(win) / 1e3 for i, f in enumerate(('total', 'wait_io', 'compute', 'other'))}
    # time outside the sink against the context position, and the decode rate of the window
    pts, rates = [], []
    for g, c in zip(groups, chats):
        ts = c.get('tok_s') or []
        for i, k in enumerate(g):
            if i < 16 or i + 1 >= len(ts) or ts[i] is None or ts[i + 1] is None or k not in arm['prof']:
                continue
            pts.append((float(k[1]), (ts[i + 1] - ts[i]) * 1e3 - arm['prof'][k][0] / 1e3))
        if len(ts) > skip + 2 and ts[skip] is not None and ts[-1] is not None:
            rates.append((len(ts) - skip - 1) / (ts[-1] - ts[skip]))
    mx, my = sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    slope = sum((x - mx) * (y - my) for x, y in pts) / sxx if sxx else 0.0
    # prefill points: passes in log order matched to the client records that carry a prompt time
    timed = [c for c in arm['client'] if (c.get('timings') or {}).get('prompt_ms')]
    # and each one's sink compute and the graph route's expert copy (bytes, read time), one record per prefill
    prefill = []
    graphs = arm['pf_graphs'] if len(arm['pf_graphs']) == len(timed) else [None] * len(timed)
    for p, c, g in zip(arm['prefill'], timed, graphs):
        prefill.append({'kind': c.get('kind', 'chat'), 'n_tokens': p['n_tokens'], 'unique': p['unique'], 'tier': p['tier'],
                        'routes': p['routes'], 'ssd_gb': round(p['ssd_bytes'] / 1e9, 2),
                        'prompt_ms': c['timings']['prompt_ms'], 'unique_per_layer': p['unique_per_layer'],
                        'graph_gb': round(g['graph_bytes'] / 1e9, 3) if g else None,
                        'graph_read_s': round(g['graph_read_us'] / 1e6, 3) if g else None,
                        'cpu_s': round(g['cpu_s'], 3) if g else None})
    rb = sum(g['graph_bytes'] for g in graphs if g)
    ru = sum(g['graph_read_us'] for g in graphs if g)
    routing = []
    for g in groups:
        L = len(arm['trace'][g[0]])
        routing.append([[arm['trace'][k][l][0] for l in sorted(arm['trace'][k])] for k in g if len(arm['trace'][k]) == L])
    adm = [x for x in arm['admission'] if x['min_count'] is not None]
    return {
        'run': run,
        'measured': {'decode_tps_window': round(sum(rates) / len(rates), 3) if rates else None,
                     'sink_ms_per_token': {k: round(v, 2) for k, v in sink.items()}, 'hits': h, 'tokens_in_window': len(win)},
        't_out_fit': {'a_ms': round(my - slope * mx, 3), 'b_ms_per_ktok': round(slope * 1e3, 4), 'points': len(pts),
                      'positions': [int(min(x for x, _ in pts)), int(max(x for x, _ in pts))], 'mean_position': round(mx),
                      'mean_ms': round(my, 2)},
        'prefetch_share': round(pre / cap, 4) if cap else 0.0,
        'ready_share_all': round(pre / nxs, 4) if nxs else 0.0,
        'ready_given_nx': {str(n): c[:n + 1] for n, c in sorted(ready.items())},
        'admission_min_count': median([x['min_count'] for x in adm]) if adm else arm.get('tier_min_count'),
        'tier_slots': arm['tier_slots_max'],
        'tier_slots_first': arm.get('tier_slots_first'),
        'tier_slot_bytes': round(arm['tier_bytes'][0]) if arm['tier_bytes'] else None,
        'tier_base_bytes': arm['tier_bytes'][1] if arm['tier_bytes'] else None,
        'decode_threads_log': arm['threads'] or {},
        'layer_table': {f'{a},{b},{c}': [v[0], round(v[1] / v[0], 4), round(v[2] / v[0], 4)] for (a, b, c), v in sorted(cls.items())},
        'layer_beta': [round(b, 4) for b in lstsq(X, Y)],
        'prefill': prefill,
        'graph_read_gbps': round(rb / ru / 1e3, 4) if ru else None,
        'memory': arm['memory'],
        'checkpoint_align': arm['checkpoint_align'],
        'ledger': arm['ledger'],
        'routing': routing,
        'mrc': arm['mrc'],
        'prefill_ubatch': max((g.get('ubatch') or 0 for g in arm['pf_graphs']), default=0) or None,
        'drive_us_per_mb': round(drive_us_per_mb, 2) if drive_us_per_mb else None,
        'cache_load': arm['cache_load'],
        'cache_new_slot_gb': [round(x / 1e9, 3) for x in arm['new_slot_windows']],
        'load_s': arm['load_s'],
        'pf_k': pf_k,
        'prefetch': pf or None,
        'drive_other_bytes': other,
        'ram_slots': arm['ram_slots'],
        # the decode plan (CPU or GPU for each non-resident expert) and the page-locking it copies from
        'decode_place': arm.get('decode_place'),
        'host_register': arm.get('host_register'),
        'host': arm.get('host'),
    }


def measure_graph(run):
    """The graph arm's prefills, matched to its client's records by their tokens; none when its server did not start
    or the prefills failed (the graph route is then not measured, and 'why' says so)."""
    try:
        arm = read_arm(run, 0)
    except OSError as e:
        return {'run': run, 'passes': [], 'memory': {}, 'why': f'no server log ({e})'}
    graphs = [g for g in arm['pf_graphs'] if g['graph_bytes'] and g['graph_read_us']]
    passes = []
    for c in arm['client']:
        t = c.get('timings') or {}
        g = next((g for g in graphs if g['n_tokens'] == t.get('prompt_n')), None) if t.get('prompt_ms') else None
        if g:
            graphs.remove(g)
            passes.append({'n_tokens': g['n_tokens'], 'passes': g['graph_passes'] or 1, 'bytes': g['graph_bytes'],
                           'read_s': round(g['graph_read_us'] / 1e6, 3), 'prompt_s': round(t['prompt_ms'] / 1e3, 3)})
    return {'run': run, 'passes': passes, 'memory': arm['memory'],
            'why': None if passes else 'no graph prefill (the server did not start, or its prefills failed)'}


ap = argparse.ArgumentParser()
ap.add_argument('--run', action='append', required=True, help='placement=run_dir; the first one is the reference')
ap.add_argument('--graph', help="the graph arm's run directory (adaptive.sh step 2g)")
ap.add_argument('--ubatch-probe', help='ubatch.json from adaptive.sh: tier slots the engine allotted at start with '
                'each larger prefill ubatch (null: it did not start)')
ap.add_argument('--skip', type=int, default=128)
ap.add_argument('-o', '--out', required=True)
ap.add_argument('--profile', required=True)
a = ap.parse_args()
placements = {}
for spec in a.run:
    name, run = spec.split('=', 1)
    placements[name] = measure(run, a.skip)
first = next(iter(placements.values()))
union = prefill.union_points(first['routing'])
routing = [seg for m in placements.values() for seg in m.pop('routing')]
n_exp = max(e for seg in routing for tok in seg for ex in tok for e in ex) + 1


def working_set(taus):
    out = {}
    for tau in taus:
        tot = cnt = 0
        for seg in routing:
            for st in range(0, len(seg) - tau + 1, max(1, tau // 4)):
                for li in range(len(seg[0])):
                    tot += len({e for tok in seg[st:st + tau] for e in tok[li]})
                    cnt += 1
        if cnt:
            out[tau] = round(tot / cnt, 2)
    return out


prof = {'schema': 2, 'layers': len(routing[0][0]), 'n_experts': n_exp, 'topk': len(routing[0][0][0]),
        'tokens': [len(s) for s in routing], 'working_set_mean': working_set([1, 2, 4, 8, 16, 32, 64, 128, 256]),
        'mrc_online': first['mrc'], 'routing': routing}
json.dump(prof, open(a.profile, 'w'), separators=(',', ':'))
# every placement's decode table at one drive state, the reference's: each bootstrap ran at its own (the drive
# drifts between runs: 12 then 6 GiB/s on one machine), and the part of a layer's time that waits on the drive scales with
# its service time per byte, the CPU compute part does not
gref = placements[next(iter(placements))].get('drive_us_per_mb')
for name, m in placements.items():
    g = m.get('drive_us_per_mb')
    f = gref / g if gref and g else 1.0
    m['drive_scale'] = round(f, 4)
    if f != 1.0:
        m['layer_table_raw'] = m['layer_table']
        m['layer_table'] = {k: [v[0], round(v[2] + (v[1] - v[2]) * f, 4), v[2]] for k, v in m['layer_table'].items()}
        seen = [v for v in m['layer_table_raw'].values() if v[0] >= 5]
        cpu_share = sum(v[0] * v[2] for v in seen) / max(1e-9, sum(v[0] * v[1] for v in seen))
        m['layer_beta'] = [round(b * (cpu_share + (1 - cpu_share) * f), 4) for b in m['layer_beta']]
# larger prefill ubatches: each measured CPU placement again with a smaller tier; everything else of the placement
# holds, the prefill passes take more tokens each.  The probe (ubprobe.py) and the reference bootstrap (the engine
# default, as the probe) give the same quantity, the tier after its first decode growth; the difference is what the
# larger ubatch's buffers take from the tier for decode, and each placement's tier loses it.  A probe of the older
# format (one number, the most the tier held) is that tier.  Neither prices the graph route's workspace, which a
# graph prefill takes back from the tier on unified memory and decode refills afterwards (design F_g).
if a.ubatch_probe:
    ref_first = placements[next(iter(placements))].get('tier_slots_first')
    for ub, v in sorted(json.load(open(a.ubatch_probe)).items(), key=lambda kv: int(kv[0])):
        probe_first, held = (v.get('first'), v.get('held')) if isinstance(v, dict) else (None, v)
        if not (probe_first or held):
            continue
        for name in [n for n in placements if '@' not in n]:
            m = placements[name]
            slots = (m['tier_slots'] or 0) - (ref_first - probe_first) if probe_first and ref_first else held
            if not slots or slots >= (m['tier_slots'] or 0):
                continue                  # no smaller tier: the probe saw no cost (or a noisy start); keep it out
            placements[f'{name}@{ub}'] = dict(m, tier_slots=float(slots), prefill_ubatch=int(ub), derived_from=name)
# the largest prompt chunk one prefill pass takes (the engine's ubatch); a longer prompt is prefilled in passes of it
ubs = [m['prefill_ubatch'] for m in placements.values() if m.get('prefill_ubatch') and 'derived_from' not in m]
ins = {'schema': 3, 'skip': a.skip, 'pf_k': first['pf_k'], 'reference': next(iter(placements)), 'layers': prof['layers'],
       'prefill_ubatch': max(ubs) if ubs else None,
       'n_experts': n_exp, 'topk': prof['topk'], 'union': union, 'placements': placements,
       'graph': measure_graph(a.graph) if a.graph else None}
json.dump(ins, open(a.out, 'w'), indent=1)
for name, m in placements.items():
    print(f"== placement {name}: {m['run']}")
    print('  measured', json.dumps(m['measured']))
    print(f"  drive {m.get('drive_us_per_mb')} us/MB read in decode, table scaled x{m.get('drive_scale')} to the reference's drive state")
    print(f"  t_out {m['t_out_fit']}  prefetch {m['prefetch_share']} (of all non-tier {m['ready_share_all']})  min_count {m['admission_min_count']}  tier {m['tier_slots']}"
          f"  threads {m['decode_threads_log']}")
    for p in m['prefill']:
        print('  prefill', {k: v for k, v in p.items() if k != 'unique_per_layer'})
    if m['graph_read_gbps']:
        print(f"  the graph route's expert copy read {m['graph_read_gbps']} GB/s")
    print(f"  memory at load: {m['memory']}; checkpoints after the batch at multiples of {m['checkpoint_align']}; "
          f"host-memory ledger {m['ledger']}")
    load = m.get('cache_load')
    print(f"  RAM cache at load: " + (f"{int(load['slots'])} slots, {load['committed_bytes'] / 1e9:.2f} GB paged in in "
          f"{load['us'] / 1e6:.2f} s" if load else 'none (at its first decode token)')
          + "; decode committed {} GB of new slots (first touches) in tokens 1-128 and {} GB after".format(
              *(m.get('cache_new_slot_gb') or [None, None]))
          + (f"; the server loaded in {m['load_s']} s" if m.get('load_s') is not None else
             '; the server\'s load time not recorded'))
    pf = m['prefetch'] or {}
    print(f"  prefetch depth {m['pf_k']}: {pf.get('layers')} hinted layers, idle window p50 {pf.get('window_us_p50')} us, "
          f"novel {pf.get('novel_r')}, novel and routed {pf.get('novel_routed_r')} of {pf.get('predicted_r')}; "
          f"other readers {m['drive_other_bytes']} B in decode; RAM cache {m['ram_slots']} slots")
    dp, hr, ho = m.get('decode_place') or {}, m.get('host_register') or {}, m.get('host') or {}
    if dp:
        copy = ho.get('copy_us_per_mb')
        print(f"  decode plan: {dp.get('gpu_per_token')} GPU experts a token from host memory (in place {ho.get('in_place')}, "
              f"copy {round(1e3 / copy, 1) if copy else None} GB/s), refused {dp.get('refused')}; predicted lanes' time "
              f"error p50 {dp.get('error_p50_pct')}% p90 {dp.get('error_p90_pct')}%; page-locked slots {hr.get('registered')} "
              f"(failures {hr.get('failures')}, {hr.get('us_mean')} us each)")
    for key, v in sorted(m['layer_table'].items(), key=lambda kv: -kv[1][0])[:5]:
        print(f'  layer class {key}: n={v[0]} mean {v[1]:.3f} ms (cpu {v[2]:.3f})')
    need = ('free_mib', 'dense_mib', 'compute_mib', 'n_ubatch')
    if not all(m['memory'].get(k) for k in need):
        print(f"  warning: the server log has no {', '.join(k for k in need if not m['memory'].get(k))} at load: "
              'no graph pass is admitted (prefill.graph_pass_max), every pass stays on the sink')
g = ins['graph']
if g:
    print(f"== the GPU graph's prefill: {g['run']}")
    for x in g['passes']:
        print(f"  {x['n_tokens']} tokens in {x['passes']} pass(es): {x['bytes'] / 1e9:.1f} GB copied in {x['read_s']} s "
              f"({x['bytes'] / 1e9 / x['read_s']:.2f} GB/s), prompt {x['prompt_s']} s")
    print(f"  {g['why'] or 'memory at load: ' + str(g['memory'])}")
print(f"model profile: {sum(prof['tokens'])} decode tokens in {len(routing)} segments, {prof['layers']} layers x {n_exp} "
      f"experts, top-{prof['topk']}; W(tau) {prof['working_set_mean']}; U(n) of the reference {union}")
