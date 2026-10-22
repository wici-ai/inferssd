#!/usr/bin/env python3
"""The four conversations of adaptive.sh validate side by side.  Per conversation: wall time from the first
request to the last token (restarts included), per turn the configuration, prefill time and decode rate of the
window; the static plan's prediction for the static ones; for the loops every decision with its prediction of the
next turn against what that turn measured.  For the static ones also each prefill against the prefill model, by
the route the server took for it (its log) and the tier state it met (the first turn cold, the others warm): a
prefill predicted more than 15 % off has a wrong input (design 3.3).  The flag is per turn and per route and state,
so that an accurate long prefill does not hide the errors of the short ones in a total.
usage: compare_online.py PIPELINE_DIR"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prefill  # noqa: E402
from logparse import ArmParser, _lines  # noqa: E402


def drive_gibs(conv):
    """Read rate of the conversation's drive check (reads.sh: ioreplay's or fio's JSON; fio may print warnings
    before it), 0 if there is none."""
    for path in (f'{conv}/drive.json', f'{conv}/fio.json'):
        try:
            txt = open(path).read()
            return json.loads(txt[txt.index('{'):])['jobs'][0]['read']['bw_bytes'] / 2 ** 30
        except (OSError, ValueError, KeyError, IndexError):
            pass
    return 0.0


def routes(conv):
    """(tokens, route) of every prefill the conversation's servers ran, in order: the graph where its expert copy
    read for it."""
    out = []
    for inst in sorted(glob.glob(f'{conv}/inst*')):
        log = f'{inst}/server.log.gz' if os.path.exists(f'{inst}/server.log.gz') else f'{inst}/server.log'
        if not os.path.exists(log):
            continue
        a = ArmParser()
        for line in _lines(log):
            if '[kimi_cpu_prefill_graph] event=' in line or line.startswith('[moe_prefill_copy]'):
                a.feed(line)
        out += [(g['n_tokens'], 'graph' if g['graph_bytes'] else 'sink') for g in a.pf_graphs]
    return out


FLAG = 15.0     # percent: a prefill predicted further off than this has a wrong input (design 3.3)


def route_check(m, conv, turns):
    """Each turn's prefill against the model's (neutral, the first turn on a cold tier), flagged per turn, then
    per route and tier state.  Returns the flags."""
    rs, ri, by, flags = routes(conv), 0, {}, []
    for i, t in enumerate(turns):
        n = t['prompt_n']
        j = next((j for j in range(ri, len(rs)) if rs[j][0] == n), None)
        route = rs[j][1] if j is not None else 'unknown route'
        ri = j + 1 if j is not None else ri
        pred = prefill.seconds(m.PF, t['cpus'], n, t['tier'], i == 0, threshold=m.route(t['cpus']),
                               ub=m.ubatch(t['cpus']))[0]
        state = 'cold' if i == 0 else 'warm'
        by.setdefault((route, state), []).append((pred, t['prefill_s']))
        err = (pred / t['prefill_s'] - 1) * 100 if t['prefill_s'] else 0.0
        flag = abs(err) > FLAG
        if flag:
            flags.append(f"turn {t['turn']}: {n} tokens on the {state} {route} {err:+.1f}%")
        print(f"   turn {t['turn']} prefill of {n} tokens on the {state} {route}: model {pred:.1f} s, measured "
              f"{t['prefill_s']:.1f} s ({err:+.1f}%)" + (f'; more than {FLAG:g} % off' if flag else ''))
    for (route, state), v in sorted(by.items()):
        p, x = sum(a for a, _ in v), sum(b for _, b in v)
        err = (p / x - 1) * 100 if x else 0.0
        off = sum(abs(a / b - 1) * 100 > FLAG for a, b in v if b)
        print(f"   {state} {route}: {len(v)} prefills, model {p:.1f} s against {x:.1f} s measured ({err:+.1f}%)"
              + (f'; {off} of them more than {FLAG:g} % off: an input of this route and state is wrong' if off else ''))
    return flags


d = sys.argv[1]
plan = json.load(open(f'{d}/plan.json'))
try:
    from machine import Machine
    model = Machine(d)
except (OSError, KeyError, ValueError) as e:
    model = None
    print(f'no prefill model for the route check: {e!r}')
D = plan['workload']['decode']
table = {(r['tier'], r['cpus']): r for r in plan['table']}
res = {}
for c in sorted(glob.glob(f'{d}/conv-*')):
    label = os.path.basename(c)[5:]
    ev = [json.loads(x) for x in open(f'{c}/events.jsonl')] if os.path.exists(f'{c}/events.jsonl') else []
    turns = [e for e in ev if e['event'] == 'turn' and 'error' not in e]
    if not turns:
        print(f'{label}: no turns')
        continue
    conv0 = next(e for e in ev if e['event'] == 'conversation')
    done = next((e for e in ev if e['event'] == 'done'), None)
    first_req = conv0['wall']
    wall = (done['wall'] if done else turns[-1]['wall']) - first_req
    model_t = sum(t['prefill_s'] + D / t['decode_tps'] for t in turns if t['decode_tps'])
    res[label] = {'wall_s': wall, 'turn_model_s': model_t, 'turns': turns, 'events': ev}
    cfgs = ' -> '.join(dict.fromkeys(f"{t['tier']},{t['cpus']}" for t in turns))
    fio = drive_gibs(c)
    print(f"== {label}: {len(turns)} turns, configs {cfgs}; wall {wall:.1f} s (first request to last token, restarts "
          f"included); prefill + D/rate {model_t:.1f} s; drive gate {fio:.2f} GiB/s")
    for t in turns:
        print(f"   turn {t['turn']}: {t['tier']},{t['cpus']:4s} prompt {t['prompt_n']} (cached {t['cache_n']}) prefill "
              f"{t['prefill_s']:6.1f} s  decode {t['decode_tps'] or 0:.2f} tok/s  wall {t['wall_s']:6.1f} s")
    if label.startswith('static'):
        cfg = (turns[0]['tier'], turns[0]['cpus'])
        p = table.get(cfg)
        if p:
            err = (p['total_s'] / model_t - 1) * 100
            print(f"   static plan predicted {p['total_s']:.1f} s ({err:+.1f}%): " + ', '.join(
                f"{x['prefill_s']:.1f}/{x['decode_tps']:.2f}" for x in p['turns']))
            res[label]['pred_s'] = p['total_s']
        if model:
            res[label]['route_flags'] = route_check(model, c, turns)
    for e in ev:
        if e['event'] == 'decision':
            dd = e['decision']
            nxt = next((t for t in turns if t['turn'] == e['before_turn']), None)
            pick = next(r for r in dd['candidates'] if [r['tier'], r['cpus']] == dd['to'])
            stay = next(r for r in dd['candidates'] if r['stay'])
            s = (f"   before turn {e['before_turn']}: {'SWITCH to ' + str(dd['to']) if dd['switch'] else 'stay'}; rest predicted "
                 f"stay {stay['rest_s']:.0f} s, best {dd['candidates'][0]['rest_s']:.0f} s ({dd['candidates'][0]['cpus']}, "
                 f"tier {dd['candidates'][0]['tier']}); feedback {dd['feedback']}")
            if nxt:
                pt = pick['turns'][0]
                s += (f"; next turn predicted {pt['prefill_s']:.1f} s / {pt['decode_tps']:.2f} tok/s, measured "
                      f"{nxt['prefill_s']:.1f} s / {nxt['decode_tps'] or 0:.2f} tok/s")
            print(s)
        elif e['event'] == 'switched':
            print(f"   switch took {e['switch_s']:.1f} s")
PLACE_FLAG = 15.0   # percent: the decode plan's median error of its predicted lanes' time per layer


def decode_place(conv, pinned_gbs):
    """The decode plan of the conversation's servers (their destroy receipts): GPU experts a token from host memory,
    the prediction's error, the page-locking, the engine's copy rate, the CPU beside the GPU's host jobs, and the
    tier's final admission point (page-locked slots make admission uploads cheaper; the tier then admits more).
    Flags a median error above PLACE_FLAG, a page-locked copy rate below half the probe's (registration failed, or
    the link is shared), and page-locked slots without a measured copy (the plan cannot use them)."""
    out = []
    for inst in sorted(glob.glob(f'{conv}/inst*')):
        log = f'{inst}/server.log.gz' if os.path.exists(f'{inst}/server.log.gz') else f'{inst}/server.log'
        if not os.path.exists(log):
            continue
        a = ArmParser()
        for line in _lines(log):
            if line.startswith(('[kimi_decode_place] reason=', '[kimi_host_register] reason=destroy', '[kimi_vram_host]',
                                '[kimi_vram_admission]')):
                a.feed(line)
        dp, hr, ho = a.decode_place or {}, a.host_register or {}, a.host or {}
        adm = a.adm[-1] if a.adm else {}
        if not dp:
            continue
        copy_gbs = 1e3 / ho['copy_us_per_mb'] if ho.get('copy_us_per_mb') else None
        flags = []
        if (dp.get('error_p50_pct') or 0) > PLACE_FLAG:
            flags.append(f"plan error p50 {dp['error_p50_pct']:.0f}%")
        if copy_gbs and pinned_gbs and copy_gbs < 0.5 * pinned_gbs:
            flags.append(f'copy {copy_gbs:.1f} GB/s < half the probe page-locked {pinned_gbs:.1f}')
        if hr.get('failures'):
            flags.append(f"{hr['failures']:.0f} page-lock failures")
        if hr.get('registered') and not copy_gbs:
            flags.append(f"{hr['registered']:.0f} slots page-locked, no copy measured")
        print(f"   {os.path.basename(inst)}: decode plan {dp.get('gpu_per_token')} GPU experts/token (refused "
              f"{dp.get('refused')}), error p50 {dp.get('error_p50_pct')}% p90 {dp.get('error_p90_pct')}%, page-locked "
              f"{hr.get('registered')} slots, copy {round(copy_gbs, 1) if copy_gbs else None} GB/s, CPU beside host jobs "
              f"x{dp.get('cpu_moved')}, admission min_count {adm.get('min_count')} upload {adm.get('upload_us')} us"
              + (' FLAG ' + '; '.join(flags) if flags else ''))
        out += [f'{os.path.basename(conv)} {os.path.basename(inst)} {f}' for f in flags]
    return out


try:
    PINNED_GBS = json.load(open(f'{d}/probe/device.json'))['gpu'].get('h2d_pinned_gbs')
except (OSError, ValueError, KeyError):
    PINNED_GBS = None
print('== decode placement')
place_flags = []
for c in sorted(glob.glob(f'{d}/conv-*')):
    place_flags += decode_place(c, PINNED_GBS)
print()
for k, v in sorted(res.items(), key=lambda kv: kv[1]['wall_s']):
    ts = [t for t in v['turns'] if t['decode_tps']]
    pf = sum(t['prefill_s'] for t in v['turns'])
    rate = sum(t['decode_tps'] for t in ts) / len(ts) if ts else 0.0
    print(f"{k:13s} wall {v['wall_s']:7.1f} s   prefill + D/rate {v['turn_model_s']:7.1f} s (prefill {pf:.1f} s, "
          f"decode {rate:.2f} tok/s)" + (f"   static plan predicted {v['pred_s']:.1f} s" if 'pred_s' in v else ''))
flags = [f'{k} {x}' for k, v in sorted(res.items()) for x in v.get('route_flags', [])]
print(f"route check: {len(flags)} prefills more than {FLAG:g} % off" + (': ' + '; '.join(flags) if flags else ''))
print(f"decode placement check: {len(place_flags)} flags" + (': ' + '; '.join(place_flags) if place_flags else ''))
c = plan['plan']
print(f"plan decision: tier {c['tier']} cpus {c['cpus']} env {c['env'] or '(engine defaults)'}")
json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'events'} for k, v in res.items()},
          open(f'{d}/compare-online.json', 'w'), indent=1)
