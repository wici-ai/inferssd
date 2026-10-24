#!/usr/bin/env python3
"""The plan (adaptive.sh step 4): the configuration with the least time for a conversation, before anything is
served.  The workload is turns of N new prompt tokens and D decode tokens each on a freshly started server; every
configuration of the machine model (machine.py) is priced over it with the bootstrap's own routing.  The engine's
runtime rules (admission, its decode thread A/B when not pinned, the elastic tier) stay on.  Each configuration's
prefill runs with the settings settings.py derives for its placement and the workload (the prefill threshold and
ubatch, and the context where the workload is known: --known, the conversation the servers will serve rather than
a forecast), except those the model's conf holds (--pending, its PENDING).
plan.json: the choice ('plan'), the best configuration with another CPU placement ('other'), every configuration
('table'), in-sample checks of the model against what each bootstrap measured ('calibration'), and the derived
settings of the choice ('derived', with the pending names, which settings.py and the loop take from here), and the
configurations set aside ('rejected': priced at a time or a rate at or below zero, or on a placement whose layer
table prices a class so; machine.row_faults, layer_faults), which the plan never chooses.
usage: plan.py RUN_DIR --prompts 3631,3632 --decode 512 --window 129-512 [--known] [--pending 'NAME ...'] -o plan.json"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from machine import Machine, layer_faults, row_faults  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('run')
ap.add_argument('--prompts', required=True, help='new prompt tokens of each turn, comma separated')
ap.add_argument('--decode', type=int, default=512)
ap.add_argument('--window', default='129-512')
ap.add_argument('--known', action='store_true', help='the workload is the conversation to be served, not a forecast')
ap.add_argument('--pending', default='', help="settings derived but held for this model (its conf's PENDING)")
ap.add_argument('-o', '--out', required=True)
a = ap.parse_args()
t_start = time.time()
w0, w1 = (int(x) for x in a.window.split('-'))
prompts = [int(x) for x in a.prompts.split(',')]
m = Machine(a.run, pending=a.pending.split(), workload={'prompts': prompts, 'decode': a.decode, 'known': a.known})

table = []
for tier, p in m.configs:
    turns = m.predict(m.routing, tier, p, 0, prompts, a.decode, w0, w1, cold=True, key='bootstrap')
    table.append({'tier': tier, 'cpus': p, 'total_s': round(sum(t['prefill_s'] + a.decode / t['decode_tps'] for t in turns), 2),
                  'prefill_s': round(sum(t['prefill_s'] for t in turns), 2),
                  'decode_tps': round(sum(t['decode_tps'] for t in turns) / len(turns), 3), 'hit_v': round(turns[0]['hit_v'], 4),
                  'turns': [{'prefill_s': round(t['prefill_s'], 2), 'decode_tps': round(t['decode_tps'], 3)} for t in turns],
                  'env': m.env(tier, p)})
# a configuration priced at a time or a rate at or below zero, or on a placement whose layer table prices a class at
# or below zero, has an input gone wrong upstream: it is set aside with its reasons, and the plan chooses among the
# rest; with none left there is no plan
k = m.ins['topk']
bad_layers = {p: layer_faults(m.P[p], k) for p in m.P}
rejected = []
for r in table:
    why = row_faults(r) + ([f"layer classes priced at or below zero: {bad_layers[r['cpus']][:4]}"
                            f"{' ...' if len(bad_layers[r['cpus']]) > 4 else ''}"] if bad_layers[r['cpus']] else [])
    if why:
        rejected.append({'tier': r['tier'], 'cpus': r['cpus'], 'total_s': r['total_s'], 'why': why})
table = [r for r in table if not any(x['tier'] == r['tier'] and x['cpus'] == r['cpus'] for x in rejected)]
for x in rejected:
    print(f"rejected {x['cpus']} tier {x['tier']} (total {x['total_s']} s): {'; '.join(x['why'])}")
if not table:
    sys.exit('plan: every configuration was rejected (above); no plan')
table.sort(key=lambda r: r['total_s'])
best = table[0]
other = next((r for r in table if r['cpus'].split('@')[0] != best['cpus'].split('@')[0]), None)

# in-sample: each placement at its bootstrap's tier, decode at that run's mean position, against what it measured
calib = {}
for p, mp in m.P.items():
    sink, _, hv = m.sink(m.routing, min(m.budget, m.caps[p]), p, key='bootstrap')
    calib[p] = {'decode_tps_model': round(1000.0 / (sink + m.t_out(p, mp['t_out_fit']['mean_position'])), 3),
                'decode_tps_measured': mp['measured']['decode_tps_window'], 'hit_v_model': round(hv, 4),
                'hit_v_measured': round(mp['measured']['hits']['v'], 4)}

out = {'reference': m.ref, 'budget_tier_slots': m.budget, 'admission_min_count': m.mc,
       't_out_slope_ms_per_ktok': round(m.slope * 1e3, 4), 'prefill_model': m.PF,
       'workload': {'prompts': prompts, 'decode': a.decode, 'window': a.window, 'known': a.known},
       'plan': best, 'other': other, 'table': table, 'rejected': rejected, 'calibration': calib,
       'derived': {'pending': m.pending, 'plan': [s._asdict() for s in m.derived.get(best['cpus'], [])]},
       'plan_seconds': round(time.time() - t_start, 1)}
json.dump(out, open(a.out, 'w'), indent=1)
print(json.dumps({k: out[k] for k in ('reference', 'budget_tier_slots', 'admission_min_count', 'workload', 'calibration',
                                       'derived', 'plan_seconds')}, indent=1))
for label, r in (('plan', best), ('other', other)):
    if r:
        print(f"{label:6s} {r['cpus']} tier {r['tier']}: total {r['total_s']:.1f} s, prefill {r['prefill_s']:.1f} s, "
              f"decode {r['decode_tps']:.2f} tok/s, hit_v {r['hit_v']}; env: {r['env'] or '(engine defaults)'}")
print('all configurations:')
for r in table:
    print('  ', {k: v for k, v in r.items() if k not in ('turns', 'env')})
