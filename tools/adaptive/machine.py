"""The model of one machine, built from its bootstrap (adaptive.sh steps 1-3: probe/device.json, insitu.json,
model.json): the time of the turns of a conversation in any configuration, a GPU-tier size times a CPU placement
(optionally with a larger prefill ubatch, 'all@16384').  plan.py asks it before anything is served, online.py again
at every request with what the served turns showed.

  decode   a token is the MoE sink's time (the routing replayed through the tier, sim.py, priced with the placement's
           per-layer table) plus the time outside the sink (a level per placement and a slope against the context
           position).  The sink's time splits into its CPU compute part and the rest (waiting on the drive, the GPU).
  prefill  pass by pass on the route the prefill threshold gives it, the CPU sink (the larger of its reads and the
           placement's CPU compute, plus its time a token besides) or the ggml graph on the GPU (its reads and its
           time beyond them, from the bootstrap's graph prefills) (prefill.py); the threshold and the ubatch are the ones settings.py derives for the placement, unless
           the model's conf holds them (PENDING).
  feedback scales the parts (online.feedback): the CPU compute and the outside level per placement; the rest of the
           sink and the time of prefill's reads by the drive's state, which every placement shares; the bytes a
           prefill reads.  NEUTRAL is none: the bootstrap's model as it is.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prefill  # noqa: E402
import settings  # noqa: E402
from sim import Replay, price  # noqa: E402

NEUTRAL = {'k_cpu': {}, 'out0': {}, 'k_drive': 1.0, 'k_bytes': 1.0, 'pf_cpu': {}, 'pf_drive': 1.0}


def layer_faults(pl, k):
    """The layer classes placement pl (insitu.json's placements[p]) prices at or below zero time: a layer of top-k
    experts, nv in the tier, nr host-ready and nm read (nv + nr + nm = k), at the class's measured mean when the
    bootstrap saw it often enough, else at the linear fit of the rare classes (sim.price).  Every layer takes time,
    so such a price is a measurement or a fit gone wrong (a singular fit priced (0, 0, 6) at -6.2 ms on the Spark
    ysy9), and the placement's decode prices cannot be used.  [(nv, nr, nm, ms)]"""
    table, beta = pl['layer_table'], pl['layer_beta']
    out = []
    for nv in range(k + 1):
        for nr in range(k + 1 - nv):
            nm = k - nv - nr
            e = table.get(f'{nv},{nr},{nm}')
            t = e[1] if e and e[0] >= 5 else beta[0] + beta[1] * nv + beta[2] * nr + beta[3] * nm + beta[4] * (nm > 0)
            if t <= 0:
                out.append((nv, nr, nm, round(t, 3)))
    return out


def row_faults(row):
    """Why a configuration the plan priced cannot be right: a total, a prefill or a decode rate at or below zero, in
    all or in a turn (an input gone wrong upstream: the one-block ubatch probe and the singular layer fit on the
    Spark ysy9 priced a 16-slot tier at -4.84 tok/s and -373 s).  [] when there is none."""
    out = [f"{k} {row[k]}" for k in ('total_s', 'decode_tps') if row[k] <= 0]
    out += [f"prefill_s {row['prefill_s']}"] if row['prefill_s'] < 0 else []
    for i, t in enumerate(row.get('turns') or []):
        out += [f"turn {i} {k} {t[k]}" for k in ('decode_tps',) if t[k] <= 0]
        out += [f"turn {i} prefill_s {t['prefill_s']}"] if t['prefill_s'] < 0 else []
    return out


class Machine:
    def __init__(self, run, pending=None, workload=None):
        """pending: the settings the model's conf holds (PENDING); workload: the turns' new prompt tokens and replies
        ({'prompts': [...], 'decode': D}).  None: those plan.py was given (plan.json), and nothing derived for a plan
        that predates derived settings."""
        self.ins = json.load(open(f'{run}/insitu.json'))
        self.dev = json.load(open(f'{run}/probe/device.json'))
        self.routing = json.load(open(f'{run}/model.json'))['routing']
        self.P, self.ref = self.ins['placements'], self.ins['reference']
        ref = self.P[self.ref]
        self.budget = int(ref['tier_slots'])
        self.mc = ref['admission_min_count'] if ref['admission_min_count'] is not None else 1.0
        # time outside the sink against the context: the reference bootstrap's slope when its tokens span enough
        # positions to tell (a few hundred do not), else flat until a conversation spans enough (online.fit_slope)
        f = ref['t_out_fit']
        self.slope = max(0.0, f['b_ms_per_ktok'] / 1e3) if f['positions'][1] - f['positions'][0] >= 2000 else 0.0
        self.PF = prefill.build(self.ins, self.dev, self.routing)
        # configurations: the tier at quarters of the budget, each capped by what the placement leaves (a larger
        # prefill ubatch leaves less GPU memory to the tier)
        self.caps = {p: int(m['tier_slots']) for p, m in self.P.items()}
        tiers = {max(1, int(self.budget * f)) for f in (0.25, 0.5, 0.75, 1.0)}
        self.configs = [(t, p) for p in self.P for t in sorted({min(t, self.caps[p]) for t in tiers})]
        self._replays, self._sinks = {}, {}
        if pending is None and os.path.exists(f'{run}/plan.json'):
            plan = json.load(open(f'{run}/plan.json'))
            pending = (plan.get('derived') or {}).get('pending')
            workload = workload or plan.get('workload')
        self.pending, self.workload = pending, workload
        self.derived = {p: settings.derive(self, p, pending, workload) for p in self.P} if pending is not None else {}

    def _derived(self, p, name):
        """The value of setting `name` the server of placement p runs with (derived and not pending), else None."""
        s = next((s for s in self.derived.get(p, ()) if s.name == name and not s.pending), None)
        return s.value if s else None

    def route(self, p):
        """The prefill threshold of placement p (prefill.seconds); None: every pass on the sink."""
        t = self._derived(p, 'GGML_KIMI_CPU_PREFILL_MAX_TOKENS')
        return None if t is None or t == settings.EVERY_PASS else t

    def ubatch(self, p):
        """The prefill ubatch of placement p when settings.py derives it; None: the placement's."""
        return self._derived(p, 'UBATCH')

    def env(self, tier, p):
        """The launcher environment of a configuration, as settings.py writes it."""
        return settings.env(settings.configuration(self, tier, p) + self.derived.get(p, []))

    def t_out(self, p, x, fb=NEUTRAL):
        """ms outside the sink per decode token at context position x."""
        if p in fb['out0']:
            return fb['out0'][p] + self.slope * x
        f = self.P[p]['t_out_fit']
        return f['mean_ms'] + self.slope * (x - f['mean_position'])

    def price(self, rows, p):
        """(sink ms, its CPU compute part) per token of replayed rows on placement p."""
        r = price(rows, dict(self.P[p], pf_k=self.P[p].get('pf_k', self.ins['pf_k']), t_out_ms=0.0), 1.0)
        return r['sink_ms'], r['cpu_ms']

    def replay(self, key, segs, tier):
        """The routing segments replayed through a tier of `tier` slots, carried on from the last call under `key`
        when segs extends what that call replayed (the turns of a conversation), else from the start."""
        r = self._replays.get(key)
        try:
            if r is None or r.tier.cap != tier:
                raise ValueError
            r.advance(segs)
        except ValueError:
            r = self._replays[key] = Replay(tier, self.mc, self.ins['skip']).advance(segs)
        return r

    def sink(self, segs, tier, p, key='conversation'):
        """(sink ms, its CPU part, tier hit share) per decode token with this routing on (tier, p); priced once per
        routing, whatever the turns asked about."""
        r = self.replay((key, tier), segs, tier)
        k = (key, tier, p, len(r.rows))
        if k not in self._sinks:
            self._sinks[k] = self.price(r.rows, p) + (r.stats()['hit_v'],)
        return self._sinks[k]

    def predict(self, segs, tier, p, ctx, prompts, decode, w0, w1, cold, fb=NEUTRAL, key='conversation'):
        """Turns of `prompts` new tokens and `decode` replies each, from context position ctx, on (tier, p): per turn
        the prefill seconds and the decode rate of tokens w0..w1.  cold: the first turn meets an empty tier."""
        sink, cpu, hv = self.sink(segs, tier, p, key)
        ms = fb['k_cpu'].get(p, 1.0) * cpu + fb['k_drive'] * (sink - cpu)
        x, out = ctx, []
        for j, n in enumerate(prompts):
            pf = prefill.seconds(self.PF, p, n, tier, cold and j == 0, fb['k_bytes'], fb['pf_cpu'].get(p, 1.0),
                                 fb['pf_drive'], self.route(p), self.ubatch(p))[0]
            x += n
            out.append({'prefill_s': pf, 'decode_tps': 1000.0 / (ms + self.t_out(p, x + (w0 + w1) / 2, fb)),
                        'hit_v': hv})
            x += decode
        return out
