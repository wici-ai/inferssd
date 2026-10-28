#!/usr/bin/env python3
"""settings.py's rules on small machine models (no GPU, no server): what a configuration writes, the prefill model
and its threshold (prefill.py), the receipt format run-ssd-llama.sh reads; the decode model's fit of the per-layer
table (sim.lstsq) and the tier the ubatch probe reads (ubprobe.py).
usage: python3 framework/tests/compat/test_settings.py   (unittest; exit 1 on a failure)"""
import math
import os
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../fixtures/adaptive'))
import logparse  # noqa: E402
import machine as machine_mod  # noqa: E402
import placement  # noqa: E402
import prefill  # noqa: E402
import settings  # noqa: E402
import ubprobe  # noqa: E402
from sim import Replay, lstsq  # noqa: E402

# two CPU groups, so that 'split' and 'pin' exist
DEV = {'coresets': {'g1': {'cpus': [0, 1, 2, 3]}, 'g2': {'cpus': [4, 5, 6, 7]}}, 'default_threads': 8}
LINE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)=(\S*)  # source=([a-z-]+); input=(\S.*)$')


def machine(**kw):
    m = SimpleNamespace(caps={'all': 336, 'split': 330, 'all@16384': 300}, ref='all', dev=DEV,
                        P={'all': {'tier_slot_bytes': 18800640, 'tier_base_bytes': 21e9}})
    m.__dict__.update(kw)
    return m


class Configuration(unittest.TestCase):
    def test_whole_tier_default_placement_writes_nothing(self):
        self.assertEqual(settings.configuration(machine(), 336, 'all'), [])

    def test_smaller_tier_caps_and_lowers_the_base(self):
        s = {x.name: x for x in settings.configuration(machine(), 84, 'all')}
        self.assertEqual(s['GGML_KIMI_CPU_SINK_VRAM_ELASTIC_MAX_SLOTS'].value, 84)
        self.assertEqual(s['GGML_KIMI_CPU_SINK_VRAM_BYTES'].value, 84 * 18800640)
        self.assertTrue(all(x.source == 'decided' for x in s.values()))

    def test_base_kept_when_the_tier_is_above_it(self):
        s = settings.configuration(machine(P={'all': {'tier_slot_bytes': 18800640, 'tier_base_bytes': 1e9}}), 84, 'all')
        self.assertEqual([x.name for x in s], ['GGML_KIMI_CPU_SINK_VRAM_ELASTIC_MAX_SLOTS'])

    def test_unified_memory_caps_only(self):
        # the tier starts from one block there (derive), whatever base the bootstrap ran with
        s = settings.configuration(machine(dev={**DEV, 'gpu': {'integrated': True}}), 84, 'all')
        self.assertEqual([(x.name, x.value) for x in s], [('GGML_KIMI_CPU_SINK_VRAM_ELASTIC_MAX_SLOTS', 84)])

    def test_placement_and_ubatch(self):
        names = [x.name for x in settings.configuration(machine(), 330, 'split')]
        self.assertIn('GGML_KIMI_CPU_SINK_DECODE_PARTICIPANTS', names)
        s = settings.configuration(machine(), 300, 'all@16384')
        self.assertEqual([(x.name, x.value) for x in s], [('UBATCH', '16384')])

    def test_env_is_the_old_machine_env(self):
        # machine.py's environment before settings.py: the tier's knobs, then placement.env of the whole name
        m = machine()
        for tier, p in ((336, 'all'), (84, 'all'), (330, 'split'), (100, 'split'), (300, 'all@16384'), (10, 'all@16384')):
            e = []
            if tier < m.caps[p]:
                e.append(f'GGML_KIMI_CPU_SINK_VRAM_ELASTIC_MAX_SLOTS={tier}')
                if tier * 18800640 < 21e9:
                    e.append(f'GGML_KIMI_CPU_SINK_VRAM_BYTES={tier * 18800640}')
            pe = placement.env(DEV, p)
            self.assertEqual(settings.env(settings.configuration(m, tier, p)), ' '.join(e + ([pe] if pe else [])))


def model(c=150e-6, rate=10e9, union=(), E=256, k=8, L=62, B_e=30e6, memory=None, d=2e-3, cov=0.0, link=None,
          role=11534336):
    """A prefill model (prefill.build's fields) of one placement 'all' with the same cold and warm state; role the
    largest role slice of an expert; the memory by default the Spark's with Muse (the sink's buffers at load, a graph
    pass's reserve)."""
    st = {'n': 4000, 'c': c, 'r': k * L, 'd': d}
    mem = memory if memory is not None else {'free_mib': 119000, 'dense_mib': 20000, 'kv_mib': 2000,
                                              'compute_mib': 1664, 'n_ubatch': 8192,
                                              'reserves': [[8192, 1664.16, 1024.31, 0], [8192, 5760.16, 1024.31, 1]]}
    return {'E': E, 'k': k, 'L': L, 'B_e': B_e, 'R_d': rate, 'R_io': rate and min(rate, link or rate),
            'union': [list(x) for x in union], 'cov_warm': cov, 'tier_at_warm': 100,
            'pass': {'all': {'cold': st, 'warm': st}, 'all@16384': {'cold': st, 'warm': st}},
            'memory': {'all': mem, 'all@16384': mem}, 'ubatch': 8192, 'ubatch_p': {'all': 8192, 'all@16384': 16384},
            'role_mib': E * role / 2 ** 20}


class Union(unittest.TestCase):
    def test_windows_across_a_runs_requests(self):
        # one layer; token t selects expert t mod 4: n tokens touch min(n, 4) experts, across the two requests too
        segs = [[[[t % 4]] for t in range(6)], [[[t % 4]] for t in range(6, 12)]]
        self.assertEqual(prefill.union_points(segs), [[1, 1.0], [2, 2.0], [4, 4.0], [8, 4.0], [12, 4.0]])

    def test_coverage_through_the_points_and_under_uniform(self):
        m = model(union=[(1, 8), (100, 80), (1000, 160)])
        self.assertAlmostEqual(prefill.coverage(m, 100), 80)
        self.assertAlmostEqual(prefill.coverage(m, 316), 80 * (316 / 100) ** 0.30103, places=3)   # log-log
        self.assertAlmostEqual(prefill.coverage(m, 2000), 160 * 2 ** 0.30103, places=3)          # the last slope
        self.assertEqual(prefill.coverage(m, 0), 0.0)
        u = model()                                                                               # no points: uniform
        self.assertAlmostEqual(prefill.coverage(u, 669), 256 * (1 - (1 - 8 / 256) ** 669))
        self.assertLessEqual(prefill.coverage(model(union=[(1, 8), (2, 16), (4, 40)]), 4), 256 * (1 - (1 - 8 / 256) ** 4))


class Threshold(unittest.TestCase):
    def brute(self, m):
        def cost(n, g):
            return prefill.pass_seconds(m, 'all', n, g, False, 100)[0]
        return max(n for n in range(0, 20000) if cost(n, True) >= cost(n, False))

    def test_bisection_is_the_crossing_of_the_two_routes(self):
        for c, rate, cov, link in ((150e-6, 10e9, 0.0, None), (40e-6, 7e9, 0.3, 6.9e9), (500e-6, 12e9, 0.0, None)):
            m = model(c=c, rate=rate, cov=cov, link=link, union=[(1, 8), (64, 100), (512, 170), (4221, 222)])
            self.assertEqual(prefill.threshold(m, 'all'), self.brute(m))

    def test_closed_form_at_the_threshold(self):
        # past T the sink's compute n k c exceeds the graph's reads U(n) B_e / R_io, at T it does not
        m = model(union=[(1, 8), (64, 100), (512, 170), (4221, 222)])
        T = prefill.threshold(m, 'all')
        def gap(n):
            return n * 8 * 150e-6 - prefill.coverage(m, n) * 30e6 / 10e9
        self.assertLessEqual(gap(T), 0)
        self.assertGreater(gap(T + 1), 0)
        self.assertLess(T, 256 * 30e6 / (8 * 150e-6 * 10e9))                                    # U < E

    def test_graph_from_the_first_token_and_no_inputs(self):
        self.assertEqual(prefill.threshold(model(c=1.0), 'all'), 0)
        self.assertIsNone(prefill.threshold(model(c=0.0), 'all'))
        self.assertIsNone(prefill.threshold(model(rate=None), 'all'))

    def test_passes_take_their_routes(self):
        m = model(union=[(1, 8), (64, 100), (512, 170), (4221, 222)])
        whole = prefill.seconds(m, 'all', 1000, 100, False, threshold=300, ub=400)
        parts = [prefill.pass_seconds(m, 'all', n, g, False, 100) for n, g in ((400, True), (400, True), (200, False))]
        for i in range(3):
            self.assertAlmostEqual(whole[i], sum(p[i] for p in parts))
        self.assertEqual(prefill.seconds(m, 'all', 1000, 100, False, ub=400)[1],
                         sum(prefill.pass_seconds(m, 'all', n, False, False, 100)[1] for n in (400, 400, 200)))

    def test_drive_factor_inverts_the_reads(self):
        def time_of(f):
            return max(f * 30.0, 20.0) + 5.0
        self.assertAlmostEqual(prefill.drive_factor(time_of, 50.0), 1.5, places=6)
        self.assertIsNone(prefill.drive_factor(lambda f: 25.0, 30.0))                             # compute-bound
        self.assertIsNone(prefill.drive_factor(time_of, 4.0))


class GraphRoute(unittest.TestCase):
    """The graph route priced from the bootstrap's graph prefills: the Spark's two (Muse, 669 and 4221 tokens)."""
    GRAPH = {'passes': [{'n_tokens': 669, 'passes': 1, 'bytes': 379930084864, 'read_s': 32.083, 'prompt_s': 36.727},
                        {'n_tokens': 4221, 'passes': 1, 'bytes': 419034384896, 'read_s': 34.91, 'prompt_s': 49.641}],
             'memory': {'free_mib': 119273, 'dense_mib': 20106.78, 'kv_mib': 1984.0, 'compute_mib': 5760.16,
                        'host_compute_mib': 1024.31, 'n_ubatch': 8192}}

    def ins(self, graph=None, union=((1, 8.0), (64, 96.5), (256, 151.2), (511, 176.2))):
        pt = {'n_tokens': 4218, 'unique': 222.4 * 62, 'unique_per_layer': [222] * 62, 'tier': 0.0, 'routes': 2058392.0,
              'ssd_gb': 419.5, 'prompt_ms': 333537.0, 'cpu_s': 322.3}
        return {'placements': {'all': {'prefill': [pt], 'tier_slots': 2554, 'memory': {}}}, 'reference': 'all',
                'n_experts': 256, 'layers': 62, 'topk': 8, 'union': [list(x) for x in union], 'graph': graph}

    DEV = {'ssd': {'3': {'gibs': 11.72}}, 'ssd_knee_qd': 3, 'expert_read_bytes': 30420992,
           'bytes_per_role': {'down': 11534336, 'gate': 9437184, 'up': 9437184},
           'gpu': {'integrated': True, 'h2d_pinned_gbs': 58.7}}

    def test_the_rest_beyond_the_reads(self):
        def close(x, y):
            self.assertAlmostEqual(x[0], y[0], places=9)
            self.assertAlmostEqual(x[1], y[1], places=12)
        close(prefill.rest_fit([(1, 100, 3.0), (1, 1100, 5.0)]), (2.8, 0.002))
        close(prefill.rest_fit([(1, 669, 36.727 - 32.083), (1, 4221, 49.641 - 34.91)]),
              (4.644 - 669 * 10.087 / 3552, 10.087 / 3552))                   # 2.744 s a pass, 2.840 ms a token
        close(prefill.rest_fit([(1, 500, 2.0)]), (0.0, 0.004))                # one prefill: per token
        close(prefill.rest_fit([(1, 100, 3.0), (1, 1100, 1.0)]), (2.0, 0.0))   # d < 0: per pass
        close(prefill.rest_fit([(1, 100, 1.0), (1, 1100, 21.0)]), (0.0, 22.0 / 1200))   # a < 0: per token
        close(prefill.rest_fit([(1, 1000, 3.0), (4, 4000, 12.0)]), (0.0, 0.003))       # passes in proportion

    def test_the_model_takes_the_graphs_prefills(self):
        m = prefill.build(self.ins(self.GRAPH), self.DEV)
        g = m['graph']
        self.assertAlmostEqual(g['R'] / 1e9, (379.930084864 + 419.034384896) / (32.083 + 34.91), places=6)
        self.assertEqual(g['n'], [669, 4221])
        self.assertEqual(m['graph_memory'], self.GRAPH['memory'])
        # U: the graph's bytes at 669 and 4221, the sink's at 4218; the decode points below 669 scaled to it
        u669 = 379930084864 / (62 * 30420992)
        self.assertEqual(m['union_scale'][0], 669)
        dec = prefill.coverage({'E': 256, 'k': 8, 'union': [[1, 8.0], [64, 96.5], [256, 151.2], [511, 176.2]]}, 669)
        self.assertAlmostEqual(m['union_scale'][1], round(u669 / dec, 4))
        self.assertEqual([n for n, _ in m['union']], [1, 64, 256, 511, 669, 4218, 4221])
        self.assertAlmostEqual(dict(m['union'])[256], round(151.2 * m['union_scale'][1], 2))
        self.assertAlmostEqual(prefill.coverage(m, 669), round(u669, 2))
        # a graph pass is its reads at the copy's rate, a per pass and d per token
        b = 62 * prefill.coverage(m, 1000) * 30420992
        self.assertAlmostEqual(prefill.pass_seconds(m, 'all', 1000, True, False, 2554)[0],
                               b / g['R'] + g['a'] + 1000 * g['d'])
        self.assertAlmostEqual(prefill.pass_seconds(m, 'all', 4221, True, True, 0)[0], 49.64, delta=0.5)

    def test_the_sink_at_its_own_prefill(self):
        # Muse's last layer computes the output token only: 488 routes a prompt token, not k L = 496; the sink's
        # pass at the bootstrap's own prefill is that prefill
        m = prefill.build(self.ins(), self.DEV)
        self.assertAlmostEqual(m['pass']['all']['cold']['r'], 2058392 / 4218)
        self.assertAlmostEqual(prefill.pass_seconds(m, 'all', 4218, False, True, 0)[0], 333.537, places=6)
        self.assertAlmostEqual(prefill.pass_seconds(m, 'all', 4218, False, True, 0)[1], 322.3, places=6)

    def test_without_graph_prefills_the_old_model(self):
        m = prefill.build(self.ins(), self.DEV)
        self.assertIsNone(m['graph'])
        self.assertEqual(m['union'], [[1, 8.0], [64, 96.5], [256, 151.2], [511, 176.2], [4218, 222.4]])
        st = m['pass']['all']['warm']
        self.assertAlmostEqual(prefill.pass_seconds(m, 'all', 1000, True, False, 2554)[0],
                               62 * prefill.coverage(m, 1000) * 30420992 / m['R_io'] + 1000 * st['d'])

    def brute(self, m):
        def cost(n, g):
            return prefill.pass_seconds(m, 'all', n, g, False, m['tier_at_warm'])[0]
        return max(n for n in range(0, 20000) if cost(n, True) >= cost(n, False))

    def test_the_threshold_is_the_crossing(self):
        m = prefill.build(self.ins(self.GRAPH), self.DEV)
        T = prefill.threshold(m, 'all')
        self.assertEqual(T, self.brute(m))
        self.assertTrue(350 < T < 450)
        for a, d in ((0.0, 0.001), (10.0, 0.002), (2.7, 0.07)):
            g = dict(m, graph=dict(m['graph'], a=a, d=d))
            self.assertEqual(prefill.threshold(g, 'all'), self.brute(g))

    def test_a_graph_slower_a_token_never_wins(self):
        m = prefill.build(self.ins(self.GRAPH), self.DEV)
        st = m['pass']['all']['warm']
        slow = dict(m, graph=dict(m['graph'], d=8 * 62 * st['c'] + st['d']))
        self.assertEqual(prefill.threshold(slow, 'all'), math.inf)
        s = {x.name: x for x in settings.prefill_route(SimpleNamespace(PF=slow), 'all', 4864)}
        self.assertEqual(s['GGML_KIMI_CPU_PREFILL_MAX_TOKENS'].value, settings.EVERY_PASS)
        self.assertIn('the graph is never faster', s['GGML_KIMI_CPU_PREFILL_MAX_TOKENS'].input)

    def test_the_graphs_memory_and_the_hosts_on_unified_memory(self):
        # the graph's prefills hold one down tensor of a layer's experts (256 x 11534336 B = 2816 MiB) and a slope a
        # token from their reserve at load (8192 tokens, every pass the graph's); on unified memory the host's too
        m = prefill.build(self.ins(self.GRAPH), self.DEV)
        self.assertEqual(m['role_mib'], 2816.0)
        g = prefill.graph_line(m, 'all')
        self.assertEqual(g['n'], 8192)
        self.assertAlmostEqual(g['slope'], (5760.16 - 2816) / 8192)
        self.assertAlmostEqual(g['host'], 1024.31 / 8192)
        room = 119273 - 20106.78 - 1984.0 - 1024 - 2816
        self.assertEqual(prefill.graph_pass_max(m, 'all', 1 << 20), int(room / (g['slope'] + g['host'])))
        discrete = dict(m, integrated=False)
        self.assertEqual(prefill.graph_pass_max(discrete, 'all', 1 << 20), int(room / g['slope']))
        self.assertEqual(prefill.graph_pass_max(dict(m, graph_memory=None), 'all', 4096), 0)   # the bootstrap's: none

    def test_the_line_from_one_reserve_predicts_the_others(self):
        # the Spark's reserves (2026-10-21): Muse's graph at 8192 tokens predicts 669 and 4221, DeepSeek-V4.1-Flash's
        # (384 x 6266880 B role slices) at 8192 predicts 4104, each within 1%
        for role, reserves, others in (
                (11534336, [[1, 0.91, 0.13, 0], [8192, 5760.16, 1024.31, 1]], {669: 3056.43, 4221: 4333.0}),
                (6266880 * 384 / 256, [[1, 18.77, 0.13, 0], [8192, 5847.22, 585.95, 1]], {4104: 4087.82})):
            mem = {'free_mib': 119000, 'dense_mib': 20000, 'kv_mib': 2000, 'compute_mib': 5000, 'n_ubatch': 8192,
                   'reserves': reserves}
            m = dict(model(memory=mem, role=role), graph_memory=mem)
            g = prefill.graph_line(m, 'all')
            for n, mib in others.items():
                self.assertLess(abs(g['role'] + g['slope'] * n - mib) / mib, 0.01)
            self.assertAlmostEqual(g['host'] * 8192, reserves[-1][2])
        # the sink's reserves (DeepSeek's bootstrap on an RTX 5060 Ti, every pass the sink's) say nothing of the
        # graph's: no line, and no graph pass is admitted
        mem = {'free_mib': 15712, 'dense_mib': 8530.2, 'kv_mib': 970, 'compute_mib': 4640.22, 'n_ubatch': 8192,
               'reserves': [[1, 18.77, 0.13, 0], [3323, 1726.49, 237.72, 0], [8192, 4640.22, 585.95, 0]]}
        m = model(memory=mem, E=384, role=6266880)
        self.assertIsNone(prefill.graph_line(m, 'all'))
        self.assertEqual(prefill.graph_pass_max(m, 'all', 8192), 0)
        # a graph pass's reserve below the role copy did not hold one: not the graph's line either
        mem = dict(mem, reserves=[[1, 18.77, 0.13, 0], [64, 300.0, 2.0, 1]])
        self.assertIsNone(prefill.graph_line(model(memory=mem, E=384, role=6266880), 'all'))
        # the bootstrap's own graph passes (a threshold below its prefills) give the line: Muse on an RTX 5060 Ti
        mem = {'free_mib': 15712, 'dense_mib': 9906.78, 'kv_mib': 1116.0, 'compute_mib': 4290.83, 'n_ubatch': 4352,
               'reserves': [[1, 68.18, 0.10, 0], [669, 3042.71, 65.36, 1], [4352, 4290.83, 425.17, 1]]}
        g = prefill.graph_line(model(memory=mem), 'all')
        self.assertEqual(g['n'], 4352)
        self.assertLess(abs(g['role'] + g['slope'] * 669 - 3042.71) / 3042.71, 0.01)

    def test_the_receipt_says_what_priced_the_graph(self):
        m = prefill.build(self.ins(self.GRAPH), self.DEV)
        (t, u) = settings.prefill_route(SimpleNamespace(PF=m), 'all', 4864)
        self.assertIn("the graph's copy at 11.93 GB/s, 2.74 s a pass and 2.84 ms a token besides (its prefills of "
                      "669, 4221 tokens)", t.input)
        self.assertIn('prompt tokens: the decode trace times', t.input)
        self.assertIn("KV 1984.0; the graph's compute 2816.00 + 0.3594 a token + 0.1250 a token on the host (its largest "
                      "role copy, the slope from its reserve at 8192 tokens) + 1 GiB", u.input)
        old = prefill.build(self.ins(), dict(self.DEV, gpu={'integrated': False}))
        (t, _) = settings.prefill_route(SimpleNamespace(PF=dict(old, memory={'all': self.GRAPH['memory']})), 'all', 4864)
        self.assertIn("provisional, the graph's prefill not measured", t.input)


class TierStates(unittest.TestCase):
    """The threshold for a workload whose turns meet two tier states: DeepSeek-V4.1-Flash-1017 on the Spark, whose
    sink computes the tier's experts on the GPU and does not read them.  The bootstrap's two sink prefills (cold 4104
    tokens in 40.0 s; warm 4616 tokens in 33.8 s, 5418 of 13193 experts in the tier) and the graph's (523 tokens in
    17.2 s, 4104 in 31.1 s): the warm sink wins 4104 tokens narrowly, the cold one loses it by 9 s."""
    INS = {'placements': {'all': {'tier_slots': 5637.0, 'memory': {}, 'prefill': [
        {'n_tokens': 4104, 'unique': 12951.0, 'unique_per_layer': [324] * 40, 'tier': 0.0, 'routes': 984960.0,
         'ssd_gb': 243.65, 'prompt_ms': 40037.718, 'cpu_s': 18.367},
        {'n_tokens': 4616, 'unique': 13193.0, 'unique_per_layer': [330] * 40, 'tier': 5418.0, 'routes': 1107840.0,
         'ssd_gb': 146.2, 'prompt_ms': 33846.812, 'cpu_s': 20.856}]}},
        'reference': 'all', 'n_experts': 384, 'layers': 40, 'topk': 6, 'prefill_ubatch': 8192,
        'union': [[1, 6.0], [32, 73.14], [128, 141.55], [256, 177.93], [511, 221.3]],
        'graph': {'passes': [{'n_tokens': 523, 'passes': 1, 'bytes': 183346181760, 'read_s': 14.6, 'prompt_s': 17.198},
                             {'n_tokens': 4104, 'passes': 1, 'bytes': 243057776640, 'read_s': 19.395,
                              'prompt_s': 31.119}],
                  'memory': {'free_mib': 119446, 'dense_mib': 8530.2, 'kv_mib': 970.0, 'compute_mib': 5847.22,
                             'host_compute_mib': 585.95, 'n_ubatch': 8192}}}
    DEV = {'ssd': {'6': {'gibs': 11.181}}, 'ssd_knee_qd': 6, 'expert_read_bytes': 18812928,
           'bytes_per_role': {'down': 6266880, 'gate': 6266880, 'up': 6266880},
           'gpu': {'integrated': True, 'h2d_pinned_gbs': 58.78}}

    def model(self):
        return prefill.build(self.INS, self.DEV)

    def gap(self, m, n, cold):
        """The sink's time less the graph's for a pass of n tokens, weighed over the states."""
        return sum(w * (prefill.pass_seconds(m, 'all', n, False, first, 5637)[0] -
                        prefill.pass_seconds(m, 'all', n, True, first, 5637)[0])
                   for w, first in ((cold, True), (1 - cold, False)))

    def brute(self, m, cold):
        return max(n for n in range(0, 12000) if self.gap(m, n, cold) <= 0)

    def test_the_states_route_4104_tokens_apart(self):
        m = self.model()
        self.assertAlmostEqual(prefill.pass_seconds(m, 'all', 4104, False, True, 5637)[0], 40.0, delta=0.1)
        self.assertAlmostEqual(prefill.pass_seconds(m, 'all', 4104, False, False, 5637)[0], 30.1, delta=0.1)
        self.assertAlmostEqual(prefill.pass_seconds(m, 'all', 4104, True, True, 5637)[0], 31.1, delta=0.1)
        warm, cold = prefill.break_even(m, 'all'), prefill.break_even(m, 'all', 1.0)
        self.assertEqual(warm, prefill.threshold(m, 'all'))           # no workload: a serving server's warm tier
        self.assertEqual((warm, cold), (self.brute(m, 0.0), self.brute(m, 1.0)))
        for cold_share in (1 / 2, 1 / 4, 1 / 8):
            self.assertEqual(prefill.break_even(m, 'all', cold_share), self.brute(m, cold_share))
        self.assertGreater(warm, 4104)                      # a serving server's warm tier: the sink
        self.assertLess(cold, 523)                          # a fresh server's first turn: the graph, 523 tokens too

    def time(self, m, turns, T):
        return sum(prefill.seconds(m, 'all', n, 5637, i == 0, threshold=T)[0] for i, n in enumerate(turns))

    def test_the_threshold_is_the_workloads_least_time(self):
        # any workload: no threshold gives it less time (its first turn on a cold tier, the others on the warm one),
        # and of those that give it as little, the threshold is the one nearest the break-even over the states its
        # passes the GPU can take meet
        m = self.model()
        for turns in ([4104], [4104] * 4, [4104, 28, 22, 23], [523, 28, 22, 23], [4104, 3800, 3800, 3800],
                      [1500, 2000, 1400, 3000], [28, 22, 23], [9000, 300, 2500]):
            T = prefill.threshold(m, 'all', turns)
            ps = [(q, i == 0) for i, n in enumerate(turns) for q in prefill.cut(m, 'all', n) if q >= 32]
            t = prefill.break_even(m, 'all', sum(f for _, f in ps) / len(ps) if ps else 0.0)
            times = {x: self.time(m, turns, x) for x in range(0, 9100)}
            best = min(times.values())
            self.assertEqual(self.time(m, turns, T), best, turns)
            self.assertEqual(T, min((x for x in times if times[x] == best), key=lambda x: (abs(x - t), -x)), turns)

    def test_a_long_first_prompt_and_short_follow_ups(self):
        # the protocol's conversation: the first prompt on a cold tier and three follow-ups the GPU does not take,
        # so the cold break-even and the graph for the first prompt of 523 tokens and more; turns of one size weigh
        # the states 1 : turns - 1
        m = self.model()
        cold = prefill.break_even(m, 'all', 1.0)
        for n0 in (523, 1000, 2000, 3000, 3700, 4104):
            self.assertEqual(prefill.threshold(m, 'all', [n0, 28, 22, 23]), cold)
            self.assertLess(self.time(m, [n0, 28, 22, 23], 0), self.time(m, [n0, 28, 22, 23], None))
        self.assertEqual(prefill.threshold(m, 'all', [4104] * 4), prefill.break_even(m, 'all', 0.25))
        self.assertLess(prefill.threshold(m, 'all', [4104] * 4), 4104)     # the plan's learned workload: 4 such turns
        self.assertEqual(prefill.threshold(m, 'all', [28, 22, 23]), prefill.break_even(m, 'all'))

    def test_a_follow_up_may_prefill_its_reply_again(self):
        # with replies of 512 tokens each follow-up is also priced as that reply prefilled again with it (the cache
        # parts inside a reply whose text gives back other tokens), a warm pass of 540, 534 and 535 tokens: the warm
        # sink takes it in 10.6 s, the graph in 17.3 s.  The two forms weigh 1/2 each, so the states weigh
        # 1 : 1.5 and T leaves the reply passes on the sink and the 4104 prompt on the graph
        m = self.model()
        self.assertEqual(prefill.served([4104, 28], 512), [(4104, True, 1.0), (28, False, 0.5), (540, False, 0.5)])
        self.assertEqual(prefill.served([4104, 28]), [(4104, True, 1.0), (28, False, 1.0)])
        conv = [4104, 28, 22, 23]
        T = prefill.threshold(m, 'all', conv, reply=512)
        self.assertEqual(T, prefill.break_even(m, 'all', 0.4))
        self.assertTrue(540 <= T < 4104)
        self.assertLess(prefill.pass_seconds(m, 'all', 540, False, False, 5637)[0] + 6,
                        prefill.pass_seconds(m, 'all', 540, True, False, 5637)[0])

        def served_time(x):
            return sum(w * prefill.seconds(m, 'all', n, 5637, f, threshold=x)[0]
                       for n, f, w in prefill.served(conv, 512))
        times = {x: served_time(x) for x in range(0, 6000, 7)}
        self.assertAlmostEqual(served_time(T), min(times.values()), places=9)
        # the threshold without replies (313) sends the reply passes to the graph: 3 x 1/2 x 6.8 s more
        self.assertAlmostEqual(served_time(prefill.threshold(m, 'all', conv)) - served_time(T), 10.1, delta=0.2)
        # a short first prompt: the cold sink costs it 0.6 s, the reply passes on the graph would cost 10 s
        self.assertEqual(prefill.threshold(m, 'all', [523, 28, 22, 23], reply=512), T)

    def test_the_offload_minimum_keeps_short_passes_on_the_sink(self):
        m = self.model()
        self.assertEqual(prefill.seconds(m, 'all', 28, 5637, False, threshold=0),
                         prefill.seconds(m, 'all', 28, 5637, False, threshold=None))
        self.assertNotEqual(prefill.seconds(m, 'all', 32, 5637, False, threshold=0),
                            prefill.seconds(m, 'all', 32, 5637, False, threshold=None))

    def test_settings_weigh_the_workloads_turns(self):
        m = SimpleNamespace(PF=self.model(), P={'all': {}}, dev={'gpu': {'integrated': True}})
        conv = {x.name: x for x in settings.derive(m, 'all', (), {'prompts': [4104, 28, 22, 23], 'decode': 512,
                                                                   'known': True})}
        t = conv['GGML_KIMI_CPU_PREFILL_MAX_TOKENS']
        self.assertEqual(t.value, prefill.break_even(m.PF, 'all', 0.4))
        self.assertIn("the least time for the workload's turns of 4104,28,22,23 new tokens, the first on a cold tier "
                      'and the rest on the warm one, each follow-up also as the 512-token reply before it prefilled '
                      "again with it (540,534,535 tokens: the server's cache parts inside a reply whose text gives back "
                      'other tokens), the two forms weighed 1/2 each, and between their sizes the break-even over the '
                      "tier states their passes of at least 32 tokens meet (1 cold, 1.5 warm): the sink's compute "
                      '18.6 us a route, 240.0 routes a token and 4.81 ms a token besides (cold prefill of 4104 tokens), '
                      '18.8 us a route', t.input)
        # no replies: the follow-ups as counted, below the offload minimum, and the cold break-even
        bare = {x.name: x for x in settings.derive(m, 'all', (), {'prompts': [4104, 28, 22, 23], 'decode': 0,
                                                                   'known': True})}
        t = bare['GGML_KIMI_CPU_PREFILL_MAX_TOKENS']
        self.assertEqual(t.value, prefill.break_even(m.PF, 'all', 1.0))
        self.assertIn('meet (1 cold, 0 warm)', t.input)
        four = {x.name: x for x in settings.derive(m, 'all', (), {'prompts': [4104] * 4, 'decode': 512})}
        t = four['GGML_KIMI_CPU_PREFILL_MAX_TOKENS']
        self.assertEqual(t.value, prefill.break_even(m.PF, 'all', 0.25))
        self.assertIn("(1 cold, 3 warm): the sink's compute 18.6 us a route, 240.0 routes a token and 4.81 ms a token "
                      'besides (cold prefill of 4104 tokens), 18.8 us a route', t.input)
        uneven = {x.name: x for x in settings.derive(m, 'all', (), {'prompts': [4104, 3800, 3800, 3800],
                                                                     'decode': 512})}
        t = uneven['GGML_KIMI_CPU_PREFILL_MAX_TOKENS']
        self.assertEqual(t.value, 3800)                     # the warm follow-ups on the sink, the first on the graph
        self.assertIn(f"the break-even alone, {prefill.break_even(m.PF, 'all', 0.25)}, would not give the workload "
                      'its least time', t.input)
        none = {x.name: x for x in settings.derive(m, 'all')}
        self.assertEqual(none['GGML_KIMI_CPU_PREFILL_MAX_TOKENS'].value, prefill.break_even(m.PF, 'all'))
        self.assertIn('(warm prefill of 4616 tokens)', none['GGML_KIMI_CPU_PREFILL_MAX_TOKENS'].input)


class Route(unittest.TestCase):
    U = [(1, 8), (64, 100), (512, 170), (4221, 222)]

    def test_graph_pass_fits_by_memory(self):
        m = model()
        self.assertEqual(prefill.graph_pass_max(m, 'all', 4608), 4608)
        # DeepSeek on an RTX 5060 Ti: the sink's reserves only, the graph's compute not measured
        small = model(memory={'free_mib': 15712, 'dense_mib': 8530, 'kv_mib': 970, 'compute_mib': 4640, 'n_ubatch': 8192,
                              'reserves': [[8192, 4640.0, 585.95, 0]]})
        self.assertEqual(prefill.graph_pass_max(small, 'all', 8192), 0)
        # the role copy alone does not fit: 13000 - 8530 - 970 - 1024 < 2816
        small = model(memory={'free_mib': 13000, 'dense_mib': 8530, 'kv_mib': 970, 'compute_mib': 4640, 'n_ubatch': 8192,
                              'reserves': [[8192, 4640.0, 585.95, 1]]})
        self.assertEqual(prefill.graph_pass_max(small, 'all', 8192), 0)
        tight = model(memory={'free_mib': 40000, 'dense_mib': 20000, 'kv_mib': 0, 'compute_mib': 8192, 'n_ubatch': 8192,
                              'reserves': [[8192, 8192 + 2816.0, 0.0, 1]]})
        self.assertEqual(prefill.graph_pass_max(tight, 'all', 1 << 20), int(40000 - 20000 - 1024 - 2816))
        self.assertEqual(prefill.graph_pass_max(model(memory={}), 'all', 4096), 0)

    def test_threshold_and_ubatch_where_a_graph_pass_fits(self):
        m = SimpleNamespace(PF=model(union=self.U))
        s = {x.name: x for x in settings.prefill_route(m, 'all', 4608)}
        self.assertEqual(s['GGML_KIMI_CPU_PREFILL_MAX_TOKENS'].value, prefill.threshold(m.PF, 'all'))
        self.assertEqual(s['UBATCH'].value, 4608)
        self.assertTrue(all(x.source == 'derived' for x in s.values()))

    def test_every_pass_on_the_sink_where_none_fits_or_none_wins(self):
        small = {'free_mib': 15712, 'dense_mib': 8530, 'kv_mib': 970, 'compute_mib': 4640, 'n_ubatch': 8192}
        for pf, limit in ((model(union=self.U, memory=small), 8192), (model(union=self.U, c=1e-6), 8192),
                          (model(union=self.U, c=0.0), 8192)):
            s = {x.name: x for x in settings.prefill_route(SimpleNamespace(PF=pf), 'all', limit)}
            self.assertEqual(s['GGML_KIMI_CPU_PREFILL_MAX_TOKENS'].value, settings.EVERY_PASS)
            self.assertEqual(s['UBATCH'].value, limit)

    def test_a_configuration_with_its_own_ubatch_keeps_it(self):
        s = settings.prefill_route(SimpleNamespace(PF=model(union=self.U)), 'all@16384', 8192)
        self.assertEqual([x.name for x in s], ['GGML_KIMI_CPU_PREFILL_MAX_TOKENS'])

    def test_pending_is_a_comment_and_not_passed(self):
        m = SimpleNamespace(PF=model(union=self.U), P={'all': {}}, dev={})
        s = settings.derive(m, 'all', pending={'UBATCH'})
        self.assertEqual([x.pending for x in s], [False, True])
        self.assertNotIn('UBATCH=', settings.env(s))
        lines = settings.render(s).splitlines()
        self.assertRegex(lines[0], LINE)
        self.assertTrue(lines[1].startswith('# pending UBATCH=8192  # source=derived; input='))
        self.assertRegex(lines[1][len('# pending '):], LINE)


class Workload(unittest.TestCase):
    U = [(1, 8), (64, 100), (512, 170), (4221, 222)]
    BENCH = {'prompts': [4221], 'decode': 256, 'known': True}

    def machine(self, integrated=True, align=1, ledger=None, **kw):
        P = {'all': {'checkpoint_align': align, 'ledger': ledger}}
        return SimpleNamespace(PF=model(union=self.U, **kw), P=P, dev={'gpu': {'integrated': integrated}})

    def test_context_holds_the_conversation(self):
        self.assertEqual(settings.context(self.BENCH), 4608)                      # 4221 + 256, up to 256
        self.assertEqual(settings.context({'prompts': [3323] * 4, 'decode': 512}), 15360)
        self.assertEqual(settings.context({'prompts': [256], 'decode': 256}), 512)

    def test_bench_workload_on_unified_memory(self):
        s = {x.name: x for x in settings.derive(self.machine(), 'all', (), self.BENCH)}
        self.assertEqual(s['CONTEXT'].value, 4608)
        self.assertEqual(s['UBATCH'].value, 4608)                                 # the context, a graph pass fits
        self.assertEqual(s['GGML_KIMI_CPU_PREFILL_MAX_TOKENS'].value, prefill.threshold(model(union=self.U), 'all'))
        self.assertEqual(s['CTX_CHECKPOINTS'].value, 0)                           # one turn restores nothing
        # the tier's start and the margin's floor are the launcher's on unified memory; nothing was learned
        self.assertEqual(list(s), ['CONTEXT', 'GGML_KIMI_CPU_PREFILL_MAX_TOKENS', 'UBATCH', 'CTX_CHECKPOINTS'])

    def test_a_conversation_takes_one_checkpoint(self):
        conv = {'prompts': [4217, 28, 22, 23], 'decode': 512, 'known': True}
        s = {x.name: x for x in settings.derive(self.machine(), 'all', (), conv)}
        self.assertEqual((s['CTX_CHECKPOINTS'].value, s['CONTEXT'].value), (1, 6400))
        self.assertIn('at multiples of 1 tokens', s['CTX_CHECKPOINTS'].input)
        self.assertNotIn('CTX_CHECKPOINTS', [x.name for x in settings.derive(self.machine(align=None), 'all', (), conv)])

    def test_a_forecast_bounds_no_context(self):
        # the plan's forecast (WORKLOAD=learned) is no bound on a conversation: the launcher keeps its context
        s = {x.name: x for x in settings.derive(self.machine(), 'all', (), {'prompts': [4221] * 4, 'decode': 512})}
        self.assertNotIn('CONTEXT', s)
        self.assertEqual(s['UBATCH'].value, 8192)                                 # the ubatch the bootstrap ran
        self.assertIn('the ubatch a server of this machine started with, 8192', s['UBATCH'].input)
        self.assertEqual(s['CTX_CHECKPOINTS'].value, 1)

    def test_the_sink_ubatch_is_the_context_too(self):
        small = {'free_mib': 15712, 'dense_mib': 8530, 'kv_mib': 970, 'compute_mib': 4640, 'n_ubatch': 8192}
        s = {x.name: x.value for x in settings.derive(self.machine(memory=small), 'all', (), self.BENCH)}
        self.assertEqual((s['GGML_KIMI_CPU_PREFILL_MAX_TOKENS'], s['UBATCH']), (settings.EVERY_PASS, 4608))
        long = {x.name: x.value for x in settings.derive(self.machine(memory=small), 'all', (),
                                                         {'prompts': [20000], 'decode': 512, 'known': True})}
        self.assertEqual((long['CONTEXT'], long['UBATCH']), (20736, 8192))        # at most the bootstrap's ubatch

    def test_the_learned_margin_is_carried(self):
        GiB = 2 ** 30
        learned = {'unified': 1.0, 'margin': 6 * GiB + 1, 'margin_updates': 3.0}
        s = {x.name: x for x in settings.derive(self.machine(ledger=learned), 'all', (), self.BENCH)}
        self.assertEqual((s['GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB'].value,     # MiB, rounded up
                          s['GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB'].source), (6145, 'self-tuned'))
        given = {'unified': 1.0, 'margin': 16 * GiB, 'margin_updates': 0.0}      # a start value, nothing learned
        later = ('RUN/conv-static-plan/inst0', {'unified': 1.0, 'margin': 8 * GiB, 'margin_updates': 1.0})
        s = {x.name: x for x in settings.derive(self.machine(ledger=given), 'all', (), self.BENCH, [later])}
        self.assertEqual(s['GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB'].value, 8192)
        self.assertIn('conv-static-plan/inst0: 8192 MiB', s['GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB'].input)
        s = settings.derive(self.machine(ledger=given), 'all', (), self.BENCH)  # nothing learned: the launcher's
        self.assertNotIn('GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB', [x.name for x in s])
        learned_off_uma = {'unified': 0.0, 'margin': 6 * GiB, 'margin_updates': 3.0}
        s = settings.derive(self.machine(integrated=False, ledger=learned_off_uma), 'all', (), self.BENCH)
        self.assertNotIn('GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB', [x.name for x in s])

    def test_discrete_gpu_and_no_after_batch_checkpoints(self):
        names = [x.name for x in settings.derive(self.machine(integrated=False, align=None), 'all', (), self.BENCH)]
        self.assertEqual(names, ['CONTEXT', 'GGML_KIMI_CPU_PREFILL_MAX_TOKENS', 'UBATCH'])

    def test_deepseeks_conf_holds_every_derived_value(self):
        conf = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../fixtures/adaptive', 'models', 'dsv41.conf')).read()
        pending = next(x for x in conf.splitlines() if x.startswith('PENDING=')).split('=', 1)[1].split()
        s = settings.derive(self.machine(integrated=False), 'all', pending, self.BENCH)
        self.assertTrue(s and all(x.pending for x in s))
        self.assertEqual(settings.env(s), '')
        self.assertTrue(all(x.startswith('# pending ') for x in settings.render(s).splitlines()))
        # on unified memory, which DeepSeek's deployment does not use, it takes a margin a run learned
        learned = {'unified': 1.0, 'margin': 5 * 2 ** 30, 'margin_updates': 1.0}
        s = settings.derive(self.machine(ledger=learned), 'all', pending, self.BENCH)
        self.assertEqual(settings.env(s), 'GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB=5120')


class PrefetchLog(unittest.TestCase):
    def test_the_log_gives_the_depth_and_the_statistics(self):
        a = logparse.ArmParser()
        for line in ('[kimi_cpu_sink_prefetch] enabled topk=2 readers=3',
                     '[kimi_cpu_sink_cache] auto_budget=3802624000 unified=1 slot_bytes=30420992 cgroup_surplus=1',
                     '[kimi_cpu_sink_prefetch] reason=periodic topk=2 layers=8192 windows=8191 window_us_p50=3481 '
                     'window_us_mean=3565 reader_bytes=5 reader_busy_us=6 width=2 predicted_r=8191,8191 novel_r=1,2 '
                     'novel_routed_r=1,1 issued_r=1,2 used_r=0,1',
                     '[kimi_cpu_sink_prefetch] reason=destroy topk=2 layers=15555 windows=15555 window_us_p50=3406 '
                     'window_us_mean=3489 reader_bytes=7 reader_busy_us=8 width=2 predicted_r=15555,15555 '
                     'novel_r=4006,4379 novel_routed_r=3327,3311 issued_r=3000,2000 used_r=900,800'):
            a.feed(line + '\n')
        pf = a.prefetch
        self.assertEqual((pf['topk'], pf['readers'], pf['layers'], pf['window_us_p50']), (2, 3, 15555, 3406))
        self.assertEqual((pf['novel_r'], pf['used_r']), ([4006, 4379], [900, 800]))
        self.assertEqual(a.ram_slots, 125)
        # the report's tier skips and the reads' time past their windows (an engine that measures them)
        a.feed('[kimi_cpu_sink_prefetch] reason=destroy topk=2 layers=15555 skip_vram=40 vram_late=0 skip_vram_lost=3 '
               'windows=15555 window_us_p50=3406 window_us_mean=3489 reader_bytes=7 reader_busy_us=8 spill_us=1234 '
               'spills=56 width=2 predicted_r=15555,15555 novel_r=4006,4379 novel_routed_r=3327,3311 issued_r=3000,2000 '
               'used_r=900,800 started_r=2990,1900 spill_us_r=400,1500\n')
        pf = a.prefetch
        self.assertEqual((pf['skip_vram'], pf['skip_vram_lost'], pf['spill_us'], pf['spills']), (40, 3, 1234, 56))
        self.assertEqual((pf['started_r'], pf['spill_us_r']), ([2990, 1900], [400, 1500]))
        d = logparse.ArmParser()                                                  # a discrete GPU: the costed size
        for line in ('[kimi_cpu_sink_cache] auto_budget=4133711872 unified=0 slot_bytes=18812928 cgroup_surplus=1',
                     '[kimi_cpu_sink_cache] auto_capacity_bytes=4082405376 capacity_cost_bytes=4131987456',
                     '[kimi_cpu_sink_cache] auto_capacity_bytes=1 capacity_cost_bytes=1'):
            d.feed(line + '\n')
        self.assertEqual((d.ram_slots, d.prefetch), (217, None))

    def test_the_ram_caches_storage_at_load_and_decodes_first_touches(self):
        a = logparse.ArmParser()
        profile = ('[kimi_cpu_sink_profile] {"schema":1,"event":"layer","phase":"decode","epoch":1,"token":%d,'
                   '"layer":0,"total_us":5.0,"compute_wall_us":1.0,"wait_io_us":2.0,"other_wall_us":1.0,"workers":[]}')
        for line in ('[kimi_cpu_sink_cache] reason=load slots=1512 of=1512 slot_bytes=18812928 '
                     'committed_bytes=28444947456 us=812345 threads=24 pages=populated',
                     profile % 10, '[kimi_cpu_sink_layer] layer=0 calls=1 io_us=900 bytes=37625856 cache_new_slot_bytes=0',
                     profile % 11, '[kimi_cpu_sink_layer] layer=0 calls=2 io_us=700 bytes=18812928 '
                     'cache_new_slot_bytes=18812928',
                     '[kimi_cpu_sink_layer] layer=0 calls=3 io_us=7 bytes=1 cache_new_slot_bytes=99'):   # not decode's
            a.feed(line + '\n')
        r = a.result('/nonexistent', skip=0)
        self.assertEqual(r['cache_load'], {'slots': 1512.0, 'slot_bytes': 18812928.0, 'committed_bytes': 28444947456.0,
                                           'us': 812345.0})
        self.assertEqual(r['new_slot_bytes'], 18812928.0)
        self.assertEqual((r['new_slot_windows'], r['load_s']), ([18812928.0, 0.0], None))
        self.assertIsNone(logparse.ArmParser().result('/nonexistent')['cache_load'])
        # tokens 1-128 and after of each request, and the server's load time from arm.sh's meta.txt
        b = logparse.ArmParser()
        for t in range(200):
            b.feed(profile % (4104 + t) + '\n')
            b.feed(f'[kimi_cpu_sink_layer] layer=0 calls=1 io_us=1 bytes=1 cache_new_slot_bytes={1000 + t}\n')
        with tempfile.TemporaryDirectory() as d:
            open(f'{d}/meta.txt', 'w').write('label=x\nhealth_ok=1 load_s=41.5\n')
            r = b.result(d, skip=0)
        self.assertEqual(r['new_slot_windows'], [sum(range(1000, 1128)), sum(range(1128, 1200))])
        self.assertEqual(r['load_s'], 41.5)


class LoadLog(unittest.TestCase):
    # a server at the threshold 0 on the Spark (Muse, 2026-10-21): its lines at load and one graph-route prefill
    LINES = ('llama_model_load_from_file_impl: using device CUDA0 (NVIDIA GB10) (000f:01:00.0) - 119273 MiB free',
             'load_tensors:   CPU_Mapped model buffer size = 46219.71 MiB',
             'load_tensors:        CUDA0 model buffer size = 20106.78 MiB',
             'llama_context: n_ubatch      = 8192',
             'llama_context:  CUDA_Host  output buffer size =     0.77 MiB',
             'llama_kv_cache:      CUDA0 KV buffer size =   512.00 MiB',
             'llama_kv_cache:      CUDA0 KV buffer size =  1472.00 MiB',
             'sched_reserve:      CUDA0 compute buffer size =  5760.16 MiB',
             'sched_reserve:  CUDA_Host compute buffer size =  1024.31 MiB',
             'sched_reserve: graph nodes  = 5127 (with bs=8192), 4011 (with bs=1)',
             '[kimi_prefill_route] n_tokens=2 route=sink max_sink_tokens=0 offloads=0',
             'sched_reserve:      CUDA0 compute buffer size =  4333.00 MiB',
             'sched_reserve:  CUDA_Host compute buffer size =   527.79 MiB',
             '[kimi_cpu_prefill_graph] event=activate requested_tokens=4221 selected_ubatch=4221 configured_max=8192 '
             'granularity=1 transition_ms=675.227',
             '[kimi_prefill_route] n_tokens=4221 route=graph max_sink_tokens=0 offloads=1',
             '[moe_prefill_copy] tokens=4221 passes=1 expert_slices=41340 bytes=419034384896 read_us=34909866 '
             'read_gbs=12.00 fallbacks=0',
             '[kimi_cpu_prefill_graph] event=release selected_ubatch=4221 decode_graph_restored=1 transition_ms=168.752')

    def test_the_memory_at_load_and_the_graphs_copy(self):
        a = logparse.ArmParser()
        for line in self.LINES:
            a.feed(line + '\n')
        r = a.result('/nonexistent')
        self.assertEqual(r['memory'], {'device': 'CUDA0', 'free_mib': 119273, 'dense_mib': 20106.78, 'n_ubatch': 8192,
                                       'kv_mib': 1984.0, 'compute_mib': 5760.16, 'host_compute_mib': 1024.31})
        (g,) = r['pf_graphs']
        self.assertEqual((g['n_tokens'], g['ubatch'], g['graph_bytes'], g['graph_read_us'], g['graph_passes'],
                          g['graph_tokens']), (4221, 8192, 419034384896.0, 34909866.0, 1, 4221))

    def test_a_prefill_in_smaller_passes_is_its_own(self):
        a = logparse.ArmParser()
        for line in self.LINES[:13] + (
                '[kimi_cpu_prefill_graph] event=fallback requested_tokens=4104 target_ubatch=4104 selected_ubatch=1100 '
                'route=graph max_sink_tokens=0 transition_ms=900.000',
                '[moe_prefill_copy] tokens=4104 passes=4 expert_slices=1 bytes=1000 read_us=2000 read_gbs=0.50 '
                'fallbacks=0'):
            a.feed(line + '\n')
        (g,) = a.result('/nonexistent')['pf_graphs']
        self.assertEqual((g['n_tokens'], g['graph_passes'], g['graph_bytes']), (4104, 4, 1000.0))

    def test_each_requests_reserve(self):
        # a reserve is the graph route's (it copies the experts) when its tokens exceed the sink's threshold, which the
        # route lines give (the reserve at load comes before the first); the sink's and decode's are not; a reserve
        # seen again (every decode token's) is kept once
        a = logparse.ArmParser()
        for line in self.LINES[:1] + (
                '[prefill_reserve_outputs] n_tokens=8192 n_outputs=1',
                'ggml_backend_cuda_graph_reserve: device 0 memory pool needs 291.02 MiB for attn_wo_a-0 (MUL_MAT)',
                'sched_reserve:      CUDA0 compute buffer size =  5760.16 MiB',
                'sched_reserve:  CUDA_Host compute buffer size =  1024.31 MiB',
                'sched_reserve: graph splits = 187 (with bs=8192), 126 (with bs=1)',
                '[kimi_prefill_route] n_tokens=2 route=sink max_sink_tokens=4096 offloads=0',
                '[prefill_reserve_outputs] n_tokens=669 n_outputs=1',
                'sched_reserve:      CUDA0 compute buffer size =   135.90 MiB',
                'sched_reserve:  CUDA_Host compute buffer size =    83.65 MiB',
                'sched_reserve: graph splits = 126',
                '[kimi_prefill_route] n_tokens=669 route=sink max_sink_tokens=4096 offloads=1') + (
                '[prefill_reserve_outputs] n_tokens=1 n_outputs=1',
                'sched_reserve:      CUDA0 compute buffer size =     0.91 MiB',
                'sched_reserve:  CUDA_Host compute buffer size =     0.13 MiB',
                'sched_reserve: graph splits = 126') * 2:
            a.feed(line + '\n')
        self.assertEqual(a.result('/nonexistent')['memory']['reserves'],
                         [[1, 0.91, 0.13, 0], [669, 135.9, 83.65, 0], [8192, 5760.16, 1024.31, 1]])
        # every pass the sink's: a reserve with more splits at its tokens than at one (weights in RAM, copied from 32
        # tokens on) is still the sink's
        a = logparse.ArmParser()
        for line in self.LINES[:1] + (
                '[prefill_reserve_outputs] n_tokens=8192 n_outputs=1',
                'sched_reserve:      CUDA0 compute buffer size =  1664.16 MiB',
                'sched_reserve: graph splits = 190 (with bs=8192), 126 (with bs=1)',
                '[kimi_prefill_route] n_tokens=2 route=sink max_sink_tokens=all offloads=0'):
            a.feed(line + '\n')
        self.assertEqual(a.result('/nonexistent')['memory']['reserves'], [[8192, 1664.16, 0.0, 0]])

    def test_no_memory_lines_no_graph_pass(self):
        self.assertEqual(logparse.ArmParser().result('/nonexistent')['memory'], {})


class DecodeLog(unittest.TestCase):
    @staticmethod
    def token(position, us, layers=2):
        for layer in range(layers):
            yield ('[kimi_cpu_sink_trace] {"schema":1,"event":"layer","phase":"decode","epoch":1,"token":%d,'
                   '"layer":%d,"selected_experts":[1,2],"vram_hit":[false,true],"cache_hit":[false,false]}'
                   % (position, layer))
            yield ('[kimi_cpu_sink_profile] {"schema":1,"event":"layer","phase":"decode","epoch":1,"token":%d,'
                   '"layer":%d,"total_us":%.1f,"compute_wall_us":1.0,"wait_io_us":2.0,"other_wall_us":3.0,'
                   '"workers":[]}' % (position, layer, us))

    def test_a_request_that_decodes_positions_again_is_its_own(self):
        # the second request's prompt cut the cache back: it decodes 12-21 after the first's 10-19, and a third
        # decodes from 21, the second's last position
        a = logparse.ArmParser()
        for positions, us in ((range(10, 20), 100.0), (range(12, 22), 200.0), (range(21, 24), 300.0)):
            for p in positions:
                for line in self.token(p, us):
                    a.feed(line + '\n')
        r = a.result('/nonexistent')
        self.assertEqual([(g[0][1], len(g)) for g in r['groups']], [(10, 10), (12, 10), (21, 3)])
        self.assertEqual([r['prof'][g[0]][::4] for g in r['groups']], [[200.0, 2], [400.0, 2], [600.0, 2]])
        self.assertEqual(r['lprof'][(*r['groups'][1][0], 1)], (200.0, 1.0))


class Prefetch(unittest.TestCase):
    NU =[0.26, 0.28, 0.32, 0.35, 0.39, 0.42, 0.45, 0.48]
    P = [0.83, 0.76, 0.65, 0.54, 0.45, 0.36, 0.29, 0.23]

    def test_the_window_and_the_slots_bound_the_depth(self):
        # (k*, k_win, k_mem, k_pay); a read that starts inside the window costs nothing here (no spill)
        self.assertEqual(settings.depth(self.NU, self.P, 1.03, [0.01] * 8), (4, 4, 8, 4))    # 0.86 < 1.03 <= 1.21
        self.assertEqual(settings.depth(self.NU, self.P, 99.0, None), (8, 8, 8, 8))    # never filled: top-k
        self.assertEqual(settings.depth(self.NU, self.P, 99.0, [0.15] * 8), (6, 8, 6, 8))   # rank 6 saves 0.1512, 7 0.1305
        self.assertEqual(settings.depth(self.NU, self.P, 99.0, [0.152] * 8), (5, 8, 5, 8))
        self.assertEqual(settings.depth(self.NU, self.P, 99.0, [0.3] * 8), (0, 8, 0, 8))
        self.assertEqual(settings.depth(self.NU, self.P, 99.0, [0.0] + [0.3] * 7), (1, 8, 1, 8))   # rank 1 takes none
        # a slot nearer the front of a small cache serves more: rank 3 saves 0.208 against 0.25
        self.assertEqual(settings.depth(self.NU, self.P, 99.0, [0.05, 0.1, 0.25, 0.3, 0.4] + [math.inf] * 3),
                         (2, 8, 2, 8))

    def test_a_read_pays_or_the_prefetch_is_off(self):
        # no idle window: every read costs its own transfer, which a used share below 1 never earns back: off
        self.assertEqual(settings.depth(self.NU, self.P, 0.0, None), (0, 0, 8, 0))
        self.assertEqual(settings.depth(self.NU, self.P, -0.2, [0.01] * 8), (0, 0, 8, 0))
        # inside the window a read costs its spill: rank 1 (p 0.83) pays a spill of 0.8 reads, rank 2 (0.76) does not
        self.assertEqual(settings.depth(self.NU, self.P, 99.0, None, spill=0.8), (1, 8, 8, 1))
        self.assertEqual(settings.depth(self.NU, self.P, 99.0, None, spill=0.9), (0, 8, 8, 0))
        # the demand time a read saves weighs the used share: 1.25 reads make rank 3 (0.65) pay 0.8, 4 (0.54) not
        self.assertEqual(settings.depth(self.NU, self.P, 99.0, None, gain=1.25, spill=0.8), (3, 8, 8, 3))
        # a demand read slower than the prefetch's own may pay its transfer past the window: 0.83 * 1.3 > 1
        self.assertEqual(settings.depth(self.NU, self.P, 0.0, None, gain=1.3), (1, 0, 8, 1))

    def test_the_stored_runs_keep_their_depth(self):
        # the stored DGX Spark bootstraps (DeepSeek-V4.1-Flash-1017 and Muse-Spark-Open), as their
        # settings receipts give them, with the demand time a read saves from their layer fits ([nm] - [nr]) and the
        # readers' time from their reports (10.42 s and 44.41 s, against windows of 19929 x 3004 and 31171 x 3345 us):
        # no spill, so every rank that starts inside the window pays, and the depths stay 6 and 5
        dsv = settings.read_spill({'issued': 4796, 'reader_busy_us': 10422618, 'windows': 19929,
                                   'window_us_mean': 3004}, 1.0)[0]
        muse = settings.read_spill({'issued': 13902, 'reader_busy_us': 44414290, 'windows': 31171,
                                    'window_us_mean': 3345}, 0.0)[0]
        self.assertEqual((dsv, muse), (0.0, 0.0))
        t_dsv, t_muse = 18812928 / (10.568 * 2 ** 30) * 1e6, 30420992 / (11.72 * 2 ** 30) * 1e6
        g_dsv = settings.read_gain({'layer_beta': [0.6291, 0.1813, 0.492, 2.0397, 0.2085]}, 18812928, t_dsv)[0]
        g_muse = settings.read_gain({'layer_beta': [6.3644, -0.5301, -0.2674, 2.0358, -0.3163]}, 30420992, t_muse)[0]
        self.assertAlmostEqual(g_dsv, 1547.7, places=1)
        self.assertEqual(settings.depth([0.114, 0.140, 0.175, 0.213, 0.247, 0.288],
                                        [0.876, 0.712, 0.536, 0.412, 0.303, 0.207], 1.77,
                                        [0.0, 0.0063, 0.0074, 0.0066, 0.0076, 0.0063], g_dsv / t_dsv, dsv / t_dsv)[0], 6)
        self.assertEqual(settings.depth([0.218, 0.253, 0.293, 0.336, 0.377, 0.418, 0.449, 0.474],
                                        [0.859, 0.766, 0.662, 0.540, 0.461, 0.377, 0.298, 0.235], 1.32,
                                        [0.0, 0.0443, 0.0448, 0.0480, 0.0495, 0.0545, 0.0523, 0.0551],
                                        g_muse / t_muse, muse / t_muse)[0], 5)
        # an RTX 5090, the same model, its bootstrap (placement all, depth 2): the readers' 24.87 s against windows
        # of 19929 x 927 us spill 649 us a read over 9858 reads, less than a used read saves (1.29 ms, the layer fit)
        # at ranks 1 and 2; rank 3 starts after the window (700 us, 0.46 reads of 1535 us): 2, as before
        spill, how = settings.read_spill({'issued': 9858, 'reader_busy_us': 24872641, 'windows': 19929,
                                          'window_us_mean': 927}, 0.0)
        self.assertAlmostEqual(spill, 649.1, delta=0.5)
        self.assertIn('a lower bound, the engine not measuring it', how)
        t_e = 18812928 / (11.42 * 2 ** 30) * 1e6
        g = settings.read_gain({'layer_beta': [10.481, -1.6327, -1.2876, 0.0, 0.4909]}, 18812928, t_e)[0]
        self.assertEqual(settings.depth([0.237, 0.261, 0.293, 0.332, 0.382, 0.421],
                                        [0.915, 0.769, 0.605, 0.48, 0.344, 0.251], 700 / t_e, None, g / t_e,
                                        spill / t_e), (2, 2, 6, 2))

    def test_the_engines_spill_measure_prices_each_rank(self):
        # the engine's measure: 2.0 s of union time past the windows, shared out by the ranks' own times (0, 1, 3 s
        # of reads that started 1000, 1000, 500 times): 0, 500 and 3000 us a read; ranks 4-6 not read (depth 3)
        pf = {'issued': 3000, 'reader_busy_us': 1e6, 'windows': 1000, 'window_us_mean': 3000, 'spill_us': 2e6,
              'spills': 900, 'started_r': [1000, 1000, 500, 0, 0, 0], 'spill_us_r': [0, 1e6, 3e6, 0, 0, 0]}
        per, how = settings.read_spill(pf, 0.0)
        self.assertEqual(per[3:], [None] * 3)
        self.assertEqual([round(x) for x in per[:3]], [0, 500, 3000])
        self.assertIn("2.00 s past the windows (900 role reads ran past one) over 2500 experts read, an expert's read of "
                      "each rank 0,500,3000,-,-,- us", how)
        # the aggregate would see no spill here: the readers' 1 s is less than the windows' 3 s
        self.assertEqual(settings.read_spill({k: v for k, v in pf.items() if k not in ('spill_us', 'started_r',
                                                                                        'spill_us_r')}, 0.0)[0], 0.0)
        # no spill measured: every read rank costs 0
        self.assertEqual(settings.read_spill(dict(pf, spill_us=0.0, spill_us_r=[0] * 6), 0.0)[0],
                         [0.0, 0.0, 0.0, None, None, None])
        # depth: a measured rank costs its measure (no transfer term), an unread one the worst measure plus the
        # transfer past the window: with t_e 1000 us and a gain of 1.0 read, rank 3 (p 0.65 against 3.0) stops it
        nu, p = [0.2] * 6, [0.9, 0.8, 0.65, 0.6, 0.5, 0.4]
        price = [None if x is None else x / 1000 for x in per]
        self.assertEqual(settings.depth(nu, p, 99.0, None, 1.0, price), (2, 6, 6, 2))
        # a rank inside the window that was not read costs the worst measured rank (here 0.5) without the transfer:
        # ranks 3 and 4 (0.65, 0.6) pay it, rank 5 (0.5) does not
        self.assertEqual(settings.depth(nu, p, 99.0, None, 1.0, [0.0, 0.5, None, None, None, None]), (4, 6, 6, 4))
        # past the window the transfer adds a whole read: rank 3 (p 0.65) pays 0.5 inside it, not 1.5 past it
        self.assertEqual(settings.depth(nu, p, 0.3, None, 1.0, [0.0, 0.5, None, None, None, None]), (2, 2, 6, 2))

    def test_the_ram_cache_behind_the_tier_is_the_engines(self):
        import random
        rnd = random.Random(7)
        routing = [[[rnd.sample(range(12), 3) for _ in range(2)] for _ in range(200)]]
        r = Replay(4, 1.0, 20, behind=True).advance(routing)
        tier = Replay(4, 1.0, 20).advance(routing)
        self.assertEqual(r.rows, tier.rows)                                       # the tier is the same replay
        hit = {}
        for slots in range(0, 7):
            # the engine's count_decay2 written out: every count of the tier's misses halves in 2 tokens; a miss takes
            # a free slot, else the lowest count's (the oldest touch of equals) unless its own count is lower
            t, hits, clock = Replay(4, 1.0, 0), 0, 0
            count, res = [{}, {}], [{}, {}]
            for i, tok in enumerate(routing[0]):
                for li, ex in enumerate(tok):
                    miss = [e for e, h in zip(ex, t.tier.access(li, ex)) if not h]
                    c, rl = count[li], res[li]
                    for e in c:
                        c[e] *= 2 ** -0.5 if i else 1.0
                    for e in miss:
                        c[e] = c.get(e, 0.0) + 1.0
                    held = {e for e in miss if e in rl}
                    hits += len(held) if i >= 20 else 0
                    for e in miss:
                        if e in held:
                            clock += 1
                            rl[e] = clock
                    for e in miss:
                        if e in rl or not slots:
                            continue
                        if len(rl) >= slots:
                            free = sorted((x for x in rl if x not in held), key=lambda x: (c[x], rl[x]))
                            if not free or c[e] < c[free[0]]:
                                continue
                            del rl[free[0]]
                        clock += 1
                        rl[e] = clock
                        held.add(e)
                t.tier.step += 1
            hit[slots] = hits / 180
            self.assertAlmostEqual(r.behind_hits(slots), hit[slots])
        self.assertEqual(hit[0], 0.0)
        self.assertGreater(hit[6], hit[1])

    def machine(self, integrated=True, ram_slots=None, prefetch=True, other=0.0):
        import random
        rnd = random.Random(3)
        routing = [[[rnd.sample(range(32), 4) for _ in range(4)] for _ in range(300)]]
        reps = {}

        def replay(key, segs, tier):
            if key not in reps:
                reps[key] = Replay(tier, 1.0, 50).advance(segs)
            return reps[key]
        pf = {'topk': 2, 'readers': 3, 'layers': 1200, 'windows': 1200, 'window_us_p50': 3400.0,
              'predicted_r': [1200] * 4, 'novel_r': [300, 360, 420, 480], 'novel_routed_r': [250, 260, 240, 200]}
        P = {'all': {'prefetch': pf if prefetch else None, 'pf_k': 2, 'drive_other_bytes': other, 'ram_slots': ram_slots}}
        dev = {'gpu': {'integrated': integrated}, 'ssd': {'1': {'gibs': 8.0}, '3': {'gibs': 9.0}, '6': {'gibs': 10.0}}}
        return SimpleNamespace(P=P, dev=dev, caps={'all': 40}, routing=routing, replay=replay, mc=1.0,
                               ins={'skip': 50}, PF={'L': 4, 'B_e': 30420992})

    def test_the_depth_from_the_bootstraps_report(self):
        m = self.machine(other=1200 * 0.5e-3 * 9.0 * 2 ** 30)                     # other readers: 500 us a layer
        (s,) = settings.prefetch_depth(m, 'all')
        t_e = 30420992 / (9.0 * 2 ** 30) * 1e6                                    # the probe at QD 3, 3 readers
        nu = [0.25, 0.3, 0.35, 0.4]
        p = [250 / 300, 260 / 360, 240 / 420, 200 / 480]
        # the bootstrap reached 40 slots at depth 2, 4 layers: 44 at depths 0 and 1, 36 at 3, 32 at 4
        h = {c: Replay(c, 1.0, 50).advance(m.routing).stats()['hits_per_token'] for c in (44, 40, 36, 32)}
        cost = [0.0, (h[44] - h[40]) / 4, (h[40] - h[36]) / 4, (h[36] - h[32]) / 4]
        self.assertEqual(settings.rank_costs(m, 'all', 4)[0], [max(0.0, x) for x in cost])
        k = settings.depth(nu, p, (3400.0 - 500.0) / t_e, [max(0.0, x) for x in cost])
        self.assertEqual((s.name, s.value, s.source), ('GGML_KIMI_CPU_SINK_PREFETCH', k[0], 'derived'))
        self.assertIn(f'min(k_pay {k[3]}, k_mem {k[2]}) of top-k 4', s.input)
        self.assertIn(f'(k_win {k[1]} start inside it)', s.input)
        self.assertIn('spill into the next layer\'s demand reads, 0 (no reads to measure it)', s.input)
        self.assertIn('less 500 us of other readers', s.input)
        self.assertIn('the probe at QD 3 for 3 readers', s.input)
        self.assertIn(f'{h[44]:.2f} hits a token at depth 1 (44 slots), {h[32]:.2f} at depth 4 (32)', s.input)
        self.assertRegex(settings.render([s]).strip(), LINE)
        tiny = self.machine()
        tiny.caps = {'all': 6}                                                    # 10, 10, 6, 2, -2 slots by depth
        self.assertEqual(settings.rank_costs(tiny, 'all', 4)[0][3], math.inf)

    def test_a_discrete_gpu_prices_the_ram_cache(self):
        m = self.machine(integrated=False, ram_slots=12)                          # 3 slots a layer
        (s,) = settings.prefetch_depth(m, 'all')
        r = Replay(40, 1.0, 50, behind=True).advance(m.routing)
        H = [r.behind_hits(x) for x in range(4)]
        cost, why = settings.rank_costs(m, 'all', 4)
        self.assertEqual(cost, [max(0.0, (H[3 - j] - H[2 - j]) / 4) for j in range(3)] + [math.inf])
        self.assertIn(f'{H[3]:.2f}, {H[2]:.2f}, {H[1]:.2f}, 0.00 hits a token at 3 down to 0 slots a layer', s.input)
        self.assertIn('no slot cost', settings.prefetch_depth(self.machine(integrated=False), 'all')[0].input)

    def test_no_statistics_no_depth(self):
        self.assertEqual(settings.prefetch_depth(self.machine(prefetch=False), 'all'), [])

    def test_deepseek_holds_the_depth(self):
        conf = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../fixtures/adaptive', 'models', 'dsv41.conf')).read()
        self.assertIn('GGML_KIMI_CPU_SINK_PREFETCH', next(x for x in conf.splitlines() if x.startswith('PENDING=')))


class LaneModel(unittest.TestCase):
    """The sink pass with the GPU lane (B6): prefill.lane_layer is kimi_place_layer's streamed form (kimi-prefill-cost.h);
    the golden values are the C++ header's on the same layers (scratchpad lane-vm/golden.cpp at the lane branch): a
    page-locked ring, a pageable one (the staging pipelined), in place where the GPU reads host memory and that is
    no slower, copies where it is, and a page-locked ring without a measured copy (nothing priced)."""
    GOLDEN = [
        {'n': 4221, 'U': 220, 'R': 0, 'routes': 150, 'mb': 30.408704, 'in_place': False, 'ring_locked': True, 'route_bytes': 8, 'lane': {'cpu_fixed_us': 175, 'cpu_route_us': 409, 'gpu_fixed_us': 121.5, 'gpu_route_us': 1.664, 'gpu_host_fixed_us': 0, 'gpu_host_route_us': 0, 'link_gbps': 6.91, 'pageable_gbps': 5, 'drive_gbps': 7, 'drive_fixed_us': 12000, 'gpu_token_us': 2.3}, 'us': 1071099.049515, 'moves': 206, 'chosen_in_place': False},
        {'n': 4221, 'U': 220, 'R': 40, 'routes': 150, 'mb': 30.408704, 'in_place': False, 'ring_locked': True, 'route_bytes': 8, 'lane': {'cpu_fixed_us': 175, 'cpu_route_us': 409, 'gpu_fixed_us': 121.5, 'gpu_route_us': 1.664, 'gpu_host_fixed_us': 0, 'gpu_host_route_us': 0, 'link_gbps': 6.91, 'pageable_gbps': 5, 'drive_gbps': 7, 'drive_fixed_us': 12000, 'gpu_token_us': 2.3}, 'us': 882266.328414, 'moves': 168, 'chosen_in_place': False},
        {'n': 669, 'U': 205, 'R': 0, 'routes': 26, 'mb': 30.408704, 'in_place': True, 'ring_locked': False, 'route_bytes': 8, 'lane': {'cpu_fixed_us': 1046, 'cpu_route_us': 118, 'gpu_fixed_us': 212, 'gpu_route_us': 2.527, 'gpu_host_fixed_us': 173.6, 'gpu_host_route_us': 2.186, 'link_gbps': 0, 'pageable_gbps': 52.6, 'drive_gbps': 12.14, 'drive_fixed_us': 3000, 'gpu_token_us': 0.4}, 'us': 521636.539794, 'moves': 134, 'chosen_in_place': True},
        {'n': 4104, 'U': 300, 'R': 10, 'routes': 80, 'mb': 18.812928, 'in_place': False, 'ring_locked': True, 'route_bytes': 5448, 'lane': {'cpu_fixed_us': 520, 'cpu_route_us': 297, 'gpu_fixed_us': 60, 'gpu_route_us': 25, 'gpu_host_fixed_us': 0, 'gpu_host_route_us': 0, 'link_gbps': 6.91, 'pageable_gbps': 5, 'drive_gbps': 7, 'drive_fixed_us': 12000, 'gpu_token_us': 7}, 'us': 889702.502788, 'moves': 260, 'chosen_in_place': False},
        {'n': 4221, 'U': 220, 'R': 0, 'routes': 150, 'mb': 30.408704, 'in_place': False, 'ring_locked': True, 'route_bytes': 8, 'lane': {'cpu_fixed_us': 1, 'cpu_route_us': 0.01, 'gpu_fixed_us': 121.5, 'gpu_route_us': 1.664, 'gpu_host_fixed_us': 0, 'gpu_host_route_us': 0, 'link_gbps': 6.91, 'pageable_gbps': 5, 'drive_gbps': 7, 'drive_fixed_us': 12000, 'gpu_token_us': 2.3}, 'us': 972046.226286, 'moves': 0, 'chosen_in_place': False},
        {'n': 4221, 'U': 220, 'R': 0, 'routes': 150, 'mb': 30.408704, 'in_place': False, 'ring_locked': False, 'route_bytes': 8, 'lane': {'cpu_fixed_us': 175, 'cpu_route_us': 409, 'gpu_fixed_us': 121.5, 'gpu_route_us': 1.664, 'gpu_host_fixed_us': 0, 'gpu_host_route_us': 0, 'link_gbps': 6.91, 'pageable_gbps': 5.83, 'drive_gbps': 7, 'drive_fixed_us': 12000, 'gpu_token_us': 2.3}, 'us': 1159561.621193, 'moves': 203, 'chosen_in_place': False},
        {'n': 4221, 'U': 220, 'R': 20, 'routes': 150, 'mb': 30.408704, 'in_place': True, 'ring_locked': False, 'route_bytes': 8, 'lane': {'cpu_fixed_us': 175, 'cpu_route_us': 409, 'gpu_fixed_us': 121.5, 'gpu_route_us': 1.664, 'gpu_host_fixed_us': 4000, 'gpu_host_route_us': 30, 'link_gbps': 0, 'pageable_gbps': 52.6, 'drive_gbps': 12.14, 'drive_fixed_us': 3000, 'gpu_token_us': 0.4}, 'us': 508996.875476, 'moves': 196, 'chosen_in_place': False},
        {'n': 4221, 'U': 220, 'R': 0, 'routes': 150, 'mb': 30.408704, 'in_place': False, 'ring_locked': True, 'route_bytes': 8, 'lane': {'cpu_fixed_us': 175, 'cpu_route_us': 409, 'gpu_fixed_us': 121.5, 'gpu_route_us': 1.664, 'gpu_host_fixed_us': 0, 'gpu_host_route_us': 0, 'link_gbps': 0, 'pageable_gbps': 5.83, 'drive_gbps': 7, 'drive_fixed_us': 12000, 'gpu_token_us': 2.3}, 'us': 13539844.109150, 'moves': 0, 'chosen_in_place': False},
    ]

    def test_the_port_is_the_engines_choice(self):
        for g in self.GOLDEN:
            us, _, moves = prefill.lane_layer(g['lane'], g['n'], g['U'], g['routes'], g['R'] / g['U'], g['mb'] * 1e6,
                                              g['in_place'], g['route_bytes'], ring_locked=g['ring_locked'])
            self.assertEqual(moves, g['moves'])
            self.assertAlmostEqual(us / g['us'], 1.0, places=9)

    def test_coupled_is_the_headers(self):
        self.assertEqual(prefill.coupled(1000.0, 0.0, 8), 1000.0)
        self.assertAlmostEqual(prefill.coupled(1000.0, 1000.0, 8), 1125.0)
        self.assertAlmostEqual(prefill.coupled(1000.0, 100.0, 8), 1000.0, places=4)

    def lane_model(self, d_s=1.9e-3, graph=None):
        # the Spark with Muse: the lane in place, the drive 12.14 GB/s, the CPU 1046 + 118 us a route
        m = model(c=150e-6, rate=12.14e9, d=d_s, B_e=30408704)
        lane = dict(self.GOLDEN[2]['lane'])
        for s in ('cold', 'warm'):
            m['pass']['all'][s] = dict(m['pass']['all'][s], lane=lane, d=d_s)
        m['lane'] = {'in_place': True, 'route_bytes': 8, 'engine': 'batched'}
        m['graph'] = graph
        return m

    def test_the_sink_pass_with_the_lane(self):
        # far below the CPU-only sink at 4221 tokens: the GPU takes the most-routed experts; reads stay the floor
        m = self.lane_model()
        lane_s, cpu_s, read = prefill.pass_seconds(m, 'all', 4221, False, True, 0)
        plain = model(c=150e-6, rate=12.14e9, d=1.9e-3, B_e=30408704)
        plain_s = prefill.pass_seconds(plain, 'all', 4221, False, True, 0)[0]
        self.assertLess(lane_s, 0.5 * plain_s)
        self.assertGreaterEqual(lane_s, read / 12.14e9 + 4221 * 1.9e-3)
        self.assertLess(cpu_s, prefill.pass_seconds(plain, 'all', 4221, False, True, 0)[1])

    def test_the_break_even_needs_a_graph_arm(self):
        self.assertIsNone(prefill.break_even(self.lane_model(), 'all'))

    def test_a_graph_slower_a_token_never_wins_against_the_lane(self):
        # the graph's copy at the drive's rate and 3.5 ms a token besides (its serialized compute): never faster
        m = self.lane_model(graph={'R': 12.0e9, 'a': 0.5, 'd': 3.5e-3, 'n': [669, 4221]})
        self.assertEqual(prefill.break_even(m, 'all'), math.inf)

    def test_the_crossing_with_the_lane(self):
        # a graph whose time a token besides is far below the sink's: it wins from some pass size on, and the
        # break-even is where the two routes cross
        m = self.lane_model(d_s=6e-3, graph={'R': 12.0e9, 'a': 0.5, 'd': 0.5e-3, 'n': [669, 4221]})
        t = prefill.break_even(m, 'all')
        self.assertTrue(0 < t < 1 << 20)
        at = lambda n, g: prefill.pass_seconds(m, 'all', n, g, False, m['tier_at_warm'])[0]   # noqa: E731
        self.assertLessEqual(at(t, False), at(t, True))
        self.assertGreater(at(t + 1, False), at(t + 1, True))

    def test_the_check_prices_each_pass_with_the_lane(self):
        # prefill.py check: each pass with the lane, in sample and from the other state's table
        m = self.lane_model()
        for st in m['pass']['all'].values():
            st['prompt_s'] = 40.0
        rows = prefill.check(m)
        self.assertEqual([r[:3] for r in rows], [('all', 'cold', 4000), ('all', 'warm', 4000)])
        self.assertTrue(all(r[4] > 0 and r[5] is not None for r in rows))
        self.assertEqual(prefill.check(model()), [])          # no lane: nothing to check

    def test_the_bootstraps_table_is_carried(self):
        lane = {'cpu_fixed_us': 1046.0, 'cpu_route_us': 118.0, 'gpu_fixed_us': 212.0, 'gpu_route_us': 2.527,
                'gpu_host_fixed_us': 0.0, 'gpu_host_route_us': 0.0, 'link_gbps': 0.0, 'drive_gbps': 12.14,
                'drive_fixed_us': 3000.0, 'gpu_token_us': -1.0}
        m = machine(P={'all': {'prefill': [{'n_tokens': 4221, 'tier': 0}, {'n_tokens': 4221, 'tier': 900, 'lane': lane,
                                                                          'layers_costed': 61}]}})
        s = settings.prefill_carried(m, 'all')
        self.assertEqual((s.name, s.source), ('GGML_KIMI_CPU_PREFILL_COST', 'derived'))
        t = dict(x.split('=') for x in s.value.split(','))
        self.assertEqual(sorted(t), ['cpu_fixed_us', 'cpu_route_us', 'drive_fixed_us', 'drive_gbps', 'gpu_fixed_us',
                                     'gpu_route_us'])   # no zero rate, no uncalibrated overhead
        self.assertIsNone(settings.prefill_carried(machine(P={'all': {'prefill': [{'n_tokens': 4221, 'tier': 0}]}}), 'all'))

    def test_the_log_gives_each_pass_its_table(self):
        lines = ['[kimi_vram_prefill_engine] format=q4_K/q5_K engine=batched in_place=1 route_bytes=8 token_bytes=16384 '
                 'output_bytes=16384 fixed_bytes=23400000 bytes_per_token=32768 reserved_rows=4221 device_bytes=1 host_bytes=1']
        for layer in range(3):
            lines.append(f'[kimi_cpu_prefill_cost] calibrated={layer + 1} tokens=4221 samples={layer + 1} cpu_fixed_us=1046.0 '
                         f'cpu_route_us=118.0 gpu_fixed_us=212.0 gpu_route_us=2.5 gpu_host_fixed_us=173.6 gpu_host_route_us=2.2 '
                         f'link_gbps=0.000 pageable_gbps=52.600 drive_gbps=12.140 drive_fixed_us=3000.0 gpu_token_us=0.4 '
                         f'cpu_one_us=1300.0 predicted_us=600000.0 measured_us=610000.0')
            lines.append('[kimi_cpu_prefill_schedule] format=batched rule=makespan drive_us=1.0 fill_us=1.0 gpu_overhead_us=1.0 '
                         'ring=12 form=streamed ring_locked=0 in_place=1 makespan_before_us=1.0 makespan_after_us=1.0')
            lines.append(f'[kimi_cpu_prefill_layer] layer={layer} n_tokens=4221 slots=8 unique_experts=200 routed_slots=33768 '
                         'gpu_experts=0 gpu_ssd_transient_experts=150 ssd_experts=200 ssd_bytes=6000000000')
        a = logparse.ArmParser()
        for line in lines:
            a.feed(line)
        with tempfile.TemporaryDirectory() as d:
            r = a.result(d)
        (p,) = r['prefill']
        self.assertEqual(p['lane']['drive_gbps'], 12.14)
        self.assertAlmostEqual(p['layers_s'], 1.83)
        self.assertEqual(p['layers_costed'], 3)
        self.assertEqual(r['prefill_engine']['engine'], 'batched')
        self.assertEqual((r['prefill_engine']['ring'], r['prefill_engine']['ring_locked']), (12, 0))   # the schedule's ring
        # an engine without the receipt (every stored run): no lane, the old model
        b = logparse.ArmParser()
        for line in lines[1:]:
            b.feed(line)
        with tempfile.TemporaryDirectory() as d:
            self.assertNotIn('lane', b.result(d)['prefill'][0])


class PrefillStart(unittest.TestCase):
    """The cost table's start values from the probe's lanes (B4)."""
    # DGX Spark (GB10) at Muse's shape: busy lines, the copy, the drive at its knee
    LANES = {'format': 'q4_K/q5_K', 'latent': 4096, 'hidden': 4096, 'threads': 20,
             'gpu': {'fixed_us': 212.0, 'per_route_us': 2.527, 'at150_us': 591.0, 'load': 'busy'},
             'gpu_host': {'fixed_us': 173.6, 'per_route_us': 2.186, 'at150_us': 501.6, 'load': 'busy'},
             'cpu': {'fixed_us': 1045.9, 'per_route_us': 118.1, 'at150_us': 18755.0, 'load': 'busy'},
             'copy_gbps': 58.82}
    SSD = {'ssd': {'3': {'gibs': 11.307, 'lat_ms': 1.0, 'gibs_runs': [11.279, 11.335]}}, 'ssd_knee_qd': 3}

    def terms(self, s):
        return dict(x.split('=') for x in s.value.split(','))

    def test_the_lanes_give_the_start_values(self):
        dev = {**DEV, **self.SSD, 'lanes': self.LANES, 'gpu': {'integrated': True, 'h2d_pinned_gbs': 52.6}}
        (s,) = settings.prefill_start(machine(dev=dev))
        self.assertEqual((s.name, s.source), ('GGML_KIMI_CPU_PREFILL_COST', 'measured'))
        t = self.terms(s)
        self.assertEqual((t['cpu_fixed_us'], t['cpu_route_us'], t['gpu_fixed_us'], t['gpu_route_us']),
                         ('1045.9', '118.1', '212', '2.527'))
        self.assertEqual((t['gpu_host_fixed_us'], t['gpu_host_route_us']), ('173.6', '2.186'))
        self.assertNotIn('link_gbps', t)                   # unified memory: no link
        self.assertEqual(t['pageable_gbps'], '58.82')      # the copy the in-place choice weighs: the probe's
        self.assertAlmostEqual(float(t['drive_gbps']), 12.14, places=2)
        self.assertRegex(settings.render([s]).strip(), LINE)

    def test_a_discrete_gpu_takes_the_copies(self):
        lanes = {k: v for k, v in self.LANES.items() if k != 'gpu_host'}
        dev = {**DEV, 'lanes': lanes, 'gpu': {'integrated': False, 'h2d_pinned_gbs': 6.91, 'h2d_pageable_gbs': 5.83}}
        t = self.terms(settings.prefill_start(machine(dev=dev))[0])
        self.assertEqual((t['link_gbps'], t['pageable_gbps']), ('58.82', '5.83'))
        self.assertNotIn('gpu_host_fixed_us', t)
        self.assertNotIn('drive_gbps', t)                  # no drive measured

    def test_a_format_on_its_own_engine_has_no_gpu_line(self):
        dev = {**DEV, 'lanes': {**self.LANES, 'format': 'mxfp4/mxfp4'}, 'gpu': {'integrated': False}}
        t = self.terms(settings.prefill_start(machine(dev=dev))[0])
        self.assertIn('cpu_fixed_us', t)
        self.assertFalse(any(k.startswith('gpu_') for k in t))

    def test_no_lanes_no_start_values(self):
        # a probe before the lane timing (every stored run's): nothing written, the settings as they were
        dev = {**DEV, **self.SSD, 'gpu': {'integrated': False, 'h2d_pinned_gbs': 6.91}}
        self.assertEqual(settings.prefill_start(machine(dev=dev)), [])

    def test_the_probe_parses_the_lanes(self):
        # device_profile.py on a probe directory with the timing's output (the busy line of each lane)
        here = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../fixtures/adaptive')
        with tempfile.TemporaryDirectory() as d:
            def write(name, text):
                with open(f'{d}/{name}', 'w') as f:
                    f.write(text)
            write('cpus.txt', '0 0 3000000 0\n1 0 3000000 1\n')
            write('meminfo.txt', 'MemTotal: 131072000 kB\nMemAvailable: 120000000 kB\n')
            write('memprobe.txt', 'integrated=1 gpu_read_bw=200.0 pinned=52.6 pageable=40.1\n')
            write('coresets.txt', 'all 0,1\n')
            write('experts.json', '{"expert_read_bytes": 30408704, "bytes_per_role": {}}')
            fit = ('[kimi_vram_group_fit] lane={} source={} load={} threads=20 participants=19 format=q4_K/q5_K '
                   'latent=4096 hidden=4096 points_us=8:230 fixed_us={} per_route_us={} at150_us={}\n')
            lanes_txt = (
                  fit.format('gpu', 'device', 'idle', 184.2, 2.022, 487.5) + fit.format('gpu', 'device', 'busy', 212.0, 2.527, 591.0) +

                  '[kimi_vram_group_copy] bytes=30408704 idle_us=516.6 idle_gbps=58.86 busy_us=517.0 busy_gbps=58.82\n' +
                  fit.format('gpu', 'host', 'idle', 180.4, 2.017, 483.0) + fit.format('gpu', 'host', 'busy', 173.6, 2.186, 501.6) +
                  fit.format('cpu', 'host', 'idle', 1617.4, 100.7, 16722) + fit.format('cpu', 'host', 'busy', 1045.9, 118.1, 18755) +
                  'KIMI_VRAM_GROUP_TIME_DONE\n')
            write('lanes.txt', lanes_txt)
            import json
            import subprocess
            profile = lambda: json.loads(subprocess.run(   # noqa: E731
                [sys.executable, '-B', os.path.join(here, 'device_profile.py'), d, d],
                capture_output=True, text=True, check=True).stdout)['lanes']
            lanes = profile()
            # a negative term never reaches the engine (it stops on one): a busy line of -3.2 us is read as 0
            write('lanes.txt', lanes_txt + fit.format('cpu', 'host', 'busy', -3.2, 120.0, 17997))
            self.assertEqual(profile()['cpu']['fixed_us'], 0.0)
        self.assertEqual((lanes['gpu']['fixed_us'], lanes['gpu']['load']), (212.0, 'busy'))
        self.assertEqual((lanes['gpu_host']['per_route_us'], lanes['cpu']['fixed_us']), (2.186, 1045.9))
        self.assertEqual((lanes['format'], lanes['threads'], lanes['copy_gbps']), ('q4_K/q5_K', 20, 58.82))


class Render(unittest.TestCase):
    def test_every_value_line_has_a_source_and_an_input(self):
        text = settings.render(settings.configuration(machine(), 84, 'split'), ['a header'])
        lines = text.splitlines()
        self.assertEqual(lines[0], '# a header')
        values = [x for x in lines if not x.startswith('#')]
        self.assertTrue(values)
        for x in values:
            self.assertRegex(x, LINE)


class LayerFit(unittest.TestCase):
    # the per-layer table of DeepSeek-V4.1-Flash-1017's bootstrap on a DGX Spark (placement all, depth 2, top-6):
    # class (tier hits, ready, read): [layers, mean ms]
    TABLE = {(6, 0, 0): [10569, 1.8732], (5, 0, 1): [3438, 3.7726], (4, 0, 2): [585, 5.3636], (5, 1, 0): [415, 2.0285],
             (4, 1, 1): [175, 3.845], (3, 0, 3): [69, 7.5655], (3, 1, 2): [42, 6.131], (2, 1, 3): [11, 8.0256],
             (4, 2, 0): [7, 2.4874], (2, 0, 4): [3, 9.4339], (2, 2, 2): [2, 7.3981], (3, 2, 1): [2, 4.9227],
             (1, 1, 4): [1, 9.9388], (1, 2, 3): [1, 8.2054]}

    def fit(self, table):
        X, Y = [], []
        for (nv, nr, nm), (n, ms) in table.items():
            X += [[1.0, nv, nr, nm, 1.0 if nm else 0.0]] * n
            Y += [ms] * n
        b = lstsq(X, Y)
        return lambda nv, nr, nm: b[0] + b[1] * nv + b[2] * nr + b[3] * nm + b[4] * (1 if nm else 0)

    def test_dependent_columns_fit_the_classes(self):
        # the classes' experts add up to the top-k: the constant and the three counts are dependent; the fit still
        # reproduces the common classes and prices an all-miss layer above the slowest seen one (the solver with an
        # absolute tolerance gave 6.55 ms for (6, 0, 0) and -16 ms on the bootstrap's own rows)
        f = self.fit(self.TABLE)
        self.assertAlmostEqual(f(6, 0, 0), 1.872, places=2)
        self.assertAlmostEqual(f(5, 0, 1), 3.758, places=2)
        self.assertAlmostEqual(f(0, 0, 6), 12.50, places=1)
        self.assertGreater(f(1, 0, 5), f(2, 0, 4))

    def test_exact_on_a_linear_table(self):
        g = lambda nv, nr, nm: 0.3 + 0.2 * nv + 0.5 * nr + 1.9 * nm + 0.4 * (1 if nm else 0)
        f = self.fit({c: [7 + sum(c[1:]), g(*c)] for c in [(6, 0, 0), (5, 0, 1), (4, 1, 1), (5, 1, 0), (3, 0, 3), (2, 2, 2)]})
        for c in [(0, 0, 6), (1, 2, 3), (6, 0, 0)]:
            self.assertAlmostEqual(f(*c), g(*c), places=6)


class PlanGuard(unittest.TestCase):
    """plan.py sets aside a configuration priced at a time or a rate at or below zero, or on a placement whose layer
    table prices a class so: the first decision on a DGX Spark (the singular fit's betas, a 16-slot tier from the
    one-block ubatch probe) chose all@16384 at -373 s."""
    TABLE = {'6,0,0': [10569, 1.8732, 0.0131], '5,0,1': [3438, 3.7726, 0.2267], '4,1,1': [175, 3.845, 0.5455]}

    def test_a_layer_class_priced_below_zero(self):
        singular = {'layer_table': self.TABLE, 'layer_beta': [-12.3738, -0.7374, -0.5369, 1.0109, 0.1379]}
        bad = machine_mod.layer_faults(singular, 6)
        self.assertEqual(bad[0], (0, 0, 6, -6.17))                # an all-miss layer
        self.assertEqual(len(bad), 25)                            # every fitted class of the 28: 3 are measured
        self.assertNotIn((6, 0, 0), [x[:3] for x in bad])         # measured often enough: its own mean, 1.87 ms
        fixed = {'layer_table': self.TABLE, 'layer_beta': [12.3617, -1.7483, -1.5478, 0.0, 0.1379]}
        self.assertEqual(machine_mod.layer_faults(fixed, 6), [])

    def test_a_row_priced_below_zero(self):
        first = {'tier': 16, 'cpus': 'all@16384', 'total_s': -372.97, 'prefill_s': 50.22, 'decode_tps': -4.839,
                 'turns': [{'prefill_s': 40.0, 'decode_tps': -4.839}, {'prefill_s': 2.4, 'decode_tps': -4.839}]}
        self.assertEqual(machine_mod.row_faults(first), ['total_s -372.97', 'decode_tps -4.839',
                                                        'turn 0 decode_tps -4.839', 'turn 1 decode_tps -4.839'])
        sane = {'total_s': 358.0, 'prefill_s': 47.2, 'decode_tps': 6.59,
                'turns': [{'prefill_s': 40.9, 'decode_tps': 6.59}]}
        self.assertEqual(machine_mod.row_faults(sane), [])


class UbatchProbe(unittest.TestCase):
    UMA = ['[kimi_vram_tier] reason=allocated budget_bytes=1 capacity_bytes=300810240 slots=16 hits=0',
           '[kimi_vram_elastic] event=decode_grow generation=0 status=pass epoch=1 token=194 decode_token=2 '
           'us=3940937 before_slots=16 target_slots=5739 after_slots=5632 added_slots=5616 blocks=352',
           '[kimi_vram_elastic] event=watermark_grow token=195 free_before=229060608 high_bytes=67108864 '
           'added_slots=8 slots=5640 grows=1']

    def test_unified_memory_reads_the_decode_growth(self):
        self.assertEqual(ubprobe.reached(self.UMA), (5632, 5640))

    def test_discrete_reads_the_allotted_tier(self):
        # a discrete GPU grows the tier before it reports the allotment: both lines carry the grown tier
        log = ['x', '[kimi_vram_elastic] event=decode_grow generation=0 status=pass epoch=1 token=1 decode_token=0 '
               'us=10 before_slots=1100 target_slots=1116 after_slots=1116 added_slots=16 blocks=70',
               '[kimi_vram_tier] reason=allocated budget_bytes=21000000000 capacity_bytes=20998193152 slots=1116 hits=0',
               '[kimi_vram_elastic] event=watermark_grow token=3 free_before=1 high_bytes=1 added_slots=4 slots=1120 grows=1']
        self.assertEqual(ubprobe.reached(log), (1116, 1120))
        self.assertEqual(ubprobe.reached(['[kimi_vram_tier] reason=allocated budget_bytes=21000000000 '
                                          'capacity_bytes=20998193152 slots=1116 hits=0']), (1116, 1116))

    def test_no_tier(self):
        self.assertEqual(ubprobe.reached(['llama_context: n_ubatch = 32768']), (None, None))

    def test_the_bootstrap_reads_the_same_quantity(self):
        a = logparse.ArmParser()
        for line in self.UMA:
            a.feed(line)
        self.assertEqual(a.tier_first, ubprobe.reached(self.UMA)[0])


if __name__ == '__main__':
    unittest.main()
