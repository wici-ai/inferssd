"""Read one arm: the server log (sink trace/profile, tier and admission lines) and the client's token times."""
import gzip
import json
import re
import statistics
from collections import defaultdict

from prefill import OFFLOAD_MIN


def _lines(path):
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt', errors='replace') as f:
        yield from f


def _kv(line, key):
    m = re.search(rf'\b{key}=([-\d.]+)', line)
    return float(m.group(1)) if m else None


class ArmParser:
    """The state read_arm builds from a server log, fed line by line, so a live server's log can be read in steps."""

    def __init__(self):
        # a decode token's key is ((epoch, run), position): see decode_key
        self.trace = defaultdict(dict)       # key -> layer -> (selected, vram_hit, cache_hit)
        self.prof = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])   # total, wait_io, compute, other, layers
        self.lprof = {}                      # (key[0], key[1], layer) -> (total_us, compute_wall_us)
        self.lio = defaultdict(lambda: [0.0, 0.0])   # key -> decode read time (us), bytes read
        self.last_decode = None
        self.steps = {}                      # per line kind: the last decode layer (epoch, position, layer), its run
        self.adm, self.threads, self.host, self.slots, self.mrc = [], None, None, [], None
        self.tier_min_count = None
        self.tier_bytes = None              # (bytes per tier slot, base budget bytes) from the first allocated tier line
        self.tier_first = None               # the tier after its first decode growth, else as first allotted (ubprobe)
        self.tier_grown = False
        self.pf_layers = []                  # [kimi_cpu_prefill_layer] records in log order
        self.pf_graphs = []                  # prefills: tokens, the sink's compute and reads, the graph's expert copy
        self.memory = {}                     # the device at load: free, dense weights, KV cache, compute buffers
                                             # (and the host's compute buffer, the same DRAM on unified memory);
                                             # reserves: each graph reserve [tokens, device MiB, host MiB, graph]
        self.reserve = None                  # the graph reserve being read
        self.sink_max = None                 # the sink's threshold (max_sink_tokens) of the route lines so far
        self.unrouted = []                   # reserves read before the first route line
        self.checkpoint_align = None         # the memory saves an earlier state after each prompt batch, at multiples
        self.ledger = None                   # the host-memory ledger at destroy: unified, margin, margin_updates
        self.prefetch = None                 # the next-layer prefetch's last report: depth, window, per-rank counts
        self.ram_slots = None                # the RAM expert cache's slots when it first sized itself
        self.cache_load = None               # the RAM cache's storage at load: slots, bytes committed, us
        self.new_slot_bytes = 0.0            # bytes of RAM cache slots decode committed (first touches)
        self.new_slot = defaultdict(float)   # key -> those bytes of one decode token
        self.ram_slot_bytes = self.ram_costed = None
        self.decode_place = None             # the decode plan's summary at destroy (kimi-cpu-sink.cpp place_report)
        self.host_register = None            # the RAM cache's page-locking at destroy (the decode registrar)

    def decode_key(self, kind, r):
        """The key of the decode token of trace or profile record r.  The engine numbers a token by its position in
        an epoch, and a request whose prompt does not extend the previous prompt and reply (the server cuts its
        cache back to what they share) decodes positions of the epoch again: a layer at a position not past the
        last one's starts a run of its own, so that the two requests' tokens are not merged."""
        e, t, layer = r['epoch'], r['token'], r['layer']
        last, run = self.steps.get(kind, (None, 0))
        if last and last[0] == e and (t < last[1] or (t == last[1] and layer <= last[2])):
            run += 1
        self.steps[kind] = ((e, t, layer), run)
        return (e, run), t

    def feed(self, line):
        i = line.find('[kimi_')
        if i < 0:
            self._feed_other(line)
            return
        line = line[i:]
        if line.startswith('[kimi_cpu_sink_trace] '):
            if '"phase":"decode"' not in line[:120]:
                return                    # prefill traces: long, and only decode's routing is used
            r = json.loads(line[22:])
            if r.get('phase') == 'decode' and r.get('event') == 'layer':
                self.trace[self.decode_key('trace', r)][r['layer']] = (r['selected_experts'], r['vram_hit'], r['cache_hit'])
        elif line.startswith('[kimi_cpu_sink_profile] '):
            head = line[24:].split(',"workers"')[0].rstrip()
            try:
                r = json.loads(head if head.endswith('}') else head + '}')
            except json.JSONDecodeError:
                return
            if r.get('phase') == 'prefill' and r.get('event') == 'layer' and self.pf_graphs:
                self.pf_graphs[-1]['cpu_s'] += r['compute_wall_us'] / 1e6     # the sink's CPU compute in this prefill
            if r.get('phase') == 'decode' and r.get('event') == 'layer':
                k = self.decode_key('profile', r)
                p = self.prof[k]
                p[0] += r['total_us']; p[1] += r['wait_io_us']; p[2] += r['compute_wall_us']; p[3] += r['other_wall_us']
                p[4] += 1
                self.lprof[(*k, r['layer'])] = (r['total_us'], r['compute_wall_us'])
                self.last_decode = k
            else:
                self.last_decode = None
        elif line.startswith('[kimi_cpu_sink_layer] ') and self.last_decode:
            # the decode layer's own reads (after its profile line): drive service time per byte
            d = self.lio[self.last_decode]
            d[0] += _kv(line, 'io_us') or 0.0
            d[1] += _kv(line, 'bytes') or 0.0
            self.new_slot_bytes += _kv(line, 'cache_new_slot_bytes') or 0.0
            self.new_slot[self.last_decode] += _kv(line, 'cache_new_slot_bytes') or 0.0
            self.last_decode = None
        elif line.startswith('[kimi_vram_admission]'):
            self.adm.append({k: _kv(line, k) for k in ('min_count', 'upload_us', 'miss_us', 'hit_us')})
        elif line.startswith('[kimi_cpu_sink_threads] decode_participants='):
            m = re.search(r'decode_participants=(\d+) us_per_expert=([\d.]+) \((\d+) participants: ([\d.]+)\)', line)
            if m:
                self.threads = {'participants': int(m.group(1)), 'us_per_expert': float(m.group(2)),
                           'other': int(m.group(3)), 'other_us_per_expert': float(m.group(4))}
        elif line.startswith('[kimi_cpu_sink_threads] participants='):
            m = re.search(r'participants=(\d+) fast_cluster_cpus=(\d+) decode_candidates=(\d+),(\d+)', line)
            if m and self.threads is None:
                self.threads = {'participants': int(m.group(3)) if m.group(3) == m.group(4) else None,
                           'candidates': [int(m.group(3)), int(m.group(4))]}
        elif line.startswith('[kimi_vram_host]') and 'resident_us_per_expert' in line:
            self.host = {'resident_us': _kv(line, 'resident_us_per_expert'), 'host_us': _kv(line, 'host_us_per_expert')}
            # host API v2: in place or copied, the page-locked copy rate (us per MB) and the jobs
            for k in ('in_place', 'jobs', 'copies', 'reserved_copies', 'refused', 'copy_us_per_mb', 'resident_call_us',
                      'resident_expert_us', 'host_call_us', 'host_expert_us'):
                if _kv(line, k) is not None:
                    self.host[k] = _kv(line, k)
        elif line.startswith('[kimi_decode_place] reason='):
            self.decode_place = {k: _kv(line, k) for k in (
                'layers', 'tokens', 'gpu_experts', 'gpu_per_token', 'planned', 'candidates', 'refused', 'wanted',
                'error_p50_pct', 'error_p90_pct', 'predicted_ms', 'measured_ms', 'cpu_only_ms', 'cpu_expert_us', 'early',
                'fixed_us', 'decode_read_gbs', 'cpu_alone_us', 'cpu_shared_us', 'cpu_moved')}
        elif line.startswith('[kimi_host_register] reason=destroy'):
            self.host_register = {k: _kv(line, k) for k in ('registered', 'failures', 'unregistered', 'signals', 'wanted',
                                                            'us_mean', 'us_max', 'us_total')}
        elif line.startswith('[kimi_vram_tier] reason=allocated') and self.tier_bytes is None:
            c, n, b = _kv(line, 'capacity_bytes'), _kv(line, 'slots'), _kv(line, 'budget_bytes')
            if c and n:
                self.tier_bytes = (c / n, b)
            if n and not self.tier_grown:
                self.tier_first = int(n)
            self.tier_min_count = _kv(line, 'min_count')    # the fixed admission point of an engine without adaptive admission
        elif line.startswith('[kimi_vram_elastic]') and 'slots=' in line:
            v = _kv(line, 'slots')
            if v is not None:
                self.slots.append(v)
            if line.startswith('[kimi_vram_elastic] event=decode_grow') and not self.tier_grown:
                g = _kv(line, 'after_slots')
                if g is not None:
                    self.tier_first, self.tier_grown = int(g), True
        elif line.startswith('[kimi_host_ledger] reason=destroy'):
            self.ledger = {k: _kv(line, k) for k in ('unified', 'margin', 'margin_updates')}
        elif line.startswith('[kimi_cpu_sink_prefetch] enabled'):
            # the depth the engine runs (it starts its readers, and prints this, from depth 1)
            self.prefetch = dict(self.prefetch or {}, topk=int(_kv(line, 'topk')), readers=int(_kv(line, 'readers')))
        elif line.startswith('[kimi_cpu_sink_prefetch] reason='):
            # cumulative over the run's decode: the last report (destroy, else the latest periodic) holds it all
            r = {k: _kv(line, k) for k in ('topk', 'layers', 'issued', 'windows', 'window_us_p50', 'window_us_mean',
                                           'reader_bytes', 'reader_busy_us', 'width', 'skip_vram', 'vram_late',
                                           'skip_vram_lost', 'spill_us', 'spills')}
            for k in ('predicted_r', 'novel_r', 'novel_routed_r', 'issued_r', 'used_r', 'started_r', 'spill_us_r'):
                m = re.search(rf'\b{k}=([\d,]+)', line)
                r[k] = [int(x) for x in m.group(1).split(',')] if m else None
            self.prefetch = dict(self.prefetch or {}, **{k: v for k, v in r.items() if v is not None})
        elif line.startswith('[kimi_cpu_sink_cache] reason=load '):
            self.cache_load = {k: _kv(line, k) for k in ('slots', 'slot_bytes', 'committed_bytes', 'us')}
        elif line.startswith('[kimi_cpu_sink_cache] auto_budget=') and self.ram_slot_bytes is None:
            # its first size: on unified memory the budget (the slots the prefetch lands in), elsewhere the cgroup's
            # room less what admission costs (the line after)
            self.ram_slot_bytes = _kv(line, 'slot_bytes') or 0.0
            if self.ram_slot_bytes:
                self.ram_slots = int(_kv(line, 'auto_budget') // self.ram_slot_bytes)
        elif line.startswith('[kimi_cpu_sink_cache] auto_capacity_bytes=') and self.ram_slot_bytes and not self.ram_costed:
            self.ram_slots, self.ram_costed = int(_kv(line, 'auto_capacity_bytes') // self.ram_slot_bytes), True
        elif line.startswith('[kimi_cpu_sink_mrc] '):
            self.mrc = json.loads(line[20:])
        elif line.startswith(('[kimi_cpu_prefill_graph] event=activate ', '[kimi_cpu_prefill_graph] event=fallback ')):
            # a prefill's graph at its own size, or at a smaller one that fits (fallback: the prompt in passes)
            self.pf_graphs.append({'n_tokens': int(_kv(line, 'requested_tokens') or 0),
                                   'ubatch': int(_kv(line, 'configured_max') or 0), 'cpu_s': 0.0, 'sink_bytes': 0.0,
                                   'graph_bytes': 0.0, 'graph_read_us': 0.0, 'graph_passes': 0, 'graph_tokens': 0})
        elif line.startswith('[kimi_cpu_prefill_layer] '):
            self.pf_layers.append({k: _kv(line, k) for k in ('layer', 'n_tokens', 'unique_experts', 'routed_slots', 'gpu_experts',
                                                         'gpu_ssd_transient_experts', 'ssd_experts', 'ssd_bytes')})
            if self.pf_graphs:
                self.pf_graphs[-1]['sink_bytes'] += self.pf_layers[-1]['ssd_bytes'] or 0.0   # what the sink read
        elif line.startswith('[kimi_prefill_route] '):
            self._reserve(line)

    def _reserve(self, line):
        """Each graph reserve: [prefill_reserve_outputs] n_tokens=N, then the device's and the host's compute buffers;
        the graph route's (it copies the experts to the device) when N exceeds the sink's threshold, the
        max_sink_tokens of the engine's route lines ('all': every pass the sink's), which the reserve at load precedes
        and every request's follows, and the GPU takes the experts' product at N tokens (OFFLOAD_MIN)."""
        if line.startswith('[kimi_prefill_route] '):
            m = re.search(r'\bmax_sink_tokens=(\w+)', line)
            if m:
                self.sink_max = float('inf') if m.group(1) == 'all' else int(m.group(1))
                for x in self.unrouted:
                    x[3] = self._graph_pass(x[0])
                self.unrouted = []
            return
        if line.startswith('[prefill_reserve_outputs] ') and _kv(line, 'n_tokens') is not None:
            self.reserve = {'n_tokens': int(_kv(line, 'n_tokens'))}
            return
        r = self.reserve
        if r is None:
            return
        m = re.search(r'(\S+) compute buffer size = +([\d.]+) MiB', line)
        if m:
            r['device_mib' if m.group(1) == self.memory.get('device') else 'host_mib'] = float(m.group(2))
        if 'graph splits = ' in line:
            n = r['n_tokens']
            x = [n, r.get('device_mib', 0.0), r.get('host_mib', 0.0), None if self.sink_max is None else self._graph_pass(n)]
            seen = self.memory.setdefault('reserves', [])
            if x[:3] not in [y[:3] for y in seen]:     # the same reserve again (every decode token's): once
                seen.append(x)
                seen.sort(key=lambda y: y[:3])
                if x[3] is None:
                    self.unrouted.append(x)
            self.reserve = None

    def _graph_pass(self, n):
        return int(n > self.sink_max and n >= OFFLOAD_MIN)

    def _feed_other(self, line):
        self._reserve(line)
        if 'context checkpoints are taken after each prompt batch' in line:
            m = re.search(r'at multiples of (\d+) tokens', line)
            self.checkpoint_align = int(m.group(1)) if m else None
        elif line.startswith('[moe_prefill_copy] '):
            if self.pf_graphs:          # the graph route's expert copy, one line per prefill
                g = self.pf_graphs[-1]
                g['graph_bytes'] += _kv(line, 'bytes') or 0.0
                g['graph_read_us'] += _kv(line, 'read_us') or 0.0
                g['graph_passes'] += int(_kv(line, 'passes') or 0)
                g['graph_tokens'] += int(_kv(line, 'tokens') or 0)
        elif self.memory.get('compute') == 'device':
            # the host's compute buffer, printed right after the device's (a line of any other kind ends the load)
            m = re.search(r'(\S+) compute buffer size = +([\d.]+) MiB', line)
            if m and m.group(1) != self.memory.get('device'):
                self.memory['host_compute_mib'] = float(m.group(2))
            self.memory['compute'] = True
        elif 'compute' not in self.memory:
            # llama.cpp's lines at load, up to the first compute buffers (the configured ubatch's)
            m = re.search(r'using device (\S+) .* - (\d+) MiB free', line)
            if m and 'free_mib' not in self.memory:
                self.memory.update(device=m.group(1), free_mib=int(m.group(2)))
            m = re.search(r'(\S+) (model|KV|compute) buffer size = +([\d.]+) MiB', line)
            if m and m.group(1) == self.memory.get('device'):
                key = {'model': 'dense_mib', 'KV': 'kv_mib', 'compute': 'compute_mib'}[m.group(2)]
                self.memory[key] = self.memory.get(key, 0.0) + float(m.group(3))
                if key == 'compute_mib':
                    self.memory['compute'] = 'device'
            m = re.search(r'n_ubatch += (\d+)', line)
            if m:
                self.memory['n_ubatch'] = int(m.group(1))

    def result(self, run_dir, skip=128):
        # requests in order: tokens of one epoch and run sorted by position, a new request where positions jump
        ks = sorted(self.trace)
        groups = []
        for k in ks:
            if groups and groups[-1][-1][0] == k[0] and k[1] == groups[-1][-1][1] + 1:
                groups[-1].append(k)
            else:
                groups.append([k])
        segments = [[self.trace[k] for k in g] for g in groups]
        window = [k for g in groups for k in g[skip:]]
        clients = [json.loads(x) for x in open(f'{run_dir}/client.jsonl')] if _exists(f'{run_dir}/client.jsonl') else []
        # the server's load: from its start to its health check passing (arm.sh's meta.txt)
        meta = open(f'{run_dir}/meta.txt').read() if _exists(f'{run_dir}/meta.txt') else ''
        load_s = re.search(r'\bload_s=([\d.]+)', meta)
        # decode's new RAM cache slots per window of each request: its tokens 1-128, then the rest
        new_slot = [0.0, 0.0]
        firsts = {}
        for (run, t) in sorted(self.new_slot):
            firsts.setdefault(run, t)
            new_slot[t - firsts[run] >= 128] += self.new_slot[(run, t)]
        # prefill passes: one per ubatch, layers 0..L-1 in order; the engine's 2-token warm-up pass is dropped
        passes = []
        for r in self.pf_layers:
            if not passes or r['layer'] <= passes[-1][-1]['layer']:
                passes.append([])
            passes[-1].append(r)
        prefill = []
        for ps in passes:
            if ps[0]['n_tokens'] < 64:
                continue
            prefill.append({'n_tokens': int(ps[0]['n_tokens']), 'layers': len(ps),
                            'unique': sum(r['unique_experts'] for r in ps), 'tier': sum(r['gpu_experts'] for r in ps),
                            'routes': sum(r['routed_slots'] for r in ps), 'ssd_bytes': sum(r['ssd_bytes'] for r in ps),
                            'unique_per_layer': [int(r['unique_experts']) for r in ps]})
        return {'cache_load': self.cache_load, 'new_slot_bytes': self.new_slot_bytes, 'new_slot_windows': new_slot,
                'load_s': float(load_s.group(1)) if load_s else None,
                'segments': segments, 'groups': groups, 'window': window, 'lprof': self.lprof, 'prefill': prefill, 'trace': self.trace, 'prof': self.prof, 'admission': self.adm,
                'threads': self.threads, 'host': self.host, 'tier_slots_max': max(self.slots) if self.slots else None, 'tier_slots_first': self.tier_first, 'tier_bytes': self.tier_bytes, 'mrc': self.mrc, 'tier_min_count': self.tier_min_count, 'lio': dict(self.lio),
                'client': clients, 'pf_graphs': [g for g in self.pf_graphs if g['n_tokens'] >= 64],
                'memory': {k: v for k, v in self.memory.items() if k != 'compute'},
                'checkpoint_align': self.checkpoint_align, 'ledger': self.ledger, 'prefetch': self.prefetch,
                'ram_slots': self.ram_slots, 'decode_place': self.decode_place, 'host_register': self.host_register}


def read_arm(run_dir, skip=128, parser=None):
    """Decode tokens after `skip` per request: hits, sink time, wall time; engine-measured unit costs.  parser: an
    ArmParser already fed with the log (a live server's, read in steps)."""
    if parser is None:
        parser = ArmParser()
        log = f'{run_dir}/server.log.gz' if _exists(f'{run_dir}/server.log.gz') else f'{run_dir}/server.log'   # live server: plain
        for line in _lines(log):
            parser.feed(line)
    return parser.result(run_dir, skip)


def ledger(run_dir):
    """The host-memory ledger's record at destroy in a server's log (None without one, or without a log)."""
    log = f'{run_dir}/server.log.gz' if _exists(f'{run_dir}/server.log.gz') else f'{run_dir}/server.log'
    rec = None
    if _exists(log):
        for line in _lines(log):
            if line.startswith('[kimi_host_ledger] reason=destroy'):
                rec = {k: _kv(line, k) for k in ('unified', 'margin', 'margin_updates')}
    return rec


def _exists(p):
    try:
        open(p).close()
        return True
    except OSError:
        return False


def hits(arm, keys=None):
    v = r = m = 0
    for k in keys if keys is not None else arm['window']:
        for sel, vh, ch in arm['trace'][k].values():
            for a, b in zip(vh, ch):
                if a:
                    v += 1
                elif b:
                    r += 1
                else:
                    m += 1
    n = v + r + m or 1
    return {'v': v / n, 'r': r / n, 'm': m / n, 'experts': n}


def window_rate(tok_s, first, last):
    """Tokens per second between decode tokens `first` and `last` (1-based, inclusive ends as milestones)."""
    if not tok_s or len(tok_s) < last or tok_s[first - 1] is None or tok_s[last - 1] is None:
        return None
    dt = tok_s[last - 1] - tok_s[first - 1]
    return (last - first) / dt if dt > 0 else None


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None
