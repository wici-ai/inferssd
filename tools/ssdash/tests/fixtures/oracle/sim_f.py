"""The routing replayed through the engine's GPU tier, and a token priced from the bootstrap's measured per-layer
table.

Tier (the sink's GPU tier): one pool over all layers holding whole experts, scored by selection counts that
decay with a 16-token half-life.  A missed expert is admitted once its decayed count reaches min_count, replacing
the lowest-scored resident expert if that one scores lower.
RamCache (the sink's RAM expert cache behind the tier, its count_decay2 policy and uniform allocation): per layer
a few slots of the experts the tier missed, scored by their selection counts among those misses, which halve every
2 tokens; a miss takes a free slot, else the slot of the lowest-scored expert (the least recently used of equals)
unless it scores lower itself; this token's hits and admissions are not displaced.
Price: see price().
"""
import heapq
import math


class Tier:
    def __init__(self, cap, min_count, half_life=16.0):
        self.cap, self.min, self.k = cap, min_count, 1.0 / half_life
        self.ls = {}        # key -> log2(count) + step / half_life: comparable across keys at any step
        self.res = set()
        self.heap = []      # (ls, key) of resident keys; entries whose ls is stale are skipped
        self.step = 0
        self.admissions = 0

    def count(self, key):
        v = self.ls.get(key)
        return 0.0 if v is None else 2.0 ** (v - self.step * self.k)

    def _bump(self, key):
        v = math.log2(self.count(key) + 1.0) + self.step * self.k
        self.ls[key] = v
        if key in self.res:
            heapq.heappush(self.heap, (v, key))

    def _add(self, key):
        self.res.add(key)
        heapq.heappush(self.heap, (self.ls[key], key))
        self.admissions += 1

    def access(self, layer, experts):
        keys = [(layer, e) for e in experts]
        for k in set(keys):
            self._bump(k)
        hits = [k in self.res for k in keys]
        if self.cap <= 0:
            return hits
        locked = {k for k, h in zip(keys, hits) if h}
        for k in sorted({k for k, h in zip(keys, hits) if not h}, key=lambda k: -self.ls[k]):
            if self.count(k) < self.min:
                continue
            if len(self.res) < self.cap:
                self._add(k)
                locked.add(k)
                continue
            held, victim = [], None
            while self.heap:
                v, x = heapq.heappop(self.heap)
                if x not in self.res or self.ls[x] != v:
                    continue
                if x in locked:
                    held.append((v, x))
                    continue
                victim = (v, x)
                break
            for h in held:
                heapq.heappush(self.heap, h)
            if victim is None:
                continue
            if self.ls[k] < victim[0]:
                heapq.heappush(self.heap, victim)
                continue
            self.res.discard(victim[1])
            self._add(k)
            locked.add(k)
        return hits


class RamCache:
    """One layer of the RAM expert cache (module doc).  A count is kept as log2(count) + step / 2, comparable
    across experts at any step, as Tier keeps its scores."""

    def __init__(self, slots):
        self.slots, self.ls, self.res, self.clock = slots, {}, {}, 0    # res: expert -> last touch

    def access(self, step, experts):
        """The token at `step` (decode tokens since the start) misses `experts` in the tier: their hits here."""
        for e in dict.fromkeys(experts):
            c = 2.0 ** (self.ls[e] - step / 2.0) if e in self.ls else 0.0
            self.ls[e] = math.log2(c + 1.0) + step / 2.0
        hits = [e for e in experts if e in self.res]
        held = set(hits)
        for e in hits:
            self.clock += 1
            self.res[e] = self.clock
        for e in experts:
            if e in self.res or self.slots <= 0:
                continue
            if len(self.res) >= self.slots:
                free = [x for x in self.res if x not in held]
                if not free:
                    continue
                victim = min(free, key=lambda x: (self.ls[x], self.res[x]))
                if self.ls[e] < self.ls[victim]:
                    continue
                del self.res[victim]
            self.clock += 1
            self.res[e] = self.clock
            held.add(e)
        return len(hits)


class Replay:
    """simulate() carried on as segments are appended (the turns of a conversation): the tier's state stays between
    calls, so each call replays only the new segments.  seg_rows[i] is the slice of rows segment i produced.
    behind=True also keeps, per layer, the experts the tier missed at each token: the hits a RAM cache of any number
    of slots a layer behind the tier would get (behind_hits, RamCache)."""

    def __init__(self, cap, min_count, skip, behind=False):
        self.tier, self.skip = Tier(cap, min_count), skip
        self.rows, self.seg_rows, self.done = [], [], []
        self.uploads = self.v = self.n_exp = 0
        self.behind = {} if behind else None    # layer -> [(token step, the tier's misses, counted)]
        self.steps, self._behind_hits = 0, {}

    def advance(self, routing):
        if len(routing) < len(self.done) or any(len(s) != n for s, n in zip(routing, self.done)):
            raise ValueError('routing does not extend what was replayed')
        tier = self.tier
        for seg in routing[len(self.done):]:
            start = len(self.rows)
            for ti, tok in enumerate(seg):
                a0 = tier.admissions
                row = []
                for li, ex in enumerate(tok):
                    hit = tier.access(li, ex)
                    nv = sum(hit)
                    row.append((nv, len(ex) - nv))
                    if self.behind is not None:
                        self.behind.setdefault(li, []).append(
                            (self.steps, [e for e, h in zip(ex, hit) if not h], ti >= self.skip))
                tier.step += 1
                self.steps += 1
                if ti >= self.skip:
                    self.rows.append(row)
                    self.uploads += tier.admissions - a0
                    self.v += sum(r[0] for r in row)
                    self.n_exp += sum(len(ex) for ex in tok)
            self.seg_rows.append((start, len(self.rows)))
            self.done.append(len(seg))
            self._behind_hits = {}
        return self

    def behind_hits(self, slots):
        """Hits per token (after skip) of a RAM cache of `slots` per layer behind the tier."""
        if slots not in self._behind_hits:
            hits = 0
            for misses in self.behind.values():
                c = RamCache(slots)
                for step, experts, counted in misses:
                    n = c.access(step, experts)
                    hits += n if counted else 0
            self._behind_hits[slots] = hits / max(1, len(self.rows))
        return self._behind_hits[slots]

    def stats(self):
        return {'hit_v': self.v / (self.n_exp or 1), 'uploads_per_token': self.uploads / max(1, len(self.rows)),
                'tokens': len(self.rows), 'hits_per_token': self.v / max(1, len(self.rows))}


def simulate(routing, cap, min_count, skip):
    """routing: segments (requests of one conversation, cache kept between them) of tokens of per-layer expert
    lists.  Returns the per-token rows [(nv, nx) per layer] after `skip` tokens of each segment, and stats."""
    r = Replay(cap, min_count, skip).advance(routing)
    return r.rows, r.stats()


def lstsq(X, Y):
    """Least squares by the normal equations (small, dense), for the linear fit of the per-layer table's rare classes
    (insitu.py; price()): eliminated in column order without row exchanges (the matrix is positive semidefinite).  A
    column that the columns before it already span gets no coefficient: its pivot falls to the rounding error of its
    own scale.  The fit's columns are such a set: a layer's tier hits, ready and read experts add up to the model's
    top-k, so the constant and the three counts are dependent, and every least-squares solution prices each class
    alike."""
    n = len(X[0])
    A = [[sum(x[i] * x[j] for x in X) for j in range(n)] + [sum(x[i] * y for x, y in zip(X, Y))] for i in range(n)]
    scale = [A[i][i] for i in range(n)]
    for c in range(n):
        if A[c][c] <= 1e-9 * scale[c]:
            A[c] = [0.0] * (n + 1)
            continue
        for r in range(n):
            if r != c:
                f = A[r][c] / A[c][c]
                A[r] = [u - f * v for u, v in zip(A[r], A[c])]
    return [A[i][n] / A[i][i] if A[i][i] else 0.0 for i in range(n)]


def price(rows, ins, cpu_scale):
    """Mean ms per token for rows from simulate(), from the engine's measured per-layer response (insitu.py):
    a layer with nv tier hits and nx other experts has nr of those host-ready (prefetched, or in a RAM cache) with
    the bootstrap's own distribution of nr given nx (any mechanism); for an nx the bootstrap saw too rarely,
    Binomial(min(nx, pf_k), share) when readiness never exceeded pf_k a layer (prefetch only), else Binomial(nx,
    share of all non-tier experts). The rest are read from the SSD; a layer's time is the bootstrap's mean for its
    class (linear fit for rare classes), with the CPU compute part scaled for another CPU set; cpu_ms is that CPU
    compute part of the sink time (the rest waits on the drive or the GPU)."""
    table, beta = ins['layer_table'], ins['layer_beta']
    a, k = min(1.0, ins['prefetch_share']), ins['pf_k']
    rd = ins.get('ready_given_nx') or {}
    capped = all(j <= k for c in rd.values() for j, x in enumerate(c) if x)
    a_all = ins.get('ready_share_all', a)
    memo = {}

    seen = [e for e in table.values() if e[0] >= 5]
    cpu_share = sum(e[0] * e[2] for e in seen) / max(1e-9, sum(e[0] * e[1] for e in seen))

    def cls_ms(nv, nr, nm):
        # (layer time, its CPU compute part); a rare class: the linear fit, split at the table's CPU share
        e = table.get(f'{nv},{nr},{nm}')
        if e and e[0] >= 5:
            return e[1] + (cpu_scale - 1.0) * e[2], cpu_scale * e[2]
        t = beta[0] + beta[1] * nv + beta[2] * nr + beta[3] * nm + beta[4] * (1 if nm else 0)
        return t, cpu_share * t

    def ready_dist(nx):
        c = rd.get(str(nx))
        if c and sum(c) >= 20:
            return [x / sum(c) for x in c]
        m, q = (min(nx, k), a) if capped else (nx, a_all)
        return [math.comb(m, j) * q ** j * (1 - q) ** (m - j) for j in range(m + 1)]

    def layer_ms(nv, nx):
        key = (nv, nx)
        if key not in memo:
            c = [(pj, cls_ms(nv, j, nx - j)) for j, pj in enumerate(ready_dist(nx)) if pj]
            memo[key] = (sum(pj * t for pj, (t, _) in c), sum(pj * u for pj, (_, u) in c))
        return memo[key]

    tot = cpu = 0.0
    for row in rows:
        for nv, nx in row:
            t, u = layer_ms(nv, nx)
            tot += t
            cpu += u
    n = max(1, len(rows))
    sink = tot / n
    ms = ins['t_out_ms'] + sink
    return {'ms_per_token': ms, 'tok_s': 1000.0 / ms, 'sink_ms': sink, 'cpu_ms': cpu / n}
