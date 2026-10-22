#!/usr/bin/env python3
"""Resumable parallel range downloader for the DeepSeek-V4.1-Flash Q8_0 GGUF shards.

usage: hfdl.py <endpoint> <outdir> <workers> [shard numbers...]
Each shard is preallocated and filled by fixed-size chunks; finished chunk ids are
appended to <file>.done so a restart skips them. When every chunk is in, the shard's
sha256 is checked against the Hugging Face LFS oid and <file>.ok is written.
"""
import hashlib, os, sys, threading, time, urllib.request, queue

REPO = "vcruz305/DeepSeek-V4.1-Flash-GGUF"
REV = "c4a085541cb53f67ee5e57b63d255e80cef286e7"
SHARDS = {
    1: (43322444352, "167304a8d1123d8a9b7d53a82c01aab659eba51be6d58763f769eb294f62617e"),
    2: (43316676000, "d9f95f3ec801d07bddb374b3943a7bdb484e48dd8b24c5eae9218b091e146f52"),
    3: (43316676032, "c53687cc23e2e32ab3ad5dbd8052de07801e12a8b3ce278967dfa6c2eb2983b4"),
    4: (43316676032, "ba84c194c41feb6f7fb65fb4a0a4e188ee2cb85516b86ba2818e16cec2017343"),
    5: (43316676032, "66fdac198238c64c7310de22e46a1f9a15284fbde65f6a020f6e94395155b8b5"),
    6: (43316676032, "19a2b0e85cce09a47c826d8a670f832792752fdb5a996445fee797e0ce0d100f"),
    7: (28877784064, "f04e8d27188e9a9cb4f7999a7fc099c5f52e043e23dd1b333a2294f1c1a0af84"),
    8: (104449677888, "dc05657987ace747fabdf6c9c9b498fd46253329c78e7e795533b1885c292330"),
    9: (104452537696, "f8fe6989ba5956981e1f896167b495b343ce184c24bfdff9eec981c8c541da9d"),
    10: (10268402880, "f7f00c58e282623f08e43544dc31581f0e03a527bd05330054bb6bc2cfce384b"),
}
CHUNK = 64 << 20

endpoint, outdir, workers = sys.argv[1].rstrip("/"), sys.argv[2], int(sys.argv[3])
order = [int(x) for x in sys.argv[4:]] or [10, 1, 2, 3, 4, 5, 6, 7, 8, 9]
os.makedirs(outdir, exist_ok=True)

def name(n):
    return f"DeepSeek-V4.1-Flash-Q8_0-{n:05d}-of-00010.gguf"

lock = threading.Lock()
redirect = {}  # shard -> (url, time)
got = [0]
t0 = time.time()

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

opener_noredir = urllib.request.build_opener(NoRedirect)

def resolve(n, force=False):
    with lock:
        u = redirect.get(n)
        if u and not force and time.time() - u[1] < 1800:
            return u[0]
    url = f"{endpoint}/{REPO}/resolve/{REV}/{name(n)}"
    try:
        with opener_noredir.open(urllib.request.Request(url, method="HEAD"), timeout=30):
            loc = url
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            loc = e.headers["Location"]
            if loc.startswith("/"):
                loc = endpoint + loc
        else:
            raise
    with lock:
        redirect[n] = (loc, time.time())
    return loc

jobs = queue.Queue()
state = {}
for n in order:
    size, sha = SHARDS[n]
    path = os.path.join(outdir, name(n))
    if os.path.exists(path + ".ok"):
        continue
    nchunks = (size + CHUNK - 1) // CHUNK
    done = set()
    if os.path.exists(path + ".done"):
        done = {int(x) for x in open(path + ".done").read().split()}
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    if os.fstat(fd).st_size != size:
        os.ftruncate(fd, size)
    state[n] = {"fd": fd, "left": nchunks - len(done), "path": path, "donef": open(path + ".done", "a")}
    for c in range(nchunks):
        if c not in done:
            jobs.put((n, c))

def verify(n):
    size, sha = SHARDS[n]
    path = state[n]["path"]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
    ok = h.hexdigest() == sha
    print(f"[verify] {name(n)} {'OK' if ok else 'MISMATCH ' + h.hexdigest()}", flush=True)
    if ok:
        open(path + ".ok", "w").write(sha + "\n")

verifiers = []

def worker():
    while True:
        try:
            n, c = jobs.get_nowait()
        except queue.Empty:
            return
        size = SHARDS[n][0]
        lo = c * CHUNK
        hi = min(size, lo + CHUNK) - 1
        for attempt in range(1000):
            try:
                url = resolve(n, force=attempt > 0 and attempt % 3 == 0)
                req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    if r.status != 206:
                        raise IOError(f"status {r.status}")
                    off = lo
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        os.pwrite(state[n]["fd"], b, off)
                        off += len(b)
                        with lock:
                            got[0] += len(b)
                if off != hi + 1:
                    raise IOError(f"short {off - lo} of {hi + 1 - lo}")
                break
            except Exception:
                time.sleep(min(30, 1 + attempt))
        else:
            print(f"[fail] {name(n)} chunk {c}", flush=True)
            continue
        with lock:
            st = state[n]
            st["donef"].write(f"{c}\n")
            st["donef"].flush()
            st["left"] -= 1
            finished = st["left"] == 0
        if finished:
            os.fsync(st["fd"])
            t = threading.Thread(target=verify, args=(n,))
            t.start()
            verifiers.append(t)

def report():
    last, lt = 0, time.time()
    while any(th.is_alive() for th in threads):
        time.sleep(30)
        now = time.time()
        with lock:
            g = got[0]
            left = {n: s["left"] for n, s in state.items() if s["left"]}
        print(f"[{time.strftime('%H:%M:%S')}] {g/1e9:.1f} GB in {now-t0:.0f}s, now {(g-last)/(now-lt)/1e6:.1f} MB/s, chunks left {left}", flush=True)
        last, lt = g, now

threads = [threading.Thread(target=worker) for _ in range(workers)]
for th in threads:
    th.start()
rep = threading.Thread(target=report, daemon=True)
rep.start()
for th in threads:
    th.join()
for t in verifiers:
    t.join()
print("[done]", flush=True)
