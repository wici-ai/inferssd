#!/usr/bin/env python3
"""Resumable parallel range downloader driven by a manifest.

usage: dl2.py <manifest.json> <url_template> <outdir> <workers>
manifest: [{"path": "...", "size": N, "sha256": "..."}]; url_template contains {path}.
Files are preallocated and filled by 32 MiB chunks; finished chunk ids go to <file>.done, so a restart
skips them.  A chunk that stalls (no byte for 60 s, or 600 s in total) is retried on a new connection.
When a file is complete its sha256 is checked and <file>.ok is written.
"""
import hashlib, json, os, queue, sys, threading, time, urllib.request

manifest, template, outdir, workers = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3], int(sys.argv[4])
CHUNK = 32 << 20
os.makedirs(outdir, exist_ok=True)
lock = threading.Lock()
got = [0]
jobs = queue.Queue()
state = {}
for f in manifest:
    path = os.path.join(outdir, f["path"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path + ".ok"):
        continue
    n = max(1, (f["size"] + CHUNK - 1) // CHUNK)
    done = set()
    if os.path.exists(path + ".done"):
        done = {int(x) for x in open(path + ".done").read().split()}
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    if os.fstat(fd).st_size != f["size"]:
        os.ftruncate(fd, f["size"])
    state[f["path"]] = dict(fd=fd, left=n - len(done), path=path, donef=open(path + ".done", "a"), size=f["size"], sha=f["sha256"])
    for c in range(n):
        if c not in done:
            jobs.put((f["path"], c))
verifiers = []


def verify(key):
    st = state[key]
    h = hashlib.sha256()
    with open(st["path"], "rb") as fh:
        while True:
            b = fh.read(16 << 20)
            if not b:
                break
            h.update(b)
    ok = h.hexdigest() == st["sha"]
    print(f"[verify] {key} {'OK' if ok else 'MISMATCH ' + h.hexdigest()}", flush=True)
    if ok:
        open(st["path"] + ".ok", "w").write(st["sha"] + "\n")


def worker():
    while True:
        try:
            key, c = jobs.get_nowait()
        except queue.Empty:
            return
        st = state[key]
        lo = c * CHUNK
        hi = min(st["size"], lo + CHUNK) - 1
        url = template.format(path=key)
        for attempt in range(1000):
            try:
                if st["size"] == 0:
                    break
                req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=60) as r:
                    if r.status not in (200, 206) or (r.status == 200 and (lo != 0 or hi + 1 != st["size"])):
                        raise IOError(f"status {r.status}")
                    off = lo
                    while off <= hi:
                        b = r.read(min(1 << 20, hi + 1 - off))
                        if not b:
                            break
                        os.pwrite(st["fd"], b, off)
                        off += len(b)
                        with lock:
                            got[0] += len(b)
                        if time.time() - t0 > 600:
                            raise IOError("chunk too slow")
                if off != hi + 1:
                    raise IOError(f"short {off - lo}")
                break
            except Exception:
                time.sleep(min(20, 1 + attempt))
        with lock:
            st["donef"].write(f"{c}\n"); st["donef"].flush()
            st["left"] -= 1
            finished = st["left"] == 0
        if finished:
            os.fsync(st["fd"])
            t = threading.Thread(target=verify, args=(key,)); t.start(); verifiers.append(t)


def report():
    last, lt, t0 = 0, time.time(), time.time()
    while any(th.is_alive() for th in threads):
        time.sleep(30)
        now = time.time()
        with lock:
            g = got[0]
            left = sum(s["left"] for s in state.values())
        print(f"[{time.strftime('%H:%M:%S')}] {g/1e9:.1f} GB in {now-t0:.0f}s, now {(g-last)/(now-lt)/1e6:.1f} MB/s, chunks left {left}", flush=True)
        last, lt = g, now


threads = [threading.Thread(target=worker) for _ in range(workers)]
for th in threads:
    th.start()
threading.Thread(target=report, daemon=True).start()
for th in threads:
    th.join()
for t in verifiers:
    t.join()
print("[done]", flush=True)
