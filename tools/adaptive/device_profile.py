#!/usr/bin/env python3
"""Device profile from the raw probe outputs of probe.sh.  usage: device_profile.py PROBE_DIR EXPERT_SOURCE > device.json

ssd: per queue depth (role reads in flight) the read rate of every repeat and their mean, with the mean latency of
one read; ssd_knee_qd: QD*, the smallest depth whose best repeat reaches the worst repeat of the fastest depth (as
fast as the drive gets, within the repeat spread).  expert_files: every file that holds experts, read alone (GiB/s);
slow_files: those below half the median, which sit on degraded drive blocks and read normally again once rewritten
(copied and moved over the original).  Reports are fio's JSON or ioreplay's (the same fields)."""
import glob
import json
import os
import re
import sys

d = sys.argv[1]
experts = json.load(open(f'{sys.argv[2]}/experts.json'))
dev = {'schema': 2, 'expert_read_bytes': experts.get('expert_read_bytes'), 'bytes_per_role': experts.get('bytes_per_role')}

cpus = {}
for line in open(f'{d}/cpus.txt'):
    f = line.split()
    cpus[int(f[0])] = {'capacity': float(f[1]), 'max_khz': float(f[2])}
dev['cpus'] = cpus
mem = {}
for line in open(f'{d}/meminfo.txt'):
    k, v = line.split(':')
    mem[k] = int(v.split()[0]) / 2 ** 20
dev['mem_gib'] = {'total': round(mem['MemTotal'], 2), 'available': round(mem['MemAvailable'], 2)}

mp = open(f'{d}/memprobe.txt').read()
m = re.search(r'integrated=(\d)', mp)
dev['gpu'] = {'integrated': bool(int(m.group(1))) if m else None}
# the first (whole-expert) H2D line: page-locked and pageable, which the engine's copies run at once the RAM cache's
# slots are page-locked, and before (an admission upload from a pageable slot)
for key, pat in (('read_gbs', r'gpu_read_bw=([\d.]+)'), ('h2d_pinned_gbs', r'pinned=([\d.]+)'),
                 ('h2d_pageable_gbs', r'pageable=([\d.]+)')):
    m = re.search(pat, mp)
    dev['gpu'][key] = float(m.group(1)) if m else None


def read_report(p):
    try:
        txt = open(p).read()
        return json.loads(txt[txt.index('{'):])['jobs'][0]['read']
    except (OSError, ValueError, KeyError, IndexError):
        return None


runs = {}   # qd -> [(GiB/s, mean latency ms)]
for p in glob.glob(f'{d}/reads-qd*.json') + glob.glob(f'{d}/fio-qd*.json'):
    j = read_report(p)
    if j and j['bw_bytes'] > 0:
        runs.setdefault(int(re.search(r'qd(\d+)', p).group(1)), []).append((j['bw_bytes'] / 2 ** 30, j['lat_ns']['mean'] / 1e6))
ssd = {qd: {'gibs': round(sum(g for g, _ in r) / len(r), 3), 'lat_ms': round(sum(t for _, t in r) / len(r), 3),
            'gibs_runs': sorted(round(g, 3) for g, _ in r)} for qd, r in sorted(runs.items())}
dev['ssd'] = ssd
if ssd:
    fastest = max(ssd, key=lambda q: ssd[q]['gibs'])
    dev['ssd_knee_qd'] = min(q for q in ssd if ssd[q]['gibs_runs'][-1] >= ssd[fastest]['gibs_runs'][0])
files = {}
for line in open(f'{d}/files.txt') if os.path.exists(f'{d}/files.txt') else []:
    i, path = line.split(maxsplit=1)
    j = read_report(f'{d}/reads-file{i}.json') or read_report(f'{d}/fio-file{i}.json')
    if j:
        files[path.strip()] = round(j['bw_bytes'] / 2 ** 30, 3)
dev['expert_files'] = files
median = sorted(files.values())[len(files) // 2] if files else 0
dev['slow_files'] = [p for p, g in files.items() if g < 0.5 * median]

dev['coresets'] = {name: {'cpus': [int(c) for c in s.split(',')]}
                   for name, s in (line.split() for line in open(f'{d}/coresets.txt'))}
json.dump(dev, sys.stdout, indent=1)
