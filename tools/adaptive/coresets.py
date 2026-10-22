#!/usr/bin/env python3
"""Candidate CPU sets for the sink's decode workers, from this machine's CPU labels.

Labels (cpu_capacity, else the maximum frequency) only group CPUs that look alike; which placement is fastest is
measured by the bootstraps (placement.py builds them from these sets).  SMT siblings after the first are left
out.  Candidates, CPUs of a set fastest first: each group alone, the fastest group less one CPU (kept
for the I/O thread), and the fastest-first unions of groups up to all CPUs.
usage: coresets.py cpus.txt   (lines: cpu capacity max_khz thread_siblings)   ->   lines: name cpu,cpu,...
"""
import sys


def parse_list(s):
    out = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += range(int(a), int(b) + 1)
        elif part.strip():
            out.append(int(part))
    return out


cpus = []
for line in open(sys.argv[1]):
    f = line.split()
    cpu, cap, khz = int(f[0]), float(f[1]), float(f[2])
    sib = parse_list(f[3]) if len(f) > 3 and f[3] != '-' else [cpu]
    if min(sib) != cpu:
        continue  # a second hardware thread of a core already listed
    cpus.append((cap if cap > 0 else khz, cpu))
cpus.sort(key=lambda t: (-t[0], t[1]))
groups = []
for value, cpu in cpus:
    if groups and value >= 0.92 * groups[-1][0][0]:
        groups[-1].append((value, cpu))
    else:
        groups.append([(value, cpu)])
groups = [[c for _, c in g] for g in groups]
cands = []
for i, g in enumerate(groups):
    cands.append((f'g{i + 1}', g))
if len(groups[0]) >= 3:
    cands.append(('g1-1', groups[0][:-1]))
acc = []
for i, g in enumerate(groups):
    acc = acc + g
    if i >= 1:
        cands.append(('all' if i == len(groups) - 1 else '+'.join(f'g{j + 1}' for j in range(i + 1)), list(acc)))
seen = set()
for name, s in cands:
    key = tuple(sorted(s))
    if key in seen:
        continue
    seen.add(key)
    print(name, ','.join(map(str, s)))
