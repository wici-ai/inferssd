#!/usr/bin/env python3
"""CPU placements the bootstrap measures.
  all    every CPU, unpinned, the engine's own decode A/B (its default)
  split  per-phase pools: as many threads as 'all', pinned fastest group first; decode computes on the fastest
         group less one CPU, which keeps the I/O thread; prefill computes on all of them
         (GGML_KIMI_CPU_SINK_DECODE_PARTICIPANTS)
  pin    the fastest group less one CPU for every phase and that last CPU for the I/O thread (prefill loses the
         other groups)
'split' and 'pin' exist on a machine with more than one CPU group and at least 3 CPUs in the fastest; 'split' also
needs an engine with per-phase pools (the model's conf lists the placements its engine supports).
THREADS is the DeepSeek launcher's; GGML_KIMI_CPU_SINK_THREADS reaches a launcher that fixes its own (EXTRA_ENV).
usage: placement.py device.json NAME  -> the launcher environment ('' if none)"""
import json
import os
import sys


def parse_list(s):
    out = []
    for part in s.strip().split(','):
        if '-' in part:
            a, b = part.split('-')
            out += range(int(a), int(b) + 1)
        elif part:
            out.append(int(part))
    return out


def default_threads(dev):
    """The launcher's thread count: the P and E cores where the kernel lists both (an Intel hybrid CPU's low-power
    island is left out), else every CPU."""
    if dev.get('default_threads'):
        return int(dev['default_threads'])
    try:
        cpus = set()
        for f in ('/sys/devices/cpu_core/cpus', '/sys/devices/cpu_atom/cpus'):
            cpus |= set(parse_list(open(f).read()))
        if cpus:
            return len(cpus)
    except OSError:
        pass
    return os.cpu_count()


def groups_of(dev):
    groups, i = [], 1
    while f'g{i}' in dev['coresets']:
        groups.append(dev['coresets'][f'g{i}']['cpus'])
        i += 1
    return groups


def pin_env(dev):
    groups = groups_of(dev)
    if len(groups) < 2 or len(groups[0]) < 3:
        return ''
    cpus, io = groups[0][:-1], groups[0][-1]
    return (f'GGML_KIMI_CPU_SINK_DECODE_THREAD_CHOICE=0 THREADS={len(cpus) + 1} GGML_KIMI_CPU_SINK_THREADS={len(cpus) + 1} '
            f"GGML_KIMI_CPU_SINK_COMPUTE_CPUS={','.join(map(str, cpus))} GGML_KIMI_CPU_SINK_IO_CPUS={io}")


def split_env(dev):
    groups = groups_of(dev)
    if len(groups) < 2 or len(groups[0]) < 3:
        return ''
    t = default_threads(dev)
    order = [c for g in groups for c in g][:t]
    io = groups[0][-1]
    compute = [c for c in order if c != io]
    if len(compute) != t - 1:
        return ''
    return (f"THREADS={t} GGML_KIMI_CPU_SINK_THREADS={t} GGML_KIMI_CPU_SINK_COMPUTE_CPUS={','.join(map(str, compute))} "
            f"GGML_KIMI_CPU_SINK_IO_CPUS={io} GGML_KIMI_CPU_SINK_DECODE_PARTICIPANTS={len(groups[0]) - 1}")


def env(dev, name):
    """name: a CPU placement, optionally '@N' for the engine's prefill ubatch (UBATCH=N: a longer prompt takes fewer
    passes, each reading the experts its tokens touch, at the cost of the prefill buffers' GPU memory)."""
    base, _, ub = name.partition('@')
    e = {'pin': pin_env, 'split': split_env}.get(base, lambda d: '')(dev)
    return ' '.join(x for x in (e, f'UBATCH={ub}' if ub else '') if x)


if __name__ == '__main__':
    print(env(json.load(open(sys.argv[1])), sys.argv[2]))
