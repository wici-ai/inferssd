#!/usr/bin/env python3
"""Tier slots each prefill ubatch leaves (adaptive.sh step 2u): from each probe's server log the tier after its first
decode growth ('first': the first '[kimi_vram_elastic] event=decode_grow ... after_slots=N'; a tier that does not
grow in decode, the slots allotted at the first request, '[kimi_vram_tier] reason=allocated ... slots=N'), the
quantity logparse reads from a bootstrap as tier_slots_first, so that insitu.py compares the two alike; and the most
the tier held in the probe ('held', also after watermark growth, for the record).  null when the server did not get
there.  A discrete GPU grows the tier when it allots it at the first request (the allotted tier is the grown one); on
unified memory the tier starts from one block and grows only after the first decode token, so the probe's request
decodes past it (adaptive.sh).  On unified memory this reading reflects the buffers that stay allocated in decode
(the host's compute buffer); the GPU compute buffer of a larger ubatch is taken from the tier only when a graph
prefill runs, and decode refills it afterwards (design F_g), which this probe does not see.
usage: ubprobe.py PIPELINE_DIR UB...  -> JSON {ub: {"first": slots, "held": slots}}"""
import gzip
import json
import re
import sys

TIER = re.compile(r'\[kimi_vram_tier\] reason=allocated .*? slots=(\d+)'
                  r'|\[kimi_vram_elastic\] event=decode_grow .*? after_slots=(\d+)'
                  r'|\[kimi_vram_elastic\] event=watermark_grow .*? slots=(\d+)')


def reached(lines):
    """(first, held) of a server log's lines: the tier after its first decode growth (else as first allotted) and
    the most slots it held; (None, None) when it was never allotted."""
    first = held = None
    grown = False
    for line in lines:
        m = TIER.search(line)
        if not m:
            continue
        alloc, grow, mark = (int(g) if g else None for g in m.groups())
        n = alloc or grow or mark
        held = n if held is None else max(held, n)
        if grow is not None and not grown:
            first, grown = grow, True
        elif alloc is not None and first is None:
            first = alloc
    return first, held


def main():
    out, ubs = sys.argv[1], sys.argv[2:]
    res = {}
    for ub in ubs:
        try:
            run = open(f'{out}/ubprobe-{ub}.out').readline().strip()
            first, held = reached(gzip.open(f'{run}/server.log.gz', 'rt', errors='replace'))
            res[ub] = {'first': first, 'held': held} if held is not None else None
        except OSError:
            res[ub] = None
    print(json.dumps(res))


if __name__ == '__main__':
    main()
