#!/usr/bin/env bash
# One replay of a read plan (expert_source.py iolog) at queue depth QD, as a fio-shaped JSON report (jobs[0].read:
# io_bytes, bw_bytes, lat_ns.mean, ...).  The reader is ioreplay (probe/ioreplay.c), on every machine: it reads the
# way the engine reads experts (O_DIRECT, one read per widened role range, io_uring by raw syscalls), needs only a
# C compiler (the Spark has no fio), and transfers exactly the plan.  fio's replay of the same plans does not: on
# the Spark (2026-10-20, Muse plans of the probe's length, 180-576 reads of 9-11 MiB) it left one queue of
# reads uncounted at every depth (6, 18 and 48 reads: 61, 202 and 487 MB), and at those depths it reported
# 7.6-10.7 GiB/s with 1-20 ms longer read latencies where ioreplay moved every byte at 10.1-11.8 GiB/s.  At depth 1
# the two agree (7.66/8.25 vs 7.76/8.33 GiB/s, 1.23/1.14 vs 1.22/1.13 ms).  PROBE_READER=fio replays the same plan
# with fio, for a cross-check (pending on 176).
# usage: reads.sh IOLOG QD OUT.json      env: PROBE_READER (ioreplay|fio), IOREPLAY (the binary; built on first use)
set -uo pipefail
iolog=$1; qd=$2; out=$3
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ ${PROBE_READER:-ioreplay} == fio ]]; then
  exec fio --name="qd$qd" --read_iolog="$iolog" --replay_no_stall=1 --direct=1 --iodepth="$qd" --ioengine=io_uring \
    --thread --output-format=json --output="$out" > /dev/null 2>&1 < /dev/null
fi
IOREPLAY=${IOREPLAY:-${XDG_CACHE_HOME:-$HOME/.cache}/ssd-llama/ioreplay}
if [[ ! -x $IOREPLAY || $IOREPLAY -ot $here/probe/ioreplay.c ]]; then
  mkdir -p "$(dirname "$IOREPLAY")" && cc -O2 -o "$IOREPLAY.$$" "$here/probe/ioreplay.c" -lm &&
    mv "$IOREPLAY.$$" "$IOREPLAY" || exit 1
fi
"$IOREPLAY" "$iolog" "$qd" "qd$qd" > "$out.tmp" < /dev/null && mv "$out.tmp" "$out"
