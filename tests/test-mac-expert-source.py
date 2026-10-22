#!/usr/bin/env python3
"""Small deterministic fixtures; never reads real model weights."""
import csv
import os
import re
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

binary = Path(sys.argv[1]).resolve()
prefill_only = "--prefill-only" in sys.argv[2:]
case_count = 0
with tempfile.TemporaryDirectory(prefix="mac-expert-source-") as directory:
    root = Path(directory)
    shards = []
    for index, size in enumerate((1400001, 1300003)):
        data = bytes((i * 17 + index * 23) % 256 for i in range(size))
        path = root / f"shard-{index}.gguf"
        path.write_bytes(data)
        shards.append((path, data))
    extents = [(0, 33, 614401), (1, 4099, 614401), (0, 700003, 614401), (0, 1399964, 37)]
    names = ["blk.0.ffn_up_exps.weight"] * 3 + ["blk.1.ffn_up_exps.weight"]
    header = "tensor\texpert_idx\tnbytes\tsource_path\toffset\n"
    rows = []
    args = []
    payloads = []
    for index, (shard, offset, size) in enumerate(extents):
        path, data = shards[shard]
        expected = root / f"expected-{index}"
        payload = data[offset:offset + size]
        expected.write_bytes(payload)
        payloads.append(payload)
        args.extend((names[index], str(index), str(size), str(expected)))
        rows.append(f"{names[index]}\t{index}\t{size}\t{path}\t{offset}\n")
    alias = root / "source.tsv"
    alias.write_text(header + "".join(rows))
    pack = root / "source.pack"
    content = bytearray(4096)
    entries = []
    for index, payload in enumerate(payloads):
        entries.append(struct.pack("<128siIQQ", names[index].encode(), index, 0, len(content), len(payload)))
        content.extend(payload)
        content.extend(bytes((-len(content)) % 4096))
    content[:40] = struct.pack("<16sIIQQ", b"GGMLMOEPACKv1\0\0\0", 1, 40, len(entries), 4096)
    content[40:40 + len(entries) * 152] = b"".join(entries)
    pack.write_bytes(content)

    def run(source, cache, expected_error=None, both=False, commands=None, depth=0, alignment=1, trace=False):
        global case_count
        case_count += 1
        env = os.environ.copy()
        env.pop("GGML_MOE_EXPERT_GGUF_ALIAS_TSV", None)
        env.pop("SSD_LLAMA_MAC_EXPERT_PACK", None)
        env.pop("SSD_LLAMA_MAC_EXPERT_TRACE", None)
        env["SSD_LLAMA_MAC_EXPERT_CACHE_MIB"] = str(cache)
        env["SSD_LLAMA_MAC_EXPERT_READ_AHEAD"] = str(depth)
        env["SSD_LLAMA_MAC_EXPERT_READ_ALIGNMENT"] = str(alignment)
        if depth == 4 or trace:
            env["SSD_LLAMA_MAC_EXPERT_TRACE"] = str(root / "trace.txt")
        env["GGML_MOE_EXPERT_GGUF_ALIAS_TSV" if source.suffix == ".tsv" else "SSD_LLAMA_MAC_EXPERT_PACK"] = str(source)
        if both:
            env["SSD_LLAMA_MAC_EXPERT_PACK"] = str(pack)
        result = subprocess.run([str(binary), *(commands if commands is not None else args + args)], env=env, text=True, capture_output=True)
        if expected_error:
            assert result.returncode != 0 and expected_error in result.stderr, (source, result.returncode, result.stderr)
        else:
            assert result.returncode == 0, (source, result.returncode, result.stderr)
            stats = dict(re.findall(r"([a-z_]+)=([0-9.]+)", result.stderr.split("reason=shutdown", 1)[-1]))
            assert int(stats.get("peak_staging_bytes", "0")) <= 64 * 1024**2
            values = {key: int(stats[key]) for key in ("requests", "hits", "misses", "read_requests", "read_bytes", "evictions")}
            if trace:
                with (root / "trace.txt.requests.tsv").open() as traces:
                    values["decode_hits"] = [(row["tensor"], row["expert"], row["hit"]) for row in csv.DictReader(traces, delimiter="\t") if row["cache_updates"] == "1"]
            return values

    if not prefill_only:
        for source in (pack, alias):
            for cache in (0, 1):
                for alignment in (1, 4096, 16384):
                    reference = run(source, cache, depth=0, alignment=alignment)
                    for depth in (1, 2, 4):
                        assert run(source, cache, depth=depth, alignment=alignment) == reference, (source, cache, depth, alignment)
    for source in (pack, alias):
        for depth in (0, 2):
            reference = run(source, 2, depth=depth, alignment=4096, trace=True)
            mixed = run(source, 2, depth=depth, alignment=4096, trace=True,
                        commands=["--prefill"] + args + ["--decode"] + args + ["--prefill"] + args + ["--decode"] + args)
            assert mixed["decode_hits"] == reference["decode_hits"] and mixed["evictions"] == reference["evictions"], (source, depth, mixed, reference)
    if not prefill_only:
        run(alias, 24576)
        for invalid in (-1, " 1", "1x", 18446744073709551615):
            run(alias, invalid, "CACHE_MIB must be")
        bad = root / "bad.tsv"
        for body, message in [
            (header + "".join(rows) + rows[0], "duplicate"),
            (header + rows[0].replace("\t33\n", "\t1400001\n"), "exceeds source"),
            (header + rows[0].replace("\t614401\t", "\t-1\t"), "unsigned integer"),
            ("bad header\n" + rows[0], "header"),
            (header, "invalid expert alias index"),
        ]:
            bad.write_text(body)
            run(bad, 1, message)
        run(alias, 1, "select one expert source", both=True)
        for alignment in (1, 4096, 16384):
            for depth in (0, 2):
                shards[1][0].write_bytes(shards[1][1])
                run(alias, 1, "short/failed expert-pack read", commands=args[:4] + ["--truncate", str(shards[1][0])] + args[4:8], alignment=alignment, depth=depth)
        pack.write_bytes(content[:-4090])
        run(pack, 1, "invalid expert-pack index")

        # Three 40 MiB slices cannot all fit in the 64 MiB staging budget. Use
        # sparse zero-filled fixtures so this exercises backpressure cheaply.
        large = 40 * 1024**2
        large_source, large_expected, large_alias = (root / name for name in ("large.gguf", "large.expected", "large.tsv"))
        for path, length in ((large_source, large * 3), (large_expected, large)):
            with path.open("wb") as output:
                output.truncate(length)
        large_alias.write_text(header + "".join(
            f"blk.0.ffn_up_exps.weight\t{i}\t{large}\t{large_source}\t{i * large}\n" for i in range(3)))
        large_args = [word for i in range(3) for word in ("blk.0.ffn_up_exps.weight", str(i), str(large), str(large_expected))]
        reference = run(large_alias, 0, commands=large_args, depth=0)
        assert run(large_alias, 0, commands=large_args, depth=4) == reference
print(f"{case_count} source, pipeline, alignment, prefill-cache isolation, bounds, short-read, duplicate and pin cases passed")
