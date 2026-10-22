#!/usr/bin/env python3
"""Small deterministic fixtures; never reads real model weights."""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

binary = Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory(prefix="mac-expert-source-") as directory:
    root = Path(directory)
    shards = []
    for index, size in enumerate((22001, 9012)):
        data = bytes((i * 17 + index * 23) % 256 for i in range(size))
        path = root / f"shard-{index}.gguf"
        path.write_bytes(data)
        shards.append((path, data))
    extents = [(0, 33, 8191), (1, 4099, 4913), (0, 21964, 37)]
    names = ["blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight", "blk.1.ffn_up_exps.weight"]
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

    def run(source, cache, expected_error=None, both=False, commands=None):
        env = os.environ.copy()
        env.pop("GGML_MOE_EXPERT_GGUF_ALIAS_TSV", None)
        env.pop("SSD_LLAMA_MAC_EXPERT_PACK", None)
        env.pop("SSD_LLAMA_MAC_EXPERT_TRACE", None)
        env["SSD_LLAMA_MAC_EXPERT_CACHE_MIB"] = str(cache)
        env["GGML_MOE_EXPERT_GGUF_ALIAS_TSV" if source.suffix == ".tsv" else "SSD_LLAMA_MAC_EXPERT_PACK"] = str(source)
        if both:
            env["SSD_LLAMA_MAC_EXPERT_PACK"] = str(pack)
        result = subprocess.run([str(binary), *(commands if commands is not None else args + args)], env=env, text=True, capture_output=True)
        if expected_error:
            assert result.returncode != 0 and expected_error in result.stderr, (source, result.returncode, result.stderr)
        else:
            assert result.returncode == 0, (source, result.returncode, result.stderr)

    for source in (pack, alias):
        for cache in (0, 1):
            run(source, cache)
    bad = root / "bad.tsv"
    for body, message in [
        (header + "".join(rows) + rows[0], "duplicate"),
        (header + rows[0].replace("\t33\n", "\t22001\n"), "exceeds source"),
        (header + rows[0].replace("\t8191\t", "\t-1\t"), "unsigned integer"),
        ("bad header\n" + rows[0], "header"),
        (header, "invalid expert alias index"),
    ]:
        bad.write_text(body)
        run(bad, 1, message)
    run(alias, 1, "select one expert source", both=True)
    run(alias, 1, "short/failed expert-pack read", commands=args[:4] + ["--truncate", str(shards[1][0])] + args[4:8])
    pack.write_bytes(content[:-4090])
    run(pack, 1, "invalid expert-pack index")
print("12 source, bounds, short-read, duplicate, pin and cache cases passed")
