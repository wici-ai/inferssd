#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gguf-py"))

from gguf import GGUFReader  # noqa: E402

MAGIC = b"GGMLMOEPACKv1\0\0\0"
HEADER_STRUCT = struct.Struct("<16sIIQQ")
ENTRY_STRUCT = struct.Struct("<128siIQQ")
COPY_CHUNK = 16 * 1024 * 1024


@dataclass(frozen=True)
class SliceRef:
    file: Path
    tensor: str
    expert_idx: int
    offset: int
    nbytes: int


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(p) for p in glob.glob(pattern)]
        if matches:
            paths.extend(matches)
        else:
            paths.append(Path(pattern))
    unique = sorted({p.resolve() for p in paths})
    missing = [str(p) for p in unique if not p.is_file()]
    if missing:
        raise SystemExit(f"missing input GGUF file(s): {', '.join(missing)}")
    return unique


def tensor_sort_key(name: str) -> tuple[int, int, str]:
    layer_match = re.search(r"\bblk\.(\d+)\.", name)
    layer = int(layer_match.group(1)) if layer_match else 1_000_000
    if ".ffn_up_exps." in name:
        kind = 0
    elif ".ffn_gate_exps." in name:
        kind = 1
    elif ".ffn_down_exps." in name:
        kind = 2
    elif ".ffn_gate_up_exps." in name:
        kind = 3
    else:
        kind = 9
    return layer, kind, name


def slice_sort_key(slice_ref: SliceRef, layout: str) -> tuple[object, ...]:
    layer, kind, tensor = tensor_sort_key(slice_ref.tensor)
    if layout == "expert-major":
        return layer, slice_ref.expert_idx, kind, tensor
    if layout == "tensor-major":
        return layer, kind, tensor, slice_ref.expert_idx
    raise ValueError(f"unknown expert-pack layout: {layout}")


def collect_slices(inputs: list[Path], layout: str) -> list[SliceRef]:
    slices: list[SliceRef] = []
    tensor_patterns = (
        ".ffn_up_exps.",
        ".ffn_gate_exps.",
        ".ffn_down_exps.",
        ".ffn_gate_up_exps.",
    )

    for path in inputs:
        reader = GGUFReader(path, "r")
        tensors = sorted(
            (t for t in reader.tensors if any(p in t.name for p in tensor_patterns)),
            key=lambda t: tensor_sort_key(t.name),
        )
        for tensor in tensors:
            shape = [int(x) for x in tensor.shape.tolist()]
            if len(shape) < 3:
                continue
            n_expert = shape[2]
            if n_expert <= 0 or tensor.n_bytes % n_expert != 0:
                print(f"skip non-divisible expert tensor: {tensor.name}", file=sys.stderr)
                continue
            expert_bytes = tensor.n_bytes // n_expert
            for expert in range(n_expert):
                base = int(tensor.data_offset) + expert * expert_bytes
                if ".ffn_gate_up_exps." in tensor.name and expert_bytes % 2 == 0:
                    half = expert_bytes // 2
                    slices.append(SliceRef(path, tensor.name + ":gate", expert, base, half))
                    slices.append(SliceRef(path, tensor.name + ":up", expert, base + half, half))
                else:
                    slices.append(SliceRef(path, tensor.name, expert, base, expert_bytes))
    return sorted(slices, key=lambda s: slice_sort_key(s, layout))


def copy_slice(src_path: Path, src_offset: int, nbytes: int, out) -> None:
    with src_path.open("rb") as src:
        src.seek(src_offset)
        remaining = nbytes
        while remaining:
            chunk = src.read(min(COPY_CHUNK, remaining))
            if not chunk:
                raise RuntimeError(f"short read from {src_path} at {src_offset}")
            out.write(chunk)
            remaining -= len(chunk)


def write_pack(slices: list[SliceRef], output: Path, alignment: int) -> None:
    if not slices:
        raise SystemExit("no expert tensors found")
    if alignment != 4096:
        raise SystemExit("CPU sink expert packs require --align 4096")
    for s in slices:
        if len(s.tensor.encode("utf-8")) > 127:
            raise SystemExit(f"tensor name too long for pack index: {s.tensor}")

    index_size = ENTRY_STRUCT.size * len(slices)
    data_start = align_up(HEADER_STRUCT.size + index_size, alignment)
    entries: list[tuple[SliceRef, int]] = []

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("wb") as out:
        out.write(b"\0" * data_start)
        for i, s in enumerate(slices, 1):
            pos = out.tell()
            pad = align_up(pos, alignment) - pos
            if pad:
                out.write(b"\0" * pad)
            dst_offset = out.tell()
            copy_slice(s.file, s.offset, s.nbytes, out)
            tail = align_up(out.tell(), alignment) - out.tell()
            if tail:
                out.write(b"\0" * tail)
            entries.append((s, dst_offset))
            if i % 1024 == 0:
                print(f"packed {i}/{len(slices)} expert slices", file=sys.stderr)

        out.seek(0)
        out.write(HEADER_STRUCT.pack(MAGIC, 1, HEADER_STRUCT.size, len(entries), data_start))
        for s, dst_offset in entries:
            name = s.tensor.encode("utf-8")
            name_buf = name + b"\0" * (128 - len(name))
            out.write(ENTRY_STRUCT.pack(name_buf, s.expert_idx, 0, dst_offset, s.nbytes))

    shutil.move(str(tmp), str(output))


def inspect_pack(path: Path, limit: int) -> None:
    if not path.is_file():
        raise SystemExit(f"missing expert pack: {path}")
    with path.open("rb") as f:
        header = f.read(HEADER_STRUCT.size)
        if len(header) != HEADER_STRUCT.size:
            raise SystemExit("pack header is truncated")
        magic, version, header_size, n_entries, data_start = HEADER_STRUCT.unpack(header)
        if magic != MAGIC or version != 1:
            raise SystemExit("not a v1 MoE expert pack")
        print(f"version={version} header_size={header_size} entries={n_entries} data_start={data_start}")
        counts: dict[str, int] = {}
        total = 0
        shown = 0
        for _ in range(n_entries):
            raw = f.read(ENTRY_STRUCT.size)
            if len(raw) != ENTRY_STRUCT.size:
                raise SystemExit("pack index is truncated")
            name_raw, expert_idx, _reserved, offset, nbytes = ENTRY_STRUCT.unpack(raw)
            name = name_raw.split(b"\0", 1)[0].decode("utf-8")
            counts[name] = counts.get(name, 0) + 1
            total += nbytes
            if shown < limit:
                print(f"entry[{shown}] tensor={name} expert={expert_idx} offset={offset} nbytes={nbytes}")
                shown += 1
        print(f"payload_bytes={total} payload_gib={total / (1024**3):.2f} unique_tensors={len(counts)}")
        for name, count in sorted(counts.items())[:limit]:
            print(f"tensor_count tensor={name} entries={count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a contiguous sidecar pack for GGUF MoE expert tensors.")
    parser.add_argument("gguf", nargs="*", help="Input GGUF shard(s); globs are accepted.")
    parser.add_argument("-o", "--output", required=True, help="Output expert pack path.")
    parser.add_argument("--align", type=int, default=4096, help="Alignment for packed expert slices (CPU sink requires 4096).")
    parser.add_argument(
        "--layout",
        choices=("tensor-major", "expert-major"),
        default="expert-major",
        help="Physical expert-slice order (default: expert-major).",
    )
    parser.add_argument("--inspect", action="store_true", help="Inspect an existing expert pack instead of creating one.")
    parser.add_argument("--limit", type=int, default=16, help="Rows to show for --inspect.")
    args = parser.parse_args()

    if args.inspect:
        inspect_pack(Path(args.output), args.limit)
        return

    if args.align != 4096:
        raise SystemExit("CPU sink expert packs require --align 4096")

    if not args.gguf:
        raise SystemExit("at least one input GGUF file is required unless --inspect is used")

    if args.align <= 0 or args.align & (args.align - 1):
        raise SystemExit("--align must be a positive power of two")

    inputs = expand_inputs(args.gguf)
    slices = collect_slices(inputs, args.layout)
    total = sum(s.nbytes for s in slices)
    print(
        f"packing {len(slices)} expert slices from {len(inputs)} GGUF file(s), "
        f"{total / (1024**3):.2f} GiB, layout={args.layout}"
    )
    write_pack(slices, Path(args.output), args.align)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
