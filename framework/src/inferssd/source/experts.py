"""F source indexes, dense placement artifacts and fio text; no drive execution.

JSON mappings remain unvalidated legacy views. Metadata does not certify the
payload, full shard set, current native format support or Mac's strict alias path.
"""

from __future__ import annotations

import json
import random
import re
import struct
import sys
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Optional, Protocol, TypeVar, cast

from inferssd.source import header as gguf_header
from inferssd.source.files import LOCAL_FILES, SourceFiles
from inferssd.source.header import Tensor

Record = dict[str, Any]
Row = tuple[str, int, int, str, int]
T = TypeVar("T")


class RandomChoice(Protocol):
    def choice(self, values: Sequence[T]) -> T: ...


PAGE = 4096
ROLE = re.compile("^blk\\.(\\d+)\\.ffn_(gate|up|down)_exps\\.weight$")


def is_role(t: Tensor) -> bool:
    return bool(ROLE.match(t.name)) and len(t.shape) == 3


def gguf_tensors(model: str, *, filesystem: SourceFiles = LOCAL_FILES) -> list[Tensor]:
    """Routed tensors of every shard, one per layer and role, from headers."""
    return [
        t
        for h in gguf_header.read_model(model, filesystem=filesystem)
        for t in h.tensors
        if is_role(t)
    ]


def role_slice(tensors: Iterable[Tensor]) -> int:
    """Largest role slice among the routed tensors (one expert part of a role)."""
    return max(t.nbytes // t.shape[-1] for t in tensors if is_role(t))


def blocks(rows: int, nbytes: int, largest: int) -> Optional[int]:
    """Fewest equal whole-row blocks, each no larger than the largest role slice.

    Search n in [n0, 2*n0], where n0 = ceil(nbytes / largest), dividing rows.
    There is no short/padded final block; return None if no split exists.
    """
    n0 = max(1, -(-nbytes // largest))
    return next((n for n in range(n0, 2 * n0 + 1) if rows % n == 0), None)


def dense_facts(tensors: Iterable[Tensor], largest: int) -> list[Record]:
    out = []
    for t in tensors:
        m = re.match("^blk\\.(\\d+)\\.", t.name)
        rows = 1
        for d in t.shape[1:]:
            rows *= d
        out.append(
            {
                "name": t.name,
                "layer": int(m.group(1)) if m else None,
                "type": t.type_name,
                "bytes": t.nbytes,
                "rows": rows,
                "row_bytes": t.nbytes // rows,
                "n_blocks": blocks(rows, t.nbytes, largest) if len(t.shape) == 2 else None,
            }
        )
    return out


def gguf_entries(tensors: Iterable[Tensor], *, filesystem: SourceFiles = LOCAL_FILES) -> list[Row]:
    rows = []
    for t in tensors:
        n = t.shape[-1]
        nb = t.nbytes // n
        rows += [(t.name, e, nb, filesystem.abspath(t.path), t.offset + e * nb) for e in range(n)]
    return rows


def widened(o: int, nb: int) -> int:
    return (o + nb + PAGE - 1) // PAGE * PAGE - o // PAGE * PAGE


def pack_entries(pack: str, *, filesystem: SourceFiles = LOCAL_FILES) -> list[Row]:
    with filesystem.open(pack, "rb") as f:
        (magic, version, _, n, _) = struct.unpack("<16sIIQQ", f.read(40))
        assert magic.startswith(b"GGMLMOEPACKv1") and version == 1, f"{pack}: not an expert pack"
        raw = f.read(n * 152)
    path = filesystem.abspath(pack)
    rows = []
    for i in range(n):
        (name, e, _, off, nb) = struct.unpack_from("<128siIQQ", raw, i * 152)
        rows.append((name.split(b"\x00")[0].decode(), e, nb, path, off))
    return rows


def current(
    d: str, model: str, pack: Optional[str], *, filesystem: SourceFiles = LOCAL_FILES
) -> Optional[Record]:
    """Matching index newer than every source it indexes; otherwise None."""
    try:
        info = json.load(filesystem.open(f"{d}/experts.json"))
        made = filesystem.getmtime(f"{d}/experts.json")
        if info["model"] != filesystem.abspath(model) or info["pack"] != (
            pack and filesystem.abspath(pack)
        ):
            return None
        if info["alias"] and (not filesystem.isfile(info["alias"])):
            return None
        if any(filesystem.getmtime(p) > made for p in {filesystem.abspath(model), *info["files"]}):
            return None
        if "dense" not in info:
            return None
        return cast(Record, info)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def index(
    model: str,
    pack: Optional[str],
    d: str,
    *,
    filesystem: SourceFiles = LOCAL_FILES,
    emit: Callable[..., None] = print,
) -> None:
    info = current(d, model, pack, filesystem=filesystem)
    if info is None:
        emit(f"indexing the experts of {pack or model} into {d}", file=sys.stderr)
        info = build(model, pack, d, filesystem=filesystem)
    emit(json.dumps({k: v for (k, v) in info.items() if k not in ("files", "dense")}))


def build(
    model: str, pack: Optional[str], d: str, *, filesystem: SourceFiles = LOCAL_FILES
) -> Record:
    filesystem.makedirs(d, exist_ok=True)
    every = [t for h in gguf_header.read_model(model, filesystem=filesystem) for t in h.tensors]
    tensors = [t for t in every if is_role(t)]
    rows = (
        pack_entries(pack, filesystem=filesystem)
        if pack
        else gguf_entries(tensors, filesystem=filesystem)
    )
    assert rows, f"no routed expert tensors in {model}"
    if not pack:
        with filesystem.open(f"{d}/experts.tsv.tmp", "w") as f:
            f.write("tensor\texpert_idx\tnbytes\tsource_path\toffset\n")
            f.writelines(f"{t}\t{e}\t{nb}\t{p}\t{o}\n" for (t, e, nb, p, o) in rows)
        filesystem.replace(f"{d}/experts.tsv.tmp", f"{d}/experts.tsv")
    per_role: dict[str, int]
    types: dict[str, set[str]]
    expert: dict[tuple[int, int], int]
    read: dict[tuple[int, int], int]
    t: Any
    (per_role, types, expert, read) = ({}, {}, {}, {})
    for t in tensors:
        types.setdefault(cast(re.Match[str], ROLE.match(t.name)).group(2), set()).add(t.type_name)
    for t, e, nb, _, o in rows:
        m = cast(re.Match[str], ROLE.match(t))
        per_role[m.group(2)] = max(per_role.get(m.group(2), 0), nb)
        key = (int(m.group(1)), e)
        expert[key] = expert.get(key, 0) + nb
        read[key] = read.get(key, 0) + widened(o, nb)
    info = {
        "source": "pack" if pack else "gguf",
        "model": filesystem.abspath(model),
        "pack": pack and filesystem.abspath(pack),
        "alias": None if pack else filesystem.abspath(f"{d}/experts.tsv"),
        "roles": len(rows),
        "experts": 1 + max(e for (_, e, _, _, _) in rows),
        "layers": len({layer for (layer, _) in expert}),
        "bytes_per_role": dict(sorted(per_role.items())),
        "role_types": {r: sorted(v) for (r, v) in sorted(types.items())},
        "expert_bytes": max(expert.values()),
        "expert_read_bytes": max(read.values()),
        "total_bytes": sum(nb for (_, _, nb, _, _) in rows),
        "files": sorted({p for (*_, p, _) in rows}),
        "aligned_roles": sum(1 for (*_, o) in rows if o % PAGE == 0),
        "dense": dense_facts([t for t in every if not is_role(t)], role_slice(every)),
    }
    with filesystem.open(f"{d}/experts.json.tmp", "w") as f:
        json.dump(info, f, indent=1)
    filesystem.replace(f"{d}/experts.json.tmp", f"{d}/experts.json")
    return info


def load(d: str, *, filesystem: SourceFiles = LOCAL_FILES) -> tuple[Record, list[Row]]:
    info = json.load(filesystem.open(f"{d}/experts.json"))
    if info["pack"]:
        return (info, pack_entries(info["pack"], filesystem=filesystem))
    rows = []
    with filesystem.open(info["alias"]) as f:
        next(f)
        for line in f:
            (t, e, nb, p, o) = line.rstrip("\n").split("\t")
            rows.append((t, int(e), int(nb), p, int(o)))
    return (info, rows)


def iolog(
    d: str,
    n: int,
    seed: int,
    out: str,
    only: Optional[str] = None,
    *,
    filesystem: SourceFiles = LOCAL_FILES,
    random_factory: Callable[[int], RandomChoice] = random.Random,
) -> None:
    (info, rows) = load(d, filesystem=filesystem)
    where = {(t, e): (p, o, nb) for (t, e, nb, p, o) in rows}
    experts = sorted(
        {
            (int(cast(re.Match[str], ROLE.match(t)).group(1)), e)
            for (t, e, _, p, _) in rows
            if not only or p == filesystem.abspath(only)
        }
    )
    assert experts, f"no experts in {only}"
    rng = random_factory(seed)
    (lines, files) = ([], set())
    for layer, e in (rng.choice(experts) for _ in range(n)):
        for role in ("gate", "up", "down"):
            (p, o, nb) = where[f"blk.{layer}.ffn_{role}_exps.weight", e]
            (start, end) = (o // PAGE * PAGE, (o + nb + PAGE - 1) // PAGE * PAGE)
            lines.append(f"{p} read {start} {end - start}\n")
            files.add(p)
    with filesystem.open(out, "w") as f:
        f.write("fio version 2 iolog\n" + "".join(f"{p} add\n{p} open\n" for p in sorted(files)))
        f.writelines(lines)
        f.write("".join(f"{p} close\n" for p in sorted(files)))


def facts(
    model: str,
    alias: Optional[str] = None,
    *,
    filesystem: SourceFiles = LOCAL_FILES,
    emit: Callable[..., None] = print,
) -> None:
    types = sorted({t.type_name for t in gguf_tensors(model, filesystem=filesystem)})
    unaligned = 0
    if alias:
        with filesystem.open(alias) as f:
            next(f)
            unaligned = sum(1 for line in f if int(line.rsplit("\t", 1)[1]) % PAGE)
    emit(f"expert_types='{' '.join(types)}'\nalias_unaligned={unaligned}")


def lane_args(
    d: str, *, filesystem: SourceFiles = LOCAL_FILES, emit: Callable[..., None] = print
) -> None:
    """Lane timing arguments at the model shape; up is [latent, hidden, E]."""
    info = json.load(filesystem.open(f"{d}/experts.json"))
    tensors = gguf_tensors(info["model"], filesystem=filesystem)
    ups = [t for t in tensors if cast(re.Match[str], ROLE.match(t.name)).group(2) == "up"]
    downs = [t for t in tensors if cast(re.Match[str], ROLE.match(t.name)).group(2) == "down"]
    assert ups and downs, f"no routed up and down tensors in {info['model']}"

    def common(ts: list[Tensor]) -> str:
        names = [t.type_name for t in ts]
        return max(sorted(set(names)), key=names.count)

    emit(
        f"--latent {ups[0].shape[0]} --hidden {ups[0].shape[1]} "
        f"--up {common(ups)} --down {common(downs)}"
    )


def dense(
    model: str,
    homes: str,
    out: str,
    *,
    filesystem: SourceFiles = LOCAL_FILES,
    emit: Callable[..., None] = print,
) -> None:
    every = [t for h in gguf_header.read_model(model, filesystem=filesystem) for t in h.tensors]
    tensors = {t.name: t for t in every if not is_role(t)}
    largest = role_slice(every)
    facts = {x["name"]: x for x in dense_facts(tensors.values(), largest)}
    (rows, seen, totals, stage) = ([], set(), {"drive": [0, 0], "ram": [0, 0]}, [0, 0])
    for n, line in enumerate(filesystem.open(homes), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        (name, home) = (line.split("\t") + [""])[:2]
        if home not in totals or name in seen:
            sys.exit(
                f"{homes}:{n}: not a tensor<TAB>ram|drive line of a tensor not named before: {line}"
            )
        if name not in facts:
            sys.exit(f"{homes}:{n}: {name} is not a dense weight of {model}")
        seen.add(name)
        (x, t) = (facts[name], tensors[name])
        totals[home][0] += 1
        totals[home][1] += x["bytes"]
        if home != "drive":
            continue
        if not x["n_blocks"]:
            sys.exit(
                f"{homes}:{n}: {name} ({x['rows']} rows of {x['row_bytes']} B) "
                f"has no split into equal blocks of "
                f"whole rows of at most {largest} B: it cannot be homed on the drive"
            )
        bb = x["bytes"] // x["n_blocks"]
        rows += [
            (name, i, bb, filesystem.abspath(t.path), t.offset + i * bb)
            for i in range(x["n_blocks"])
        ]
        stage = [max(stage[0], x["n_blocks"]), max(stage[1], (bb + 2 * PAGE - 1) // PAGE * PAGE)]
    with filesystem.open(f"{out}.tmp", "w") as f:
        f.write("tensor\texpert_idx\tnbytes\tsource_path\toffset\n")
        f.writelines(f"{t}\t{e}\t{nb}\t{p}\t{o}\n" for (t, e, nb, p, o) in rows)
    filesystem.replace(f"{out}.tmp", out)
    emit(
        f"dense_drive={totals['drive'][0]}\ndense_drive_bytes={totals['drive'][1]}\n"
        f"dense_ram={totals['ram'][0]}\ndense_ram_bytes={totals['ram'][1]}\n"
        f"dense_stage_bytes={2 * stage[0] * stage[1]}"
    )
