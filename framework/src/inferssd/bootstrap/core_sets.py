"""CPU-set candidates from explicit recorded labels; no host inspection."""

from __future__ import annotations

from collections.abc import Sequence

from inferssd.bootstrap.ports import Emit, OpenText


def parse_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        elif part.strip():
            out.append(int(part))
    return out


def run_coresets(argv: Sequence[str], *, open_file: OpenText, emit: Emit) -> None:
    cpus: list[tuple[float, int]] = []
    for line in open_file(argv[1]):
        f = line.split()
        cpu, cap, khz = int(f[0]), float(f[1]), float(f[2])
        sib = parse_list(f[3]) if len(f) > 3 and f[3] != "-" else [cpu]
        if min(sib) != cpu:
            continue  # a second hardware thread of a core already listed
        cpus.append((cap if cap > 0 else khz, cpu))
    cpus.sort(key=lambda t: (-t[0], t[1]))
    groups: list[list[tuple[float, int]]] = []
    for value, cpu in cpus:
        if groups and value >= 0.92 * groups[-1][0][0]:
            groups[-1].append((value, cpu))
        else:
            groups.append([(value, cpu)])
    cpu_groups = [[c for _, c in g] for g in groups]
    cands: list[tuple[str, list[int]]] = []
    for i, g in enumerate(cpu_groups):
        cands.append((f"g{i + 1}", g))
    if len(cpu_groups[0]) >= 3:
        cands.append(("g1-1", cpu_groups[0][:-1]))
    acc: list[int] = []
    for i, g in enumerate(cpu_groups):
        acc = acc + g
        if i >= 1:
            cands.append(
                (
                    "all"
                    if i == len(cpu_groups) - 1
                    else "+".join(f"g{j + 1}" for j in range(i + 1)),
                    list(acc),
                )
            )
    seen: set[tuple[int, ...]] = set()
    for name, s in cands:
        key = tuple(sorted(s))
        if key in seen:
            continue
        seen.add(key)
        emit(name, ",".join(map(str, s)))
