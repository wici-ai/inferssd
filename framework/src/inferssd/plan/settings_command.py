"""Complete offline settings orchestration with legacy receipt bytes preserved."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional, cast

from inferssd.model.machine import Machine
from inferssd.model.machine_types import ArtifactReader, PlanArtifact
from inferssd.plan.settings import configuration, derive, render
from inferssd.plan.settings_types import HostThreads, KnownWorkload, Ledger, MachineView


@dataclass(frozen=True)
class SettingsArguments:
    run: str
    tier: Optional[int] = None
    cpus: Optional[str] = None
    prompts: Optional[str] = None
    decode: Optional[int] = None
    ledgers: Sequence[str] = ()
    out: Optional[str] = None


def run_settings(
    a: SettingsArguments,
    artifacts: ArtifactReader,
    *,
    host_threads: HostThreads,
    ledger: Callable[[str], Optional[Ledger]],
    absolute_path: Callable[[str], str],
    write_text: Callable[[str, str], None],
    stdout: Callable[[str], object],
    factory: type[Machine] = Machine,
) -> None:
    m = factory(artifacts, host_threads=host_threads)
    workload = m.workload
    if a.prompts:
        workload = cast(
            KnownWorkload,
            {
                "prompts": [int(x) for x in a.prompts.split(",")],
                "decode": a.decode
                if a.decode is not None
                else cast(Mapping[str, int], workload or {}).get("decode", 0),
                "known": True,
            },
        )
    # The original command reads the plan again even for explicit configurations.
    plan = cast(PlanArtifact, artifacts.read_json("plan.json"))
    if a.tier is not None or a.cpus:
        tier, p, why = (
            a.tier if a.tier is not None else m.caps[cast(str, a.cpus)],
            a.cpus or m.ref,
            "given",
        )
    else:
        r = plan["plan"]
        tier, p = r["tier"], r["cpus"]
        why = f"the plan's choice: {r['total_s']} s predicted for the workload {plan['workload']}"
    head = [
        f"tools/adaptive/settings.py {absolute_path(a.run)}",
        f"configuration: tier {tier} of {m.caps[p]} slots, placement {p} ({why})",
    ]
    if m.pending is None:
        head.append("no derived values: the plan predates them")
        derived = []
    else:
        head += [
            f"workload: {workload}",
            f"held by the model's conf (PENDING): {' '.join(m.pending) or 'none'}",
        ]
        derived = derive(
            cast(MachineView, m),
            p,
            m.pending,
            workload,
            [(d, ledger(d)) for d in a.ledgers],
        )
    text = render(
        configuration(cast(MachineView, m), tier, p, host_threads=host_threads) + derived, head
    )
    if a.out:
        write_text(a.out, text)
    else:
        stdout(text)
