# SPDX-License-Identifier: Apache-2.0
"""Machine topology can exceed the native probe's thread and CPU-mask capacity."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from test_machine_probes import CPU_REPORT
from test_machine_profile import Clock

from inferssd.machine.probes import ProbeError
from inferssd.machine.profile import TOPOLOGY, measure_cpu, probe_threads


def topology(ids, fast=()):
    groups = (
        [
            {"id": 0, "cpus": list(fast), "kind": "performance"},
            {"id": 1, "cpus": [cpu for cpu in ids if cpu not in fast], "kind": "efficiency"},
        ]
        if fast
        else [{"id": 0, "cpus": list(ids), "kind": "uniform"}]
    )
    return {
        "logical": [
            {
                "id": cpu,
                "core": cpu,
                "package": 0,
                "smt_siblings": [cpu],
                "capacity": None,
                "max_khz": None,
                "cluster": 0 if not fast or cpu in fast else 1,
            }
            for cpu in ids
        ],
        "clusters": groups,
        "affinity_supported": True,
        "isa": [],
    }


class ProbeContract:
    def __init__(self):
        self.calls = []

    def run(self, name, arguments):
        assert name == "cpu"
        self.calls.append(list(arguments))
        threads = [int(n) for n in arguments[arguments.index("--threads") + 1].split(",")]
        if any(not 1 <= n <= 512 for n in threads):
            raise ProbeError("--threads: expected an integer in [1, 512]")
        report = copy.deepcopy(CPU_REPORT)
        if "--cpus" in arguments:
            cpus = [int(n) for n in arguments[arguments.index("--cpus") + 1].split(",")]
            if any(not 0 <= n < 512 for n in cpus):
                raise ProbeError("--cpus: expected an integer in [0, 511]")
            report["cpus"] = cpus
            report["compute"][0]["threads"] = len(cpus)
        return report


@pytest.mark.parametrize("total", [513, 600, 1024])
def test_large_machine_retains_cpu_and_memory_measurements(monkeypatch, total):
    cpu = topology(range(total))
    original = copy.deepcopy(cpu)
    monkeypatch.setitem(TOPOLOGY, "linux", ("recorded topology", lambda _host: cpu))
    probe = ProbeContract()
    measured, memory, compute, sections = measure_cpu(
        SimpleNamespace(page_bytes=lambda: 4096),
        {"hard": {"platform": "linux", "ram_bytes": 1 << 40}},
        probe,
        Clock(),
        lambda _line: None,
    )
    assert compute["cpu"] == CPU_REPORT["compute"]
    assert memory["dram_read_Bps"] == [
        {"threads": row["threads"], "Bps": row["Bps"]} for row in CPU_REPORT["dram"]
    ]
    assert all(sections[name]["status"] == "ok" for name in ("cpu", "memory", "compute"))
    assert measured["logical"] == original["logical"]
    assert measured["clusters"] == original["clusters"]
    assert probe.calls == [["--threads", "1,2,4,8,16,32,64,128,256,512"]]


def test_cluster_size_also_respects_probe_thread_capacity():
    assert probe_threads(topology(range(1024), tuple(range(768)))) == [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
    ]


def test_unrepresentable_cpu_ids_are_not_sent_to_the_probe(monkeypatch):
    cpu = topology([0, 1, 512, 513], (0, 1))
    monkeypatch.setitem(TOPOLOGY, "linux", ("recorded topology", lambda _host: cpu))
    probe = ProbeContract()
    measured, _memory, compute, sections = measure_cpu(
        SimpleNamespace(page_bytes=lambda: 4096),
        {"hard": {"platform": "linux", "ram_bytes": 1 << 30}},
        probe,
        Clock(),
        lambda _line: None,
    )
    assert [row["id"] for row in measured["logical"]] == [0, 1, 512, 513]
    assert [row.get("cpus") for row in compute["cpu"]] == [None, [0, 1]]
    assert len(probe.calls) == 2
    assert "CPU IDs exceed the probe's 0..511 mask" in sections["compute"]["detail"]


@pytest.mark.parametrize(
    "total,expected",
    [
        (1, [1]),
        (4, [1, 2, 3, 4]),
        (20, [1, 2, 4, 8, 16, 19, 20]),
        (512, [1, 2, 4, 8, 16, 32, 64, 128, 256, 511, 512]),
    ],
)
def test_supported_machine_keeps_its_probe_curve(total, expected):
    assert probe_threads(topology(range(total))) == expected
