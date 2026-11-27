# SPDX-License-Identifier: Apache-2.0
"""The Windows platform layer on any OS: Win32 replies are parsed from bytes or replaced."""

from __future__ import annotations

import ctypes
import struct
import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest
from test_machine_identity import FakeHost

from inferssd.app import command, hub
from inferssd.configuration.settings import host_platform
from inferssd.launch.resources import BUILDS, cache_directory, data_directory
from inferssd.machine.identity import UnsupportedPlatform, drive_info, live_identity
from inferssd.machine.profile import pinned_sets, probe_threads, windows_topology
from inferssd.platform.policy import Policy
from inferssd.platform.windows import process as windows_process
from inferssd.platform.windows import resources as windows_resources
from inferssd.platform.windows import system, win32

GUID = "12345678-aaaa-bbbb-cccc-1234567890ab"
CURRENT = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
REGISTRY = {
    (r"SOFTWARE\Microsoft\Cryptography", "MachineGuid"): GUID,
    (r"HARDWARE\DESCRIPTION\System\BIOS", "SystemProductName"): "Laptop",
    (
        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        "ProcessorNameString",
    ): "  Example  CPU ",
    (CURRENT, "CurrentBuild"): "26200",
    (CURRENT, "CurrentMajorVersionNumber"): 10,
    (CURRENT, "CurrentMinorVersionNumber"): 0,
    (CURRENT, "UBR"): 7171,
    (CURRENT, "ProductName"): "Windows 10 Home",
    (CURRENT, "DisplayVersion"): "25H2",
}


@dataclass
class FakeSystem:
    """What machine profiling asks a Windows host, from tables."""

    sets: tuple[system.CPUSet, ...] = tuple(
        system.CPUSet(cpu, cpu, 1 if cpu < 4 else 0, 256 + cpu) for cpu in range(8)
    )
    counts: tuple[int, ...] = (8,)
    listed: tuple[system.Adapter, ...] = (
        system.Adapter("Intel(R) Graphics", r"PCI\VEN_8086&DEV_B090&SUBSYS_389117AA", 0),
        system.Adapter("NVIDIA GeForce", r"PCI\VEN_10DE&DEV_2B85", 16 << 30),
        system.Adapter("Microsoft Basic Render Driver", r"PCI\VEN_1414&DEV_008C", 0),
        system.Adapter("A display link", r"USB\VID_17E9", 0),
    )
    volumes: dict[str, system.Volume] = field(default_factory=dict)
    disks: dict[int, system.Disk] = field(default_factory=dict)

    def registry(self, key: str, name: str) -> Any:
        return REGISTRY.get((key, name))

    def adapters(self) -> tuple[system.Adapter, ...]:
        return self.listed

    def processors(self) -> system.Processors:
        return system.Processors(self.counts, self.sets)

    def process_cpus(self) -> tuple[int, ...]:
        return tuple(range(8))

    def memory(self) -> system.Memory:
        return system.Memory(32 << 30, 24 << 30)

    def volume(self, path: str) -> system.Volume:
        return self.volumes[path]

    def disk(self, number: int) -> system.Disk:
        return self.disks[number]


class WindowsHost(FakeHost):
    def __init__(self, root: Any, windows: FakeSystem, commands: Any = None) -> None:
        super().__init__(root, "windows", "AMD64", commands)
        self.system_layer = windows

    def windows(self) -> FakeSystem:
        return self.system_layer


def set_row(index: int, core: int, efficiency: int, group: int = 0, flags: int = 0) -> bytes:
    # Size, Type, then Id, Group, LogicalProcessorIndex, CoreIndex, LastLevelCacheIndex,
    # NumaNodeIndex, EfficiencyClass, AllFlags, padded to the record's 32 bytes.
    body = struct.pack(
        "<IHBBBBBB", 256 + 64 * group + index, group, index, core, 0, 0, efficiency, flags
    )
    return struct.pack("<II", 32, 0) + body + b"\0" * (32 - 8 - len(body))


def test_cpu_sets_qualify_cpu_and_core_by_group_and_stop_at_a_broken_record():
    rows = set_row(0, 0, 1) + set_row(1, 0, 1) + set_row(0, 4, 0, group=1)
    assert system.cpu_sets(rows) == [
        system.CPUSet(0, 0, 1, 256),
        system.CPUSet(1, 0, 1, 257),
        system.CPUSet(64, 68, 0, 320),
    ]
    assert system.cpu_sets(rows + struct.pack("<II", 4, 0)) == system.cpu_sets(rows)


def test_disk_extents_and_storage_descriptor_from_their_layouts():
    extents = struct.pack("<I4x", 3)
    for disk in (2, 0, 2):
        extents += struct.pack("<I4xQQ", disk, 0, 1)
    assert system.disk_extents(extents) == (0, 2)
    text = b"Maker\0Model X\0 SERIAL \0"
    header = (
        struct.pack("<II", 1, 1) + struct.pack("<I", 0) + struct.pack("<IIIII", 32, 38, 0, 46, 0x11)
    )
    data = header.ljust(32, b"\0") + text
    assert system.storage_descriptor(3, data) == system.Disk(3, "Maker", "Model X", "SERIAL", 0x11)
    with pytest.raises(OSError, match="PhysicalDrive3"):
        system.storage_descriptor(3, b"short")


class FakeKernel:
    """kernel32 for the calls that fill buffers with fixed layouts."""

    def GetActiveProcessorGroupCount(self) -> int:
        return 1

    def GetActiveProcessorCount(self, _group: int) -> int:
        return 8

    def GetSystemCpuSetInformation(self, buffer, size, length, _process, _flags):
        data = b"".join(set_row(cpu, cpu, 1 if cpu < 4 else 0) for cpu in range(8))
        length._obj.value = len(data)
        if buffer is None or size < len(data):
            return 0
        buffer.raw = data.ljust(len(buffer.raw), b"\0")
        return 1

    def GetCurrentProcess(self) -> int:
        return 1

    def GetProcessAffinityMask(self, _handle, process, system_mask) -> int:
        process._obj.value = 0b1010
        system_mask._obj.value = 0xFF
        return 1

    def GlobalMemoryStatusEx(self, buffer) -> int:
        assert struct.unpack_from("<I", buffer.raw)[0] == 64
        buffer.raw = struct.pack("<IIQQ", 64, 0, 32 << 30, 24 << 30).ljust(64, b"\0")
        return 1


class MultiGroupKernel(FakeKernel):
    counts = (64, 64)
    process_groups = (0, 1)
    default_sets: tuple[int, ...] = ()

    def __init__(self):
        self.flags: dict[int, int] = {}

    def GetActiveProcessorGroupCount(self) -> int:
        return len(self.counts)

    def GetActiveProcessorCount(self, group: int) -> int:
        return self.counts[group]

    def GetProcessGroupAffinity(self, _handle, size, numbers):
        assert size._obj.value >= len(self.process_groups)
        size._obj.value = len(self.process_groups)
        numbers[: len(self.process_groups)] = self.process_groups
        return 1

    def GetSystemCpuSetInformation(self, buffer, size, length, process, _flags):
        assert process.value == self.GetCurrentProcess()
        data = b"".join(
            set_row(cpu, cpu // 2, 1, group, self.flags.get(64 * group + cpu, 0))
            for group, count in enumerate(self.counts)
            for cpu in range(count)
        )
        length._obj.value = len(data)
        if buffer is None or size < len(data):
            return 0
        buffer.raw = data
        return 1

    def GetProcessDefaultCpuSets(self, _handle, ids, size, count):
        count._obj.value = len(self.default_sets)
        if size < len(self.default_sets):
            return 0
        if ids is not None:
            ids[:] = self.default_sets
        return 1


def test_multigroup_process_cpus_respect_groups_hard_mask_and_default_cpu_sets():
    kernel = MultiGroupKernel()
    layer = system.Win32(kernel, last_error=lambda: 122)
    assert layer.process_cpus() == tuple(range(128))
    kernel.default_sets = (257, 321)
    assert layer.process_cpus() == (1, 65)
    # The process is restricted to bits 1 and 3 in its only allowed group.
    kernel.process_groups = (1,)
    assert layer.process_cpus() == (65,)
    kernel.default_sets = (257,)
    assert layer.process_cpus() == (65, 67)
    kernel.default_sets = (999,)
    with pytest.raises(OSError, match="unknown process-default CPU Set ID"):
        layer.process_cpus()


def test_process_cpus_refresh_reservations_and_keep_machine_topology():
    kernel = MultiGroupKernel()
    layer = system.Win32(kernel, last_error=lambda: 122)
    # Bit 1 is Allocated and bit 2 is AllocatedToTargetProcess.
    kernel.flags = {1: 0b010, 65: 0b110}
    kernel.default_sets = (257, 321)
    assert layer.process_cpus() == (65,)
    assert layer.processors().ids == tuple(range(128))
    assert len(layer.processors().sets) == 128
    kernel.default_sets = (257,)
    elsewhere = tuple(cpu for cpu in range(128) if cpu != 1)
    assert layer.process_cpus() == elsewhere
    kernel.default_sets = ()
    assert layer.process_cpus() == elsewhere
    # Same Win32 object. The reservation moves to another process between reads.
    kernel.flags = {65: 0b010, 67: 0b110}
    kernel.process_groups = (1,)
    assert layer.process_cpus() == (67,)
    kernel.default_sets = (321,)
    assert layer.process_cpus() == (67,)
    kernel.flags[67] = 0b010
    with pytest.raises(OSError, match="no available CPUs"):
        layer.process_cpus()


def test_win32_reads_processors_affinity_and_memory():
    layer = system.Win32(FakeKernel(), lambda _key, _name: None, lambda: 0)
    found = layer.processors()
    assert found.counts == (8,) and found.logical == 8
    assert [row.efficiency for row in found.sets] == [1, 1, 1, 1, 0, 0, 0, 0]
    assert layer.process_cpus() == (1, 3)
    assert layer.memory() == system.Memory(32 << 30, 24 << 30)


def test_multigroup_availability_distinguishes_no_cpu_sets_from_a_query_error(monkeypatch):
    kernel = MultiGroupKernel()
    monkeypatch.setattr(kernel, "GetSystemCpuSetInformation", lambda *_: 1)
    layer = system.Win32(kernel, last_error=lambda: 5)
    assert layer.process_cpus() == tuple(range(128))  # a successful empty table
    monkeypatch.setattr(kernel, "GetSystemCpuSetInformation", lambda *_: 0)
    with pytest.raises(OSError, match="GetSystemCpuSetInformation.*error 5"):
        layer.process_cpus()
    # Machine topology still uses group counts when the optional CPU Sets are unavailable.
    assert layer.processors() == system.Processors((64, 64), ())


def test_win32_lists_the_display_adapters_it_finds():
    values = {
        (system.DISPLAY_CLASS + "0000", "DriverDesc"): "Intel(R) Graphics",
        (system.DISPLAY_CLASS + "0000", "MatchingDeviceId"): r"PCI\VEN_8086&DEV_B090",
        (system.DISPLAY_CLASS + "0002", "DriverDesc"): "Other GPU",
        (system.DISPLAY_CLASS + "0002", "MatchingDeviceId"): r"PCI\VEN_1002&DEV_1",
        (system.DISPLAY_CLASS + "0002", "HardwareInformation.qwMemorySize"): 8 << 30,
        (system.DISPLAY_CLASS + "0003", "DriverDesc"): "No device ID",
    }
    layer = system.Win32(FakeKernel(), lambda key, name: values.get((key, name)), lambda: 0)
    assert layer.adapters() == (
        system.Adapter("Intel(R) Graphics", r"PCI\VEN_8086&DEV_B090", 0),
        system.Adapter("Other GPU", r"PCI\VEN_1002&DEV_1", 8 << 30),
    )


def test_windows_identity(tmp_path):
    identity = live_identity(WindowsHost(tmp_path, FakeSystem()))
    hard, soft = identity["hard"], identity["soft"]
    assert hard["platform"] == "windows" and hard["arch"] == "AMD64"
    assert hard["cpu_brand"] == "Example CPU" and hard["model"] == "Laptop"
    assert hard["logical_cpus"] == 8 and hard["ram_bytes"] == 32 << 30
    # The integrated GPU is listed; NVIDIA's come from nvidia-smi, software and USB are not GPUs.
    assert hard["gpus"] == [
        {
            "backend": "Vulkan",
            "uuid": r"PCI\VEN_8086&DEV_B090&SUBSYS_389117AA",
            "name": "Intel(R) Graphics",
            "vram_bytes": 0,
        }
    ]
    # Windows 11 keeps the product name of Windows 10; its builds start at 22000.
    assert soft["os_release"] == "Windows 11 Home 25H2 (build 26200)"
    assert soft["kernel"] == "10.0.26200.7171"
    # No probe measures this GPU yet.
    assert "memory" not in soft["probe_versions"]


def test_other_platforms_stay_unsupported(tmp_path):
    with pytest.raises(UnsupportedPlatform):
        live_identity(FakeHost(tmp_path, "freebsd"))


def test_windows_topology_splits_the_clusters_by_efficiency_class(tmp_path):
    cpu = windows_topology(WindowsHost(tmp_path, FakeSystem()))
    assert cpu["affinity_supported"] is True
    assert [(row["kind"], row["cpus"]) for row in cpu["clusters"]] == [
        ("performance", [0, 1, 2, 3]),
        ("efficiency", [4, 5, 6, 7]),
    ]
    assert cpu["logical"][5]["capacity"] == 0 and cpu["logical"][5]["cluster"] == 1


def test_windows_topology_without_cpu_sets_is_one_cluster(tmp_path):
    cpu = windows_topology(WindowsHost(tmp_path, FakeSystem(sets=())))
    assert [row["cpus"] for row in cpu["clusters"]] == [list(range(8))]
    assert cpu["logical"][0]["core"] is None


def test_windows_topology_profiles_separate_groups_without_pinning(tmp_path):
    kernel = MultiGroupKernel()
    kernel.counts = (32, 64)
    processors = system.Win32(kernel).processors()
    host = WindowsHost(tmp_path, FakeSystem(counts=processors.counts, sets=processors.sets))
    cpu = windows_topology(host)
    assert processors.logical == 96
    assert [row["id"] for row in cpu["logical"]] == [*range(32), *range(64, 128)]
    assert cpu["logical"][0]["smt_siblings"] == [0, 1]
    assert cpu["logical"][32]["smt_siblings"] == [64, 65]
    assert not cpu["affinity_supported"] and not pinned_sets(cpu)
    assert probe_threads(cpu) == [1, 2, 4, 8, 16, 32, 64, 95, 96]
    assert probe_threads(cpu, 2) == [1, 2]


def test_windows_probe_bounds_worker_count_for_large_machines(tmp_path):
    host = WindowsHost(tmp_path, FakeSystem(counts=(64,) * 10, sets=()))
    cpu = windows_topology(host)
    assert len(cpu["logical"]) == 640
    assert max(probe_threads(cpu, 640)) == 512


def test_windows_drive_keys(tmp_path):
    disk = system.Disk(0, "", "SAMSUNG X", "SERIAL", 0x11)
    other = system.Disk(1, "", "Y", "", 0xB)
    layer = FakeSystem(
        volumes={
            "C:\\m.gguf": system.Volume("C:\\", "NTFS", (0,)),
            "D:\\m.gguf": system.Volume("D:\\", "NTFS", (0, 1)),
            "Z:\\m.gguf": system.Volume("\\\\server\\share\\", "SMB", ()),
        },
        disks={0: disk, 1: other},
    )
    host = WindowsHost(tmp_path, layer)
    one = drive_info(host, "C:\\m.gguf")
    assert one.key.startswith("windows:nvme:SAMSUNG X:") and "SERIAL" not in one.key
    assert one.bus == "nvme" and one.model == "SAMSUNG X"
    assert drive_info(host, "D:\\m.gguf").key.startswith("windows:array:")
    remote = drive_info(host, "Z:\\m.gguf")
    assert remote.key == "windows:fs:SMB:\\\\server\\share\\" and remote.bus == "unknown"


def policy(**limits: Optional[int]) -> Policy:
    values: dict[str, object] = {
        "platform": "windows",
        "lock_path": "C:\\lock",
        "stop_path": "C:\\stop",
        "memory_budget_bytes": 32 << 30,
        "minimum_memlock_bytes": 0,
        "cleanup_seconds": 30,
    }
    values.update(limits)
    return Policy.read(values)


@pytest.fixture
def job_host(monkeypatch):
    state = {"limit": 32 << 30, "available": 24 << 30, "compressor": 0, "pagefile": 0, "vram": 0}
    monkeypatch.setattr(windows_resources, "host_platform", lambda: "windows")
    monkeypatch.setattr(windows_resources.os, "getpid", lambda: 7)
    monkeypatch.setattr(
        win32,
        "job_memory",
        lambda _job: {"job_memory_limit": state["limit"], "peak_job_memory": 1},
    )
    monkeypatch.setattr(win32, "job_processes", lambda _job: [7, 9])
    monkeypatch.setattr(win32, "discrete_video_memory", lambda: state["vram"])
    monkeypatch.setattr(
        win32, "system_memory", lambda: {"physical_available_bytes": state["available"]}
    )
    monkeypatch.setattr(
        win32, "process_memory", lambda _pid: {"working_set_bytes": state["compressor"]}
    )
    monkeypatch.setattr(win32, "process_working_set", lambda _name: state["compressor"])
    monkeypatch.setattr(win32, "pagefile_in_use", lambda: state["pagefile"])
    return state


def test_resources_leave_out_the_limits_a_policy_does_not_declare(job_host):
    runtime = windows_resources.Resources(policy())
    first = runtime.prepare()
    assert first["available_bytes"] == 24 << 30
    job_host.update(available=1, compressor=1 << 40, pagefile=1 << 40)
    assert runtime.sample(9)["server"]["working_set_bytes"] == 1 << 40


def test_a_protected_compression_process_is_sampled_and_its_growth_is_enforced(
    job_host, monkeypatch
):
    def protected(_pid):
        raise PermissionError("OpenProcess: protected compression process")

    monkeypatch.setattr(win32, "process_memory", protected)
    runtime = windows_resources.Resources(policy(max_compressor_growth_bytes=1))
    assert runtime.prepare()["compressor_bytes"] == 0
    job_host["compressor"] = 5
    with pytest.raises(RuntimeError, match="compressor growth"):
        runtime.sample(None)


def test_an_unreadable_compression_store_is_not_reported_as_zero(job_host, monkeypatch):
    def unavailable(_name):
        raise OSError("process list unavailable")

    monkeypatch.setattr(win32, "process_working_set", unavailable)
    with pytest.raises(OSError, match="process list unavailable"):
        windows_resources.Resources(policy()).prepare()


def test_an_absent_compression_process_has_no_store(job_host, monkeypatch):
    monkeypatch.setattr(win32, "process_working_set", lambda _name: None)
    assert windows_resources.Resources(policy()).prepare()["compressor_bytes"] == 0


class Ntdll:
    """NtQuerySystemInformation over lists built at the caller's buffer address."""

    def __init__(self, lists: dict[int, Any], status: int = 0):
        self.lists, self.status, self.sizes = lists, status, []

    def NtQuerySystemInformation(self, information_class, buffer, size, written):
        self.sizes.append(size)
        if self.status:
            return self.status
        data = self.lists[information_class](ctypes.addressof(buffer))
        written._obj.value = len(data)
        if len(data) > size:
            return 0xC0000004  # STATUS_INFO_LENGTH_MISMATCH
        ctypes.memmove(buffer, data, len(data))
        return 0


def process_list(processes: list[tuple[Optional[str], int]]) -> Any:
    """SYSTEM_PROCESS_INFORMATION entries, each followed by its image name as the kernel has it."""

    def build(base: int) -> bytes:
        data = bytearray()
        for index, (name, working_set) in enumerate(processes):
            entry = win32._SystemProcess()
            encoded = (name or "").encode("utf-16-le")
            start = len(data)
            size = (ctypes.sizeof(entry) + len(encoded) + 7) // 8 * 8
            entry.NextEntryOffset = size if index + 1 < len(processes) else 0
            if name is not None:
                entry.ImageName.Length = entry.ImageName.MaximumLength = len(encoded)
                entry.ImageName.Buffer = base + start + ctypes.sizeof(entry)
            entry.UniqueProcessId = 4 * index
            entry.WorkingSetSize = working_set
            data += bytes(entry) + encoded
            data += bytes(start + size - len(data))
        return bytes(data)

    return build


def pagefile_list(pages_in_use: list[int]) -> Any:
    def build(_base: int) -> bytes:
        data = bytearray()
        for index, pages in enumerate(pages_in_use):
            entry = win32._SystemPagefile(TotalSize=1 << 20, TotalInUse=pages, PeakUsage=pages)
            entry.NextEntryOffset = ctypes.sizeof(entry) if index + 1 < len(pages_in_use) else 0
            data += bytes(entry)
        return bytes(data)

    return build


def test_process_working_set_reads_the_named_process_from_the_kernel_list(monkeypatch):
    ntdll = Ntdll(
        {5: process_list([(None, 8192), ("System", 1 << 20), ("Memory Compression", 123456)])}
    )
    monkeypatch.setattr(win32, "_dll", lambda _name: ntdll)
    assert win32.process_working_set("Memory Compression") == 123456
    assert win32.process_working_set("System") == 1 << 20
    assert win32.process_working_set("Memory") is None


def test_a_process_list_larger_than_the_first_buffer_is_read_whole(monkeypatch):
    processes = [(f"worker-{index}.exe", index) for index in range(1000)]
    ntdll = Ntdll({5: process_list(processes + [("Memory Compression", 7)])})
    monkeypatch.setattr(win32, "_dll", lambda _name: ntdll)
    assert win32.process_working_set("Memory Compression") == 7
    assert ntdll.sizes[0] == 1 << 16 and len(ntdll.sizes) == 2 and ntdll.sizes[1] > 1 << 16


def test_paging_file_use_sums_every_paging_file(monkeypatch):
    lists = {18: pagefile_list([10, 5])}
    monkeypatch.setattr(win32, "_dll", lambda _name: Ntdll(lists))
    assert win32.pagefile_in_use() == 15 * win32.mmap.PAGESIZE
    lists[18] = pagefile_list([])
    assert win32.pagefile_in_use() == 0


@pytest.mark.parametrize(
    "read, name",
    [
        (lambda: win32.process_working_set("Memory Compression"), "SystemProcessInformation"),
        (win32.pagefile_in_use, "SystemPageFileInformation"),
    ],
)
def test_a_failed_system_query_names_its_class_and_status(monkeypatch, read, name):
    monkeypatch.setattr(win32, "_dll", lambda _name: Ntdll({}, status=0xC0000022))
    with pytest.raises(OSError, match=rf"{name}\) failed with NTSTATUS 0xC0000022"):
        read()


def test_memory_samples_need_no_performance_counters(monkeypatch):
    # On a machine whose performance counters are disabled or damaged, PDH refuses every counter
    # path (PDH_CSTATUS_NO_OBJECT) and a launch used to fail at its first sample.
    class NoCounters:
        def PdhOpenQueryW(self, _source, _data, query):
            query._obj.value = 1
            return 0

        def PdhAddEnglishCounterW(self, *_arguments):
            return 0xC0000BB8

        def PdhCloseQuery(self, _query):
            return 0

    ntdll = Ntdll({5: process_list([("Memory Compression", 4096)]), 18: pagefile_list([3])})
    monkeypatch.setattr(win32, "_dll", {"pdh": NoCounters(), "ntdll": ntdll}.__getitem__)
    assert win32.pagefile_in_use() == 3 * win32.mmap.PAGESIZE
    assert windows_resources.Resources(policy()).compressor_bytes() == 4096


@pytest.mark.parametrize(
    "limits,change,message",
    [
        ({"host_reserve_bytes": 2}, {"available": 1}, "below reserve"),
        ({"max_compressor_growth_bytes": 1}, {"compressor": 5}, "compressor growth"),
        ({"max_swap_growth_bytes": 1}, {"pagefile": 5}, "swap growth"),
        ({}, {"limit": 1}, "hard memory budget"),
    ],
)
def test_resources_enforce_what_a_policy_declares(job_host, limits, change, message):
    runtime = windows_resources.Resources(policy(**limits))
    runtime.prepare()
    job_host.update(change)
    with pytest.raises(RuntimeError, match=message):
        runtime.sample(None)


def test_the_job_limit_leaves_room_for_discrete_gpu_memory(job_host):
    # WDDM charges a device allocation to commit, so the job's limit is the budget plus the VRAM.
    job_host.update(vram=24 << 30, limit=(32 << 30) + (24 << 30))
    runtime = windows_resources.Resources(policy())
    runtime.prepare()
    job_host.update(limit=32 << 30)
    with pytest.raises(RuntimeError, match="hard memory budget"):
        runtime.sample(None)


def test_resources_see_a_server_outside_the_job(job_host):
    runtime = windows_resources.Resources(policy())
    runtime.prepare()
    with pytest.raises(RuntimeError, match="escaped"):
        runtime.sample(1234)


def test_resources_refuse_a_memlock_minimum(job_host):
    with pytest.raises(ValueError, match="locked-memory"):
        windows_resources.Resources(policy(minimum_memlock_bytes=1)).prepare()


def test_listener_must_belong_to_the_exact_server(monkeypatch):
    monkeypatch.setattr(
        win32, "tcp_listeners", lambda: [("127.0.0.1", 8080, 9), ("0.0.0.0", 8081, 9)]
    )
    runtime = windows_resources.Resources(policy())
    runtime.confirm_listener(9, "127.0.0.1", 8080)
    runtime.confirm_listener(9, "0.0.0.0", 8081)
    with pytest.raises(RuntimeError, match="exact server PID"):
        runtime.confirm_listener(10, "127.0.0.1", 8080)
    with pytest.raises(RuntimeError, match="exact server PID"):
        runtime.confirm_listener(9, "127.0.0.1", 8081)


class Finished:
    pid, returncode = 9, 0

    def poll(self) -> int:
        return 0

    def wait(self, timeout: Optional[float] = None) -> int:
        return 0


def owned_job(monkeypatch, active: list[int]) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(win32, "job_active", lambda _job: active[0])

    def interrupt() -> None:
        calls.append("ctrl+c")
        active[0] = 0

    monkeypatch.setattr(win32, "interrupt_console", interrupt)

    def terminate(_job: int, _code: int) -> None:
        calls.append("terminate")
        active[0] = 0

    monkeypatch.setattr(win32, "terminate_job", terminate)
    return calls


def test_a_stop_asks_with_ctrl_c_first(monkeypatch):
    calls = owned_job(monkeypatch, [1])
    process = windows_process.Group(Finished(), 5)  # type: ignore[arg-type]
    result = windows_process.stop_owned(process, 1)
    assert calls == ["ctrl+c"] and result["group_gone"] and not result["forced_kill"]


def test_a_stop_ends_the_job_when_ctrl_c_is_ignored(monkeypatch):
    calls = owned_job(monkeypatch, [1])
    monkeypatch.setattr(win32, "interrupt_console", lambda: calls.append("ctrl+c"))
    process = windows_process.Group(Finished(), 5)  # type: ignore[arg-type]
    result = windows_process.stop_owned(process, 0.01)
    assert calls == ["ctrl+c", "terminate"] and result["forced_kill"] and result["group_gone"]


def test_a_stop_survives_a_console_it_cannot_signal(monkeypatch):
    calls = owned_job(monkeypatch, [1])

    def no_console() -> None:
        raise OSError("no console")

    monkeypatch.setattr(win32, "interrupt_console", no_console)
    process = windows_process.Group(Finished(), 5)  # type: ignore[arg-type]
    assert windows_process.stop_owned(process, 0.01)["forced_kill"]
    assert calls == ["terminate"]


def test_a_process_without_a_job_cannot_be_stopped_as_a_group():
    with pytest.raises(TypeError, match="group"):
        windows_process.stop_owned(Finished(), 1)


def test_the_guardian_job_limit_adds_the_discrete_gpus_memory(tmp_path, monkeypatch):
    jobs = []

    def create_job(*, kill_on_close: bool, memory_limit: Optional[int]) -> int:
        jobs.append((kill_on_close, memory_limit))
        return 5

    monkeypatch.setattr(win32, "create_job", create_job)
    monkeypatch.setattr(win32, "discrete_video_memory", lambda: 24 << 30)
    windows_process.JobScope(policy(), tmp_path)
    assert jobs == [(False, (32 << 30) + (24 << 30))]


def test_job_scope_proves_the_job_empty(tmp_path, monkeypatch):
    active = [2]
    monkeypatch.setattr(win32, "create_job", lambda **_options: 5)
    monkeypatch.setattr(win32, "discrete_video_memory", lambda: 0)
    monkeypatch.setattr(win32, "job_active", lambda _job: active[0])
    monkeypatch.setattr(win32, "job_memory", lambda _job: {"job_memory": 0})
    monkeypatch.setattr(win32, "job_processes", lambda _job: [11])
    terminated = []

    def terminate(job: int, code: int) -> None:
        terminated.append((job, code))
        active[0] = 0

    monkeypatch.setattr(win32, "terminate_job", terminate)
    scope = windows_process.JobScope(policy(), tmp_path)
    record = scope.stop()
    assert terminated == [(5, 1)] and record["state"] == {
        "gone": True,
        "active_processes": 0,
        "pids": [],
    }
    assert record["job"]["terminated_processes"] == 2
    scope.request_stop(1)
    assert (tmp_path / "STOP").exists()


def test_the_venv_redirector_is_bypassed_so_a_pid_is_the_owner(monkeypatch):
    monkeypatch.setattr(windows_process.sys, "executable", "C:\\venv\\Scripts\\python.exe")
    monkeypatch.setattr(
        windows_process.sys, "_base_executable", "C:\\py\\python.exe", raising=False
    )
    argv, env = windows_process.interpreter(["C:\\venv\\Scripts\\python.exe", "-I"], {"A": "1"})
    assert argv == ["C:\\py\\python.exe", "-I"]
    assert env == {"A": "1", "__PYVENV_LAUNCHER__": "C:\\venv\\Scripts\\python.exe"}
    other, same = windows_process.interpreter(["C:\\server.exe"], {"A": "1"})
    assert other == ["C:\\server.exe"] and same == {"A": "1"}


def test_an_owned_child_runs_in_its_own_kill_on_close_job(monkeypatch):
    jobs = []
    monkeypatch.setattr(win32, "in_job", lambda: True)
    monkeypatch.setattr(win32, "job_memory", lambda _job: {"job_memory_limit": 123})

    def create_job(*, kill_on_close: bool, memory_limit: Optional[int]) -> int:
        jobs.append((kill_on_close, memory_limit))
        return 5

    monkeypatch.setattr(win32, "create_job", create_job)
    monkeypatch.setattr(windows_process, "create", lambda *_a, **_o: Finished())
    process = windows_process.spawn_owned(["server"], env={}, start_new_session=True)
    assert isinstance(process, windows_process.Group) and process.job == 5
    assert jobs == [(True, 123)]


def test_cli_and_serve_name_this_platform_as_the_framework_does(tmp_path, monkeypatch):
    monkeypatch.setattr(command, "physical_memory", lambda: 1 << 30)
    assert command.default_policy(tmp_path)["platform"] == host_platform()


def test_physical_memory_comes_from_win32_on_windows(monkeypatch):
    monkeypatch.setattr(command.sys, "platform", "win32")
    monkeypatch.setattr(win32, "system_memory", lambda: {"physical_total_bytes": 7})
    assert command.physical_memory() == 7


def test_cache_directory_follows_the_platform(tmp_path):
    linux = cache_directory({"XDG_CACHE_HOME": "/c"}, tmp_path, "linux")
    assert linux == type(tmp_path)("/c/inferssd")
    assert cache_directory({}, tmp_path, "linux") == tmp_path / ".cache/inferssd"
    assert cache_directory({}, tmp_path, "darwin") == tmp_path / "Library/Caches/inferssd"
    assert cache_directory({}, tmp_path, "windows") == tmp_path / "AppData/Local/inferssd"
    assert str(cache_directory({"LOCALAPPDATA": "C:\\L"}, tmp_path, "windows")).startswith("C:")


def test_data_directory_follows_the_platform(tmp_path):
    assert data_directory({"LOCALAPPDATA": "C:\\L"}, tmp_path, "windows").name == "inferssd"
    assert str(data_directory({"LOCALAPPDATA": "C:\\L"}, tmp_path, "windows")).startswith("C:")
    assert data_directory({}, tmp_path, "windows") == tmp_path / "AppData/Local/inferssd"
    assert data_directory({}, tmp_path, "linux") == tmp_path / ".local/share/inferssd"
    assert data_directory({}, tmp_path, "darwin") == tmp_path / ".local/share/inferssd"
    assert str(data_directory({"INFERSSD_HOME": "/x"}, tmp_path, "windows")) == str(
        type(tmp_path)("/x")
    )


def test_the_windows_preset_is_the_first_build_looked_in():
    assert BUILDS["windows"][0] == "build-windows-llvm"


QWEN_FILES = [f"Q8_0/Qwen3.8-Flash-Next-Q8_0-{n:05}-of-00006.gguf" for n in range(1, 7)]


def test_the_hub_takes_qwen_shards_without_its_prediction_head():
    assert hub.selection("unsloth/Qwen3.6-35B-A3B-GGUF") == (
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "UD-Q4_K_XL",
    )
    listed = QWEN_FILES + [
        "MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf",
        "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
        "mmproj-Q8_0.gguf",
        "Q4_K_M/Qwen3.8-Flash-Next-Q4_K_M-00001-of-00004.gguf",
    ]
    assert hub.select_files(listed, "Q8_0") == QWEN_FILES


def test_win32_is_only_called_on_windows():
    if ctypes.__name__ and host_platform() != "windows":
        with pytest.raises(OSError):
            win32.fd_handle(0)
        with pytest.raises(OSError):
            win32.system_memory()
        with pytest.raises(OSError):
            win32.create_process(
                ["x"],
                env={},
                stdout=None,
                stderr=None,
                stdin=subprocess.DEVNULL,
                inherit=(),
                flags=0,
            )
