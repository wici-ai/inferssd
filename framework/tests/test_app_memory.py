# SPDX-License-Identifier: Apache-2.0
"""Regressions for real cgroup page rounding and host-reserve admission."""

from dataclasses import asdict
from pathlib import Path

import pytest

from inferssd.app import command, memory

GIB = memory.GIB


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(memory.sys, "platform", "linux")
    monkeypatch.setattr(memory.os, "sysconf", lambda _: 4096)


def test_recorded_default_budget_rounding(linux, tmp_path):
    # Captured on the 32 GB host: MemoryMax lost 2458 bytes to kernel page rounding.
    p = command.default_policy(32949071872, 26359257498, tmp_path, 600)
    assert p.memory_budget_bytes == 26359255040


@pytest.mark.parametrize("page", [4096, 16384, 65536])
def test_round_down_fractional_gib_on_real_page_size(linux, monkeypatch, page):
    monkeypatch.setattr(memory.os, "sysconf", lambda _: page)
    requested = int(19.1234567 * GIB)
    result = memory.admit(32 * GIB, 29 * GIB, requested)
    assert result.memory_budget_bytes % page == 0
    assert 0 <= requested - result.memory_budget_bytes < page


def test_automatic_budget_tracks_other_processes(linux):
    idle = memory.admit(32 * GIB, 29 * GIB)
    busy = memory.admit(32 * GIB, 21 * GIB)
    assert idle.memory_budget_bytes - busy.memory_budget_bytes == 8 * GIB
    for result in (idle, busy):
        assert (
            result.memory_budget_bytes + result.host_reserve_bytes + result.startup_margin_bytes
            <= result.available_bytes
        )
    assert idle.host_reserve_bytes == 32 * GIB // 5  # guard was not lowered


def test_recorded_host_admission(linux):
    result = memory.admit(32949071872, 29779722240)
    assert 19 * GIB < result.memory_budget_bytes < 21 * GIB
    with pytest.raises(ValueError, match="omit --memory-gib"):
        memory.admit(result.total_bytes, result.available_bytes, 24 * GIB)
    assert memory.admit(result.total_bytes, result.available_bytes, 20 * GIB).requested_bytes


@pytest.mark.parametrize("requested", [0, -1, 1, 32 * GIB])
def test_invalid_requested_budget(linux, requested):
    with pytest.raises(ValueError, match="server RAM budget"):
        memory.admit(32 * GIB, 29 * GIB, requested)


@pytest.mark.parametrize("total,available", [(GIB, GIB), (32 * GIB, 7 * GIB)])
def test_no_headroom_fails_closed(linux, total, available):
    with pytest.raises(ValueError, match="not enough available RAM"):
        memory.admit(total, available)


def test_linux_available_uses_memavailable_not_free(linux, monkeypatch):
    monkeypatch.setattr(
        Path, "read_text", lambda _: "MemTotal: 32000 kB\nMemFree: 100 kB\nMemAvailable: 23000 kB\n"
    )
    assert memory.available_memory() == 23000 * 1024
    monkeypatch.setattr(Path, "read_text", lambda _: "MemTotal: 32000 kB\n")
    with pytest.raises(ValueError, match="cannot read available RAM"):
        memory.available_memory()


def test_darwin_estimate_matches_runtime_guard(monkeypatch):
    monkeypatch.setattr(memory.sys, "platform", "darwin")
    monkeypatch.setattr(
        memory,
        "command",
        lambda _: "(page size of 16384 bytes)\nPages free: 10.\nPages inactive: 20.\n",
    )
    assert memory.available_memory() == 30 * 16384
    assert memory.aligned_budget(12345) == 12345  # Darwin has a sampled soft cap.


def test_admission_receipt_is_json_compatible(linux):
    import json

    result = memory.admit(32 * GIB, 29 * GIB)
    assert json.loads(json.dumps(asdict(result)))["requested_bytes"] is None
