# SPDX-License-Identifier: Apache-2.0
"""Commands whose output is parsed run in the C locale, whatever the user's locale."""

from __future__ import annotations

import ast
import stat
from pathlib import Path

import pytest

import inferssd
from inferssd.app import command as app_command
from inferssd.launch.live import NativeHost
from inferssd.machine.identity import NativeHost as IdentityHost
from inferssd.machine.probes import run_process
from inferssd.platform import resources, worker
from inferssd.platform.captured_command import run_captured
from inferssd.platform.policy import Policy

# As macOS does: vm.swapusage uses the decimal separator of LC_ALL, LC_NUMERIC or LANG.
SYSCTL = """#!/bin/sh
case "$2" in
hw.memsize) echo 17179869184 ;;
vm.swapusage)
    locale="${LC_ALL:-${LC_NUMERIC:-$LANG}}"
    case "$locale" in
    C|POSIX|"") echo "total = 2048.00M  used = 884.56M  free = 1163.44M  (encrypted)" ;;
    *) echo "total = 2048,00M  used = 884,56M  free = 1163,44M  (encrypted)" ;;
    esac ;;
esac
"""
VM_STAT = """#!/bin/sh
echo "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
echo "Pages free:                               100."
echo "Pages inactive:                            50."
echo "Pages occupied by compressor:              10."
"""
REPORT_LOCALE = '#!/bin/sh\necho "$LC_ALL"\n'


def install(directory: Path, name: str, body: str) -> None:
    tool = directory / name
    tool.write_text(body)
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def german(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    monkeypatch.setenv("LC_ALL", "de_DE.UTF-8")
    monkeypatch.setenv("LC_NUMERIC", "de_DE.UTF-8")
    monkeypatch.setenv("LANG", "de_DE.UTF-8")
    return tmp_path


def test_darwin_memory_counters_read_swap_under_a_decimal_comma_locale(german: Path) -> None:
    install(german, "sysctl", SYSCTL)
    install(german, "vm_stat", VM_STAT)
    policy = Policy(
        platform="darwin",
        lock_path="/unused/lock",
        stop_path="/unused/stop",
        memory_budget_bytes=1 << 40,
        host_reserve_bytes=None,
        minimum_memlock_bytes=0,
        max_compressor_growth_bytes=None,
        max_swap_growth_bytes=None,
        runtime_seconds=None,
        cleanup_seconds=1,
    )
    row = resources.Resources(policy).darwin(None)
    assert row["swap_bytes"] == int(884.56 * 1024**2)
    assert row["available_bytes"] == 150 * 16384


@pytest.mark.parametrize(
    "report",
    [
        lambda: run_captured(["probe-locale"], timeout=10).stdout,
        lambda: IdentityHost().command(["probe-locale"]).stdout,
        lambda: NativeHost().command(["probe-locale"]).stdout,
        lambda: resources.command(["probe-locale"]),
    ],
    ids=["helper", "machine-identity", "launch", "resources"],
)
def test_every_command_host_runs_tools_in_the_c_locale(german: Path, report) -> None:
    install(german, "probe-locale", REPORT_LOCALE)
    assert report().strip() == "C"


def test_systemctl_runs_in_the_c_locale(german: Path) -> None:
    install(german, "systemctl", REPORT_LOCALE)
    assert worker.systemctl("unit.scope", True, "show").stdout.strip() == "C"


def test_physical_memory_reads_sysctl_through_the_runner(
    german: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(german, "sysctl", SYSCTL)
    monkeypatch.setattr(app_command.sys, "platform", "darwin")
    assert app_command.physical_memory() == 17179869184


def test_run_captured_keeps_the_rest_of_the_environment(
    german: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INFERSSD_TEST_VALUE", "kept")
    install(german, "probe-environment", '#!/bin/sh\necho "$INFERSSD_TEST_VALUE"\n')
    assert run_captured(["probe-environment"], timeout=10).stdout.strip() == "kept"


def test_run_captured_replaces_bytes_that_are_not_utf8(german: Path) -> None:
    install(german, "probe-bytes", "#!/bin/sh\nprintf 'caf\\351'\n")
    assert run_captured(["probe-bytes"], timeout=10).stdout == "caf\ufffd"


def test_probe_runner_uses_the_c_locale_in_its_given_environment(german: Path) -> None:
    install(german, "probe-locale", REPORT_LOCALE)
    environment = {"PATH": f"{german}:/usr/bin:/bin", "LC_ALL": "de_DE.UTF-8"}
    assert run_process(["probe-locale"], 10, environment).stdout.strip() == "C"


def captures_output(call: ast.Call) -> bool:
    if not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name):
        return False
    if call.func.value.id != "subprocess":
        return False
    if call.func.attr in ("check_output", "getoutput", "getstatusoutput"):
        return True
    for keyword in call.keywords:
        if keyword.arg == "capture_output":
            return True
        if keyword.arg in ("stdout", "stderr") and isinstance(keyword.value, ast.Attribute):
            if keyword.value.attr == "PIPE":
                return True
    return False


def test_every_captured_command_goes_through_the_runner() -> None:
    package = Path(inferssd.__file__).parent
    capturing = {
        str(path.relative_to(package))
        for path in package.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call) and captures_output(node)
    }
    assert capturing == {"platform/captured_command.py"}
