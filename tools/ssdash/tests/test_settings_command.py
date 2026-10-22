from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_log_records import Files, f_api, runtime_key
from test_offline_planner import F_MACHINE, F_PLACEMENT, F_SETTINGS, Transcript, exact, state

FIXTURES = Path(__file__).parent / "fixtures"


class CommandTranscript(Transcript):
    def __init__(self, case):
        super().__init__(case)
        self.logs = Files({"files": case.get("logs", {}), "file_fault": case.get("file_fault")})
        self.logs.events = self.events
        self.read_count = 0
        self.calls = []
        self.output = io.StringIO()

    def read_json(self, path):
        self.read_count += 1
        if self.read_count == self.case.get("fail_read_number"):
            self.events.append(["read_json_fault", self.relative(path)])
            raise OSError("configured artifact read failure")
        value = super().read_json(path)
        if self.case.get("pending_iterator") and self.relative(path) == "plan.json":
            value["derived"]["pending"] = iter(value["derived"]["pending"])
        return value

    def open(self, path, mode="r"):
        if mode == "r":
            return SimpleNamespace(name=str(path))
        name = self.relative(path)
        self.events.append(["open_write", name])
        if self.case.get("fail_open"):
            raise OSError("configured output open failure")
        owner = self

        class Writer(io.StringIO):
            def write(self, text):
                owner.events.append(["write", name])
                if owner.case.get("fail_write"):
                    owner.writes[name] = text[:7]
                    raise OSError("configured output write failure")
                owner.writes[name] = owner.writes.get(name, "") + text
                return len(text)

            def close(self):
                if not self.closed:
                    owner.events.append(["close_write", name])
                    super().close()
                    if owner.case.get("fail_close"):
                        raise OSError("configured output close failure")

        return Writer()

    def stdout(self, text):
        self.events.append(["stdout"])
        if self.case.get("fail_stdout"):
            self.output.write(text[:7])
            raise OSError("configured stdout failure")
        return self.output.write(text)


def snapshot(machine):
    # Iterators are observed by their consumption in receipts, not their address.
    pending = machine.pending
    if pending is not None and not isinstance(pending, (list, tuple, set, dict, str)):
        machine.pending = "<iterator>"
        try:
            return state(machine)
        finally:
            machine.pending = pending
    return state(machine)


def observe(case, *, full_host_boundary=False):
    tr, instances = CommandTranscript(case), []
    original_path = sys.path[:]
    original_configuration, original_derive = F_SETTINGS.configuration, F_SETTINGS.derive
    original_host = F_PLACEMENT.default_threads

    def missing_host_file(*args):
        raise OSError("synthetic host sysfs unavailable")

    class Captured(F_MACHINE.Machine):
        def __init__(self, *args, **kwargs):
            instances.append(self)
            super().__init__(*args, **kwargs)

    def configuration(*args):
        tr.calls.append(["configuration", *args[1:]])
        result = original_configuration(*args)
        tr.calls.append(["configuration_return", exact(result)])
        return result

    def derive(*args):
        tr.calls.append(["derive", args[1]])
        result = original_derive(*args)
        tr.calls.append(["derive_return", exact(result)])
        return result

    try:
        with (
            f_api(tr.logs) as logs,
            patch.dict(
                sys.modules,
                machine=SimpleNamespace(Machine=Captured),
                logparse=SimpleNamespace(ledger=logs.ledger),
            ),
            patch.object(F_MACHINE, "open", tr.open, create=True),
            patch.object(F_MACHINE, "json", SimpleNamespace(load=lambda f: tr.read_json(f.name))),
            patch.object(F_MACHINE, "os", SimpleNamespace(path=SimpleNamespace(exists=tr.exists))),
            patch.object(
                F_PLACEMENT,
                "default_threads",
                original_host if full_host_boundary else lambda _device: tr.host(),
            ),
            patch.object(F_PLACEMENT, "open", missing_host_file, create=True),
            patch.object(F_PLACEMENT, "os", SimpleNamespace(cpu_count=tr.host)),
            patch.object(F_SETTINGS, "open", tr.open, create=True),
            patch.object(F_SETTINGS, "json", SimpleNamespace(load=lambda f: tr.read_json(f.name))),
            patch.object(
                F_SETTINGS,
                "os",
                SimpleNamespace(path=SimpleNamespace(abspath=lambda p: p), replace=tr.replace),
            ),
            patch.object(F_SETTINGS, "configuration", configuration),
            patch.object(F_SETTINGS, "derive", derive),
            patch.object(sys, "argv", ["settings.py", "/case/run", *case["argv"]]),
            patch.object(sys, "stdout", SimpleNamespace(write=tr.stdout)),
        ):
            F_SETTINGS.main()
        outcome = ["return"]
    except Exception as error:
        outcome = [type(error).__name__, str(error)]
    finally:
        sys.path[:] = original_path
    return {
        "outcome": outcome,
        "events": tr.events,
        "calls": tr.calls,
        "reads": tr.reads,
        "writes": tr.writes,
        "stdout": tr.output.getvalue(),
        "machines": [snapshot(m) if hasattr(m, "_replays") else exact(vars(m)) for m in instances],
    }


DATA = json.loads((FIXTURES / "settings-command-goldens.json").read_text())


@pytest.mark.parametrize("case", DATA["cases"], ids=lambda c: c["name"])
def test_f_settings_command(case):
    assert observe(case) == case["expected"][runtime_key()]


HOST_SUPPLEMENT = json.loads((FIXTURES / "settings-command-host-supplement.json").read_text())


def test_f_original_host_boundary():
    case = next(c for c in DATA["cases"] if c["name"] == "host-needed")
    assert observe(case, full_host_boundary=True) == HOST_SUPPLEMENT["expected"][runtime_key()]
