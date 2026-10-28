"""Reference-only whole-command observer. Independent of the candidate package."""

import ast
import builtins
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from bootstrap_record_cases import CASES

FIXTURES = Path(__file__).parent / "fixtures"
ORACLE = FIXTURES / "oracle"


class Stream:
    def __init__(self, host, path, text):
        self.host, self.path = host, path
        self.rows = text.splitlines(keepends=True)
        self.position = 0
        self.closed = False

    def __iter__(self):
        self.host.events.append(["iter", self.path])
        return self

    def __next__(self):
        self.host.fault(f"{self.path}:next:{self.position}")
        if self.position == len(self.rows):
            self.host.events.append(["eof", self.path])
            raise StopIteration
        row = self.rows[self.position]
        self.host.events.append(["next", self.path, self.position, row])
        self.position += 1
        return row

    def readline(self):
        self.host.events.append(["readline", self.path, self.position])
        self.host.fault(f"{self.path}:readline:{self.position}")
        if self.position == len(self.rows):
            return ""
        row = self.rows[self.position]
        self.position += 1
        return row

    def close(self):
        if not self.closed:
            self.host.events.append(["close", self.path])
            self.closed = True

    def __del__(self):
        self.close()


class Host:
    def __init__(self, case):
        self.case = case
        self.events = []
        self.output = ""
        self.writes = 0

    def fault(self, key):
        name = self.case.get("faults", {}).get(key)
        if name:
            self.events.append(["fault", key, name])
            raise getattr(builtins, name)(key)

    def open(self, path, *args, **kwargs):
        self.events.append(["open", path, list(args), kwargs])
        self.fault(f"{path}:open")
        if path not in self.case["files"]:
            raise FileNotFoundError(path)
        return Stream(self, path, self.case["files"][path])

    def gzip_open(self, path, *args, **kwargs):
        self.events.append(["gzip_open", path, list(args), kwargs])
        self.fault(f"{path}:open")
        if path not in self.case["files"]:
            raise FileNotFoundError(path)
        return Stream(self, path, self.case["files"][path])

    def write(self, text):
        self.events.append(["write", text])
        self.writes += 1
        if self.writes == self.case.get("write_fault"):
            raise OSError("synthetic stdout fault")
        self.output += text
        return len(text)

    def emit(self, *args, **kwargs):
        builtins.print(*args, file=self, **kwargs)


# The snapshot is taken while exception frames still exist. Implicit stream
# cleanup caused only by later test-GC cannot mutate a returned transcript.
def observe(case, execute=None):
    host = Host(case)
    result = None
    error = None
    try:
        result = (execute or execute_reference)(case, host)
    except Exception as exception:
        error = [type(exception).__name__, str(exception)]
        return copy.deepcopy(
            {"events": host.events, "stdout": host.output, "result": None, "error": error}
        )
    return copy.deepcopy(
        {"events": host.events, "stdout": host.output, "result": result, "error": error}
    )


def execute_reference(case, host):
    name = "coresets" if case["kind"] in ("coresets", "parse_list") else "ubprobe"
    source = (ORACLE / f"{name}_reference.py").read_text()

    def importing(module, *args, **kwargs):
        if module == "sys":
            return SimpleNamespace(argv=case.get("argv", []))
        if module == "gzip":
            return SimpleNamespace(open=host.gzip_open)
        return builtins.__import__(module, *args, **kwargs)

    namespace = {
        "__name__": "__main__",
        "open": host.open,
        "print": host.emit,
        "__builtins__": {**vars(builtins), "__import__": importing},
    }
    tree = ast.parse(source)
    if case["kind"] in ("parse_list", "reached"):
        tree.body = [
            n
            for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
            or (
                isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "TIER" for t in n.targets)
            )
        ]
        exec(compile(tree, str(ORACLE / f"{name}_reference.py"), "exec"), namespace)
        value = case["value"] if case["kind"] == "parse_list" else iter(case["lines"])
        result = namespace[case["kind"]](value)
        return list(result) if isinstance(result, tuple) else result
    exec(compile(tree, str(ORACLE / f"{name}_reference.py"), "exec"), namespace)
    return None


def runtime_key():
    version = sys.version_info[:2]
    if version not in ((3, 9), (3, 12)):
        pytest.skip("recorded reference transcripts cover Python 3.9 and 3.12 only")
    return f"{version[0]}.{version[1]}"


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_reference_bootstrap_records(case):
    expected = json.loads((FIXTURES / "bootstrap-records-goldens.json").read_text())
    assert observe(case) == expected[runtime_key()][case["name"]]


def test_original_reference_source_identity():
    manifest = json.loads((ORACLE / "bootstrap-records-manifest.json").read_text())
    for source in manifest["sources"]:
        assert (
            hashlib.sha256((ORACLE / source["oracle"]).read_bytes()).hexdigest() == source["sha256"]
        )
