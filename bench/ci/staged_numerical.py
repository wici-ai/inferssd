"""Run a bounded real backend/CPU numerical comparison before model readiness."""

import csv
import io
from pathlib import Path
import time

from ci.contract import save
from ci.impact import CASES, MARKERS
from ci.staged_smoke import collect_log


def validate_csv(raw, backend, operation=None, params=None):
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    executed = [row for row in rows if row.get("backend_name") == backend
                and row.get("test_mode") == "test" and row.get("supported") == "1"]
    if not executed:
        raise ValueError("native numerical gate executed no supported tests on the requested GPU")
    if any(row.get("error_message") or row.get("passed") not in (None, "1", "true") for row in executed):
        raise ValueError("native numerical comparison failed")
    if operation is not None and any(row.get("op_name") != operation for row in executed):
        raise ValueError("executed operator differs from the selected numerical case")
    if params and any(params not in row.get("op_params", "") for row in executed):
        raise ValueError("executed format differs from the selected numerical case")
    return len(executed)


def numerical(session, config, build, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    backend = config["backend"]
    if backend not in ("CUDA0", "MTL0"):
        raise ValueError("readiness needs an explicit supported physical backend")
    binary = config["binary"]
    identities = {image["runtime_path"]: image for image in build["images"]}
    if binary not in identities or Path(binary).name != "test-backend-ops":
        raise ValueError("numerical binary is not bound to the sealed build")
    # ADD is a small actual GPU operation checked against the CPU reference.
    # It establishes backend operation, not complete model numerical parity.
    argv = [binary, "test", "-b", backend, "-o", "ADD", "-p", "type=f32", "--output", "csv"]
    result = dict(schema_version=1, purpose="device-readiness", status="failed",
                  backend=backend, binary=identities[binary], argv=argv)
    stage = None
    try:
        stage = session.start("candidate-numerical", "test", argv, {}, config["cwd"], 120)
        completed = session.wait(stage, deadline=time.monotonic() + 120)
        save(output / "stage.json", completed)
        collect_log(session, stage, output)
        if completed.get("returncode") != 0 or completed.get("cleanup_confirmed") is not True:
            raise RuntimeError("native numerical stage failed or did not clean up")
        count = validate_csv((output / "stdout.log").read_bytes(), backend)
        result.update(status="passed", tests_executed=count, cleanup_confirmed=True)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        if stage is not None and not session.closed:
            stopped = session.call(stage, "stage.stop", timeout=50)
            save(output / "stopped.json", stopped)
    finally:
        save(output / "numerical.json", result)
    if result["status"] != "passed":
        raise RuntimeError(result["error"])
    return result


def selected_numerical(session, config, build, cases, output):
    """Each required format/operator is a distinct real CPU-oracle GPU gate."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    backend = config["backend"]
    if backend not in ("CUDA0", "MTL0") or not cases or len(cases) != len(set(cases)):
        raise ValueError("explicit GPU backend and unique nonempty cases required")
    images = {Path(item["runtime_path"]).name: item for item in build["images"]}
    results = {}
    for case in cases:
        target, operation, params = CASES[case]
        if target not in images:
            raise ValueError("selected numerical target is absent from the sealed image set")
        folder = output / case
        folder.mkdir()
        argv = [images[target]["runtime_path"]]
        if operation:
            argv += ["test", "-b", backend, "-o", operation]
            if params:
                argv += ["-p", params]
            argv += ["--output", "csv"]
        elif case.startswith("cuda-") and backend != "CUDA0":
            raise ValueError("CUDA tier oracle cannot establish a Metal numerical pass")
        stage = session.start("candidate-numerical-" + case, "test", argv, {}, config["cwd"], 900)
        completed = session.wait(stage, deadline=time.monotonic() + 900)
        save(folder / "stage.json", completed)
        collect_log(session, stage, folder)
        if completed.get("returncode") != 0 or completed.get("cleanup_confirmed") is not True:
            raise RuntimeError("selected numerical stage failed or was not clean")
        raw = (folder / "stdout.log").read_bytes()
        if operation:
            count = validate_csv(raw, backend, operation, params)
        else:
            marker = MARKERS[case]
            if sum(line.startswith(marker) for line in raw.splitlines()) != 1:
                raise ValueError("real GPU oracle success marker missing or duplicated")
            count = 1
        results[case] = dict(status="passed", tests_executed=count, binary=images[target],
                             argv=argv, cleanup_confirmed=True)
    save(output / "selected-numerical.json", results)
    return results
