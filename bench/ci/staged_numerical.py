"""Run a bounded real backend/CPU numerical comparison before model readiness."""

import csv
import io
from pathlib import Path
import time

from ci.contract import save
from ci.staged_smoke import collect_log


def validate_csv(raw, backend):
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    executed = [row for row in rows if row.get("backend_name") == backend
                and row.get("test_mode") == "test" and row.get("supported") == "1"]
    if not executed:
        raise ValueError("native numerical gate executed no supported tests on the requested GPU")
    if any(row.get("error_message") for row in executed):
        raise ValueError("native numerical comparison failed")
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
