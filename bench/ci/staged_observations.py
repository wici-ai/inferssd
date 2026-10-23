"""Read native JSONL through a bounded broker stage and use the production consumer."""

from pathlib import Path
import time

from ci.contract import digest, save
from ci.staged_smoke import collect_log


def observe(session, config, model, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = Path(config["path"])
    if not path.is_absolute() or ".." in path.parts or path.suffix != ".jsonl":
        raise ValueError("explicit reviewed native observation path required")
    stage = session.start("candidate-observations-" + model, "test",
                          ["/usr/bin/cat", str(path)], {}, config["cwd"], 30)
    completed = session.wait(stage, deadline=time.monotonic() + 30)
    save(output / "stage.json", completed)
    collect_log(session, stage, output)
    if completed.get("returncode") != 0 or completed.get("cleanup_confirmed") is not True:
        raise RuntimeError("native observation extraction failed")
    events = output / "native-events.jsonl"
    events.write_bytes((output / "stdout.log").read_bytes())
    # These imports belong to the pinned installed controller wheel, never the
    # candidate source tree. A server.log is intentionally absent here.
    from inferssd.engine.runtime import LocalObservations
    from inferssd.telemetry.controller import NativeObservationSession
    arm = NativeObservationSession(files=LocalObservations()).read_arm_cached(output, 0)
    if not arm["groups"] or not arm["trace"] or not arm["prof"]:
        raise ValueError("production observer did not consume native route and profile records")
    receipt = dict(status="passed", source="native-events.jsonl", sha256=digest(events),
                   groups=len(arm["groups"]), trace=len(arm["trace"]), prof=len(arm["prof"]),
                   legacy_log_present=False)
    save(output / "observation.json", receipt)
    return receipt
