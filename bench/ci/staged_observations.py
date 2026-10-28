"""Read native JSONL through a bounded broker stage and use the production consumer."""

from pathlib import Path
import time

from ci.contract import digest, save
from ci.staged_smoke import collect_log


def mac_cache_observations(path):
    """Validate the Mac cache's existing events, without inventing route traces."""
    from inferssd.telemetry.jsonl import effective_settings, read_events

    events = list(read_events(Path(path).read_text().splitlines()))
    settings = effective_settings(events)["values"]
    budget = settings["host.expert_cache_mib"] << 20
    samples = [event.payload for event in events if event.name == "ssd.mac.cache"]
    fields = ("requests", "hits", "misses", "cache_bytes", "budget_bytes", "read_bytes", "peak_staging_bytes")
    monotonic = ("requests", "hits", "misses", "read_bytes")
    if not samples:
        raise ValueError("native Mac cache observations are missing")
    previous = dict.fromkeys(monotonic, 0)
    for index, sample in enumerate(samples):
        if any(type(sample.get(key)) is not int or sample[key] < 0 for key in fields):
            raise ValueError("native Mac cache counters must be nonnegative integers")
        if (sample["requests"] != sample["hits"] + sample["misses"]
                or sample["budget_bytes"] != budget or sample["cache_bytes"] > budget):
            raise ValueError("native Mac cache accounting or typed budget differs")
        if any(sample[key] < previous[key] for key in monotonic):
            raise ValueError("native Mac cache counters decreased")
        if sample.get("reason") != ("shutdown" if index == len(samples) - 1 else "periodic"):
            raise ValueError("native Mac cache requires one final shutdown sample")
        previous = sample
    final = samples[-1]
    if not final["requests"] or not final["read_bytes"]:
        raise ValueError("native Mac cache did not record expert requests and reads")
    return dict(observation_kind="mac-cache", cache_samples=len(samples),
                **{key: final[key] for key in fields})


def observe(session, config, model, output, *, platform):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = Path(config["path"])
    if not path.is_absolute() or ".." in path.parts or path.suffix != ".jsonl":
        raise ValueError("explicit reviewed native observation path required")
    if platform == "darwin":
        # The Mac broker maps this to its fixed Xcode Python. No extra sandbox
        # executable permission is needed to read the worker's own telemetry.
        argv = ["/usr/bin/python3", "-I", "-c",
                "import pathlib,sys;sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())", str(path)]
    elif platform == "linux":
        argv = ["/usr/bin/cat", str(path)]
    else:
        raise ValueError("explicit supported observation platform required")
    stage = session.start("candidate-observations-" + model, "test",
                          argv, {}, config["cwd"], 30)
    completed = session.wait(stage, deadline=time.monotonic() + 30)
    save(output / "stage.json", completed)
    collect_log(session, stage, output)
    if completed.get("returncode") != 0 or completed.get("cleanup_confirmed") is not True:
        raise RuntimeError("native observation extraction failed")
    events = output / "native-events.jsonl"
    events.write_bytes((output / "stdout.log").read_bytes())
    # These imports belong to the pinned installed controller wheel, never the
    # candidate source tree. A server.log is intentionally absent here.
    if platform == "darwin":
        summary = mac_cache_observations(events)
    else:
        from inferssd.engine.runtime import LocalObservations
        from inferssd.telemetry.controller import NativeObservationSession
        arm = NativeObservationSession(files=LocalObservations()).read_arm_cached(str(output), 0)
        if not arm["groups"] or not arm["trace"] or not arm["prof"]:
            raise ValueError("production observer did not consume native route and profile records")
        summary = dict(groups=len(arm["groups"]), trace=len(arm["trace"]), prof=len(arm["prof"]))
    receipt = dict(status="passed", source="native-events.jsonl", sha256=digest(events),
                   legacy_log_present=False, **summary)
    save(output / "observation.json", receipt)
    return receipt
