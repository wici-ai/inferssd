"""One fixed 512/256 checkpoint arm inside an already leased StageSession.

This is not the final three-device gate or a run_final wrapper. The caller owns
build/model/template/settings/host identity and comparison, and retains the same
job lease across base and candidate. No completion is retried.
"""

from dataclasses import asdict
from pathlib import Path
import time

import final_chat
import run_final
from ci.contract import digest, revision, save
from ci.staged_chat import NativeHTTP, record


IDENTITY = ("worker_pid", "worker_uid", "birth_identity", "scope_identity")


def verify_runtime(stage, observed):
    if any(observed.get(key) != getattr(stage, key) for key in IDENTITY):
        raise ValueError("native runtime identity differs from the started server")
    if observed.get("running", True) is not True or observed.get("ready") is not True:
        raise ValueError("native server is not alive with its owned listener")


def benchmark(session, side, config, output):
    """Run exactly one arm, retaining raw failure and cleanup evidence.

    Config requires model, full revision, argv, environment, cwd, port,
    timeout_s (whole server stage), request_timeout_s, os_page_cache_treatment.
    Broker probe(port) must report explicit listener readiness. This function
    never opens/finishes/closes the caller's shared job. A failed result must stop
    the caller's comparison; cleanup_confirmed alone is not a performance pass.
    """
    if side not in ("base", "candidate"):
        raise ValueError("explicit base or candidate benchmark side required")
    revision(config["revision"])
    timeout, request_timeout = config["timeout_s"], config["request_timeout_s"]
    final_chat.bounded_timeout(timeout)
    final_chat.bounded_timeout(request_timeout)
    if timeout <= request_timeout:
        raise ValueError("server deadline must leave time for model startup")
    if type(config["port"]) is not int or not 1 <= config["port"] <= 65535:
        raise ValueError("explicit server port required")
    cache = config["os_page_cache_treatment"]
    if not isinstance(cache, str) or not cache.strip():
        raise ValueError("record the actual OS page cache treatment; do not assume cold")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema_version=1, purpose="ci-checkpoint", status="failed",
                  side=side, model=config["model"], revision=config["revision"],
                  tier="512", output_tokens=256, samples_per_tier=1,
                  completion_requests=0, performance_acceptance=False,
                  cleanup_confirmed=False)
    runtime = dict(schema_version=1, purpose="staged-benchmark-runtime", side=side,
                   job_id=session.job_id, observations={},
                   launch={key: config[key] for key in (
                       "argv", "environment", "cwd", "port", "timeout_s", "request_timeout_s")},
                   os_page_cache_treatment=cache,
                   cache_condition="fresh-server-first-completion; OS page cache treatment recorded")
    stage, http = None, None

    def retain(name, value):
        path = output / (name + ".json")
        save(path, value)
        runtime["observations"][name] = dict(path=path.name, sha256=digest(path))

    try:
        stage = session.start(side + "-benchmark-512", "server", config["argv"],
                              config["environment"], config["cwd"], timeout)
        retain("started", asdict(stage))
        http = NativeHTTP(session, stage, config["port"])
        startup_deadline = time.monotonic() + timeout - request_timeout
        health_attempt = 0
        while True:
            if time.monotonic() >= startup_deadline:
                raise TimeoutError("native benchmark startup deadline")
            state = session.call(stage, "stage.poll")
            if state.get("running") is not True:
                retain("startup-exit", state)
                raise RuntimeError("native server exited before readiness")
            probe = session.call(stage, "stage.probe", port=config["port"])
            if "ready" not in probe:
                retain("startup-probe", probe)
                raise ValueError("broker probe must report explicit listener readiness")
            if probe["ready"] is True:
                verify_runtime(stage, probe)
                remaining = startup_deadline - time.monotonic()
                if remaining < 1:
                    raise TimeoutError("native benchmark startup deadline")
                health = http.request("GET", "/health", None, min(2, int(remaining)),
                                      allow_health_timeout=True)
                health_attempt += 1
                retain(f"health-attempt-{health_attempt:04d}", health)
                if time.monotonic() >= startup_deadline:
                    raise TimeoutError("native benchmark startup deadline")
                if health["status"] == 200:
                    retain("health", health)
                    break
            time.sleep(0.2)
        before = session.call(stage, "stage.probe", port=config["port"])
        retain("runtime-before", before)
        verify_runtime(stage, before)
        try:
            # The recorder saves raw success/failure; never promote its return
            # value directly into the checked measurement used for comparison.
            record(http, "512", output / "result.json", request_timeout)
        finally:
            after = session.call(stage, "stage.probe", port=config["port"])
            retain("runtime-after", after)
            verify_runtime(stage, after)
        checked = run_final.read_result(output / "result.json", "512", request_timeout)
        if http.completions != 1:
            raise ValueError("benchmark must contain one completion on the fresh server")
        result.update(status="passed", result=checked,
                      result_artifact=dict(path="result.json", sha256=digest(output / "result.json")))
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        result["completion_requests"] = http.completions if http is not None else 0
        if stage is not None:
            try:
                stopped = session.call(stage, "stage.stop", timeout=50)
                retain("stopped", stopped)
                result["cleanup_confirmed"] = stopped.get("cleanup_confirmed") is True
                # Linux marks an explicit controller stop as InterruptedError
                # in its watch receipt. Its passed flag is not an exit verdict.
                if (stopped.get("running") is not False or stopped.get("returncode") != 0
                        or not result["cleanup_confirmed"] or stopped.get("state") == "quarantined"):
                    raise RuntimeError("native benchmark did not stop normally and cleanly")
            except Exception as error:
                result.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
            # Drain each stream independently even if stop or the other stream
            # failed. An empty chunk without EOF must not hide incomplete logs.
            logs, errors = {}, {}
            for stream in ("stdout", "stderr"):
                path, offset = output / (stream + ".log"), 0
                try:
                    with path.open("xb") as target:
                        while True:
                            data, offset, eof = session.output(stage, stream, offset)
                            target.write(data)
                            if offset > 64 * 1024 * 1024:
                                raise ValueError("native benchmark log bound exceeded")
                            if eof:
                                break
                            if not data:
                                raise ValueError("stopped stage output did not reach EOF")
                    logs[stream] = dict(path=path.name, bytes=offset, sha256=digest(path))
                except Exception as error:
                    errors[stream] = f"{type(error).__name__}: {error}"
            runtime["logs"] = logs
            if errors:
                result.update(status="failed", log_errors=errors)
        runtime.update(status=result["status"], completion_requests=result["completion_requests"],
                       cleanup_confirmed=result["cleanup_confirmed"])
        save(output / "runtime-receipt.json", runtime)
        result["runtime_receipt"] = dict(path="runtime-receipt.json",
                                         sha256=digest(output / "runtime-receipt.json"))
        save(output / "benchmark.json", result)
    return result
