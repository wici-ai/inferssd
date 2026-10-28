"""One fixed 512/256 checkpoint arm inside an already leased StageSession.

This is not the final three-device gate or a run_final wrapper. The caller owns
build/model/template/settings/host identity and comparison, and retains the same
job lease across base and candidate. No completion is retried.
"""

from dataclasses import asdict
from pathlib import Path
import time
import re

import final_chat
import run_final
from ci.contract import digest, revision, save
from ci import mac_host_gate
from ci.staged_chat import NativeHTTP, record


IDENTITY = ("worker_pid", "worker_uid", "birth_identity", "scope_identity")


def verify_runtime(stage, observed):
    if any(observed.get(key) != getattr(stage, key) for key in IDENTITY):
        raise ValueError("native runtime identity differs from the started server")
    if observed.get("running", True) is not True or observed.get("ready") is not True:
        raise ValueError("native server is not alive with its owned listener")


def verify_stopped(stopped):
    """Classify cleanup evidence; never promote a failed measurement.

    Linux's guardian owns systemd-run. An explicit stop can make that launcher
    return 1 even though the server was alive until the controller stopped it.
    Only its exact watch/guardian cleanup receipt admits that nonzero code.
    """
    controlled = "InterruptedError: stage stopped by trusted controller"
    receipt = stopped.get("receipt")
    receipt = receipt if isinstance(receipt, dict) else {}
    code = stopped.get("returncode")
    cleanup = stopped.get("cleanup")
    cleanup = cleanup if isinstance(cleanup, dict) else {}
    job = stopped.get("job_id", "")
    mac_requested = (
        stopped.get("state") == "failed" and stopped.get("reason") == "requested stop"
        and code == 0 and isinstance(job, str) and re.fullmatch(r"[0-9a-f]{32}", job)
        and stopped.get("job_root") == stopped.get("scope_identity") == "/Library/InferSSD-CI/jobs/" + job
        and stopped.get("worker_uid") == 62002
        and cleanup.get("confirmed") is True and cleanup.get("remaining") == []
        and cleanup.get("worker_user_domain_retired", {}).get("returncode") == 0)
    if (stopped.get("running") is not False or type(code) is not int
            or stopped.get("cleanup_confirmed") is not True
            or stopped.get("state") == "quarantined"
            or (stopped.get("state") == "failed" and not mac_requested)
            or stopped.get("error") or stopped.get("cleanup_error")
            or receipt.get("cleanup_error")
            or receipt.get("cleanup_confirmed", True) is not True
            or receipt.get("error") not in (None, "", controlled)):
        raise ValueError("native benchmark did not stop normally and cleanly")
    if code == 0:
        return "mac-requested-stop" if mac_requested else "normal-exit"
    guardian = receipt.get("guardian")
    guardian = guardian if isinstance(guardian, dict) else {}
    cleanup = guardian.get("cleanup")
    cleanup = cleanup if isinstance(cleanup, dict) else {}
    state = cleanup.get("state")
    state = state if isinstance(state, dict) else {}
    unit = receipt.get("unit")
    if (code == 1 and receipt.get("returncode") == 1
            and receipt.get("error") == controlled
            and receipt.get("cleanup_confirmed") is True
            and guardian.get("returncode") == 1
            and guardian.get("controller_disconnected") is True
            and guardian.get("cleanup_confirmed") is True and not guardian.get("error")
            and isinstance(unit, str) and unit
            and guardian.get("unit") == unit == cleanup.get("unit")
            and stopped.get("scope_identity") == "/system.slice/" + unit
            and receipt.get("cleanup") == cleanup
            and cleanup.get("processes_gone") is True
            and state.get("MainPID") == "0" and state.get("ActiveState") == "inactive"):
        return "trusted-controller-stop"
    raise ValueError("native benchmark did not stop normally and cleanly")


def benchmark(session, side, config, output, *, host_policy=None):
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
    tier = config.get("tier", "512")
    prefix = config.get("stage_prefix", side)
    if tier not in ("512", "4096") or not re.fullmatch(r"[a-z0-9-]{1,64}", prefix):
        raise ValueError("fixed prompt tier and bounded stage identity required")
    # Validate the trusted asset before a candidate process starts. record()
    # and read_result() independently retain and recheck the same fixed hash.
    final_chat.prompt_path(tier)
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
                  tier=tier, output_tokens=256, samples_per_tier=1,
                  completion_requests=0, performance_acceptance=False,
                  cleanup_confirmed=False, formal_samples=0,
                  prompt_sha256=final_chat.PROMPTS[tier][1])
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
        if host_policy is not None:
            mac_host_gate.before_launch(output, host_policy)
        stage = session.start(prefix + "-benchmark-" + tier, "server", config["argv"],
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
            probe = session.call(stage, "stage.probe", port=config["port"], light=True)
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
        if host_policy is not None:
            mac_host_gate.capture(output, "host-state-before-request.json")
        try:
            # The recorder saves raw success/failure; never promote its return
            # value directly into the checked measurement used for comparison.
            record(http, tier, output / "result.json", request_timeout)
        finally:
            after = session.call(stage, "stage.probe", port=config["port"])
            retain("runtime-after", after)
            verify_runtime(stage, after)
            if host_policy is not None:
                mac_host_gate.capture(output, "host-state-after-request.json")
        checked = run_final.read_result(output / "result.json", tier, request_timeout)
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
                result["stop_classification"] = verify_stopped(stopped)
                if host_policy is not None:
                    mac_host_gate.capture(output, "host-state-after-cleanup.json")
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
