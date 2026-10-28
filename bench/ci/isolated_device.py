#!/usr/bin/env python3
"""Call an independently provisioned trusted broker, never candidate argv on a runner."""

import argparse
import json
import os
import re
import secrets
import stat
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.contract import HASH, digest, load, policy_read, runner_config, save
from ci.private_controller import bind_context, source_manifest

REQUIRED_GATES = (
    "different_execution_uid",
    "controller_home_denied",
    "runner_credentials_denied",
    "control_write_denied",
    "evidence_write_denied",
    "sudo_denied",
    "host_sockets_denied",
    "external_network_denied",
    "user_service_escape_denied",
    "process_signal_denied",
    "readonly_model_inputs",
    "cleanup_and_reset_verified",
    "controller_disconnect_cleanup",
    "original_lock_inode",
    "stop_quarantine",
)


def protected(path, executable=False):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("broker and isolation proof paths must be absolute")
    for item in (path, *path.parents):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError(
                "broker/proof and ancestors must be root-owned, non-symlink and not group/world writable"
            )
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or executable and not info.st_mode & 0o111:
        raise ValueError("broker/proof must be a regular file of the required type")
    return path


def manifest_read_path(config):
    """Selected Mac runners read one fixed copy; all legacy paths stay unchanged."""
    logical = config["manifest_path"]
    if config.get("paths", {}).get("platform") != "darwin" or config.get("purpose") != "selected-integration":
        return Path(logical)
    match = re.fullmatch(r"/Library/InferSSD-CI/private/job-manifests/([0-9a-f]{32})\.json", logical) if isinstance(logical, str) else None
    if match is None:
        raise ValueError("Mac manifest read view requires the exact private job identity")
    return Path("/Library/InferSSD-CI/inputs/operator-" + match[1] + "/manifest.json")


def protected_manifest(config):
    path = protected(manifest_read_path(config))
    if (config.get("paths", {}).get("platform") == "darwin" and config.get("purpose") == "selected-integration"
            and path.stat().st_mode & 0o222):
        raise ValueError("Mac runner manifest copy must be root-owned and readonly")
    if digest(path) != config["manifest_sha256"]:
        raise ValueError("root broker manifest hash differs")
    return path


def isolation(config, nonce, describe, controller_uid):
    spec = config.get("isolation", {})
    if (
        spec.get("backend") != "external-reviewed-broker-v1"
        or describe.get("schema_version") != 1
        or describe.get("nonce") != nonce
        or type(describe.get("broker_uid")) is not int
        or describe.get("broker_uid") != 0
        or describe.get("controller_uid") != controller_uid
        or type(describe.get("execution_uid")) is not int
        or describe["execution_uid"] in (0, controller_uid)
        or describe.get("network") != "loopback-only"
        or describe.get("candidate_access") != "sandboxed-stages"
        or describe.get("proof_sha256") != spec.get("proof_sha256")
        or any(
            describe.get("gates", {}).get(gate) is not True for gate in REQUIRED_GATES
        )
    ):
        raise ValueError(
            "real per-stage OS isolation is not provisioned/proven; no physical execution"
        )
    return describe


def broker_call(path, request, timeout):
    # No shell, arbitrary command, job token, user HOME, DBus or loader transport.
    result = subprocess.run(
        [str(path)],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        timeout=timeout,
        env={"PATH": "/usr/bin:/bin", "LANG": "C"},
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "isolated broker refused execution (see operator broker log)"
        )
    if len(result.stdout) > 1024 * 1024:
        raise ValueError("oversized broker response")
    return json.loads(result.stdout)


def run(args, env=os.environ):
    policy = policy_read(load(args.policy))
    plan = load(args.plan)
    bind_context(plan, policy, env)
    if plan.get("policy_sha256") != digest(args.policy) or plan.get(
        "control_revision"
    ) != source_manifest(args.control):
        raise ValueError("device plan does not match trusted controller source/policy")
    config = runner_config(load(args.config), args.device)
    if digest(args.config) != plan["device_config_sha256"][args.device]:
        raise ValueError("host config differs from exact admitted identity")
    spec = config.get("isolation", {})
    broker = protected(spec.get("broker_path", ""), executable=True)
    proof = protected(spec.get("proof_path", ""))
    if (
        not HASH.fullmatch(spec.get("broker_sha256", ""))
        or digest(broker) != spec["broker_sha256"]
        or not HASH.fullmatch(spec.get("proof_sha256", ""))
        or digest(proof) != spec["proof_sha256"]
    ):
        raise ValueError("trusted broker/proof identity differs")
    evidence = load(proof)
    if (
        evidence.get("device") != args.device
        or evidence.get("platform") != config["platform"]
        or any(evidence.get("gates", {}).get(g) is not True for g in REQUIRED_GATES)
    ):
        raise ValueError("host isolation evidence is incomplete")
    if digest(args.source_bundle) != plan.get("source_bundle_sha256"):
        raise ValueError("source bundle differs from the admitted immutable inputs")
    nonce = secrets.token_hex(32)
    describe = isolation(
        config,
        nonce,
        broker_call(broker, {"operation": "describe", "nonce": nonce}, 15),
        os.getuid(),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    # Fixed protocol: the external root broker must own/validate inputs, execute
    # candidate stages under a different sandboxed UID, and protect coordinator
    # code/results. Running this directory wholesale as the worker is insufficient.
    result = broker_call(
        broker,
        {
            "operation": "run",
            "nonce": nonce,
            "plan": plan,
            "device": args.device,
            "config": str(args.config.resolve()),
            "source_bundle": str(args.source_bundle.resolve()),
            "control": str(args.control.resolve()),
            "output": str(args.output.resolve()),
        },
        43000,
    )
    if (
        result.get("nonce") != nonce
        or result.get("plan") != plan
        or result.get("device") != args.device
        or result.get("status") != "passed"
        or result.get("cleanup_confirmed") is not True
        or result.get("controller_isolation") != describe
    ):
        raise ValueError("broker execution/cleanup identity is incomplete")
    save(args.output / "isolation-receipt.json", result)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("control", "policy", "plan", "config", "source-bundle", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--device", required=True)
    raise SystemExit(run(parser.parse_args()))
