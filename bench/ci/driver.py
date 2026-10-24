#!/usr/bin/env python3
"""One real device: exact base/candidate builds, native gates, one CI sample/cell."""

import argparse
from dataclasses import replace
import fcntl
import importlib.util
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

BENCH = Path(__file__).resolve().parents[1]
if os.environ.get("SSDASH_NATIVE_CORE_MANIFEST"):
    raise RuntimeError(
        "CI requires the pinned installed control core, not an ambient bundle"
    )
sys.path.insert(0, str(BENCH))
import compare_outputs
import native_common as native
import run_final
import inferssd
from ci.contract import MODELS, digest, load, policy_read, runner_config, save
from inferssd.engine.owned import OwnedProcess
from inferssd.launch.types import Resolution
from inferssd.platform.policy import Policy
from inferssd.platform.run_directory import reserve
from inferssd.platform.worker import cleanup_confirmed

_spec = importlib.util.spec_from_file_location(
    "inferssd_device_gate", BENCH / "device/gate.py"
)
GATE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GATE)


def storage(root, config):
    used = sum(
        p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink()
    )
    if (
        used > config["max_workspace_bytes"]
        or shutil.disk_usage(root).free < config["min_free_bytes"]
    ):
        raise RuntimeError("device CI storage bound exceeded; no result accepted")


def control_environment():
    return {
        k: v
        for k, v in os.environ.items()
        if k
        in (
            "PATH",
            "HOME",
            "USER",
            "LOGNAME",
            "LANG",
            "LC_ALL",
            "TMPDIR",
            "TZ",
            "XDG_RUNTIME_DIR",
            "DBUS_SESSION_BUS_ADDRESS",
        )
    }


def verify_control_install():
    installed = Path(inferssd.__file__).resolve().parent
    source = BENCH.parent / "framework/src/inferssd"
    for path in source.rglob("*"):
        if path.is_file() and path.suffix in (".py", ".json"):
            if digest(path) != digest(installed / path.relative_to(source)):
                raise ValueError(
                    "installed ownership/framework code differs from pinned control source"
                )


def command_stage(argv, directory, config, lock_fd, timeout, stop_intent):
    reservation = reserve(str(directory))
    policy = Policy.read(config["stage_policy"])
    if (
        policy.lock_path != config["lock_path"]
        or policy.stop_path != config["stop_path"]
        or policy.platform != config["platform"]
    ):
        raise ValueError(
            "stage policy must use the same actual device lease/STOP/platform"
        )
    policy = replace(policy, runtime_seconds=timeout)
    prefix = (
        ()
        if policy.platform == "darwin"
        else (
            "systemd-run",
            "--user",
            "--scope",
            "--quiet",
            "--unit=inferssd-pending.scope",
            "-p",
            "MemoryMax=" + str(policy.memory_budget_bytes),
            "-p",
            "MemorySwapMax=0",
        )
    )
    inherited = control_environment()
    resolution = Resolution(
        tuple(argv), (), tuple(inherited.items()), prefix, (), (), None, 0
    )
    owner = OwnedProcess(
        resolution,
        str(directory),
        policy,
        reservation=reservation,
        inherited_lock_fd=lock_fd,
        command_mode=True,
    )
    failure = None
    try:
        started = time.monotonic()
        while owner.poll() is None:
            if (
                stop_intent
                or Path(config["stop_path"]).exists()
                or time.monotonic() - started > timeout + 5
            ):
                raise InterruptedError("stage interrupted, STOP or deadline")
            storage(Path(config["workspace_root"]), config)
            time.sleep(0.5)
    except BaseException as error:
        failure = error
    finally:
        try:
            owner.stop()
        except BaseException:
            if not cleanup_confirmed(directory / "ownership.json"):
                Path(config["stop_path"]).touch(exist_ok=True)
            raise
    if failure:
        raise failure
    return load(directory / "ownership.json")


def expand(argv, values):
    return [word.format_map(values) for word in argv]


def git(source, *argv):
    return subprocess.check_output(["git", "-C", str(source), *argv], text=True).strip()


def run(args):
    verify_control_install()
    policy = policy_read(load(args.policy))
    policy["_sha256"] = digest(args.policy)
    plan, config = load(args.plan), runner_config(load(args.config), args.device)
    if (
        plan.get("purpose") != "ci"
        or plan.get("repository") != policy["repository"]
        or plan.get("policy_sha256") != policy["_sha256"]
        or plan.get("device_config_sha256", {}).get(args.device) != digest(args.config)
    ):
        raise ValueError(
            "plan/config/policy identity differs from the trusted preflight"
        )
    if plan.get("tiers") not in (["512"], ["512", "4096"]):
        raise ValueError("invalid CI prompt tiers")
    repository = args.repository.resolve(strict=True)
    if git(repository, "rev-parse", "HEAD") != plan["candidate"]:
        raise ValueError("checkout is not the preflight merge candidate")
    parents = git(repository, "show", "-s", "--format=%P", plan["candidate"]).split()
    if parents != [plan["base"], plan["head"]]:
        raise ValueError(
            "candidate must merge exactly the reviewed base and current PR head"
        )
    # No lease bypass: reject a changed inode and retain the same open description
    # across builds, lifecycle gates, tiny inference and every performance cell.
    lock_path, stop_path = Path(config["lock_path"]), Path(config["stop_path"])
    if not lock_path.is_file() or lock_path.is_symlink():
        raise ValueError(
            "provision the existing regular device lock before enabling CI"
        )
    with lock_path.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if stop_path.exists():
            raise InterruptedError("device STOP exists before CI")
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        root = Path(config["workspace_root"])
        root.mkdir(parents=True, exist_ok=True)
        owned = root / ("ci-" + uuid.uuid4().hex)
        owned.mkdir()
        storage(root, config)
        (output / "runner-config.json").write_bytes(args.config.read_bytes())
        save(
            output / "ci-manifest.json",
            {
                "schema_version": 1,
                "purpose": "ci",
                "device": args.device,
                "plan": plan,
                "policy_sha256": policy["_sha256"],
                "config_sha256": digest(args.config),
                "workspace": str(owned),
                "formal_samples": 0,
            },
        )
        interrupted = []
        handlers = {
            s: signal.signal(s, lambda n, _f: interrupted.append(n))
            for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        }
        success, builds = False, {}
        try:
            for side in ("base", "candidate"):
                revision = plan[side]
                source, build = owned / (side + "-source"), owned / (side + "-build")
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(repository),
                        "worktree",
                        "add",
                        "--detach",
                        str(source),
                        revision,
                    ],
                    check=True,
                )
                values = {
                    "python": sys.executable,
                    "control": str(BENCH.parent),
                    "config": str(args.config.resolve()),
                    "source": str(source),
                    "build": str(build),
                    "revision": revision,
                    "receipt": str(output / ("build-" + side + ".json")),
                    "results": str(output / ("validation-" + side + ".xml")),
                }
                command_stage(
                    expand(config["build"]["argv"], values),
                    output / ("stage-build-" + side),
                    config,
                    lock.fileno(),
                    config["build"]["timeout_s"],
                    interrupted,
                )
                receipt = load(values["receipt"])
                if (
                    receipt.get("status") != "passed"
                    or receipt.get("revision") != revision
                ):
                    raise ValueError(
                        "build did not produce a successful exact-revision receipt"
                    )
                if git(source, "diff", "--name-only", "HEAD"):
                    raise ValueError("build modified tracked source files")
                for model in (*MODELS, "tiny"):
                    profile = native.validate_profile(receipt["profiles"][model])
                    if (
                        profile["revision"] != revision
                        or "typed_settings" not in profile
                    ):
                        raise ValueError(
                            "build profile is not this typed source revision"
                        )
                    if any(
                        not Path(x["path"]).resolve().is_relative_to(build.resolve())
                        for x in (profile["binary"], *profile["images"])
                    ):
                        raise ValueError(
                            "build receipt refers to images outside this source's owned build"
                        )
                    expected_template = (
                        source / config["build"]["template_relatives"][model]
                    ).resolve()
                    if (
                        not expected_template.is_relative_to(source.resolve())
                        or Path(profile["template"]["path"]).resolve()
                        != expected_template
                    ):
                        raise ValueError(
                            "build receipt template is not bound to this exact source revision"
                        )
                    native.verify_inputs(profile)
                builds[side] = (receipt, values)
            if builds["base"][0]["toolchain"] != builds["candidate"][0]["toolchain"]:
                raise ValueError("base/candidate actual toolchain differs")
            for side in ("base", "candidate"):
                receipt, values = builds[side]
                command_stage(
                    expand(config["validation"]["argv"], values),
                    output / ("stage-validation-" + side),
                    config,
                    lock.fileno(),
                    config["validation"]["timeout_s"],
                    interrupted,
                )
                report = GATE.compare(
                    GATE.read_results(Path(values["results"])),
                    config["validation"]["baseline"],
                )
                report.update(revision=plan[side], cleanup_confirmed=True, purpose="ci")
                save(output / ("validation-" + side + ".json"), report)
                if not report["passed"] or report["counts"]["skipped"]:
                    raise ValueError(
                        "small-model numerical/lifecycle test set failed or skipped"
                    )
            # Tiny end-to-end numerical equality is separate from big-model performance.
            # Both builds and all compile/test commands have finished before any timing.
            for model in ("tiny", *MODELS):
                for side in ("base", "candidate"):
                    receipt, values = builds[side]
                    profile_path = output / (model + "-" + side + "-profile.json")
                    save(profile_path, receipt["profiles"][model])
                    config_run = {
                        "device": args.device,
                        "model": model,
                        "revision": plan[side],
                        "argv": [
                            sys.executable,
                            str(BENCH / "native_wrapper.py"),
                            "--profile",
                            str(profile_path),
                        ],
                        "request_timeout_s": config["request_timeout_s"],
                        "arm_timeout_s": config["arm_timeout_s"],
                    }
                    target = output / model / side
                    code = run_final.run(
                        config_run,
                        target,
                        lock_path,
                        stop_path,
                        order=("512",) if model == "tiny" else tuple(plan["tiers"]),
                        purpose="ci",
                        inherited_lock_fd=lock.fileno(),
                        environment=control_environment(),
                    )
                    if code:
                        raise RuntimeError("one CI attempt failed; no automatic retry")
                if model == "tiny":
                    from ci.compare import cell, comparable

                    a, ar = cell(
                        output, "tiny", "base", "512", plan["base"], args.device
                    )
                    b, br = cell(
                        output,
                        "tiny",
                        "candidate",
                        "512",
                        plan["candidate"],
                        args.device,
                    )
                    if comparable(ar) != comparable(br):
                        raise ValueError("tiny numerical comparison conditions differ")
                    numerical = compare_outputs.compare(
                        a, b, max_input_tokens=len(a["prompt_ids"])
                    )
                    save(output / "tiny-numerical.json", numerical)
                    if not numerical["exact"]:
                        raise ValueError(
                            "tiny greedy tokens/probabilities differ; requires numerical review"
                        )
            success = True
        finally:
            save(
                output / "ci-attempt.json",
                {
                    "schema_version": 1,
                    "purpose": "ci",
                    "passed": success,
                    "interrupted_signals": interrupted,
                    "formal_samples": 0,
                    "note": "all attempts retained; build worktrees retained under storage bound; no automatic retries",
                },
            )
            for s, handler in handlers.items():
                signal.signal(s, handler)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "policy", "config", "repository", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") == "true":
        raise SystemExit("direct candidate execution on an Actions runner is forbidden; use isolated_device.py and a provisioned per-stage broker")
    try:
        return run(args)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        print("InferSSD device gate failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
