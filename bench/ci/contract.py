"""Strict CI identities and classifications. No device connections or defaults."""

import hashlib
import json
import math
from pathlib import Path
import re

DEVICES = ("cloud-5090", "dgx-spark", "local-mbp")
MODELS = ("deepseek", "muse")
SHA = re.compile(r"[0-9a-f]{40}")
HASH = re.compile(r"[0-9a-f]{64}")


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key " + key)
            result[key] = value
        return result

    def bad(value):
        raise ValueError("non-finite JSON constant " + value)

    def number(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number")
        return value

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=unique,
        parse_constant=bad,
        parse_float=number,
    )


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def positive(value, name, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(
            name + " must be a positive integer" + (" or zero" if zero else "")
        )
    return value


def revision(value):
    if not isinstance(value, str) or SHA.fullmatch(value) is None:
        raise ValueError("full lowercase 40-character source revision required")
    return value


def policy_read(value):
    if (
        value.get("schema_version") != 1
        or value.get("purpose") != "ci"
        or value.get("repository") != "wici-ai/inferssd"
        or tuple(value.get("required_devices", ())) != DEVICES
        or tuple(value.get("required_models", ())) != MODELS
    ):
        raise ValueError(
            "policy must cover InferSSD and exactly three devices/two models"
        )
    if (value.get("controller_repository") != "wici-ai/inferssd-device-ci"
            or value.get("controller_ref") != "refs/heads/main"
            or value.get("controller_workflow") != ".github/workflows/device-performance.yml"
            or value.get("staging_prefix") != "device-ci/approved/"
            or value.get("upstream_workflow") != ".github/workflows/ci.yml"):
        raise ValueError("physical CI requires the separate protected private controller policy")
    thresholds = value["thresholds"]
    if (thresholds.get("provisional") is not True
            or thresholds.get("mode") != "observed-no-slowdown"
            or thresholds.get("slowdown_outcome") != "inconclusive"):
        raise ValueError("n=1 policy must block observed slowdown as inconclusive")
    for key in ("decode_regression_fraction", "ttft_regression_fraction", "prefill_regression_fraction"):
        number = thresholds.get(key)
        if (
            type(number) not in (int, float)
            or not math.isfinite(number)
            or number != 0
        ):
            raise ValueError("no permitted regression fraction: " + key)
    for name in (
        "extended_prefixes",
        "extended_names",
        "authorized_associations",
        "documentation_suffixes",
        "documentation_names",
    ):
        if (
            not isinstance(value.get(name), list)
            or not value[name]
            or any(not isinstance(item, str) or not item for item in value[name])
        ):
            raise ValueError("policy requires explicit nonempty " + name)
    return value


def classify(paths, policy):
    paths = tuple(paths)
    if any(
        not isinstance(p, str) or not p or p.startswith("/") or ".." in Path(p).parts
        for p in paths
    ):
        raise ValueError("changed paths must be repository-relative")
    # Unknown file types are code by default; a new backend/language/build file
    # cannot silently bypass a required gate because its suffix was unlisted.
    code = any(
        Path(p).suffix not in policy["documentation_suffixes"]
        and Path(p).name not in policy["documentation_names"]
        for p in paths
    )
    extended = code and any(
        p.startswith(tuple(policy["extended_prefixes"]))
        or any(p == name or p.startswith(name) for name in policy["extended_names"])
        for p in paths
    )
    return {
        "code": code,
        "tiers": ["512", "4096"] if extended else ["512"],
        "changed_paths": list(paths),
        "reason": "resource/executor or benchmark change"
        if extended
        else "all code PRs"
        if code
        else "documentation-only; no device execution",
    }


def authorize(pr, repository, policy, *, observed_sha=None):
    if (
        repository != policy["repository"]
        or pr.get("state") != "open"
        or pr.get("draft") is not False
        or pr.get("head", {}).get("repo", {}).get("full_name") != repository
        or pr.get("base", {}).get("repo", {}).get("full_name") != repository
        or pr.get("base", {}).get("ref") != "main"
        or pr.get("author_association") not in policy["authorized_associations"]
    ):
        raise ValueError(
            "device execution requires an open trusted same-repository PR into main"
        )
    head, base, candidate = (
        revision(pr["head"]["sha"]),
        revision(pr["base"]["sha"]),
        revision(pr.get("merge_commit_sha")),
    )
    if observed_sha is not None and observed_sha not in (head, candidate):
        raise ValueError("triggering CI is stale for this pull request")
    positive(pr.get("number"), "PR number")
    return {
        "schema_version": 1,
        "purpose": "ci",
        "repository": repository,
        "pr": pr["number"],
        "head": head,
        "base": base,
        "candidate": candidate,
    }


def runner_config(value, device):
    if value.get("schema_version") != 1 or value.get("activation_ready") is not True:
        raise ValueError("device config has not been explicitly provisioned/activated")
    if value.get("device") != device or device not in DEVICES:
        raise ValueError("device config identity differs from runner slot")
    if value.get("platform") != ("darwin" if device == "local-mbp" else "linux"):
        raise ValueError("device platform differs from the configured slot")
    for key in ("lock_path", "stop_path", "workspace_root"):
        if not isinstance(value.get(key), str) or not Path(value[key]).is_absolute():
            raise ValueError("device config requires absolute " + key)
    if value["lock_path"] == value["stop_path"]:
        raise ValueError("device STOP and shared lock must differ")
    from inferssd.platform.policy import Policy

    stage = Policy.read(value["stage_policy"])
    if (
        stage.platform != value["platform"]
        or stage.lock_path != value["lock_path"]
        or stage.stop_path != value["stop_path"]
    ):
        raise ValueError(
            "stage policy must use the same actual device lease/STOP/platform"
        )
    for key in ("min_free_bytes", "max_workspace_bytes"):
        positive(value.get(key), key)
    profiles = value.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != {*MODELS, "tiny"}:
        raise ValueError("device config must pin tiny/DeepSeek/Muse profile files")
    for spec in profiles.values():
        if (
            not isinstance(spec, dict)
            or not isinstance(spec.get("path"), str)
            or not Path(spec["path"]).is_absolute()
            or not isinstance(spec.get("sha256"), str)
            or HASH.fullmatch(spec["sha256"]) is None
        ):
            raise ValueError(
                "each native profile requires an absolute path and reviewed SHA-256"
            )
    templates = value.get("build", {}).get("template_relatives")
    if (
        not isinstance(templates, dict)
        or set(templates) != {*MODELS, "tiny"}
        or any(
            not isinstance(p, str)
            or not p
            or Path(p).is_absolute()
            or ".." in Path(p).parts
            for p in templates.values()
        )
    ):
        raise ValueError(
            "each model must bind a template inside its exact source revision"
        )
    for key in ("build", "validation"):
        command = value.get(key, {})
        if (
            not isinstance(command.get("argv"), list)
            or not command["argv"]
            or any(not isinstance(x, str) or not x for x in command["argv"])
        ):
            raise ValueError("explicit argv required for " + key)
        positive(command.get("timeout_s"), key + " timeout")
        if command["timeout_s"] > 7200:
            raise ValueError("stage deadline must be <=7200 seconds")
    for name in ("request_timeout_s", "arm_timeout_s"):
        positive(value.get(name), name)
        if value[name] > 7200:
            raise ValueError("request/arm deadline must be <=7200 seconds")
    if value["arm_timeout_s"] <= value["request_timeout_s"]:
        raise ValueError("arm timeout must leave room for load/cleanup")
    baseline = value["validation"].get("baseline", {}).get("tests")
    if (
        not isinstance(baseline, dict)
        or not baseline
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(row, dict)
            or row.get("expect") != "passed"
            or row.get("kind") not in ("numerical", "lifecycle")
            for name, row in baseline.items()
        )
        or {x["kind"] for x in baseline.values()} != {"numerical", "lifecycle"}
    ):
        raise ValueError(
            "validation must enumerate passing small-model numerical and lifecycle tests"
        )
    return value
