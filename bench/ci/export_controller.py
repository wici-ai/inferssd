#!/usr/bin/env python3
"""Export reviewed control bytes for a separately protected private repository."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def export(source, revision, output, workflow="device-performance", registry_revision=None):
    if len(revision) != 40 or any(x not in "0123456789abcdef" for x in revision):
        raise ValueError("full reviewed revision required")
    output.mkdir(parents=True, exist_ok=False)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args])

    paths = (
        git("ls-tree", "-r", "--name-only", revision, "bench", "framework", "tools/ssdash", "tools/inferssd")
        .decode()
        .splitlines()
    )
    files = {}
    for relative in paths:
        mode = git("ls-tree", revision, relative).split()[0]
        if mode not in (b"100644", b"100755") or ".." in Path(relative).parts:
            raise ValueError("export only regular tracked control files")
        content = git("show", revision + ":" + relative)
        target = output / "control" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        files[relative] = hashlib.sha256(content).hexdigest()
    (output / "control/control-source.json").write_text(
        json.dumps({"revision": revision, "files": files}, indent=2) + "\n"
    )
    if registry_revision is not None:
        if len(registry_revision) != 40 or any(x not in "0123456789abcdef" for x in registry_revision):
            raise ValueError("full reviewed settings registry revision required")
        prefix = "tools/inferssd/src/inferssd/"
        if any(path.startswith((prefix, "framework/src/inferssd/")) for path in files):
            raise ValueError("control source already includes its own registry")
        registry_files = {}
        for name in ("__init__.py", "configuration/__init__.py", "configuration/settings.py", "configuration/schema.json"):
            relative = prefix + name
            content = git("show", registry_revision + ":" + relative)
            target = output / "control" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            registry_files[relative] = hashlib.sha256(content).hexdigest()
        (output / "control/settings-registry-source.json").write_text(
            json.dumps({"revision": registry_revision, "files": registry_files}, indent=2) + "\n"
        )
    templates = {
        "device-performance": "device-performance.yml",
        "integration-smoke": "integration-smoke.workflow.yml",
        "checkpoint": "checkpoint.workflow.yml",
    }
    template = templates[workflow]
    target = output / ".github/workflows" / (workflow + ".yml")
    target.parent.mkdir(parents=True)
    target.write_bytes(git("show", revision + ":bench/ci/controller/" + template))
    (output / "README.md").write_text(
        "Private device controller. Do not enable execution until the source manifest, protected refs, dedicated check App and per-stage host isolation gates are reviewed.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workflow", choices=("device-performance", "integration-smoke", "checkpoint"), default="device-performance")
    parser.add_argument("--registry-revision", help="Optional separately reviewed renamed settings registry revision")
    args = parser.parse_args()
    export(args.source, args.revision, args.output, args.workflow, args.registry_revision)
