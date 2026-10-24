#!/usr/bin/env python3
"""Prepare hash-bound operator files locally; never install them or contact a device."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.contract import digest, load, save
from ci.selection_contract import bind, bound


def prepare(template, manifest, plan, policy, output):
    config, manifest = bind(template, manifest, plan, policy)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "manifest.json", manifest)
    config["manifest_sha256"] = digest(output / "manifest.json")
    bound(config, manifest, manifest["job_id"])
    save(output / "config.json", config)
    save(output / "dispatch-inputs.json", dict(
        device=config["device"], job_id=manifest["job_id"],
        source_revision=config["revision"], config_sha256=digest(output / "config.json"),
        selection_plan_sha256=config["selection_plan_sha256"],
        control_revision=config["control_revision"],
        control_manifest_sha256=config["control_manifest_sha256"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("template", "manifest", "plan", "policy", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    prepare(*(load(getattr(args, k)) for k in ("template", "manifest", "plan", "policy")), args.output)
