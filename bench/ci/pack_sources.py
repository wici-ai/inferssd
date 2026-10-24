#!/usr/bin/env python3
"""Export immutable source objects as data on a hosted preparer, never run PR code."""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.contract import digest, load, save


def pack(source, admission, bundle, plan):
    value = load(admission)

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(source), *args], text=True
        ).strip()

    if git("rev-parse", "HEAD") != value["candidate"] or git(
        "show", "-s", "--format=%P", "HEAD"
    ).split() != [value["base"], value["head"]]:
        raise ValueError("hosted source checkout does not match exact admitted parents")
    for name in ("base", "head", "candidate"):
        git("update-ref", "refs/inferssd-ci/" + name, value[name])
    git(
        "bundle",
        "create",
        str(bundle.resolve()),
        *["refs/inferssd-ci/" + n for n in ("base", "head", "candidate")],
    )
    value["source_bundle_sha256"] = digest(bundle)
    save(plan, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("source", "admission", "bundle", "plan"):
        parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    pack(args.source, args.admission, args.bundle, args.plan)
