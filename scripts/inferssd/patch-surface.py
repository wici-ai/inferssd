#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare engine files to the immutable source snapshot, excluding path relocation."""
import argparse
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = "8884f8314ce8ea96d62cd602b6bbf7fc25758a09"


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE)
    args = parser.parse_args()
    paths = git("ls-tree", "-r", "--name-only", args.base).decode().splitlines()
    rows = []
    with tempfile.TemporaryDirectory(prefix="inferssd-upstream-") as temporary:
        before = pathlib.Path(temporary) / "before"
        after = pathlib.Path(temporary) / "after"
        for path in paths:
            if not path.startswith(("ggml/", "src/", "include/", "common/", "tools/server/")):
                continue
            original = git("show", args.base + ":" + path)
            current = ROOT / "engine" / path
            replacement = current.read_bytes() if current.is_file() else b""
            if original == replacement:
                continue
            before.write_bytes(original)
            after.write_bytes(replacement)
            result = subprocess.run(
                ["git", "diff", "--no-index", "--numstat", str(before), str(after)],
                capture_output=True, text=True, check=False,
            )
            if result.returncode not in (0, 1):
                raise RuntimeError(result.stderr)
            added, removed, _ = result.stdout.split("\t", 2)
            if added != "-":
                rows.append((path, int(added), int(removed)))
    print("| Upstream path (under engine/) | Added | Removed |")
    print("| --- | ---: | ---: |")
    for path, added, removed in rows:
        print(f"| `{path}` | {added} | {removed} |")
    print(f"\nTotal: {sum(r[1] for r in rows)} added, {sum(r[2] for r in rows)} removed "
          f"across {len(rows)} upstream paths.")


if __name__ == "__main__":
    main()
