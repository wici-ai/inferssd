#!/bin/sh
# SPDX-License-Identifier: MIT
set -eu
if [ "$#" -ne 3 ]; then
    echo "usage: $0 <seed> <upstream-commit> <manifest>" >&2
    exit 2
fi
python3 - "$@" <<'PY'
import csv
import subprocess
import sys
from pathlib import Path

seed, base, manifest = sys.argv[1:]

def git(*args):
    return subprocess.check_output(["git", *args])

def tree(ref):
    entries = {}
    for record in git("ls-tree", "-rz", ref).split(b"\0"):
        if record:
            meta, name = record.split(b"\t", 1)
            entries[name.decode()] = meta.decode()
    return entries

def matches(path, pattern):
    return path.startswith(pattern[:-2]) if pattern.endswith("/**") else path == pattern

rows = list(csv.reader((line for line in Path(manifest).read_text().splitlines()
                        if line and not line.startswith("#")), delimiter="\t"))
if any(len(row) != 4 for row in rows):
    sys.exit("FAIL: malformed manifest row")
allowed = {"keep", "carry-p6", "keep-edit", "restore-upstream", "replace", "add", "drop", "exclude"}
if any(row[2] not in allowed for row in rows):
    sys.exit("FAIL: unknown manifest disposition")
head, source, original, snapshot = (tree(ref) for ref in ("HEAD", seed, base, "upstream"))
errors = []
planned = []
for path, _, action, _ in rows:
    if action in {"keep", "carry-p6", "restore-upstream"}:
        expected = original if action == "restore-upstream" else source
        if path not in expected or head.get(path) != expected[path]:
            errors.append(f"{action}: {path} differs or is missing")
    elif action in {"keep-edit", "replace", "add"}:
        if path not in head:
            errors.append(f"{action}: {path} is missing")
        planned.append(path)
    elif action == "drop":
        errors.extend(f"drop: {name} is present" for name in head if matches(name, path))
    elif action == "exclude":
        errors.extend(f"exclude: {name} is in upstream snapshot" for name in snapshot if matches(name, path))
        for name in head:
            if matches(name, path) and not any(r[0] == name and r[2] in {"add", "replace"} for r in rows):
                errors.append(f"exclude: {name} lacks an explicit replacement")

for path in set(head) | set(original):
    if not any(matches(path, row[0]) for row in rows) and head.get(path) != original.get(path):
        errors.append(f"unlisted: {path} differs from upstream")
for path in head:
    if path.startswith(".github/") and not any(r[0] == path and r[2] == "add" for r in rows):
        errors.append(f"workflow/template not allowed: {path}")

for path in planned:
    reference = seed if path in source else base
    print(f"\n--- Planned change: {path} ---", flush=True)
    subprocess.run(["git", "diff", "--no-ext-diff", reference, "HEAD", "--", path], check=True)
if errors:
    print("\n".join(f"FAIL: {error}" for error in errors), file=sys.stderr)
    sys.exit(1)
print(f"PASS: {len(rows)} manifest rows; {len(head)} tracked paths; seed and upstream blobs verified")
PY
