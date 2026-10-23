#!/usr/bin/env python3
"""List changes to paths present in the immutable upstream snapshot."""
import pathlib
import subprocess

root = pathlib.Path(__file__).resolve().parents[2]
def git(*args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)
upstream = set(git("ls-tree", "-r", "--name-only", "upstream").splitlines())
rows = []
for line in git("diff", "--numstat", "upstream", "--", "ggml", "src", "include", "common", "tools/server").splitlines():
    added, removed, path = line.split("\t", 2)
    if path not in upstream or added == "-":
        continue
    rows.append((path, int(added), int(removed)))
print("| Upstream path | Added | Removed |")
print("| --- | ---: | ---: |")
for path, added, removed in rows:
    print(f"| `{path}` | {added} | {removed} |")
print(f"\nTotal: {sum(r[1] for r in rows)} added, {sum(r[2] for r in rows)} removed across {len(rows)} upstream paths.")
