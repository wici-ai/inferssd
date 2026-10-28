#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check a pull request title against the commit-title rules in CONTRIBUTING.md.

Pull requests are squash-merged, so the title becomes the commit subject on main.
The title is read from the argument or, when omitted, from the TITLE environment
variable. Diagnostics name the failed rule without repeating the title.
"""
import argparse
import contextlib
import importlib.util
import io
import os
import re
import sys
from pathlib import Path

AREAS = ("engine", "ggml", "cuda", "metal", "cpu", "server", "model", "framework",
         "bench", "ci", "docs", "build", "scripts", "repo")
MAX_LENGTH = 72
FORM = re.compile(r"(?P<areas>[^:]+): (?P<summary>.+)")
# Capitalized words that may open a summary because they are names.
NAMES = {"Apple", "Engram", "Linux", "Mac", "Metal", "Muse", "Python", "Spark", "Vulkan", "Windows"}
# Base verbs used to recognize third-person and past forms such as "adds" or "used".
VERBS = {
    "add", "allow", "avoid", "build", "bump", "change", "check", "clean", "clear", "compute",
    "create", "delete", "disable", "document", "drop", "enable", "ensure", "expose", "fix",
    "handle", "implement", "improve", "introduce", "keep", "limit", "load", "make", "merge",
    "move", "parse", "prefer", "read", "reduce", "refactor", "remove", "rename", "replace",
    "report", "require", "restore", "return", "reuse", "revert", "run", "set", "simplify",
    "skip", "split", "support", "test", "track", "update", "use", "validate", "write",
}
# Imperative verbs that end like past or progressive forms.
IMPERATIVE_EXCEPTIONS = {"bring", "embed", "exceed", "feed", "need", "proceed", "seed",
                         "shed", "speed", "spring", "string", "succeed", "swing"}
OTHER_OWNER_REPOSITORY = re.compile(r"\bwici-ai/(?!inferssd(?![\w-]))", re.IGNORECASE)


def load_scanner():
    spec = importlib.util.spec_from_file_location(
        "public_scan", Path(__file__).with_name("public-scan.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_name(word):
    core = word.strip("`'\"()[],")
    return core in NAMES or any(c.isupper() or c.isdigit() for c in core[1:])


def is_imperative(word):
    word = word.lower()
    if not word.isalpha() or word in IMPERATIVE_EXCEPTIONS:
        return True
    if len(word) > 4 and word.endswith(("ed", "ing")):
        return False
    if word.endswith("d") and word[:-1] in VERBS:
        return False
    if word.endswith("s") and (word[:-1] in VERBS or (word.endswith("es") and word[:-2] in VERBS)
                               or (word.endswith("ies") and word[:-3] + "y" in VERBS)):
        return False
    return True


def problems(title, scanner=None):
    found = []
    if title != title.strip():
        found.append("remove leading or trailing whitespace")
    if any(not " " <= c <= "~" for c in title):
        found.append("use printable ASCII characters only")
    if len(title) > MAX_LENGTH:
        found.append(f"shorten the title to at most {MAX_LENGTH} characters (it has {len(title)})")
    if "  " in title:
        found.append("separate words with single spaces")
    match = FORM.fullmatch(title.strip())
    if not match:
        found.append("use the form 'area: summary', for example 'docs: fix quickstart link'")
    else:
        areas = match.group("areas").split(", ")
        for area in areas:
            if area in ("test", "tests"):
                found.append("use the area of the tested module instead of 'test'")
            elif area not in AREAS:
                found.append("use areas from: " + ", ".join(AREAS) + "; separate several with ', '")
                break
        if len(set(areas)) != len(areas):
            found.append("list each area once")
        summary = match.group("summary")
        first = summary.split(" ", 1)[0]
        if first[:1].isupper() and not is_name(first):
            found.append("start the summary in lowercase unless it begins with a proper noun")
        if not is_imperative(first):
            found.append("write the summary in the imperative mood, for example 'add' rather than 'added'")
        if summary.endswith("."):
            found.append("do not end the summary with a period")
    if OTHER_OWNER_REPOSITORY.search(title):
        found.append("do not reference other repositories of the project owner")
    scanner = scanner or load_scanner()
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        blocked = scanner.inspect(title, "title", True)
    if blocked:
        found.extend("public-scan: " + line.split(": ", 1)[-1]
                     for line in output.getvalue().splitlines() if line.startswith("BLOCK"))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("title", nargs="?", help="Title to check; defaults to $TITLE")
    args = parser.parse_args()
    title = args.title if args.title is not None else os.environ.get("TITLE")
    if title is None:
        parser.error("pass a title or set TITLE")
    found = problems(title)
    for problem in found:
        print("PR title: " + problem, file=sys.stderr)
    if found:
        print("See 'Pull request titles' in CONTRIBUTING.md.", file=sys.stderr)
        return 1
    print("PR title passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
