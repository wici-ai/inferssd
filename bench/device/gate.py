#!/usr/bin/env python3
"""Compare complete device test outcomes with an explicit reviewed baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

STATUSES = {"passed", "failed", "skipped"}


def read_results(path):
    if path.suffix.lower() == ".xml":
        rows = []
        for case in ET.parse(path).getroot().iter("testcase"):
            status = "passed"
            if case.find("failure") is not None or case.find("error") is not None:
                status = "failed"
            elif case.find("skipped") is not None:
                status = "skipped"
            rows.append({"name": case.get("name"), "status": status})
    else:
        rows = json.loads(path.read_text())["tests"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("zero device test results")
    outcomes = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("every test result must be an object")
        name, status = row.get("name"), row.get("status")
        if not isinstance(name, str) or not name.strip() or not isinstance(status, str) or status not in STATUSES:
            raise ValueError("every test needs a name and passed/failed/skipped status")
        if name in outcomes:
            raise ValueError(f"duplicate result: {name}")
        outcomes[name] = status
    return outcomes


def compare(actual, baseline):
    expected = baseline["tests"]
    if not isinstance(expected, dict) or not expected:
        raise ValueError("baseline must enumerate a nonempty complete test set")
    for name, entry in expected.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(entry, dict):
            raise ValueError(f"invalid baseline entry: {name}")
        expected_status = entry.get("expect")
        if not isinstance(expected_status, str) or expected_status not in STATUSES:
            raise ValueError(f"invalid baseline entry: {name}")
        reason = entry.get("reason")
        if expected_status != "passed" and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError(f"non-passing baseline needs a reason: {name}")
    differences = []
    for name in sorted(set(actual) | set(expected)):
        if name not in actual:
            differences.append({"name": name, "error": "missing result"})
        elif name not in expected:
            differences.append({"name": name, "error": "unlisted test"})
        elif actual[name] != expected[name]["expect"]:
            differences.append({"name": name, "expected": expected[name]["expect"], "actual": actual[name]})
    # An all-disabled selection is never an executed acceptance test.
    if not any(status != "skipped" for status in actual.values()):
        differences.append({"error": "zero executed tests"})
    return {"schema_version": 1, "passed": not differences, "total": len(actual),
            "counts": {status: list(actual.values()).count(status) for status in sorted(STATUSES)},
            "differences": differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="CTest JUnit XML or normalized JSON")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare(read_results(args.results), json.loads(args.baseline.read_text()))
        report["baseline_sha256"] = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
        return 0 if report["passed"] else 1
    except (OSError, ValueError, KeyError, ET.ParseError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
