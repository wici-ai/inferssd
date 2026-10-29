#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Baseline history: verified measurements of main commits on each device.

Each new main commit gets one baseline job: one build, four timed runs per
model on every device. The hosted report of that job is replayed from sealed
evidence; only its verified cells become records here, one file per device,
model, tier, commit and history conditions (the machine, its OS, the model
files and the chat template; settings are derived per commit and not part of
them). Records live in the
`baseline-history` directory next to the reviewed device profiles
(`inputs.json`) on the preparation host. Every job seals its devices' records
into its job configuration, so the device and the hosted replay decide with
exactly the same values.

Commands: `list` shows records, `verify` re-checks every file, `prune` keeps
the newest records per device/model/tier, and `ingest` adds the records of a
verified baseline report (the operator does this itself when a baseline job
completes). This tool only reads and writes files here.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.compare import BASELINE, history_checked
from ci.contract import HISTORY_RUNS, PERFORMANCE_METRICS, load

DIRECTORY = "baseline-history"


def store_root(inputs):
    return Path(inputs).resolve().parent / DIRECTORY


def record_path(root, record):
    return (Path(root) / record["device"] / record["model"] / record["tier"]
            / (record["commit"] + "." + record["conditions_sha256"] + ".json"))


def records_from_report(report, *, device, job_id, commit, run_id, run_attempt, measured_at):
    """Records for every verified baseline cell of one device in a hosted report.

    Returns (records, skipped). A cell that is not a clean baseline (different
    tokens, a flagged host check, failed or missing evidence) is skipped with
    its reason; it never becomes history.
    """
    if (not isinstance(report, dict) or report.get("purpose") != "operator-selected-report-only"
            or any(report.get(key) != commit for key in ("base", "head", "candidate"))
            or not isinstance(report.get("cells"), list)):
        raise ValueError("report is not the hosted report of a baseline job for this commit")
    records, skipped = [], []
    for error in report.get("errors", []):
        if isinstance(error, dict) and error.get("device") == device:
            skipped.append(dict(cell=device, status="error", reasons=[str(error.get("error"))]))
    for row in report["cells"]:
        if not isinstance(row, dict) or row.get("device") != device:
            continue
        name = device + "/" + str(row.get("model"))
        if not isinstance(row.get("performance"), list):
            if row.get("status") != "not-required":
                skipped.append(dict(cell=name, status=row.get("status"),
                                    reasons=[str(row.get("error", row.get("performance", "no verified result")))]))
            continue
        if row.get("job_id") != job_id:
            raise ValueError(name + " belongs to another job")
        for entry in row["performance"]:
            cell = entry.get("cell", {})
            if (entry.get("protocol", {}).get("purpose") != BASELINE
                    or cell != dict(device=device, model=row["model"], tier=entry.get("tier"), job_id=job_id,
                                    base=commit, candidate=commit)):
                raise ValueError(name + " is not a baseline cell of this job")
            if entry.get("status") != "baseline":
                skipped.append(dict(cell=name + "/" + str(entry.get("tier")), status=entry.get("status"),
                                    reasons=entry.get("review_reasons", [])))
                continue
            metrics = entry["metrics"]
            runs = [dict(metrics={metric: metrics[metric]["values"][index] for metric in PERFORMANCE_METRICS},
                         logs={metric: metrics[metric]["logs"][index] for metric in PERFORMANCE_METRICS},
                         tokens_sha256=entry["output_workload"]["tokens_sha256"])
                    for index in range(HISTORY_RUNS)]
            record = dict(schema_version=1, purpose="baseline-history", device=device, model=row["model"],
                          tier=entry["tier"], commit=commit,
                          conditions_sha256=entry["measurement_identity"]["conditions_sha256"],
                          runs=runs, measurement_identity=entry["measurement_identity"], build=entry["build"],
                          job_id=job_id, run_id=run_id, run_attempt=run_attempt, measured_at=measured_at)
            history_checked([record])
            records.append(record)
    return records, skipped


def write(root, record):
    """Store a record once, read-only; an identical existing record is kept ("present")."""
    history_checked([record])
    root = Path(root)
    path = record_path(root, record)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("unsafe history entry " + str(path))
    if path.exists():
        if load(path) != record:
            raise ValueError("a different record already exists for " + str(path.relative_to(root)))
        return "present"
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return "stored"


def load_all(root):
    """Every record, verified; a file whose name differs from its content is refused."""
    root = Path(root)
    if not root.exists():
        return []
    records = []
    for path in sorted(root.rglob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise ValueError("unsafe history entry " + str(path))
        record = load(path)
        history_checked([record])
        if record_path(root, record) != path:
            raise ValueError("history file name differs from its record: " + str(path.relative_to(root)))
        records.append(record)
    history_checked(records)
    return records


def prune(root, keep, *, dry_run=False):
    """Keep the newest `keep` records per device/model/tier; return the removed paths."""
    groups = {}
    for record in load_all(root):
        groups.setdefault((record["device"], record["model"], record["tier"]), []).append(record)
    removed = []
    for group in groups.values():
        for record in sorted(group, key=lambda item: (item["measured_at"], item["commit"]), reverse=True)[keep:]:
            path = record_path(root, record)
            removed.append(str(path.relative_to(root)))
            if not dry_run:
                path.chmod(0o600)
                path.unlink()
    return removed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="the baseline-history directory")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    commands.add_parser("verify")
    pruning = commands.add_parser("prune")
    pruning.add_argument("--keep", type=int, required=True)
    pruning.add_argument("--dry-run", action="store_true")
    ingest = commands.add_parser("ingest")
    for name in ("report", "device", "job-id", "commit", "measured-at"):
        ingest.add_argument("--" + name, required=True)
    ingest.add_argument("--run-id", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            for record in load_all(args.root):
                values = [record["device"], record["model"], record["tier"], record["commit"][:12],
                          record["conditions_sha256"][:12], record["measured_at"], record["job_id"]]
                print("  ".join(values))
        elif args.command == "verify":
            print("verified " + str(len(load_all(args.root))) + " baseline records")
        elif args.command == "prune":
            if args.keep < 1:
                parser.error("--keep must be positive")
            for path in prune(args.root, args.keep, dry_run=args.dry_run):
                print(("would remove " if args.dry_run else "removed ") + path)
        else:
            records, skipped = records_from_report(
                load(args.report), device=args.device, job_id=args.job_id, commit=args.commit,
                run_id=args.run_id, run_attempt=1, measured_at=args.measured_at)
            for item in skipped:
                print("history: skipped " + item["cell"] + " (" + str(item["status"]) + ": "
                      + ", ".join(map(str, item["reasons"])) + ")", file=sys.stderr)
            for record in records:
                print(write(args.root, record) + " " + str(record_path(args.root, record).relative_to(args.root)))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("history: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
