#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Baseline history: verified measurements of main commits on each device.

Each new main commit gets one baseline job: one build, a machine profile and
four timed runs per model on every device. The hosted report of that job is
replayed from sealed evidence; only its verified cells become records here, one
file per device, model, tier, commit and history conditions (the machine, its
OS, the model files, the chat template and the machine policy, i.e. the memory
budget and CPU set; settings are derived per commit and not part of them).
Records live in the `baseline-history` directory next to the reviewed device
profiles (`inputs.json`) on the preparation host. The characteristics file the
job profiled is stored once per device under `characteristics/`, named by its
SHA-256, and every record names the hash its runs used. Every job seals its
devices' records into its job configuration, and a PR job also the
characteristics of each device's most recent baseline job, so the device and
the hosted replay decide with exactly the same values.

Commands: `list` shows records, `verify` re-checks every file, `prune` keeps
the newest records per device/model/tier (and the characteristics files they
name), and `ingest` adds the records of a verified baseline report (the
operator does this itself when a baseline job completes). This tool only reads
and writes files here.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.compare import BASELINE, history_checked
from ci.contract import HISTORY_RUNS, PERFORMANCE_METRICS, load

DIRECTORY = "baseline-history"
CHARACTERISTICS = "characteristics"


def store_root(inputs):
    return Path(inputs).resolve().parent / DIRECTORY


def record_path(root, record):
    return (Path(root) / record["device"] / record["model"] / record["tier"]
            / (record["commit"] + "." + record["conditions_sha256"] + ".json"))


def characteristics_path(root, device, sha256):
    return Path(root) / device / CHARACTERISTICS / (sha256 + ".json")


def records_from_report(report, *, device, job_id, commit, run_id, run_attempt, measured_at):
    """Records for every verified baseline cell of one device in a hosted report.

    Returns (records, skipped, characteristics). A cell that is not a clean
    baseline (different tokens, a flagged host check, failed or missing
    evidence) is skipped with its reason; it never becomes history.
    `characteristics` is the file the device's baseline job profiled, which
    every record names by hash, or None when no cell became a record.
    """
    if (not isinstance(report, dict) or report.get("purpose") != "operator-selected-report-only"
            or any(report.get(key) != commit for key in ("base", "head", "candidate"))
            or not isinstance(report.get("cells"), list)):
        raise ValueError("report is not the hosted report of a baseline job for this commit")
    records, skipped, profiles = [], [], set()
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
        profile = row.get("machine_profile")
        if (not isinstance(profile, dict) or not isinstance(profile.get("characteristics"), str)
                or hashlib.sha256(profile["characteristics"].encode()).hexdigest() != profile.get("sha256")):
            raise ValueError(name + " does not carry the machine characteristics its baseline job profiled")
        profiles.add(profile["characteristics"])
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
            used = entry.get("derived_launch", {}).get("sides", {}).get("candidate", {})
            if used.get("characteristics_sha256") != profile["sha256"]:
                raise ValueError(name + " did not launch from the machine characteristics its job profiled")
            metrics = entry["metrics"]
            runs = [dict(metrics={metric: metrics[metric]["values"][index] for metric in PERFORMANCE_METRICS},
                         logs={metric: metrics[metric]["logs"][index] for metric in PERFORMANCE_METRICS},
                         tokens_sha256=entry["output_workload"]["tokens_sha256"])
                    for index in range(HISTORY_RUNS)]
            record = dict(schema_version=1, purpose="baseline-history", device=device, model=row["model"],
                          tier=entry["tier"], commit=commit,
                          conditions_sha256=entry["measurement_identity"]["conditions_sha256"],
                          characteristics_sha256=profile["sha256"], runs=runs,
                          measurement_identity=entry["measurement_identity"], build=entry["build"],
                          job_id=job_id, run_id=run_id, run_attempt=run_attempt, measured_at=measured_at)
            history_checked([record])
            records.append(record)
    if len(profiles) > 1:
        raise ValueError("the baseline job of " + device + " reports more than one machine profile")
    return records, skipped, (profiles.pop() if records else None)


def stored(path, raw, root):
    """Write `raw` once, read-only; identical existing bytes are kept ("present")."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("unsafe history entry " + str(path))
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("a different record already exists for " + str(path.relative_to(root)))
        return "present"
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return "stored"


def write_characteristics(root, device, text):
    """Store a baseline job's characteristics file for its device, named by its SHA-256."""
    raw = text.encode()
    root = Path(root)
    return stored(characteristics_path(root, device, hashlib.sha256(raw).hexdigest()), raw, root)


def read_characteristics(root, device, sha256):
    """A stored characteristics file; a missing or changed file is refused."""
    path = characteristics_path(root, device, sha256)
    if path.is_symlink() or not path.is_file():
        raise ValueError("baseline machine characteristics " + device + "/" + sha256[:12] + " are missing")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("baseline machine characteristics " + device + "/" + sha256[:12] + " differ from their hash")
    return raw.decode()


def write(root, record):
    """Store a record once, read-only, after the characteristics file it names."""
    history_checked([record])
    root = Path(root)
    read_characteristics(root, record["device"], record["characteristics_sha256"])
    raw = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return stored(record_path(root, record), raw, root)


def load_all(root):
    """Every record and characteristics file, verified; a misnamed or missing file is refused."""
    root = Path(root)
    if not root.exists():
        return []
    records, files = [], set()
    for path in sorted(root.rglob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise ValueError("unsafe history entry " + str(path))
        if path.parent.name == CHARACTERISTICS and path.parent.parent.parent == root:
            read_characteristics(root, path.parent.parent.name, path.stem)
            files.add((path.parent.parent.name, path.stem))
            continue
        record = load(path)
        history_checked([record])
        if record_path(root, record) != path:
            raise ValueError("history file name differs from its record: " + str(path.relative_to(root)))
        records.append(record)
    for record in records:
        if (record["device"], record["characteristics_sha256"]) not in files:
            read_characteristics(root, record["device"], record["characteristics_sha256"])
    history_checked(records)
    return records


def latest_characteristics(root, records, device):
    """The characteristics file of the device's most recent baseline job, as a PR job seals it."""
    own = [record for record in records if record["device"] == device]
    if not own:
        raise ValueError(device + " has no baseline machine characteristics yet; a baseline job must run on it first")
    latest = max(own, key=lambda record: (record["measured_at"], record["job_id"]))
    return dict(sha256=latest["characteristics_sha256"], job_id=latest["job_id"], commit=latest["commit"],
                text=read_characteristics(root, device, latest["characteristics_sha256"]))


def prune(root, keep, *, dry_run=False):
    """Keep the newest `keep` records per device/model/tier and the files they name; return the removed paths."""
    root = Path(root)
    groups = {}
    for record in load_all(root):
        groups.setdefault((record["device"], record["model"], record["tier"]), []).append(record)
    removed, named = [], set()
    for group in groups.values():
        ordered = sorted(group, key=lambda item: (item["measured_at"], item["commit"]), reverse=True)
        named.update((record["device"], record["characteristics_sha256"]) for record in ordered[:keep])
        for record in ordered[keep:]:
            removed.append(record_path(root, record))
    removed += [path for path in sorted(root.glob("*/" + CHARACTERISTICS + "/*.json"))
                if (path.parent.parent.name, path.stem) not in named]
    if not dry_run:
        for path in removed:
            path.chmod(0o600)
            path.unlink()
    return [str(path.relative_to(root)) for path in removed]


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
                          record["conditions_sha256"][:12], record["characteristics_sha256"][:12],
                          record["measured_at"], record["job_id"]]
                print("  ".join(values))
        elif args.command == "verify":
            print("verified " + str(len(load_all(args.root))) + " baseline records")
        elif args.command == "prune":
            if args.keep < 1:
                parser.error("--keep must be positive")
            for path in prune(args.root, args.keep, dry_run=args.dry_run):
                print(("would remove " if args.dry_run else "removed ") + path)
        else:
            records, skipped, characteristics = records_from_report(
                load(args.report), device=args.device, job_id=args.job_id, commit=args.commit,
                run_id=args.run_id, run_attempt=1, measured_at=args.measured_at)
            for item in skipped:
                print("history: skipped " + item["cell"] + " (" + str(item["status"]) + ": "
                      + ", ".join(map(str, item["reasons"])) + ")", file=sys.stderr)
            if characteristics is not None:
                print(write_characteristics(args.root, args.device, characteristics) + " characteristics "
                      + hashlib.sha256(characteristics.encode()).hexdigest()[:12])
            for record in records:
                print(write(args.root, record) + " " + str(record_path(args.root, record).relative_to(args.root)))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("history: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
