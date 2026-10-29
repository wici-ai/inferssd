#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Derive performance noise bands from A/A calibration results.

Reads operator reports (`aggregate.json`) or retained per-cell `comparison.json`
files from `/ci calibrate` runs and prints the noise-bands object that the
operator keeps as `noise-bands.json` next to the reviewed device profiles
(`inputs.json`) on the preparation host. For each device/model/tier the per-pair
log ratios d of every input are pooled; because base and candidate are the same
revision, the expected d is zero, so sigma_log is the root mean square of the
pooled values. Inputs from different revisions or measured conditions are not
pooled. With `--merge`, cells from an existing bands file that the inputs do
not cover are kept. This tool only reads files; review its output before
replacing the operator's bands file.
"""

import argparse
import datetime
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.compare import CALIBRATION, log_ratio
from ci.contract import NOISE_BAND_STATISTIC, PERFORMANCE_METRICS, load, noise_band, noise_bands


def series(value):
    """Every cell result in an operator report, or the single result of a cell file."""
    if isinstance(value, dict) and isinstance(value.get("cells"), list):
        for row in value["cells"]:
            if isinstance(row, dict) and isinstance(row.get("performance"), list):
                yield from row["performance"]
    elif isinstance(value, dict) and isinstance(value.get("protocol"), dict):
        yield value
    else:
        raise ValueError("input is neither an operator report nor a cell comparison")


def differences(entry, metric, count):
    """Recompute d from the retained per-pair values instead of trusting the summary."""
    data = entry["metrics"][metric]
    base, candidate = data["base"], data["candidate"]
    if (not isinstance(base, list) or not isinstance(candidate, list)
            or len(base) != count or len(candidate) != count):
        raise ValueError("calibration result lacks one value per pair for " + metric)
    for value in (*base, *candidate):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("calibration result has an invalid value for " + metric)
    d = [log_ratio(metric, before, after) for before, after in zip(base, candidate)]
    if d != data["d"]:
        raise ValueError("retained d values differ from their recorded pair values for " + metric)
    return d


def bands(values, *, date):
    """Return (bands, skipped) for the calibration results in `values`."""
    pooled, skipped = {}, []
    for value in values:
        for entry in series(value):
            protocol, cell = entry.get("protocol", {}), entry.get("cell", {})
            name = "/".join(str(cell.get(key)) for key in ("device", "model", "tier", "job_id"))
            if protocol.get("purpose") != CALIBRATION or cell.get("base") != cell.get("candidate"):
                raise ValueError(name + " is not an A/A calibration result")
            if entry.get("status") != "calibrated":
                skipped.append(dict(cell=name, status=entry.get("status"), reasons=entry.get("review_reasons", [])))
                continue
            group = pooled.setdefault((cell["device"], cell["model"], cell["tier"]), dict(
                d={metric: [] for metric in PERFORMANCE_METRICS}, jobs=[], revisions=set(), conditions=set()))
            if cell["job_id"] in group["jobs"]:
                raise ValueError(name + " was supplied more than once")
            group["jobs"].append(cell["job_id"])
            group["revisions"].add(cell["base"])
            group["conditions"].add(entry["measurement_identity"]["conditions_sha256"])
            for metric in PERFORMANCE_METRICS:
                group["d"][metric].extend(differences(entry, metric, protocol["samples_per_side"]))
    result = {}
    for (device, model, tier), group in sorted(pooled.items()):
        if len(group["revisions"]) != 1 or len(group["conditions"]) != 1:
            raise ValueError("calibrations from different revisions or measured conditions cannot be pooled: "
                             + "/".join((device, model, tier)))
        count = len(group["d"][PERFORMANCE_METRICS[0]])
        band = dict(sigma_log={metric: math.sqrt(sum(x * x for x in group["d"][metric]) / count)
                               for metric in PERFORMANCE_METRICS},
                    pairs=count, statistic=NOISE_BAND_STATISTIC,
                    conditions_sha256=next(iter(group["conditions"])),
                    engine_revision=next(iter(group["revisions"])),
                    calibration_jobs=sorted(group["jobs"]), date=date)
        try:
            noise_band(band)
        except ValueError as error:
            raise ValueError("/".join((device, model, tier)) + ": " + str(error)) from error
        result.setdefault(device, {}).setdefault(model, {})[tier] = band
    return result, skipped


def merged(existing, derived):
    """Keep every existing cell the new calibration does not replace."""
    result = json.loads(json.dumps(noise_bands(existing)))
    for device, models in derived.items():
        for model, tiers in models.items():
            result.setdefault(device, {}).setdefault(model, {}).update(tiers)
    return noise_bands(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", type=Path, nargs="+", help="calibration aggregate.json or cell comparison.json files")
    parser.add_argument("--date", default=datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
                        help="date recorded in each band (default: today, UTC)")
    parser.add_argument("--merge", type=Path, help="existing noise-bands.json whose other cells are kept")
    args = parser.parse_args(argv)
    try:
        result, skipped = bands([load(path) for path in args.inputs], date=args.date)
        if args.merge is not None and result:
            result = merged(load(args.merge), result)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("noise_band: " + str(error), file=sys.stderr)
        return 1
    for item in skipped:
        print("noise_band: skipped " + item["cell"] + " (" + str(item["status"]) + ": "
              + ", ".join(map(str, item["reasons"])) + ")", file=sys.stderr)
    if not result:
        print("noise_band: no usable calibration result", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
