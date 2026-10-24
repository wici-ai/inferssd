#!/usr/bin/env python3
"""Compare two completed raw-runner requests with the recorded CPU reference.

This verifies artifacts, not process identity, source coverage, or cleanup.
It never executes a model or creates/replaces expected output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def verify(directory, model, mode):
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    cases = [case for case in manifest["cases"]
             if case["model"] == model and case["source_mode"] == mode]
    if len(cases) != 1:
        raise ValueError("unknown model/source case")
    case = cases[0]
    reference = []
    for row in case["rows"]:
        data = (root / row["file"]).read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("reference bytes do not match the manifest")
        values = struct.unpack("<" + str(row["n_vocab"]) + "f", data)
        if not all(float("-inf") < value < float("inf") for value in values):
            raise ValueError("reference contains a nonfinite logit")
        if max(range(len(values)), key=values.__getitem__) != row["selected_id"]:
            raise ValueError("reference argmax differs from the selected ID")
        reference.append(data)
    for request in (1, 2):
        receipt = json.loads((directory / f"request-{request}.json").read_text())
        if not isinstance(receipt, dict):
            raise ValueError("request receipt must be an object")
        if type(receipt.get("request")) is not int or receipt["request"] != request:
            raise ValueError("request identity differs")
        rows = receipt.get("rows")
        if not isinstance(rows, list) or len(rows) != len(reference):
            raise ValueError("request must contain every output row")
        for step, (actual, expected, data) in enumerate(zip(rows, case["rows"], reference)):
            if not isinstance(actual, dict):
                raise ValueError("invalid row record")
            for key in ("step", "position", "batch_tokens", "selected_id"):
                if type(actual.get(key)) is not int or actual[key] != expected[key]:
                    raise ValueError(f"request {request}, step {step}: {key} differs")
            name = f"request-{request}-step-{step}.f32"
            if actual.get("file") != name:
                raise ValueError("unexpected raw output filename")
            if (directory / name).read_bytes() != data:
                raise ValueError(f"request {request}, step {step}: full raw bytes differ")
    return {"status": "passed-recorded-bytes", "model": model, "source_mode": mode,
            "requests": 2, "rows_per_request": 4, "vocabulary": case["ready"]["n_vocab"],
            "runtime_identity_and_coverage_attested": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("pack", "alias"), required=True)
    args = parser.parse_args()
    try:
        result = verify(args.actual_dir, args.model, args.mode)
    except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
        parser.exit(1, f"CPU reference verification failed: {error}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
