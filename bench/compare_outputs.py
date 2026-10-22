#!/usr/bin/env python3
"""Compare two complete short-request records without a numerical tolerance."""
import argparse
import json
from pathlib import Path

from stream_chat import decode_json, token_list, validate_probability


def compare(a, b):
    for value in (a, b):
        if value.get("status") != "passed":
            raise ValueError("both measurements must have passed")
        if (type(value.get("input_tokens")) is not int or not 1 <= value["input_tokens"] <= 1024
                or type(value.get("output_tokens")) is not int or not 1 <= value["output_tokens"] <= 512
                or not token_list(value.get("prompt_ids"))
                or len(value["prompt_ids"]) != value["input_tokens"]):
            raise ValueError("invalid short protocol token counts")
        tokens = value.get("tokens")
        if not token_list(tokens) or len(tokens) != value.get("output_tokens") or not tokens:
            raise ValueError("incomplete token record")
        probs = value.get("probabilities")
        if not isinstance(probs, list) or len(probs) != len(tokens):
            raise ValueError("incomplete probability record")
        for probability, token in zip(probs, tokens):
            validate_probability(probability, token)
    for key in ("prompt_sha256", "rendered_prompt_sha256", "prompt_ids", "input_tokens", "output_tokens"):
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError(f"protocol mismatch: {key}")
    matches = [x == y for x, y in zip(a["tokens"], b["tokens"])]
    first = next((index for index, equal in enumerate(matches) if not equal), None)
    probabilities_equal = a["probabilities"] == b["probabilities"]
    return {"exact": all(matches) and probabilities_equal,
            "output_tokens": len(matches), "matching_token_positions": sum(matches),
            "first_token_difference_zero_based": first,
            "probabilities_exact": probabilities_equal,
            "caveat": "After the first differing token, greedy runs have different contexts; positional agreement is descriptive, not a teacher-forced numerical comparison."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(decode_json(args.a.read_bytes()), decode_json(args.b.read_bytes()))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if result["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
