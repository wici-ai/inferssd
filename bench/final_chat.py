#!/usr/bin/env python3
"""Record one fixed final-benchmark prompt with exactly 256 output tokens."""
import argparse
import math
import urllib.request
from pathlib import Path

import stream_chat

PROTOCOL = "final-256-v1"
OUTPUT_TOKENS = 256
DEFAULT_TIMEOUT = 1800
PROMPTS = {
    "512": ("prompt-512.txt", "17ac26b4a3bc13d6c51dd10b62eb54931651d48e41ca4071c587a931a662e283"),
    "4096": ("prompt-4096.txt", "7eac0ca46f379476846f8221e2ab9617d3023c05f3655769eed523a101754dec"),
}


def bounded_timeout(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 1 <= value <= 7200:
        raise ValueError("deadline must be finite and between 1 and 7200 seconds")
    return value


def prompt_path(tier):
    name, expected_hash = PROMPTS[tier]
    path = Path(__file__).resolve().parent / "prompts" / name
    if stream_chat.digest(path.read_bytes()) != expected_hash:
        raise ValueError("fixed benchmark prompt SHA-256 mismatch")
    return path


def validate_context(prompt_ids, props):
    if not stream_chat.token_list(prompt_ids) or not prompt_ids:
        raise ValueError("template must return nonempty authoritative token IDs")
    if not isinstance(props, dict) or not isinstance(props.get("default_generation_settings"), dict):
        raise ValueError("server properties and default generation settings must be objects")
    context = props["default_generation_settings"].get("n_ctx")
    if type(props.get("total_slots")) is not int or props["total_slots"] != 1:
        raise ValueError("final benchmark requires exactly one server slot")
    if type(context) is not int or context < len(prompt_ids) + OUTPUT_TOKENS:
        raise ValueError("actual server context cannot hold the prompt plus 256 outputs")
    return context


def validate_timings(result):
    if (not isinstance(result, dict) or not isinstance(result.get("timings"), dict)
            or not isinstance(result.get("metrics"), dict)):
        raise ValueError("result timings and metrics must be objects")
    value = result["timings"].get("predicted_ms")
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("final benchmark requires finite positive native decode time")
    metrics = dict(result["metrics"], server_native_decode_tps=OUTPUT_TOKENS * 1000 / value,
                   end_to_end_output_tps=OUTPUT_TOKENS / result["metrics"]["request_wall_s"])
    if not all(math.isfinite(v) and v > 0 for v in metrics.values()):
        raise ValueError("final benchmark requires finite positive metrics")
    result["metrics"] = metrics


def run(args):
    bounded_timeout(args.timeout)
    args.prompt = prompt_path(args.tier)
    args.input_tokens = None  # The label is not the tokenizer's actual count.
    args.output_tokens = OUTPUT_TOKENS

    def validate_input(prompt_ids, base):
        with urllib.request.urlopen(base + "/props", timeout=args.timeout) as response:
            props = stream_chat.decode_json(response.read())
        context = validate_context(prompt_ids, props)
        return {"server_props": props, "server_context_tokens": context}

    return stream_chat.record_chat(args, validate_input, {
        "protocol": PROTOCOL, "tier": args.tier, "samples_per_tier": 1,
        "cache_condition": "fresh-server-first-completion; OS page cache recorded by wrapper",
        "decode_intervals": {"first128": 127, "second128": 128, "overall256": 255},
    }, validate_timings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--tier", choices=PROMPTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
