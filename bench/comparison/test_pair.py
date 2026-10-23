"""Focused decision-boundary checks; no device access or model request."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pair


class ComparisonTests(unittest.TestCase):
    def test_actual_cache_failure_cannot_pass_declared_resource_match(self):
        with self.assertRaises(ValueError):
            pair.arm.cache_receipt(b"[kimi_cpu_sink_cache] paused=1 reason=mmap_failed\n", 123)
        valid = b"[kimi_cpu_sink_cache] reason=before_destroy capacity_bytes=123 mapped_bytes=123 committed_bytes=100 hits=1\n"
        self.assertEqual(pair.arm.cache_receipt(valid, 123)["values"]["hits"], 1)
        with self.assertRaises(ValueError):
            pair.arm.cache_receipt(valid, 124)
        with self.assertRaises(ValueError):
            pair.arm.cache_receipt(b"[kimi_cpu_sink_cache] auto_capacity_bytes=124\n" + valid, 123)

    def setUp(self):
        self.result = {"prompt_ids": [1, 2], "prompt_sha256": "same", "tokens": [4] * 256,
                       "input_tokens": 2, "timings": {"prompt_n": 2, "prompt_ms": 10},
                       "metrics": {"decode_first128_tps": 10, "decode_second128_tps": 12,
                                   "decode_tps": 11, "ttft_s": 1, "server_prefill_tps": 200}}

    def test_equal_samples_are_only_observed_pass(self):
        value = pair.compare(self.result, self.result)
        self.assertEqual(value["status"], "passed")
        self.assertIn("not statistical proof", value["interpretation"])

    def test_any_small_slowdown_remains_inconclusive(self):
        after = copy.deepcopy(self.result)
        after["metrics"]["decode_first128_tps"] *= .9999
        self.assertEqual(pair.compare(self.result, after)["status"], "inconclusive")
        after = copy.deepcopy(self.result)
        after["metrics"]["ttft_s"] *= 1.0001
        self.assertEqual(pair.compare(self.result, after)["status"], "inconclusive")

    def test_invalid_or_client_derived_prefill_is_rejected(self):
        for field, value in (("decode_tps", float("nan")), ("server_prefill_tps", 201)):
            after = copy.deepcopy(self.result)
            after["metrics"][field] = value
            with self.assertRaises(ValueError):
                pair.compare(self.result, after)

    def test_prompt_mismatch_is_invalid(self):
        after = copy.deepcopy(self.result)
        after["prompt_ids"] = [1, 3]
        with self.assertRaises(ValueError):
            pair.compare(self.result, after)


if __name__ == "__main__":
    unittest.main()
