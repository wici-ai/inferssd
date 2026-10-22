#!/usr/bin/env python3
"""logparse.py on the decode plan's receipts (a stored snippet of 125's DSV log, 10-22): the destroy summaries of the
plan and of the page-locking, the tier's host-path line with its v2 fields, and an engine before them (no receipt).
usage: python3 tools/adaptive/tests/test_logparse_place.py   (unittest; exit 1 on a failure)"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import logparse  # noqa: E402

SNIPPET = [
    '[kimi_host_register] reason=decided enabled=1 in_place=0 host_gpu=1 stride=18812928\n',
    '[kimi_decode_place] token=4156 layer=2 resident=3 ram=0 miss=3 locked=0 candidates=2 planned=0 predicted_us=6069.7 '
    'cpu_only_us=6069.7 measured_us=5221.9 fixed_us=141.5 total_us=5363.4\n',
    '[kimi_decode_place] reason=destroy layers=10117 tokens=253 gpu_experts=812 gpu_per_token=3.21 planned=814 '
    'candidates=13552 refused=2 wanted=51 error_p50_pct=12 error_p90_pct=34 predicted_ms=30873.8 measured_ms=32318.0 '
    'cpu_only_ms=31084.0 cpu_expert_us=446.7 early=0.687 fixed_us=123.0 decode_read_gbs=9.57 cpu_alone_us=431.0 '
    'cpu_shared_us=495.6 cpu_moved=1.150\n',
    '[kimi_host_register] reason=destroy enabled=1 locked=0 bytes=0 registered=1204 failures=0 unregistered=1204 '
    'signals=1204 wanted=1300 us_mean=6050 us_max=19717 us_total=2055283 error=-\n',
    '[kimi_vram_host] event=probe_copy bytes=18812928 us=612 gbs=30.74\n',
    '[kimi_vram_host] in_place=0 jobs=812 batches=786 copies=812 reserved_copies=412 refused=2 copy_mb=15266.1 '
    'copy_us=487525 resident_us_per_expert=74.6 host_us_per_expert=229.0 resident_call_us=195.6 resident_expert_us=16.7 '
    'host_call_us=195.6 host_expert_us=35.6 copy_us_per_mb=29.05 scratch_slots=0 host_floor=536870912\n',
]


class Place(unittest.TestCase):
    def test_receipts(self):
        a = logparse.ArmParser()
        for line in SNIPPET:
            a.feed(line)
        self.assertEqual(a.decode_place['gpu_per_token'], 3.21)
        self.assertEqual(a.decode_place['error_p50_pct'], 12)
        self.assertEqual(a.decode_place['refused'], 2)
        self.assertEqual(a.host_register['registered'], 1204)
        self.assertEqual(a.host_register['failures'], 0)
        self.assertEqual(a.host_register['wanted'], 1300)
        self.assertAlmostEqual(a.decode_place['cpu_moved'], 1.15)
        self.assertEqual(a.decode_place['cpu_shared_us'], 495.6)
        self.assertEqual(a.host['in_place'], 0)
        self.assertAlmostEqual(a.host['copy_us_per_mb'], 29.05)
        self.assertEqual(a.host['reserved_copies'], 412)
        # the fields read before the host path's version 2 keep their names
        self.assertEqual(a.host['resident_us'], 74.6)
        self.assertEqual(a.host['host_us'], 229.0)

    def test_pending(self):
        # the plan's candidates include the RAM hits a prefetch still reads: how many there were and how many moved
        a = logparse.ArmParser()
        a.feed('[kimi_decode_place] reason=destroy layers=21930 tokens=510 gpu_experts=10200 gpu_per_token=20.00 '
               'planned=10180 candidates=10500 refused=0 wanted=0 error_p50_pct=15 error_p90_pct=90 predicted_ms=32000.0 '
               'measured_ms=32100.0 cpu_only_ms=36000.0 cpu_expert_us=1600.0 early=0.650 fixed_us=120.0 '
               'decode_read_gbs=8.10 cpu_alone_us=1500.0 cpu_shared_us=1650.0 cpu_moved=1.100 pending=2400 '
               'pending_planned=2300\n')
        self.assertEqual(a.decode_place['pending'], 2400)
        self.assertEqual(a.decode_place['pending_planned'], 2300)
        # an engine before them: absent, not zero
        b = logparse.ArmParser()
        for line in SNIPPET:
            b.feed(line)
        self.assertIsNone(b.decode_place['pending'])

    def test_older_engine(self):
        # an engine before the plan: version 1's host line only on unified memory, no plan or page-locking receipt
        a = logparse.ArmParser()
        a.feed('[kimi_vram_host] jobs=10 batches=5 resident_us_per_expert=190.0 host_us_per_expert=300.0 host_floor=0\n')
        self.assertIsNone(a.decode_place)
        self.assertIsNone(a.host_register)
        self.assertEqual(a.host, {'resident_us': 190.0, 'host_us': 300.0, 'jobs': 10.0})


if __name__ == '__main__':
    unittest.main()
