"""F-first lifecycle and reviewed standalone ownership core receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from observe_engine_F import CASES, observe
from observe_ownership_F import GROUP_CASES, LEASE_CASES, group_case, lease_case

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_engine_original(case):
    expected = json.loads((FIXTURES / "engine-lifecycle-goldens.json").read_text())
    item = next(item for item in expected["cases"] if item["name"] == case["name"])
    assert observe(case) == item


@pytest.mark.parametrize("case", GROUP_CASES, ids=lambda case: case["name"])
def test_group_original(case):
    expected = json.loads((FIXTURES / "ownership-core-goldens.json").read_text())
    assert group_case(case) == next(x for x in expected["groups"] if x["name"] == case["name"])


@pytest.mark.parametrize("case", LEASE_CASES, ids=lambda case: case["name"])
def test_lease_original(case):
    expected = json.loads((FIXTURES / "ownership-core-goldens.json").read_text())
    assert lease_case(case) == next(x for x in expected["leases"] if x["name"] == case["name"])


def test_engine_frozen_bytes():
    manifest = json.loads((FIXTURES / "oracle/engine-manifest.json").read_text())
    for item in manifest["files"]:
        path = Path(__file__).parent / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
