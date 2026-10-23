"""Original guardian observations supplement the separate group/lease freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from observe_guardian_F import CASES, observe

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_original_guardian(case):
    data = json.loads((FIXTURES / "guardian-goldens.json").read_text())
    assert observe(case) == next(item for item in data["cases"] if item["name"] == case["name"])


def test_guardian_frozen_bytes():
    manifest = json.loads((FIXTURES / "oracle/guardian-manifest.json").read_text())
    for item in manifest["files"]:
        assert (
            hashlib.sha256((Path(__file__).parent / item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
        )
