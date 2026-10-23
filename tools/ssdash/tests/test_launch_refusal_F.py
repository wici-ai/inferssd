"""F-only supplement for the native refusal branch omitted by dry-run extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from observe_launch_refusal_F import CASES, ORACLE, definitions, observe

FROZEN = json.loads((Path(__file__).parent / "fixtures/launch-native-refusal-F.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_original_refusal(case):
    assert observe(case) == next(x for x in FROZEN["cases"] if x["name"] == case["name"])


def test_original_refusal_source():
    assert hashlib.sha256(ORACLE.read_bytes()).hexdigest() == FROZEN["source_sha256"]
    assert definitions() == FROZEN["definitions"]
