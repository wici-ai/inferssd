"""The new Mac path replays the original effects, including partial publication."""

import json
from pathlib import Path

import pytest
from launch_test_paths import canonical_roots
from observe_launch_candidate import mac_candidate
from observe_launch_F import mac_case

GOLDENS = json.loads((Path(__file__).parent / "fixtures/launch-mac-goldens.json").read_text())


@pytest.mark.parametrize("expected", GOLDENS, ids=lambda x: x["case"]["name"])
def test_mac_frozen(expected):
    actual = mac_candidate(expected["case"])
    assert canonical_roots(actual) == canonical_roots(expected)
    assert actual == mac_case(expected["case"])
