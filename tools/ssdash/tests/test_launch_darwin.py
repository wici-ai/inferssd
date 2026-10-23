"""The new Mac path replays the original effects, including partial publication."""

import copy
import json
from pathlib import Path

import pytest
from launch_test_paths import canonical_roots
from observe_launch_candidate import mac_candidate
from observe_launch_F import mac_case

GOLDENS = json.loads((Path(__file__).parent / "fixtures/launch-mac-goldens.json").read_text())


def original_helper_spelling(observation):
    """Normalize only the intentional installed-module entrypoint change.

    Keep original F files and observations immutable. The actual new argv and
    isolated installed-wheel execution are checked independently by alias tests.
    Every argument following the helper entrypoint, all effects, errors and
    partial-publication artifacts still compare against the original records.
    """
    result = copy.deepcopy(observation)
    runs = [event for event in result["events"] if event[0] == "run"]
    assert len(runs) <= 1
    for event in runs:
        command = event[1]
        assert command[:5] == ["@python", "-I", "-m", "ssdash.source.mac_alias", "--repo"]
        assert canonical_roots(command[5]) == "@repo"
        original = [
            "@python",
            command[5] + "/scripts/kimi-build-gguf-expert-alias-tsv.py",
            *command[6:],
        ]
        # CalledProcessError displays the actual command on generation failure;
        # its status and surrounding error text retain the original contract.
        result["stderr"] = result["stderr"].replace(repr(command), repr(original))
        event[1] = original
    return result


@pytest.mark.parametrize("expected", GOLDENS, ids=lambda x: x["case"]["name"])
def test_mac_frozen(expected):
    actual = mac_candidate(expected["case"])
    normalized = original_helper_spelling(actual)
    assert canonical_roots(normalized) == canonical_roots(expected)
    assert normalized == mac_case(expected["case"])
