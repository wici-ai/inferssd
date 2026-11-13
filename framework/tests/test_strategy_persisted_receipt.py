# SPDX-License-Identifier: Apache-2.0
"""A saved strategy must explain the settings used by its resolved launch."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from test_launch_strategy import resolved, strategic


@pytest.mark.parametrize(
    "batch,ubatch,kept",
    [
        (1024, None, ["server -b"]),
        (None, 128, ["server -ub"]),
        (1024, 128, ["server -b", "server -ub"]),
        (None, None, []),
    ],
)
def test_saved_receipt_includes_user_batch_choices(tmp_path, batch, ubatch, kept):
    inputs, spec, _, adapter, out = strategic(tmp_path)
    spec = replace(spec, batch=batch, ubatch=ubatch)
    result, output, stderr = resolved(inputs, spec, adapter, out)
    assert result.code == 0 and stderr == ""

    # RecordedArtifacts performs the production filesystem writes. Only the
    # machine observations are recorded; no native server is started here.
    saved = json.loads((out / "strategy.json").read_text(encoding="utf-8"))
    assert saved["kept"] == kept
    assert saved == output["strategy"]
    argv = list(result.argv)
    expected_batch = batch or 1048576
    expected_ubatch = ubatch or expected_batch
    assert argv[argv.index("-b") + 1] == str(expected_batch)
    assert argv[argv.index("-ub") + 1] == str(expected_ubatch)
    assert output["launch"]["batch"] == expected_batch
    assert output["launch"]["ubatch"] == expected_ubatch
