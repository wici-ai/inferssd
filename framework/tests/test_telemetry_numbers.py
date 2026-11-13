# SPDX-License-Identifier: Apache-2.0
"""Keep exact JSON integers while enforcing the native finite-number boundary."""

import sys

import pytest

from inferssd.telemetry.jsonl import read_events


@pytest.mark.parametrize(
    "number",
    [
        0,
        1,
        -1,
        2**53 - 1,
        2**53 + 1,
        -(2**53 + 1),
        2**64 - 1,
        -(2**64),
        2**1023 - 1,
        -(2**1023 - 1),
        2**1023,
        -(2**1023),
        int(sys.float_info.max),
        -int(sys.float_info.max),
        int(sys.float_info.max) + 2**970 - 1,
        -(int(sys.float_info.max) + 2**970 - 1),
    ],
)
def test_finite_integers_keep_every_digit(number):
    line = '{"v":1,"seq":0,"ev":"ssd.number","value":' + str(number) + "}"
    event = next(iter(read_events([line])))
    assert type(event.payload["value"]) is int
    assert event.payload["value"] == number


@pytest.mark.parametrize(
    "number",
    [
        int(sys.float_info.max) + 2**970,
        -(int(sys.float_info.max) + 2**970),
        2**1024,
        -(2**1024),
        10**309,
        -(10**309),
    ],
)
def test_integer_overflow_still_rejects_the_event(number):
    line = '{"v":1,"seq":0,"ev":"ssd.number","value":' + str(number) + "}"
    with pytest.raises(ValueError, match="telemetry line 1: non-finite JSON number"):
        list(read_events([line]))


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e309", "-1.0e309"])
def test_non_finite_constants_and_floats_stay_rejected(value):
    line = '{"v":1,"seq":0,"ev":"ssd.number","value":' + value + "}"
    with pytest.raises(ValueError, match="telemetry line 1: non-finite JSON number"):
        list(read_events([line]))
