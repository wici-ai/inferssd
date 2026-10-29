# SPDX-License-Identifier: Apache-2.0
"""Merging a strategy into typed settings: user values win, dependent values stay consistent."""

from __future__ import annotations

import pytest

from inferssd.configuration.settings import parse, uncomment
from inferssd.launch.typed import materialized_settings, threadpools
from inferssd.launch.types import Threads
from inferssd.strategy.derive import Choice, Inputs, Strategy, launch_conflicts
from inferssd.strategy.merge import apply, server_arguments

COSTS = '"cpu_fixed_us=150.5,cpu_route_us=40.25,drive_gbps=7.5"'


def strategy(platform="linux", threads=Threads(11, 11, (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11))):
    settings = {
        "cpu.io_cpus": Choice('"5"' if threads.cpus else '""', "placement"),
        "cpu.decode_participants": Choice("0", "the engine measures its decode participants"),
        "place.costs": Choice(COSTS, "cpu line; drive at its knee"),
    }
    if platform == "linux":
        settings["vram.budget"] = Choice("0", "no CUDA GPU is usable now")
    else:
        settings["host.budget_mib"] = Choice("8192", "the memory budget")
    return Strategy(
        settings,
        threads,
        ("--ctx-checkpoints", "0"),
        Inputs("00" * 32, ("/m/model.gguf",), platform),
        (),
    )


def lines(snapshot):
    """The generated settings file's entries as {key: (literal, receipt)}."""
    out = {}
    for line in materialized_settings(snapshot).splitlines()[1:]:
        body, receipt = uncomment(line)
        key, _, value = body.partition("=")
        out[key.strip()] = (value.strip(), receipt)
    return out


def test_defaults_take_every_proposal():
    result = apply(strategy(), parse(None, platform="linux"))
    assert (result.kept, result.dropped) == ((), ())
    assert result.threads == strategy().threads
    written = lines(result.settings)
    assert written == {
        "cpu.io_cpus": ('"5"', " source=strategy; input=placement"),
        "cpu.decode_participants": (
            "0",
            " source=strategy; input=the engine measures its decode participants",
        ),
        "place.costs": (
            '"cpu_fixed_us=150.5,cpu_route_us=40.25,drive_gbps=7.5"',
            " source=strategy; input=cpu line; drive at its knee",
        ),
        "vram.budget": ("0", " source=strategy; input=no CUDA GPU is usable now"),
    }
    item = result.settings.at("place.costs")
    assert (item.origin, item.location) == ("file", "strategy")
    assert dict(item.value) == {"cpu_fixed_us": 150.5, "cpu_route_us": 40.25, "drive_gbps": 7.5}
    # The generated file parses back to the same values and receipts.
    again = parse(materialized_settings(result.settings), "settings.ssd", platform="linux")
    for key in written:
        assert again.at(key).value == result.settings.at(key).value
        assert again.at(key).receipt == result.settings.at(key).receipt


def test_file_and_cli_values_are_kept():
    text = 'inferssd-settings 1\nplace.costs = "drive_gbps=3"  # mine\nvram.budget = auto\n'
    user = parse(text, "user.ssd", overrides=[("cpu.decode_participants", "4")], platform="linux")
    result = apply(strategy(), user)
    assert result.kept == ("cpu.decode_participants", "place.costs", "vram.budget")
    assert result.settings.at("place.costs") == user.at("place.costs")
    assert result.settings.at("vram.budget").value == "auto"
    assert result.settings.at("cpu.decode_participants").origin == "cli"
    assert result.settings.at("cpu.io_cpus").location == "strategy"
    # The user's file receipt stays as it was; CLI values stay off the generated file.
    written = lines(result.settings)
    assert written["place.costs"] == ('"drive_gbps=3"', " mine")
    assert "cpu.decode_participants" not in written


def test_darwin_budget_and_platform_check():
    result = apply(strategy("darwin", Threads(10, 10, ())), parse(None, platform="darwin"))
    assert result.settings.at("host.budget_mib").value == 8192
    assert result.threads == Threads(10, 10, ())
    with pytest.raises(ValueError, match="another platform"):
        apply(strategy("darwin", Threads(10, 10, ())), parse(None, platform="linux"))


def check(result, platform="linux"):
    io = result.settings.at("cpu.io_cpus").value
    participants = result.settings.at("cpu.decode_participants").value
    assert launch_conflicts(result.threads, io, participants, platform) == []
    threadpools(result.threads, platform)


def test_user_thread_fields_win_per_field():
    result = apply(strategy(), parse(None, platform="linux"), Threads(decode=8))
    assert result.threads == Threads(8, 11, strategy().threads.cpus)
    assert result.kept == ("threads.decode",)
    check(result)


def test_user_cpus_drop_counts_and_io_that_contradict_them():
    result = apply(strategy(), parse(None, platform="linux"), Threads(cpus=(2, 3, 4, 5)))
    # 11 threads do not fit four strict CPUs, and CPU 5 cannot be both I/O and compute.
    assert result.threads == Threads(None, None, (2, 3, 4, 5))
    assert result.kept == ("threads.cpus",)
    assert result.dropped == ("threads.decode", "threads.batch", "cpu.io_cpus")
    assert result.settings.at("cpu.io_cpus").origin == "default"
    check(result)


def test_user_counts_beyond_the_strategy_cpus_unpin_them():
    result = apply(strategy(), parse(None, platform="linux"), Threads(decode=16, batch=16))
    assert result.threads == Threads(16, 16, ())
    assert result.dropped == ("threads.cpus",)
    check(result)


def test_user_io_cpus_inside_the_strategy_cpus_unpin_them():
    user = parse(None, overrides=[("cpu.io_cpus", "3")], platform="linux")
    result = apply(strategy(), user)
    assert result.threads == Threads(11, 11, ())
    assert result.kept == ("cpu.io_cpus",)
    assert result.dropped == ("threads.cpus",)
    check(result)


def test_user_participants_beyond_the_strategy_threads():
    user = parse(None, overrides=[("cpu.decode_participants", "12")], platform="linux")
    result = apply(strategy(), user)
    # Neither the strategy's 11 threads nor its 11 strict CPUs can hold 12 participants.
    assert result.threads == Threads(None, 11, ())
    assert result.dropped == ("threads.decode", "threads.cpus")
    check(result)


def test_all_placement_merges_without_conflicts():
    unpinned = strategy(threads=Threads(60, 60, ()))
    result = apply(unpinned, parse(None, platform="linux"), Threads(cpus=(0, 1, 2)))
    assert result.threads == Threads(None, None, (0, 1, 2))
    assert result.dropped == ("threads.decode", "threads.batch")
    check(result)


@pytest.mark.parametrize(
    "extra, expected",
    [
        ((), ("--ctx-checkpoints", "0")),
        (("--temp", "0.6"), ("--ctx-checkpoints", "0")),
        (("-ctxcp", "4"), ()),
        (("--ctx_checkpoints=2",), ()),
        (("--swa-checkpoints", "1"), ()),
    ],
)
def test_server_arguments_yield_to_the_user(extra, expected):
    assert server_arguments(strategy(), extra) == expected
