# SPDX-License-Identifier: Apache-2.0
"""Which of a run's memory samples are written: all of the load, then few unless memory moves."""

from inferssd.platform.worker import (
    SAMPLES_DENSE_SECONDS,
    SAMPLES_SPARSE_SECONDS,
    keep_sample,
)

GIB = 1024**3


def sample(at: float, rss: float = 20 * GIB, available: float = 40 * GIB) -> dict:
    return {"monotonic": at, "rss_bytes": rss, "available_bytes": available}


def kept(samples: list, started: float = 0.0) -> list:
    """The samples of a run that are written, each judged against the last one written."""
    written = None
    result = []
    for value in samples:
        if keep_sample(value, started, written):
            written = value
            result.append(value)
    return result


def test_every_sample_of_the_load_is_kept():
    load = [sample(0.5 * n) for n in range(int(SAMPLES_DENSE_SECONDS * 2) + 1)]
    assert kept(load) == load


def test_a_steady_run_keeps_one_sample_every_thirty_seconds():
    hour = [sample(SAMPLES_DENSE_SECONDS + 0.5 * n) for n in range(1, 2 * 3600 + 1)]
    written = kept([sample(SAMPLES_DENSE_SECONDS)] + hour)[1:]
    assert len(written) == 3600 / SAMPLES_SPARSE_SECONDS
    gaps = {b["monotonic"] - a["monotonic"] for a, b in zip(written, written[1:])}
    assert gaps == {SAMPLES_SPARSE_SECONDS}


def test_memory_that_moves_is_written_as_it_moves():
    # the host's available memory falls by 1 GiB a second from 40 GiB: each second is over 2%
    start = SAMPLES_DENSE_SECONDS + 100
    falling = [sample(start + n, available=(40 - n) * GIB) for n in range(1, 11)]
    assert kept([sample(start)] + falling)[1:] == falling
    # the server's own memory growing counts the same
    growing = [sample(start + n, rss=(20 + n) * GIB) for n in range(1, 4)]
    assert kept([sample(start)] + growing)[1:] == growing
    # under 2% is not a move
    still = [sample(start + n, rss=20.1 * GIB, available=39.9 * GIB) for n in range(1, 11)]
    assert kept([sample(start)] + still) == [sample(start)]


def test_a_sample_without_the_figures_follows_the_clock_alone():
    start = SAMPLES_DENSE_SECONDS + 100
    first = {"monotonic": start}
    assert not keep_sample({"monotonic": start + 1}, 0.0, first)
    assert keep_sample({"monotonic": start + SAMPLES_SPARSE_SECONDS}, 0.0, first)
