# SPDX-License-Identifier: Apache-2.0
"""Native JSONL tails preserve byte offsets and consume one record at a time."""

import io
import json

import pytest

from inferssd.telemetry.observations import LocalObservations, NativeObservationSession


def line(sequence, name, **payload):
    return (
        json.dumps(dict(v=1, seq=sequence, ev=name, **payload), ensure_ascii=False) + "\n"
    ).encode()


class LineReadsOnly(io.BufferedReader):
    def read(self, size=-1):
        raise AssertionError("observation tails must not buffer the whole unread file")


class LineFiles(LocalObservations):
    def open_binary(self, path):
        return LineReadsOnly(io.FileIO(path, "r"))


def test_backlog_is_read_by_record(tmp_path):
    path = tmp_path / "native-events.jsonl"
    with path.open("wb") as stream:
        stream.write(line(0, "ssd.controller.begin", schema=1))
        for sequence in range(1, 1001):
            stream.write(
                line(sequence, "ssd.prefill.graph", requested_tokens=64, configured_max=128)
            )

    reader = NativeObservationSession(files=LineFiles())
    result = reader.read_arm_cached(str(tmp_path), 0)
    assert len(result["pf_graphs"]) == 1000
    assert reader.live[str(path)][1] == path.stat().st_size
    assert reader.read_arm_cached(str(tmp_path), 0) == result


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_unicode_separators_inside_a_json_string_are_not_record_boundaries(tmp_path, separator):
    path = tmp_path / "native-events.jsonl"
    path.write_bytes(
        line(0, "ssd.controller.begin", schema=1)
        + line(1, "ssd.settings", path="model" + separator + "directory")
        + line(2, "ssd.prefill.graph", requested_tokens=64, configured_max=128)
    )

    reader = NativeObservationSession(files=LocalObservations())
    assert len(reader.read_arm_cached(str(tmp_path), 0)["pf_graphs"]) == 1
    assert reader.live[str(path)][0].sequence == 3


def test_partial_utf8_record_resumes_from_its_first_byte(tmp_path):
    path = tmp_path / "native-events.jsonl"
    begin = line(0, "ssd.controller.begin", schema=1)
    partial = line(
        1, "ssd.settings", path="\N{CJK UNIFIED IDEOGRAPH-6A21}\N{CJK UNIFIED IDEOGRAPH-578B}"
    )
    cut = partial.index("\N{CJK UNIFIED IDEOGRAPH-6A21}".encode()) + 1
    path.write_bytes(begin + partial[:cut])

    reader = NativeObservationSession(files=LocalObservations())
    assert reader.read_arm_cached(str(tmp_path), 0)["pf_graphs"] == []
    assert reader.live[str(path)][1] == len(begin)
    with path.open("ab") as stream:
        stream.write(partial[cut:])
        stream.write(line(2, "ssd.prefill.graph", requested_tokens=64, configured_max=128))
    result = reader.read_arm_cached(str(tmp_path), 0)
    assert len(result["pf_graphs"]) == 1
    assert reader.live[str(path)][0].sequence == 3
    assert reader.live[str(path)][1] == path.stat().st_size
    assert reader.read_arm_cached(str(tmp_path), 0) == result


def test_only_newline_terminated_records_are_consumed(tmp_path):
    path = tmp_path / "native-events.jsonl"
    begin = line(0, "ssd.controller.begin", schema=1)
    graph = line(1, "ssd.prefill.graph", requested_tokens=64, configured_max=128)
    path.write_bytes(begin.replace(b"\n", b"\r\n") + graph[:-1])

    reader = NativeObservationSession(files=LocalObservations())
    assert reader.read_arm_cached(str(tmp_path), 0)["pf_graphs"] == []
    with path.open("ab") as stream:
        stream.write(b"\n")
    assert len(reader.read_arm_cached(str(tmp_path), 0)["pf_graphs"]) == 1
